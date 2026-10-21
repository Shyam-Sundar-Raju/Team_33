import time
import json
import struct
import numpy as np
import joblib
from scipy.stats import chi2
from pymodbus.client import ModbusTcpClient

# --- Configuration ---
STAGE = 1
POLL_INTERVAL = 0.1  # seconds
MAPPING_FILE = 'mapping_by_stage.json'
MODEL_PATH = f'../models/pasad_stage{STAGE}.pkl'
# ---------------------

def load_mapping_and_model():
    with open(MAPPING_FILE, 'r') as f:
        mapping_data = json.load(f)
    stage_mapping = mapping_data[str(STAGE)]
    
    print(f"Loading PASAD model from {MODEL_PATH}")
    tag_models = joblib.load(MODEL_PATH)
    
    # In PASAD, we only process continuous tags, which are the keys of tag_models
    features = sorted(list(tag_models.keys()))
    
    # Verify mapping
    for f in features:
        if f not in stage_mapping:
            raise ValueError(f"Feature {f} required by model but not found in mapping for stage {STAGE}.")
            
    return stage_mapping, tag_models, features

def read_modbus(stage_mapping, features, plcs):
    raw_values = {}
    for feat in features:
        config = stage_mapping[feat]
        ip = config['plc_ip']
        addr = config['address']
        is_discrete = config['is_discrete']
        
        client = plcs[ip]
        if is_discrete:
            res = client.read_coils(address=addr, count=1)
            if res.isError():
                raw_values[feat] = 0.0
            else:
                raw_values[feat] = 1.0 if res.bits[0] else 0.0
        else:
            res = client.read_holding_registers(address=addr, count=2)
            if res.isError():
                raw_values[feat] = 0.0
            else:
                packed = struct.pack('>HH', res.registers[0], res.registers[1])
                val = struct.unpack('>f', packed)[0]
                raw_values[feat] = val
                
    return raw_values

import socket

def main():
    fusion_ip = '127.0.0.1'
    fusion_port = 50000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stage_mapping, tag_models, features = load_mapping_and_model()
    
    # Setup PLC clients
    ips = list(set([stage_mapping[f]['plc_ip'] for f in features]))
    plcs = {}
    for ip in ips:
        client = ModbusTcpClient(ip, port=502)
        client.connect()
        plcs[ip] = client
        print(f"Connected to PLC {ip}")
        
    print(f"Starting EPASAD inference for Stage {STAGE} every {POLL_INTERVAL}s. Press Ctrl+C to stop.")
    
    # Initialize history buffers per tag
    # L is the window size. It might be different per tag.
    # tag_models[tag]['U_r'] has shape (L, r)
    history = {}
    L_dict = {}
    for tag in features:
        U_r = tag_models[tag]['U_r']
        L = U_r.shape[0]
        L_dict[tag] = L
        history[tag] = []
        
    try:
        while True:
            start_time = time.time()
            
            # Read Modbus
            raw_values = read_modbus(stage_mapping, features, plcs)
            
            # Process each tag independently
            tag_anomalies = {}
            for tag in features:
                val = raw_values[tag]
                
                # Scale value
                scaler = tag_models[tag]['scaler']
                # scaler expects 2D array
                val_scaled = scaler.transform([[val]])[0][0]
                
                history[tag].append(val_scaled)
                L = L_dict[tag]
                
                if len(history[tag]) > L:
                    history[tag].pop(0)
                    
                if len(history[tag]) == L:
                    window = np.array(history[tag])
                    U_r = tag_models[tag]['U_r']
                    centroid = tag_models[tag]['centroid']
                    cov_inv = np.linalg.inv(tag_models[tag]['cov']) # fallback if cov_inv not saved explicitly, but it should be
                    # Actually wait, let's check if cov_inv was saved. The training script returned it but maybe it wasn't saved.
                    # The script says: tag_models[tag] = {'U_r': U_r, 'centroid': centroid, 'cov': cov, 'scaler': scalers[tag], ...}
                    # It saved 'cov', not 'cov_inv'.
                    cov = tag_models[tag]['cov']
                    cov_inv = np.linalg.inv(cov)
                    
                    r = U_r.shape[1]
                    
                    # Project
                    proj = U_r.T @ window
                    diff = proj - centroid
                    
                    # EPASAD (Mahalanobis)
                    dist_epasad = diff.T @ cov_inv @ diff
                    
                    # We use Chi-Square distribution for Mahalanobis thresholding since min/max were not saved
                    chi2_thresh = chi2.ppf(0.999, r)
                    
                    if dist_epasad > chi2_thresh:
                        tag_anomalies[tag] = dist_epasad
            
            is_anomaly = len(tag_anomalies) > 0
            
            if any(len(history[tag]) == L_dict[tag] for tag in features):
                status_str = f"ATTACK DETECTED! Anomalous tags: {list(tag_anomalies.keys())}" if is_anomaly else "NORMAL"
                print(f"[{time.strftime('%H:%M:%S')}] Status: {status_str}")
            
            payload = json.dumps({"source": "local", "stage": STAGE, "is_anomaly": int(is_anomaly)})
            sock.sendto(payload.encode("utf-8"), (fusion_ip, fusion_port))
            elapsed = time.time() - start_time
            sleep_t = max(0, POLL_INTERVAL - elapsed)
            time.sleep(sleep_t)
            
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        for client in plcs.values():
            client.close()

if __name__ == '__main__':
    main()
