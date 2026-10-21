import time
import json
import struct
import numpy as np
from pymodbus.client import ModbusTcpClient

# --- Configuration ---
STAGE = 1
POLL_INTERVAL = 0.1  # seconds
MAPPING_FILE = 'mapping_by_stage.json'
MODEL_PATH = f'../models/cusum_ewma_stage{STAGE}.json'
LAMBDA_EWMA = 0.2
# ---------------------

def load_mapping_and_model():
    with open(MAPPING_FILE, 'r') as f:
        mapping_data = json.load(f)
    stage_mapping = mapping_data[str(STAGE)]
    
    print(f"Loading CUSUM/EWMA stats from {MODEL_PATH}")
    with open(MODEL_PATH, 'r') as f:
        stats = json.load(f)
        
    # The JSON only contains continuous columns, which is what CUSUM/EWMA uses
    features = sorted(list(stats.keys()))
    
    for f in features:
        if f not in stage_mapping:
            raise ValueError(f"Feature {f} required by model but not found in mapping for stage {STAGE}.")
            
    return stage_mapping, stats, features

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
    stage_mapping, stats, features = load_mapping_and_model()
    
    ips = list(set([stage_mapping[f]['plc_ip'] for f in features]))
    plcs = {}
    for ip in ips:
        client = ModbusTcpClient(ip, port=502)
        client.connect()
        plcs[ip] = client
        print(f"Connected to PLC {ip}")
        
    print(f"Starting CUSUM & EWMA inference for Stage {STAGE} every {POLL_INTERVAL}s. Press Ctrl+C to stop.")
    
    # Initialize states
    z_ewma = {f: stats[f]['mu'] for f in features}
    s_high = {f: 0.0 for f in features}
    s_low = {f: 0.0 for f in features}
    
    try:
        while True:
            start_time = time.time()
            
            raw_values = read_modbus(stage_mapping, features, plcs)
            
            ewma_anomalies = []
            cusum_anomalies = []
            
            for feat in features:
                x = raw_values[feat]
                st = stats[feat]
                
                # --- EWMA ---
                z_ewma[feat] = LAMBDA_EWMA * x + (1 - LAMBDA_EWMA) * z_ewma[feat]
                dev = abs(z_ewma[feat] - st['mu'])
                ewma_score = dev / (st['ewma_limit'] + 1e-9)
                if ewma_score > 1.0:
                    ewma_anomalies.append(feat)
                    
                # --- CUSUM ---
                s_high[feat] = max(0.0, s_high[feat] + (x - st['mu'] - st['k']))
                s_low[feat] = max(0.0, s_low[feat] + (st['mu'] - st['k'] - x))
                max_s = max(s_high[feat], s_low[feat])
                cusum_score = max_s / (st['h'] + 1e-9)
                if cusum_score > 1.0:
                    cusum_anomalies.append(feat)
            
            is_anomaly = len(ewma_anomalies) > 0 or len(cusum_anomalies) > 0
            
            if is_anomaly:
                print(f"[{time.strftime('%H:%M:%S')}] ATTACK DETECTED! EWMA flags: {ewma_anomalies} | CUSUM flags: {cusum_anomalies}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Status: NORMAL")
                
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
