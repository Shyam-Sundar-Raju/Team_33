import time
import json
import struct
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from pymodbus.client import ModbusTcpClient

# --- Configuration ---
POLL_INTERVAL = 0.1  # seconds
MAPPING_FILE = '../mapping.json'
MODEL_PATH = 'models/autoencoder_model.keras'
SCALER_PATH = 'models/autoencoder_scaler.pkl'
META_PATH = 'models/autoencoder_metadata.pkl'

BASELINE_FEATURES = [
    "FIT101", "LIT101", "MV101", "P101", "P102", "AIT201", "AIT202", "AIT203",
    "FIT201", "MV201", "P201", "P202", "P203", "P204", "P205", "P206",
    "DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304", "P301",
    "P302", "AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402", "P403",
    "P404", "UV401", "AIT501", "AIT502", "AIT503", "AIT504", "FIT501", "FIT502",
    "FIT503", "FIT504", "P501", "P502", "PIT501", "PIT502", "PIT503", "FIT601",
    "P601", "P602", "P603"
]
# ---------------------

def load_mapping_and_model():
    with open(MAPPING_FILE, 'r') as f:
        mapping_data = json.load(f)
        
    flat_mapping = {}
    for ip, plc_data in mapping_data['plcs'].items():
        for coil in plc_data.get('coils', []):
            flat_mapping[coil['feature']] = {'plc_ip': ip, 'address': coil['address'], 'is_discrete': True}
        for hr in plc_data.get('holding_registers', []):
            flat_mapping[hr['feature']] = {'plc_ip': ip, 'address': hr['address'], 'is_discrete': False}
            
    for f in BASELINE_FEATURES:
        if f not in flat_mapping:
            raise ValueError(f"Feature {f} required by baseline model but not found in mapping.json.")
            
    print(f"Loading Scaler from {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    
    print(f"Loading Autoencoder model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    
    print(f"Loading Metadata from {META_PATH}")
    metadata = joblib.load(META_PATH)
    threshold = metadata['threshold']
    
    return flat_mapping, model, scaler, threshold

def read_modbus(flat_mapping, features, plcs):
    raw_values = []
    for feat in features:
        config = flat_mapping[feat]
        ip = config['plc_ip']
        addr = config['address']
        is_discrete = config['is_discrete']
        
        client = plcs[ip]
        if is_discrete:
            res = client.read_coils(address=addr, count=1)
            if res.isError():
                raw_values.append(0.0)
            else:
                raw_values.append(1.0 if res.bits[0] else 0.0)
        else:
            res = client.read_holding_registers(address=addr, count=2)
            if res.isError():
                raw_values.append(0.0)
            else:
                packed = struct.pack('>HH', res.registers[0], res.registers[1])
                val = struct.unpack('>f', packed)[0]
                raw_values.append(val)
                
    return np.array(raw_values)

def main():
    flat_mapping, model, scaler, threshold = load_mapping_and_model()
    
    ips = list(set([flat_mapping[f]['plc_ip'] for f in BASELINE_FEATURES]))
    plcs = {}
    for ip in ips:
        client = ModbusTcpClient(ip, port=502)
        client.connect()
        plcs[ip] = client
        print(f"Connected to PLC {ip}")
        
    print(f"Starting Baseline Autoencoder inference every {POLL_INTERVAL}s. Press Ctrl+C to stop.")
    
    try:
        while True:
            start_time = time.time()
            
            raw_values = read_modbus(flat_mapping, BASELINE_FEATURES, plcs)
            
            # Scale
            scaled_values = scaler.transform([raw_values])
            
            # Predict
            reconstructed = model.predict(scaled_values, verbose=0)
            mse = np.mean(np.power(scaled_values - reconstructed, 2), axis=1)[0]
            
            is_anomaly = mse > threshold
            status_str = "ATTACK DETECTED!" if is_anomaly else "NORMAL"
            print(f"[{time.strftime('%H:%M:%S')}] MSE: {mse:.5f} / Thresh: {threshold:.5f} | Status: {status_str}")
            
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
