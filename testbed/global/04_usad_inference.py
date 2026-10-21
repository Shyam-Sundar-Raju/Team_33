import time
import json
import struct
import numpy as np
import joblib
import torch
import torch.nn as nn
from pymodbus.client import ModbusTcpClient

# --- Configuration ---
POLL_INTERVAL = 0.1  # seconds
MAPPING_FILE = '../mapping.json'
MODEL_PATH = 'models/usad_swat_model_full.pth'
SCALER_PATH = 'models/swat_scaler.pkl'

GLOBAL_FEATURES = [
    'MV101', 'FIT201', 'FIT301', 'FIT401', 'FIT501', 'FIT601', 
    'LIT101', 'LIT301', 'LIT401', 'AIT201', 'AIT202', 'AIT203', 
    'AIT402', 'AIT502', 'AIT503', 'AIT504', 'DPIT301', 'P101', 
    'MV201', 'P301', 'P401', 'P501', 'P602'
]

# Note: The threshold was NOT saved by the training notebook.
# You will need to determine the correct threshold and update this variable.
THRESHOLD = 0.1
# ---------------------

# --- Model Classes ---
input_dim = 23
hidden_dim = 25
latent_dim = 12

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)
        

class USAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        z2 = self.encoder(w1)
        w3 = self.decoder2(z2)
        return w1, w2, w3

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
            
    for f in GLOBAL_FEATURES:
        if f not in flat_mapping:
            raise ValueError(f"Feature {f} required by global model but not found in mapping.json.")
            
    print(f"Loading Scaler from {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    
    print(f"Loading USAD model from {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    
    return flat_mapping, model, scaler, device

def read_modbus(flat_mapping, features, plcs):
    raw_values = []
    for feat in features:
        config = flat_mapping[feat]
        ip = config['plc_ip']
        address = config['address']
        is_discrete = config['is_discrete']
        
        client = plcs[ip]
        if is_discrete:
            res = client.read_coils(address=address, count=1)
            if res.isError():
                raw_values.append(0.0)
            else:
                raw_values.append(1.0 if res.bits[0] else 0.0)
        else:
            res = client.read_holding_registers(address=address, count=2)
            if res.isError():
                raw_values.append(0.0)
            else:
                packed = struct.pack('>HH', res.registers[0], res.registers[1])
                val = struct.unpack('>f', packed)[0]
                raw_values.append(val)
                
    return np.array(raw_values)

import socket

def main():
    fusion_ip = '127.0.0.1'
    fusion_port = 50000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    flat_mapping, model, scaler, device = load_mapping_and_model()
    
    ips = list(set([flat_mapping[f]['plc_ip'] for f in GLOBAL_FEATURES]))
    plcs = {}
    for ip in ips:
        client = ModbusTcpClient(ip, port=502)
        client.connect()
        plcs[ip] = client
        print(f"Connected to PLC {ip}")
        
    print(f"Starting Global USAD inference every {POLL_INTERVAL}s. Press Ctrl+C to stop.")
    
    try:
        while True:
            start_time = time.time()
            
            raw_values = read_modbus(flat_mapping, GLOBAL_FEATURES, plcs)
            
            # Scale
            scaled_values = scaler.transform([raw_values])
            
            # Predict
            x_tensor = torch.tensor(scaled_values, dtype=torch.float32).to(device)
            with torch.no_grad():
                w1, w2, w3 = model(x_tensor)
                error1 = torch.mean((x_tensor - w1) ** 2, dim=1)
                error2 = torch.mean((x_tensor - w3) ** 2, dim=1)
                anomaly_score = 0.7 * error1 + 0.3 * error2
                score_val = anomaly_score.item()
                
            is_anomaly = score_val > THRESHOLD
            status_str = "ATTACK DETECTED!" if is_anomaly else "NORMAL"
            print(f"[{time.strftime('%H:%M:%S')}] Score: {score_val:.5f} / Thresh: {THRESHOLD:.5f} | Status: {status_str}")
            payload = json.dumps({"source": "global", "is_anomaly": int(is_anomaly)})
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
