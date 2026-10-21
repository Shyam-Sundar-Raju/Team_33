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
MODEL_PATH = 'models/vae_model_weights.pth'
SCALER_PATH = 'models/vae_scaler.pkl'
META_PATH = 'models/vae_metadata.pkl'

GLOBAL_FEATURES = [
    'MV101', 'FIT201', 'FIT301', 'FIT401', 'FIT501', 'FIT601', 
    'LIT101', 'LIT301', 'LIT401', 'AIT201', 'AIT202', 'AIT203', 
    'AIT402', 'AIT502', 'AIT503', 'AIT504', 'DPIT301', 'P101', 
    'MV201', 'P301', 'P401', 'P501', 'P602'
]
# ---------------------

# --- Model Class ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return self.decoder(z), mu, logvar
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
    
    print(f"Loading VAE weights from {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=len(GLOBAL_FEATURES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print(f"Loading Metadata from {META_PATH}")
    metadata = joblib.load(META_PATH)
    threshold = metadata['threshold']
    
    return flat_mapping, model, scaler, threshold, device

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
    flat_mapping, model, scaler, threshold, device = load_mapping_and_model()
    
    ips = list(set([flat_mapping[f]['plc_ip'] for f in GLOBAL_FEATURES]))
    plcs = {}
    for ip in ips:
        client = ModbusTcpClient(ip, port=502)
        client.connect()
        plcs[ip] = client
        print(f"Connected to PLC {ip}")
        
    print(f"Starting Global VAE inference every {POLL_INTERVAL}s. Press Ctrl+C to stop.")
    
    try:
        while True:
            start_time = time.time()
            
            raw_values = read_modbus(flat_mapping, GLOBAL_FEATURES, plcs)
            
            # Scale
            scaled_values = scaler.transform([raw_values])
            
            # Predict
            x_tensor = torch.tensor(scaled_values, dtype=torch.float32).to(device)
            with torch.no_grad():
                recon, _, _ = model(x_tensor)
                error = torch.mean((x_tensor - recon) ** 2, dim=1)
                score_val = error.item()
                
            is_anomaly = score_val > threshold
            status_str = "ATTACK DETECTED!" if is_anomaly else "NORMAL"
            print(f"[{time.strftime('%H:%M:%S')}] Score: {score_val:.5f} / Thresh: {threshold:.5f} | Status: {status_str}")
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
