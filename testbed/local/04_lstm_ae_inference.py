import time
import json
import struct
import numpy as np
import joblib
import torch
import torch.nn as nn
from pymodbus.client import ModbusTcpClient

# --- Configuration ---
STAGE = 1
POLL_INTERVAL = 0.1  # seconds
MAPPING_FILE = 'mapping_by_stage.json'
MODEL_PATH = f'../models/lstm_ae_stage{STAGE}.pt'
SCALER_PATH = f'../models/scaler_stage{STAGE}.pkl'
T = 60
LSTM_HIDDEN_SIZE = 32
LATENT_SIZE = 8
NUM_LAYERS = 1
# Note: Threshold was not explicitly saved in the training notebook artifacts for LSTM,
# so we use a reasonable default. In a real scenario, this would be loaded.
THRESHOLD = 0.05 
# ---------------------

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_size)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n_last = h_n[-1]
        latent = self.fc(h_n_last)
        return latent

class DecoderLSTM(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers=1):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, seq_len):
        x = x.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, latent_size, num_layers)
        self.decoder = DecoderLSTM(latent_size, hidden_size, input_size, num_layers)
        
    def forward(self, x):
        seq_len = x.size(1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent, seq_len)
        return reconstructed

def load_mapping_and_model():
    with open(MAPPING_FILE, 'r') as f:
        mapping_data = json.load(f)
    stage_mapping = mapping_data[str(STAGE)]
    
    # Sort alphabetically to exactly match the training CSV column order!
    features = sorted(list(stage_mapping.keys()))
    input_size = len(features)
    
    print(f"Loading Scaler from {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    
    print(f"Loading LSTM model from {MODEL_PATH}")
    model = LSTMAutoencoder(input_size, LSTM_HIDDEN_SIZE, LATENT_SIZE, NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    return stage_mapping, model, scaler, features, input_size

def read_modbus(stage_mapping, features, plcs):
    raw_values = []
    for feat in features:
        config = stage_mapping[feat]
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

import socket

def main():
    fusion_ip = '127.0.0.1'
    fusion_port = 50000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stage_mapping, model, scaler, features, input_size = load_mapping_and_model()
    
    ips = list(set([stage_mapping[f]['plc_ip'] for f in features]))
    plcs = {}
    for ip in ips:
        client = ModbusTcpClient(ip, port=502)
        client.connect()
        plcs[ip] = client
        print(f"Connected to PLC {ip}")
        
    print(f"Starting LSTM-AE inference for Stage {STAGE} every {POLL_INTERVAL}s. Press Ctrl+C to stop.")
    
    history = []
    
    try:
        while True:
            start_time = time.time()
            
            raw_values = read_modbus(stage_mapping, features, plcs)
            
            # Scale
            scaled_values = scaler.transform([raw_values])[0]
            
            history.append(scaled_values)
            if len(history) > T:
                history.pop(0)
                
            if len(history) == T:
                # Prepare tensor
                window = torch.tensor([history], dtype=torch.float32)
                
                with torch.no_grad():
                    reconstructed = model(window)
                    
                    # MSE over features, shape (batch, T)
                    loss = torch.mean((window - reconstructed)**2, dim=-1)
                    
                    # For this timestep (the last one in the window), we can use the window average 
                    # or just the last element. The notebook averaged over windows. We'll use the window mean.
                    mean_mse = loss.mean().item()
                    
                is_anomaly = mean_mse > THRESHOLD
                status_str = "ATTACK DETECTED!" if is_anomaly else "NORMAL"
                print(f"[{time.strftime('%H:%M:%S')}] MSE: {mean_mse:.5f} | Status: {status_str}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Buffering... {len(history)}/{T}")
                
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
