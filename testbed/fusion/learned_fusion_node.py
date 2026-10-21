import socket
import json
import time
import threading
import torch
import torch.nn as nn
import numpy as np

FUSION_IP = '0.0.0.0'
FUSION_PORT = 50000
POLL_INTERVAL = 0.1
MODEL_PATH = 'models/learned_fusion_model.pth'

class TopologyAwareFusion(nn.Module):
    def __init__(self, n_stages=6, hidden_dim=8, window=5, adjacency=None):
        super().__init__()
        self.window = window
        self.stage_encoder = nn.Linear(window, hidden_dim)
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer('adjacency', adjacency)
        self.global_weight = nn.Parameter(torch.tensor(1.0))
        self.stage_weights = nn.Linear(hidden_dim * n_stages, 1)

    def forward(self, global_score, stage_score_windows):
        h = torch.relu(self.stage_encoder(stage_score_windows))
        q, k = self.attn_q(h), self.attn_k(h)
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / (h.size(-1) ** 0.5)
        mask = self.adjacency.unsqueeze(0)
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        alpha = torch.softmax(attn_logits, dim=-1)
        h_prop = h + torch.bmm(alpha, h)
        h_flat = h_prop.flatten(start_dim=1)
        stage_contrib = self.stage_weights(h_flat)
        final_score = self.global_weight * global_score + stage_contrib.squeeze(-1)
        return final_score, alpha

state = {
    "global_flag": 0,
    "local_flags": [0, 0, 0, 0, 0, 0],
    "last_update": time.time()
}

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((FUSION_IP, FUSION_PORT))
    print(f"[*] Learned Fusion UDP Listener started on {FUSION_IP}:{FUSION_PORT}")
    
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            payload = json.loads(data.decode('utf-8'))
            
            is_anomaly = int(payload.get('is_anomaly', 0))
            
            if payload.get('source') == 'global':
                state['global_flag'] = is_anomaly
            elif payload.get('source') == 'local':
                stage = int(payload.get('stage', 1))
                if 1 <= stage <= 6:
                    state['local_flags'][stage - 1] = is_anomaly
                    
            state['last_update'] = time.time()
        except Exception as e:
            pass

def main():
    print(f"Loading TopologyAwareFusion model from {MODEL_PATH}...")
    adjacency = torch.ones(6, 6)
    model = TopologyAwareFusion(n_stages=6, hidden_dim=8, window=5, adjacency=adjacency)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    listener = threading.Thread(target=udp_listener, daemon=True)
    listener.start()
    
    print(f"[*] Starting Learned Fusion Inference every {POLL_INTERVAL}s. Press Ctrl+C to stop.")
    
    window_size = 5
    local_history = []
    
    try:
        while True:
            start_time = time.time()
            
            # Decay old state if we haven't received updates in 2 seconds
            if time.time() - state['last_update'] > 2.0:
                state['global_flag'] = 0
                state['local_flags'] = [0, 0, 0, 0, 0, 0]
                
            current_local = list(state['local_flags'])
            local_history.append(current_local)
            
            if len(local_history) > window_size:
                local_history.pop(0)
                
            # If buffer isn't full yet, just pad with zeros
            if len(local_history) < window_size:
                pad_len = window_size - len(local_history)
                padded_history = [[0]*6]*pad_len + local_history
            else:
                padded_history = local_history
                
            # Convert to tensor: shape [batch=1, n_stages=6, window=5]
            # padded_history is currently [window, n_stages]
            history_np = np.array(padded_history).T # [n_stages, window]
            X_local = torch.tensor([history_np], dtype=torch.float32)
            X_global = torch.tensor([state['global_flag']], dtype=torch.float32)
            
            with torch.no_grad():
                logits, alpha = model(X_global, X_local)
                prob = torch.sigmoid(logits).item()
                
            is_anomaly = prob >= 0.5
            status_str = "ATTACK DETECTED!" if is_anomaly else "NORMAL"
            
            print(f"[{time.strftime('%H:%M:%S')}] Global: {state['global_flag']} | Local: {state['local_flags']} | Prob: {prob:.4f} | Status: {status_str}")
            
            elapsed = time.time() - start_time
            sleep_t = max(0, POLL_INTERVAL - elapsed)
            time.sleep(sleep_t)
            
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == '__main__':
    main()
