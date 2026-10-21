import time
import json
import struct
import numpy as np
import joblib
from pymodbus.client import ModbusTcpClient

# --- Configuration ---
STAGE = 6
POLL_INTERVAL = 0.1  # seconds
MAPPING_FILE = 'mapping_by_stage.json'
MODEL_PATH = f'./models/markov_stage{STAGE}.pkl'
# ---------------------

def load_mapping_and_model():
    # Load mapping
    with open(MAPPING_FILE, 'r') as f:
        mapping_data = json.load(f)
    stage_mapping = mapping_data[str(STAGE)]
    
    # Load model
    print(f"Loading Markov model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    # We must use exactly the columns used during training in the exact order
    discrete_cols = model['discrete_cols']
    continuous_cols = model['continuous_cols']
    features = discrete_cols + continuous_cols
    
    # Verify all required features are in the mapping
    for f in features:
        if f not in stage_mapping:
            raise ValueError(f"Feature {f} required by model but not found in mapping for stage {STAGE}.")
            
    return stage_mapping, model, features, discrete_cols, continuous_cols

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

def discretize_state(raw_values, features, discrete_cols, continuous_cols, bin_edges):
    state_parts = []
    
    # Process discrete first
    for col in discrete_cols:
        val = int(raw_values[col])
        state_parts.append(str(val))
        
    # Process continuous
    for col in continuous_cols:
        val = raw_values[col]
        edges = bin_edges[col]
        # Equivalent to KBinsDiscretizer(strategy='quantile', encode='ordinal')
        # We find which bin it falls into. 
        # digitize returns 1-indexed bin, we want 0-indexed, and cap at max bin
        bin_idx = np.digitize(val, edges[1:-1])
        state_parts.append(str(bin_idx))
        
    return "|".join(state_parts)

import socket

def main():
    fusion_ip = '192.168.200.170'
    fusion_port = 50000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stage_mapping, model, features, discrete_cols, continuous_cols = load_mapping_and_model()
    
    # Setup PLC clients
    ips = list(set([v['plc_ip'] for v in stage_mapping.values()]))
    plcs = {}
    for ip in ips:
        client = ModbusTcpClient(ip, port=502)
        client.connect()
        plcs[ip] = client
        print(f"Connected to PLC {ip}")
        
    print(f"Starting inference for Stage {STAGE} every {POLL_INTERVAL}s. Press Ctrl+C to stop.")
    
    # Retrieve model parameters (using 1st order)
    s2i = model['s2i_1']
    log_p_raw = model['log_p_1']
    if isinstance(log_p_raw, tuple):
        log_p, fallback_prob = log_p_raw
        is_dense = False
    else:
        log_p = log_p_raw
        fallback_prob = log_p.min() if len(log_p) > 0 else -10.0
        is_dense = True
    thresh = model['thresh_1']
    nll_min, nll_max = model['nll_min_1'], model['nll_max_1']
    dwell_bounds = model['dwell_bounds']
    global_bound = model['global_bound']
    window_size = model['window_size']
    bin_edges = model['bin_edges']
    
    state_history = []
    
    try:
        while True:
            start_time = time.time()
            
            # 1. Read Modbus
            raw_values = read_modbus(stage_mapping, features, plcs)
            
            # 2. Convert to discrete state string
            curr_state = discretize_state(raw_values, features, discrete_cols, continuous_cols, bin_edges)
            state_history.append(curr_state)
            
            # Keep history bounded (we need window_size + enough for dwell time check)
            # Actually, dwell time requires arbitrary history. Let's keep a reasonable buffer (e.g. 1000)
            if len(state_history) > 1000:
                state_history.pop(0)
                
            if len(state_history) < 2:
                time.sleep(max(0, POLL_INTERVAL - (time.time() - start_time)))
                continue
                
            # 3. Anomaly Inference
            # Dwell time
            duration = 1
            for j in range(len(state_history)-2, -1, -1):
                if state_history[j] == curr_state:
                    duration += 1
                else:
                    break
                    
            bound = dwell_bounds.get(curr_state, {}).get('bound', global_bound)
            dwell_anomaly = duration > bound
            
            # Sequence likelihood (over window_size)
            window = state_history[-window_size:] if len(state_history) >= window_size else state_history
            nll = 0
            for j in range(1, len(window)):
                s_prev, s_curr = window[j-1], window[j]
                if is_dense:
                    if s_prev in s2i and s_curr in s2i:
                        nll -= log_p[s2i[s_prev], s2i[s_curr]]
                    else:
                        nll -= fallback_prob
                else:
                    if s_prev in log_p:
                        nll -= log_p[s_prev].get(s_curr, log_p[s_prev].get('__unseen__', fallback_prob))
                    else:
                        nll -= fallback_prob
                    
            nll_norm = max(0, (nll - nll_min) / (nll_max - nll_min + 1e-9))
            seq_anomaly = nll_norm > thresh
            
            is_anomaly = dwell_anomaly or seq_anomaly
            
            status_str = "ATTACK DETECTED!" if is_anomaly else "NORMAL"
            print(f"[{time.strftime('%H:%M:%S')}] State: {curr_state} | NLL_norm: {nll_norm:.3f} | Dwell: {duration}/{bound:.0f} | Status: {status_str}")
            
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
