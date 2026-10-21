import time
import json
import struct
from pymodbus.client import ModbusTcpClient

# --- Configuration ---
STAGE = 1
POLL_INTERVAL = 0.1  # seconds
MAPPING_FILE = 'mapping_by_stage.json'
MODEL_PATH = f'../models/invariants_stage{STAGE}.json'
# ---------------------

# --- Invariant Rules ---
def evaluate_invariants_stage1(row, prev_row, thresh):
    violations = []
    if prev_row is not None:
        if row.get('MV101') == 1 and (row.get('LIT101', 0) - prev_row.get('LIT101', 0)) > thresh.get('LIT101_max_increase', 10):
            violations.append("LIT101 increasing while MV101 is closed")
    if row.get('LIT101', 0) < thresh.get('LIT101_min', 0) or row.get('LIT101', 0) > thresh.get('LIT101_max', 1000):
        violations.append("LIT101 out of calibrated normal bounds")
    return len(violations) > 0, violations

def evaluate_invariants_stage2(row, prev_row, thresh):
    violations = []
    if row.get('AIT202', 0) < thresh.get('AIT202_min', 0) or row.get('AIT202', 0) > thresh.get('AIT202_max', 100):
        violations.append("AIT202 out of calibrated normal bounds")
    return len(violations) > 0, violations

def evaluate_invariants_stage3(row, prev_row, thresh):
    violations = []
    if row.get('LIT301', 1000) < thresh.get('LIT301_low', 0) and row.get('MV301') == 1:
        violations.append("MV301 is closed while LIT301 is extremely low")
    if row.get('MV304', 1) == 1 and row.get('DPIT301', 0) > thresh.get('DPIT301_max', 100):
        violations.append("DPIT301 exceeds max normal during non-backwash")
    return len(violations) > 0, violations

def evaluate_invariants_stage4(row, prev_row, thresh):
    violations = []
    if row.get('LIT401', 0) < thresh.get('LIT401_min', 0) or row.get('LIT401', 0) > thresh.get('LIT401_max', 1000):
        violations.append("LIT401 out of calibrated normal bounds")
    if row.get('P401') == 2 and row.get('FIT401', 0) < thresh.get('FIT401_min_on', 0):
        violations.append("FIT401 too low while P401 is ON")
    return len(violations) > 0, violations

def evaluate_invariants_stage5(row, prev_row, thresh):
    violations = []
    if row.get('AIT504', 0) < thresh.get('AIT504_min', 0) or row.get('AIT504', 0) > thresh.get('AIT504_max', 100):
        violations.append("AIT504 out of calibrated normal bounds")
    if row.get('P501') == 2 and row.get('FIT501', 0) < thresh.get('FIT501_min_on', 0):
        violations.append("FIT501 too low while P501 is ON")
    return len(violations) > 0, violations

def evaluate_invariants_stage6(row, prev_row, thresh):
    violations = []
    if row.get('P601') == 2 or row.get('P602') == 2:
        if row.get('FIT601', 0) < thresh.get('FIT601_min_on', 0) or row.get('FIT601', 0) > thresh.get('FIT601_max_on', 1000):
            violations.append("FIT601 out of normal bounds while P601/P602 ON")
    if row.get('P601', 1) == 1 and row.get('P602', 1) == 1:
        if row.get('FIT601', 0) > thresh.get('FIT601_max_off', 0):
            violations.append("FIT601 exceeds off-state max while P601/P602 OFF")
    return len(violations) > 0, violations

STAGE_INVARIANTS = {
    1: evaluate_invariants_stage1,
    2: evaluate_invariants_stage2,
    3: evaluate_invariants_stage3,
    4: evaluate_invariants_stage4,
    5: evaluate_invariants_stage5,
    6: evaluate_invariants_stage6
}
# ------------------------

def load_mapping_and_model():
    with open(MAPPING_FILE, 'r') as f:
        mapping_data = json.load(f)
    stage_mapping = mapping_data[str(STAGE)]
    
    print(f"Loading Invariant thresholds from {MODEL_PATH}")
    with open(MODEL_PATH, 'r') as f:
        thresh = json.load(f)
        
    features = sorted(list(stage_mapping.keys()))
    return stage_mapping, thresh, features

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
    stage_mapping, thresh, features = load_mapping_and_model()
    
    # Setup PLC clients
    ips = list(set([stage_mapping[f]['plc_ip'] for f in features]))
    plcs = {}
    for ip in ips:
        client = ModbusTcpClient(ip, port=502)
        client.connect()
        plcs[ip] = client
        print(f"Connected to PLC {ip}")
        
    print(f"Starting Invariant inference for Stage {STAGE} every {POLL_INTERVAL}s. Press Ctrl+C to stop.")
    
    eval_func = STAGE_INVARIANTS[STAGE]
    prev_row = None
    
    try:
        while True:
            start_time = time.time()
            
            raw_values = read_modbus(stage_mapping, features, plcs)
            
            is_anomaly, violations = eval_func(raw_values, prev_row, thresh)
            
            if is_anomaly:
                print(f"[{time.strftime('%H:%M:%S')}] ATTACK DETECTED! Violations: {violations}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Status: NORMAL")
                
            prev_row = raw_values
            
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
