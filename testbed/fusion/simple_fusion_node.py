import socket
import json
import time
import threading

FUSION_IP = '0.0.0.0'
FUSION_PORT = 50000
POLL_INTERVAL = 0.1

state = {
    "global_flag": 0,
    "local_flags": [0, 0, 0, 0, 0, 0],
    "last_update": time.time()
}

def evaluate_fusion_alert(global_flag, local_flags):
    """
    Evaluates fusion alert ruleset based on 1 global model binary and 6 local model binaries.
    """
    any_local = any(local_flags)
    stages_with_local_flags = [i+1 for i, flag in enumerate(local_flags) if flag]
    
    if global_flag and any_local:
        return {
            "alert": 1,
            "likely_category": "corroborated (local+global agree)",
            "localized_to": stages_with_local_flags
        }
    elif global_flag and not any_local:
        return {
            "alert": 1,
            "likely_category": "possible multi-subsystem attack, OR local miss, OR global false alarm",
            "localized_to": []
        }
    elif not global_flag and any_local:
        return {
            "alert": 1,
            "likely_category": "possible single-subsystem attack, OR global miss",
            "localized_to": stages_with_local_flags
        }
    else:
        return {
            "alert": 0,
            "likely_category": "Normal",
            "localized_to": []
        }

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((FUSION_IP, FUSION_PORT))
    print(f"[*] Simple Fusion UDP Listener started on {FUSION_IP}:{FUSION_PORT}")
    
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
    listener = threading.Thread(target=udp_listener, daemon=True)
    listener.start()
    
    print(f"[*] Starting Simple Fusion Inference every {POLL_INTERVAL}s. Press Ctrl+C to stop.")
    
    try:
        while True:
            start_time = time.time()
            
            # Decay old state if we haven't received updates in 2 seconds
            if time.time() - state['last_update'] > 2.0:
                state['global_flag'] = 0
                state['local_flags'] = [0, 0, 0, 0, 0, 0]
            
            result = evaluate_fusion_alert(state['global_flag'], state['local_flags'])
            
            status_str = "ATTACK DETECTED!" if result['alert'] else "NORMAL"
            
            print(f"[{time.strftime('%H:%M:%S')}] Global: {state['global_flag']} | Local: {state['local_flags']} | {status_str} | Cat: {result['likely_category']} | Stages: {result['localized_to']}")
            
            elapsed = time.time() - start_time
            sleep_t = max(0, POLL_INTERVAL - elapsed)
            time.sleep(sleep_t)
            
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == '__main__':
    main()
