import time
import json
import struct
import re
from pymodbus.client import ModbusTcpClient

STAGE = 4

# Load mapping
with open('mapping.json', 'r') as f:
    mapping = json.load(f)

# Find the PLC for this stage
target_ip = None
for ip in mapping["plcs"].keys():
    if STAGE in [1, 2] and ip == '192.168.200.219':
        target_ip = ip
    elif STAGE in [3, 4] and ip == '192.168.200.220':
        target_ip = ip
    elif STAGE in [5, 6] and ip == '192.168.200.221':
        target_ip = ip

if not target_ip:
    print(f"Could not find PLC IP for Stage {4}")
    exit(1)

client = ModbusTcpClient(target_ip, port=502)
client.connect()

print(f"Starting to read Stage 4 from PLC {target_ip}. Press Ctrl+C to stop.")
try:
    previous_state = None
    while True:
        feature_dict = {}
        config = mapping["plcs"][target_ip]
        
        # Filter coils for this stage
        coils_config = [c for c in config["coils"] if re.search(r'\d+', c["feature"]) and int(re.search(r'\d+', c["feature"]).group(0)[0]) == STAGE]
        
        if coils_config:
            max_addr = max(c["address"] for c in coils_config)
            result = client.read_coils(address=0, count=max_addr + 1)
            if not result.isError():
                for c in coils_config:
                    # All coils in this setup (Pumps, UVs) have states 1 and 2
                    val = 2 if result.bits[c["address"]] else 1
                    feature_dict[c["feature"]] = val
            else:
                print(f"[{target_ip}] Error reading coils.")
        
        # Filter holding registers for this stage
        hr_config = [hr for hr in config["holding_registers"] if re.search(r'\d+', hr["feature"]) and int(re.search(r'\d+', hr["feature"]).group(0)[0]) == STAGE]
        
        if hr_config:
            max_addr = max(hr["address"] for hr in hr_config) if hr_config else 0
            count = max_addr + 2
            result = client.read_holding_registers(address=0, count=count)
            if not result.isError():
                regs = result.registers
                for hr in hr_config:
                    idx = hr["address"]
                    if idx + 1 < len(regs):
                        r1, r2 = regs[idx], regs[idx + 1]
                        packed_bytes = struct.pack('>HH', r1, r2)
                        val = struct.unpack('>f', packed_bytes)[0]
                        # Use the is_discrete flag straight from the JSON mapping!
                        if hr.get("is_discrete", False):
                            val = int(round(val))
                        feature_dict[hr["feature"]] = val
            else:
                print(f"[{target_ip}] Error reading holding registers.")
                
        # Only output if the state has changed
        if feature_dict != previous_state:
            print(f"\n--- Stage 4 Live Values ---")
            # Print alphabetically for neatness
            for k in sorted(feature_dict.keys()):
                if isinstance(feature_dict[k], float):
                    print(f"{k}: {feature_dict[k]:.4f}")
                else:
                    print(f"{k}: {feature_dict[k]}")
            previous_state = feature_dict.copy()
            
        time.sleep(1)
        
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    client.close()
