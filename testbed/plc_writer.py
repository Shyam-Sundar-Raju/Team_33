import time
import json
import struct
import pandas as pd
import numpy as np
from scipy.io import arff
from pymodbus.client import ModbusTcpClient

# Load mapping
with open('mapping.json', 'r') as f:
    mapping = json.load(f)

# Initialize PLC clients
plcs = {}
for ip in mapping["plcs"].keys():
    client = ModbusTcpClient(ip, port=502)
    client.connect()
    plcs[ip] = client

# Load Dataset
print("Loading dataset for playback...")
# Uncomment ONE of the following lines to choose between normal or attack data:
data, meta = arff.loadarff('multiclass/normal_dataset.arff')
# data, meta = arff.loadarff('multiclass/attack_dataset.arff')

df = pd.DataFrame(data)
X = df.drop(columns=['marker'], errors='ignore')

for col in X.columns:
    if pd.api.types.is_object_dtype(X[col]):
        X[col] = X[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        X[col] = pd.to_numeric(X[col], errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

print("Starting to write to PLCs. Press Ctrl+C to stop.")
try:
    for index, row in X.iterrows():
        print(f"\n--- Writing Row {index} ---")
        for ip, config in mapping["plcs"].items():
            client = plcs[ip]
            
            # Write Coils (Digital)
            coils_config = config["coils"]
            if coils_config:
                for c in coils_config:
                    # For pumps/valves, any value > 0 can be considered True (active/open/on)
                    # For P: 1 is usually off, 2 is on. We'll map > 1 to True. 
                    # For MV: 0 is closed, 1/2 is open. We'll map > 0 to True.
                    val_raw = float(row[c["feature"]])
                    if c["feature"].startswith('P') and not c["feature"].startswith('PIT'):
                        val = bool(val_raw > 1.0)
                    else:
                        val = bool(val_raw > 0.0)
                        
                    client.write_coil(address=c["address"], value=val)
                print(f"[{ip}] Wrote {len(coils_config)} coils.")
            
            # Write Holding Registers (Analog)
            hr_config = config["holding_registers"]
            if hr_config:
                payload = []
                for hr in hr_config:
                    val = float(row[hr["feature"]])
                    # Pack float into 4 bytes (Big-Endian), unpack as two 16-bit integers
                    packed_bytes = struct.pack('>f', val)
                    reg1, reg2 = struct.unpack('>HH', packed_bytes)
                    payload.extend([reg1, reg2])
                
                client.write_registers(address=0, values=payload)
                print(f"[{ip}] Wrote {len(hr_config)} floats (to {len(payload)} registers).")
        
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    for client in plcs.values():
        client.close()
