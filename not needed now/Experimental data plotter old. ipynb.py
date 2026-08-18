import serial
import serial.tools.list_ports
import csv
import time
import os
import numpy as np

# ============================================================
# SETTINGS & PATHS
# ============================================================
BAUD = 115200
N_SAMPLES = 1000 
NUM_CHANNELS = 16
SPACING = 0.012  # 1.2 cm distance between IR sensors

# Path from your screenshot
user_profile = os.path.expanduser("~")
target_folder = os.path.join(user_profile, "OneDrive - Personal", "playing data sypt 2026")

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

timestr = time.strftime("%Y%m%d_%H%M%S")
filename = f"experiment_{timestr}.csv"
full_path = os.path.join(target_folder, filename)

# ============================================================
# AUTO-DETECT PORT
# ============================================================
def find_arduino():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ["arduino", "ch340", "cp210", "silabs"]):
            return p.device
    return ports[0].device if ports else None

# ============================================================
# RECORDING DATA
# ============================================================
PORT = find_arduino()
if not PORT:
    raise Exception("No Arduino found. Check your USB cable!")

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2) 
ser.reset_input_buffer()

values = []
timestamps = []
print(f"Recording to: {filename}")

start_time = time.time()
while len(values) < N_SAMPLES:
    raw = ser.readline()
    if not raw: continue
    line = raw.decode("ascii", errors="ignore").strip()
    
    if line.startswith("<") and line.endswith(">"):
        payload = line[1:-1].split(",")
        if len(payload) == NUM_CHANNELS:
            try:
                nums = [int(x) for x in payload]
                values.append(nums)
                timestamps.append(time.time() - start_time)
            except ValueError:
                continue

ser.close()

# ============================================================
# CALCULATE ACCELERATION
# ============================================================
# 1. Find the time when the object passed each sensor (t_enter)
# We look for the moment the signal drops (DIP) because of the object blocking the IR
data_array = np.array(values).T  # Shape: (16, N_SAMPLES)
t_array = np.array(timestamps)
t_enters = []
x_positions = []

for ch in range(NUM_CHANNELS):
    y = data_array[ch]
    threshold = np.median(y) - (np.std(y) * 3) # Simple event detection
    indices = np.where(y < threshold)[0]
    
    if len(indices) > 0:
        t_enters.append(t_array[indices[0]])
        x_positions.append(ch * SPACING)

# 2. Fit to Quadratic Equation: x = At^2 + Bt + C
if len(t_enters) >= 3:
    coeffs = np.polyfit(t_enters, x_positions, 2)
    acceleration = 2 * coeffs[0] # Acceleration is 2 * the t^2 coefficient
    result_text = f"Calculated Acceleration: {acceleration:.4f} m/s^2"
else:
    acceleration = None
    result_text = "Not enough sensors triggered to calculate acceleration."

print(f"\n{result_text}")

# ============================================================
# SAVE TO CSV
# ============================================================
with open(full_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["SUMMARY", result_text]) # Put result at the top
    writer.writerow([]) # Empty line
    writer.writerow(["Timestamp"] + [f"CH{i}" for i in range(NUM_CHANNELS)])
    
    for t, val_row in zip(timestamps, values):
        writer.writerow([round(t, 4)] + val_row)

print(f"Done! Data and acceleration saved to OneDrive.")