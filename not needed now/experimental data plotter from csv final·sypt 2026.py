import pandas as pd
import numpy as np
import os

# 1. Define the folder path once
script_folder = os.path.dirname(os.path.abspath(__file__))

fn = os.path.join(script_folder, '4CR_R4.csv')
excel_out = os.path.join(script_folder, 'runs.xlsx')

def process_run_data(csv_filename, excel_filename):
    # Load Data
    df = pd.read_csv(csv_filename)
    
    # Physics Constants
    dt = 0.022 
    sensor_spacing = 0.015  # 15mm spacing
    mux_cols = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
    
    # Centroid Tracking
    ambient_baseline = df[mux_cols].values.max()
    weights = (ambient_baseline - df[mux_cols]).clip(lower=0)
    indices = np.arange(len(mux_cols))
    total_weight = weights.sum(axis=1)
    
    valid_mask = total_weight > (ambient_baseline * 0.1)
    df['pos_idx'] = np.nan
    df.loc[valid_mask, 'pos_idx'] = (weights[valid_mask] * indices).sum(axis=1) / total_weight[valid_mask]
    df['pos_m'] = df['pos_idx'] * sensor_spacing
    
    # Differentiate
    df_clean = df.dropna(subset=['pos_m']).copy()
    if len(df_clean) < 3:
        print("Not enough data points found.")
        return
        
    df_clean['velocity'] = df_clean['pos_m'].diff() / dt
    df_clean['accel'] = df_clean['velocity'].diff() / dt
    avg_accel = df_clean['accel'].mean()
    
    # Save to Excel
    if os.path.exists(excel_filename):
        runs_df = pd.read_excel(excel_filename)
    else:
        runs_df = pd.DataFrame()

    new_col_name = f"Run_{len(runs_df.columns) + 1}_Accel"
    
    if runs_df.empty:
        runs_df = pd.DataFrame({new_col_name: [avg_accel]})
    else:
        # We use loc to ensure the value is placed in the first row of the new column
        runs_df.loc[0, new_col_name] = avg_accel
        
    runs_df.to_excel(excel_filename, index=False)
    print(f"Success! Accel = {avg_accel:.4f} m/s^2")
    print(f"File saved at: {excel_filename}")

# Execute using the full path for both
process_run_data(fn, excel_out)