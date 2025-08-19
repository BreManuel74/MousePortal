import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# File paths (update these if your files are in a different location)
treadmill_path = r'Kaufman_Project/BM13/Session 4/beh/1750955432treadmill.csv'
capacitive_path = r'Kaufman_Project/BM14/Session 4/beh/1750960112capacitive.csv'

# Read the CSV files into pandas DataFrames
treadmill_df = pd.read_csv(treadmill_path, comment='/', engine='python')
capacitive_df = pd.read_csv(capacitive_path, comment='/', engine='python')

# Safe literal eval function
def safe_literal_eval(val):
    try:
        if isinstance(val, list):
            return val
        if pd.isna(val) or val == '':
            return []
        # If it's a number, wrap in a list
        if isinstance(val, (int, float)):
            return [val]
        # If it's a string that is not a list, try to convert to float
        if isinstance(val, str) and not (val.strip().startswith("[") and val.strip().endswith("]")):
            try:
                return [float(val)]
            except Exception:
                return [val]
        return ast.literal_eval(val)
    except Exception:
        return []
    
# Interpolate treadmill distance to match capacitive elapsed_time
treadmill_interp = pd.Series(
    data=np.interp(
        capacitive_df['elapsed_time'],
        treadmill_df['global_time'],
        treadmill_df['speed']
    ),
    index=capacitive_df['elapsed_time']
)


plt.figure(figsize=(12, 4))
plt.plot(capacitive_df['elapsed_time'], capacitive_df['capacitive_value'], label='Capacitive Value', color='blue')
plt.xlabel('Elapsed Time (s)')
plt.ylabel('Capacitive Value')
plt.title('Capacitive Value Over Time')
plt.ylim(bottom=0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- New Plot: Interpolated Treadmill Speed Only ---
plt.figure(figsize=(12, 4))
plt.plot(capacitive_df['elapsed_time'], treadmill_interp, label='Treadmill Speed (interpolated)', color='purple')
plt.xlabel('Elapsed Time (s)')
plt.ylabel('Treadmill Speed')
plt.title('Interpolated Treadmill Speed Over Time')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()