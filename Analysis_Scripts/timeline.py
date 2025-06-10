import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# File paths (update these if your files are in a different location)
trial_log_path = r'Kaufman_Project/Algernon/Session2/beh/1749576021trial_log.csv'
treadmill_path = r'Kaufman_Project/Algernon/Session2/beh/1749576021treadmill.csv'
capacitive_path = r'Kaufman_Project/Algernon/Session2/beh/1749576021capacitive.csv'

# Read the CSV files into pandas DataFrames
trial_log_df = pd.read_csv(trial_log_path, engine='python')
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

# Remove the is_list_like filter and just use safe_literal_eval
texture_history = trial_log_df['texture_history'].apply(safe_literal_eval)
texture_change_time = trial_log_df['texture_change_time'].apply(safe_literal_eval)
revert_time = trial_log_df['texture_revert'].apply(safe_literal_eval)

max_len = max(
    texture_history.apply(len).max(),
    texture_change_time.apply(len).max(),
    revert_time.apply(len).max()
)

def pad_list(lst, length):
    return lst + [np.nan] * (length - len(lst))

texture_history_padded = np.array(texture_history.apply(lambda x: pad_list(x, max_len)).tolist())
texture_change_time_padded = np.array(texture_change_time.apply(lambda x: pad_list(x, max_len)).tolist())
revert_time_padded = np.array(revert_time.apply(lambda x: pad_list(x, max_len)).tolist())

combined_array = np.stack(
    [texture_history_padded, texture_change_time_padded, revert_time_padded],
    axis=1
)

# combined_array now has shape (num_rows, 3, max_len)
# combined_array[:, 0, :] = texture_history
# combined_array[:, 1, :] = texture_change_time
# combined_array[:, 2, :] = revert_time

# Create boolean masks for each asset type
is_punish = texture_history_padded[:, 0] == "assets/punish_mean100.jpg"
is_reward = texture_history_padded[:, 0] == "assets/reward_mean100.jpg"

# Select rows for each type
punish_array = combined_array[is_punish]
reward_array = combined_array[is_reward]

# Now punish_array and reward_array contain only rows starting with the respective asset
# For punish:
punish_texture_change_time = punish_array[:, 1, :]
punish_revert_time = punish_array[:, 2, :]

# And for reward:
reward_texture_change_time = reward_array[:, 1, :]
reward_revert_time = reward_array[:, 2, :]

# Interpolate treadmill distance to match capacitive elapsed_time
treadmill_interp = pd.Series(
    data=np.interp(
        capacitive_df['elapsed_time'],
        treadmill_df['global_time'],
        treadmill_df['distance']
    ),
    index=capacitive_df['elapsed_time']
)


# Plot both on the same graph (capacitive only, with reward and puff events)
plt.figure(figsize=(12, 5))
plt.plot(capacitive_df['elapsed_time'], capacitive_df['capacitive_value'], label='Capacitive Value')

# Add vertical colored bars at each reward_event time (if not NaN)
reward_times = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
for i, rt in enumerate(reward_times):
    plt.axvline(x=rt, color='green', linestyle='-', alpha=0.7, label='Reward Event' if i == 0 else "")

# Add vertical colored bars at each puff_event time (if not NaN)
if 'puff_event' in trial_log_df.columns:
    puff_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna()
    for i, pt in enumerate(puff_times):
        plt.axvline(x=pt, color='red', linestyle='-', alpha=0.7, label='Puff Event' if i == 0 else "")

# Add vertical colored bars at each prob_time (if not NaN)
if 'probe_time' in trial_log_df.columns:
    probe_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna()
    for i, pt in enumerate(probe_times):
        plt.axvline(x=pt, color='black', linestyle='-', alpha=0.7, label='Probe Event' if i == 0 else "")

plt.xlabel('Elapsed Time (s)')
plt.ylabel('Capacitive Value')
plt.title('Capacitive Sensor Over Time with Reward and Puff Events')
plt.legend()
plt.tight_layout()
#plt.show()

# Highlight reward intervals on capacitive plot
for trial_idx in range(reward_texture_change_time.shape[0]):
    for seg_idx in range(reward_texture_change_time.shape[1]):
        try:
            start = float(reward_texture_change_time[trial_idx, seg_idx])
            end = float(reward_revert_time[trial_idx, seg_idx])
            if not np.isnan(start) and not np.isnan(end):
                plt.axvspan(start, end, color='green', alpha=0.15)
        except (ValueError, TypeError):
            continue

# Highlight punish intervals on capacitive plot
for trial_idx in range(punish_texture_change_time.shape[0]):
    for seg_idx in range(punish_texture_change_time.shape[1]):
        try:
            start = float(punish_texture_change_time[trial_idx, seg_idx])
            end = float(punish_revert_time[trial_idx, seg_idx])
            if not np.isnan(start) and not np.isnan(end):
                plt.axvspan(start, end, color='red', alpha=0.15)
        except (ValueError, TypeError):
            continue

plt.xlabel('Elapsed Time (s)')
plt.ylabel('Capacitive Value')
plt.title('Capacitive Sensor Over Time with Reward and Puff Events')
plt.legend()
plt.tight_layout()
#plt.show()

# Interpolate treadmill speed to match capacitive elapsed_time
treadmill_speed_interp = pd.Series(
    data=np.interp(
        capacitive_df['elapsed_time'],
        treadmill_df['global_time'],
        treadmill_df['speed']
    ),
    index=capacitive_df['elapsed_time']
)

# Plot interpolated treadmill speed over time
plt.figure(figsize=(10, 4))
plt.plot(capacitive_df['elapsed_time'], treadmill_speed_interp, label='Treadmill Speed (interpolated)')

# Add vertical colored bars at each reward_event time (if not NaN)
reward_times = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
for i, rt in enumerate(reward_times):
    plt.axvline(x=rt, color='green', linestyle='-', alpha=0.7, label='Reward Event' if i == 0 else "")

# Add vertical colored bars at each puff_event time (if not NaN)
if 'puff_event' in trial_log_df.columns:
    puff_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna()
    for i, pt in enumerate(puff_times):
        plt.axvline(x=pt, color='red', linestyle='-', alpha=0.7, label='Puff Event' if i == 0 else "")

# Add vertical colored bars at each prob_time (if not NaN)
if 'probe_time' in trial_log_df.columns:
    probe_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna()
    for i, pt in enumerate(probe_times):
        plt.axvline(x=pt, color='black', linestyle='-', alpha=0.7, label='Probe Event' if i == 0 else "")

# Highlight reward intervals on treadmill speed plot
for trial_idx in range(reward_texture_change_time.shape[0]):
    for seg_idx in range(reward_texture_change_time.shape[1]):
        try:
            start = float(reward_texture_change_time[trial_idx, seg_idx])
            end = float(reward_revert_time[trial_idx, seg_idx])
            if not np.isnan(start) and not np.isnan(end):
                plt.axvspan(start, end, color='green', alpha=0.15)
        except (ValueError, TypeError):
            continue

# Highlight punish intervals on treadmill speed plot
for trial_idx in range(punish_texture_change_time.shape[0]):
    for seg_idx in range(punish_texture_change_time.shape[1]):
        try:
            start = float(punish_texture_change_time[trial_idx, seg_idx])
            end = float(punish_revert_time[trial_idx, seg_idx])
            if not np.isnan(start) and not np.isnan(end):
                plt.axvspan(start, end, color='red', alpha=0.15)
        except (ValueError, TypeError):
            continue

plt.xlabel('Elapsed Time (s)')
plt.ylabel('Speed')
plt.title('Interpolated Treadmill Speed Over Time with Reward and Puff Events')
plt.legend()
plt.tight_layout()
plt.show()
