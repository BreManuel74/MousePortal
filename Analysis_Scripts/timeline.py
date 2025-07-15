import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# File paths (update these if your files are in a different location)
trial_log_path = r'Kaufman_Project/BM15/Session 17/beh/1752604904trial_log.csv'
treadmill_path = r'Kaufman_Project/BM15/Session 17/beh/1752604904treadmill.csv'
capacitive_path = r'Kaufman_Project/BM15/Session 17/beh/1752604904capacitive.csv'
# output_folder = r"Kaufman_Project/BM14/Session 9/beh"
# output_path = f"{output_folder}\\timeline.svg"

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
        treadmill_df['speed']
    ),
    index=capacitive_df['elapsed_time']
)

# Plot both on the same graph (capacitive only, with reward and puff events)
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# --- Plot 1: Capacitive Value ---
axs[0].plot(capacitive_df['elapsed_time'], capacitive_df['capacitive_value'], label='Capacitive Value')

# Reward events
reward_times = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
for i, rt in enumerate(reward_times):
    axs[0].axvline(x=rt, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Reward Event' if i == 0 else "")

# Puff events
if 'puff_event' in trial_log_df.columns:
    puff_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna()
    for i, pt in enumerate(puff_times):
        axs[0].axvline(x=pt, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Puff Event' if i == 0 else "")

# Probe events
if 'probe_time' in trial_log_df.columns:
    probe_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna()
    for i, pt in enumerate(probe_times):
        axs[0].axvline(x=pt, color='black', linestyle='-', alpha=0.7, linewidth=2, label='Probe Event' if i == 0 else "")

# Highlight reward intervals
for trial_idx in range(reward_texture_change_time.shape[0]):
    for seg_idx in range(reward_texture_change_time.shape[1]):
        try:
            start = float(reward_texture_change_time[trial_idx, seg_idx])
            end = float(reward_revert_time[trial_idx, seg_idx])
            if not np.isnan(start) and not np.isnan(end):
                axs[0].axvspan(start, end, color='green', alpha=0.15)
        except (ValueError, TypeError):
            continue

# Highlight punish intervals
for trial_idx in range(punish_texture_change_time.shape[0]):
    for seg_idx in range(punish_texture_change_time.shape[1]):
        try:
            start = float(punish_texture_change_time[trial_idx, seg_idx])
            end = float(punish_revert_time[trial_idx, seg_idx])
            if not np.isnan(start) and not np.isnan(end):
                axs[0].axvspan(start, end, color='red', alpha=0.15)
        except (ValueError, TypeError):
            continue

axs[0].set_ylabel('Capacitive Value')
axs[0].set_title('Capacitive Sensor Over Time with Reward and Puff Events')
axs[0].legend(loc='upper right')
axs[0].set_ylim(bottom=0)  # Set the bottom y-axis limit to 0

# Remove top and right borders for both subplots
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- Plot 2: Treadmill Speed ---
axs[1].plot(
    capacitive_df['elapsed_time'],
    treadmill_interp,
    label='Treadmill Speed (interpolated)',
    color='purple'  # Set treadmill speed line to purple
)

# Reward events
for i, rt in enumerate(reward_times):
    axs[1].axvline(x=rt, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Reward Event' if i == 0 else "")

# Puff events
if 'puff_event' in trial_log_df.columns:
    for i, pt in enumerate(puff_times):
        axs[1].axvline(x=pt, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Puff Event' if i == 0 else "")

# Probe events
if 'probe_time' in trial_log_df.columns:
    for i, pt in enumerate(probe_times):
        axs[1].axvline(x=pt, color='black', linestyle='-', alpha=0.7, linewidth=2, label='Probe Event' if i == 0 else "")

# Highlight reward intervals
for trial_idx in range(reward_texture_change_time.shape[0]):
    for seg_idx in range(reward_texture_change_time.shape[1]):
        try:
            start = float(reward_texture_change_time[trial_idx, seg_idx])
            end = float(reward_revert_time[trial_idx, seg_idx])
            if not np.isnan(start) and not np.isnan(end):
                axs[1].axvspan(start, end, color='green', alpha=0.15)
        except (ValueError, TypeError):
            continue

# Highlight punish intervals
for trial_idx in range(punish_texture_change_time.shape[0]):
    for seg_idx in range(punish_texture_change_time.shape[1]):
        try:
            start = float(punish_texture_change_time[trial_idx, seg_idx])
            end = float(punish_revert_time[trial_idx, seg_idx])
            if not np.isnan(start) and not np.isnan(end):
                axs[1].axvspan(start, end, color='red', alpha=0.15)
        except (ValueError, TypeError):
            continue

axs[1].set_xlabel('Elapsed Time (s)')
axs[1].set_ylabel('Speed')
axs[1].set_title('Interpolated Treadmill Speed Over Time with Reward and Puff Events')
axs[1].legend(loc='upper right')

# Set x-axis limits to the data range
xmin = capacitive_df['elapsed_time'].min()
xmax = capacitive_df['elapsed_time'].max()
for ax in axs:
    ax.set_xlim([xmin, xmax])

plt.tight_layout()
#plt.savefig(output_path, format="svg")
#plt.show()

window = 5  # seconds before and after
reward_times_flat = reward_texture_change_time.flatten()
reward_times_flat = pd.to_numeric(reward_times_flat, errors='coerce')
reward_times_flat = reward_times_flat[~np.isnan(reward_times_flat)]

cap_time = capacitive_df['elapsed_time'].values
cap_val = capacitive_df['capacitive_value'].values

cap_windows = []
for rt in reward_times_flat:
    mask = (cap_time >= rt - window) & (cap_time <= rt + window)
    cap_segment = cap_val[mask]
    cap_windows.append(cap_segment)

# Pad all segments to the same length (max found)
max_len = max(len(seg) for seg in cap_windows)
cap_windows_padded = np.array([
    np.pad(seg.astype(float), (0, max_len - len(seg)), constant_values=np.nan)
    for seg in cap_windows
])

# cap_windows_padded is your 2D array: shape (num_reward_events, num_timepoints)
# Each row: capacitive values from 5s before to 5s after each reward_texture_change_time

# Example: print shape
#print("Shape of cap_windows_padded:", cap_windows_padded.shape)

# Create a common time axis centered at 0
dt = np.median(np.diff(cap_time))  # Estimate sampling interval
window_len = cap_windows_padded.shape[1]
aligned_time = np.linspace(-window, window, window_len)

# plt.figure(figsize=(10, 6))

n_rewards = cap_windows_padded.shape[0]  # Number of reward events

# Plot mean and SEM
# mean_vals = np.nanmean(cap_windows_padded, axis=0)
# sem_vals = np.nanstd(cap_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(cap_windows_padded), axis=0))
# plt.plot(aligned_time, mean_vals, color='blue', label=f'Mean (n={n_rewards})')
# plt.fill_between(aligned_time, mean_vals - sem_vals, mean_vals + sem_vals, color='blue', alpha=0.2, label='SEM')

# plt.axvline(0, color='red', linestyle='--', label='Reward Onset (t=0)')
# plt.xlabel('Time from Reward Zone Onset (s)')
# plt.ylabel('Capacitive Value')
# plt.title('Capacitive Value Aligned to Reward Zone Onset')
# plt.legend()
# plt.xticks(np.arange(-5, 6, 1))  # Set x-axis ticks from -5 to 5 with step 1
# plt.xlim(-5, 5)   
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.tick_params(axis='both', direction='out')
# plt.tight_layout()
# #plt.show()

# --- Interpolated Treadmill Speed aligned to reward_times_flat ---

# Get interpolated speed as numpy array
speed_val = treadmill_interp.values

speed_windows = []
for rt in reward_times_flat:
    mask = (cap_time >= rt - window) & (cap_time <= rt + window)
    speed_segment = speed_val[mask]
    speed_windows.append(speed_segment)

# Pad all segments to the same length (max found)
max_speed_len = max(len(seg) for seg in speed_windows)
speed_windows_padded = np.array([
    np.pad(seg.astype(float), (0, max_speed_len - len(seg)), constant_values=np.nan)
    for seg in speed_windows
])

# Create a common time axis centered at 0 for speed
aligned_time_speed = np.linspace(-window, window, max_speed_len)

n_rewards_speed = speed_windows_padded.shape[0]

# Plot mean and SEM for speed
mean_speed = np.nanmean(speed_windows_padded, axis=0)
sem_speed = np.nanstd(speed_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_windows_padded), axis=0))

# --- Capacitive Value aligned to reward_event_times_flat ---

reward_event_times_flat = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
reward_event_times_flat = reward_event_times_flat[~np.isnan(reward_event_times_flat)]

window_event = 5  # seconds before and after for capacitive analysis

# Ensure reward_event_times_flat is a numpy array of floats
reward_event_times_flat = np.array(reward_event_times_flat, dtype=float)

cap_event_windows = []
for rt in reward_event_times_flat:
    mask = (cap_time >= rt - window_event) & (cap_time <= rt + window_event)
    cap_segment = cap_val[mask]
    cap_event_windows.append(cap_segment)

#print(len(cap_event_windows))

# Pad all segments to the same length (max found)
max_event_len = max(len(seg) for seg in cap_event_windows)
cap_event_windows_padded = np.array([
    np.pad(seg.astype(float), (0, max_event_len - len(seg)), constant_values=np.nan)
    for seg in cap_event_windows
])

# Apply moving median filter (21-point window) to each row before averaging
def apply_moving_median_21point(data):
    """Apply 21-point moving median filter to 2D array along axis 1"""
    filtered = np.copy(data)
    half_window = 10  # 10 points before + current + 10 points after = 21 total
    
    for i in range(data.shape[0]):
        row = data[i, :]
        for j in range(half_window, len(row) - half_window):
            window = row[j-half_window:j+half_window+1]  # +1 to include current point
            if not np.isnan(window).any():
                filtered[i, j] = np.median(window)
    return filtered

cap_event_windows_filtered = apply_moving_median_21point(cap_event_windows_padded)

# Create a common time axis centered at 0
aligned_time_event = np.linspace(-window_event, window_event, max_event_len)

n_rewards_event = cap_event_windows_filtered.shape[0]

# Plot mean and SEM using filtered data
mean_event_vals = np.nanmean(cap_event_windows_filtered, axis=0)
sem_event_vals = np.nanstd(cap_event_windows_filtered, axis=0) / np.sqrt(np.sum(~np.isnan(cap_event_windows_filtered), axis=0))

# --- Combined Subplots: Treadmill Speed and Capacitive Value aligned to reward_event_times_flat ---

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- Plot 1: Treadmill Speed aligned to reward_times_flat ---
axs[0].plot(aligned_time_speed, mean_speed, color='purple', label=f'Mean Speed (n={n_rewards_speed})')
axs[0].fill_between(aligned_time_speed, mean_speed - sem_speed, mean_speed + sem_speed, color='purple', alpha=0.2, label='SEM')
axs[0].axvline(0, color='red', linestyle='--', label='Reward Onset (t=0)')
axs[0].set_ylabel('Treadmill Speed (interpolated)')
axs[0].set_title('Treadmill Speed Aligned to Reward Zone Onset')
axs[0].legend()
axs[0].set_xlim(-5, 5)
#axs[0].set_ylim(bottom=0)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].tick_params(axis='both', direction='out')

# --- Plot 2: Capacitive Value aligned to reward_event_times_flat (5s window) ---
axs[1].plot(aligned_time_event, mean_event_vals, color='green', label=f'Mean (n={n_rewards_event})')
axs[1].fill_between(aligned_time_event, mean_event_vals - sem_event_vals, mean_event_vals + sem_event_vals, color='green', alpha=0.2, label='SEM')
axs[1].axvline(0, color='red', linestyle='--', label='Reward Event (t=0)')
axs[1].set_xlabel('Time from Reward Event (s)')
axs[1].set_ylabel('Capacitive Value')
axs[1].set_title('Capacitive Value Aligned to Reward Event')
axs[1].legend()
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(bottom=0)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].tick_params(axis='both', direction='out')
axs[1].set_xticks(np.arange(-5, 6, 1))

plt.tight_layout()


# --- Probe Event Analysis: Treadmill Speed and Capacitive Value aligned to probe events ---

# Check if probe events exist
if 'probe_time' in trial_log_df.columns:
    probe_event_times_flat = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna()
    probe_event_times_flat = probe_event_times_flat[~np.isnan(probe_event_times_flat)]
    
    if len(probe_event_times_flat) > 0:
        window = 5  # seconds before and after
        
        # Ensure probe_event_times_flat is a numpy array of floats
        probe_event_times_flat = np.array(probe_event_times_flat, dtype=float)
        
        # --- Capacitive Value aligned to probe events ---
        cap_probe_windows = []
        for pt in probe_event_times_flat:
            mask = (cap_time >= pt - window) & (cap_time <= pt + window)
            cap_segment = cap_val[mask]
            cap_probe_windows.append(cap_segment)
        
        # Pad all segments to the same length (max found)
        max_probe_len = max(len(seg) for seg in cap_probe_windows) if cap_probe_windows else 0
        if max_probe_len > 0:
            cap_probe_windows_padded = np.array([
                np.pad(seg.astype(float), (0, max_probe_len - len(seg)), constant_values=np.nan)
                for seg in cap_probe_windows
            ])
            
            # Apply moving median filter (21-point window) to probe capacitive data
            cap_probe_windows_filtered = apply_moving_median_21point(cap_probe_windows_padded)
            
            # --- Treadmill Speed aligned to probe events ---
            speed_probe_windows = []
            for pt in probe_event_times_flat:
                mask = (cap_time >= pt - window) & (cap_time <= pt + window)
                speed_segment = speed_val[mask]
                speed_probe_windows.append(speed_segment)
            
            # Pad speed segments to the same length
            speed_probe_windows_padded = np.array([
                np.pad(seg.astype(float), (0, max_probe_len - len(seg)), constant_values=np.nan)
                for seg in speed_probe_windows
            ])
            
            # Create a common time axis centered at 0
            aligned_time_probe = np.linspace(-window, window, max_probe_len)
            
            # --- Combined Subplots: Treadmill Speed and Capacitive Value aligned to probe events ---
            
            fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            n_probes = cap_probe_windows_padded.shape[0]
            
            # --- Plot 1: Treadmill Speed aligned to probe events ---
            mean_speed_probe = np.nanmean(speed_probe_windows_padded, axis=0)
            sem_speed_probe = np.nanstd(speed_probe_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_probe_windows_padded), axis=0))
            axs[0].plot(aligned_time_probe, mean_speed_probe, color='purple', label=f'Mean Speed (n={n_probes})')
            axs[0].fill_between(aligned_time_probe, mean_speed_probe - sem_speed_probe, mean_speed_probe + sem_speed_probe, color='purple', alpha=0.2, label='SEM')
            axs[0].axvline(0, color='black', linestyle='--', label='Probe Event (t=0)')
            axs[0].set_ylabel('Treadmill Speed (interpolated)')
            axs[0].set_title('Treadmill Speed Aligned to Probe Events')
            axs[0].legend()
            axs[0].set_xlim(-5, 5)
            axs[0].spines['top'].set_visible(False)
            axs[0].spines['right'].set_visible(False)
            axs[0].tick_params(axis='both', direction='out')
            
            # --- Plot 2: Capacitive Value aligned to probe events ---
            mean_cap_probe = np.nanmean(cap_probe_windows_filtered, axis=0)
            sem_cap_probe = np.nanstd(cap_probe_windows_filtered, axis=0) / np.sqrt(np.sum(~np.isnan(cap_probe_windows_filtered), axis=0))
            axs[1].plot(aligned_time_probe, mean_cap_probe, color='green', label=f'Mean (n={n_probes})')
            axs[1].fill_between(aligned_time_probe, mean_cap_probe - sem_cap_probe, mean_cap_probe + sem_cap_probe, color='green', alpha=0.2, label='SEM')
            axs[1].axvline(0, color='black', linestyle='--', label='Probe Event (t=0)')
            axs[1].set_xlabel('Time from Probe Event (s)')
            axs[1].set_ylabel('Capacitive Value')
            axs[1].set_title('Capacitive Value Aligned to Probe Events')
            axs[1].legend()
            axs[1].set_xlim(-5, 5)
            axs[1].set_ylim(bottom=0)
            axs[1].spines['top'].set_visible(False)
            axs[1].spines['right'].set_visible(False)
            axs[1].tick_params(axis='both', direction='out')
            axs[1].set_xticks(np.arange(-5, 6, 1))
            
            plt.tight_layout()
            #plt.show()
        else:
            print("No valid probe event data found for alignment analysis.")
    else:
        print("No probe events found in the data.")
else:
    print("No 'probe_time' column found in trial_log_df.")

plt.tight_layout()

# --- Prepare data for heatmap with 2-second window ---
window_heatmap = 2  # seconds before and after for heatmap

cap_event_windows_heatmap = []
for rt in reward_event_times_flat:
    mask = (cap_time >= rt - window_heatmap) & (cap_time <= rt + window_heatmap)
    cap_segment = cap_val[mask]
    cap_event_windows_heatmap.append(cap_segment)

# Pad all segments to the same length (max found)
max_event_len_heatmap = max(len(seg) for seg in cap_event_windows_heatmap)
cap_event_windows_padded_heatmap = np.array([
    np.pad(seg.astype(float), (0, max_event_len_heatmap - len(seg)), constant_values=np.nan)
    for seg in cap_event_windows_heatmap
])

# --- Heatmap of cap_event_windows (2s window) ---

plt.figure(figsize=(12, 8))

# Create the heatmap using 2-second window data
im = plt.imshow(cap_event_windows_padded_heatmap, aspect='auto', cmap='gray_r', interpolation='nearest')

# Set up the time axis labels for 2-second window
n_timepoints = cap_event_windows_padded_heatmap.shape[1]
#print("Number of timepoints:", n_timepoints)
time_labels = np.linspace(-window_heatmap, window_heatmap, n_timepoints)
tick_indices = np.linspace(0, n_timepoints-1, 11, dtype=int)  # 11 ticks 
tick_labels = [f'{time_labels[i]:.1f}' for i in tick_indices]

plt.xticks(tick_indices, tick_labels)
plt.xlabel('Time from Reward Event (s)')
plt.ylabel('Reward Event #')
plt.title(f'Heatmap: Capacitive Value Across All Reward Events (2s window, n={n_rewards_event})')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Capacitive Value')

# Add vertical line at t=0
plt.axvline(x=n_timepoints//2, color='red', linestyle='--', alpha=0.8, linewidth=2)

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
#plt.show()

# --- Match probe_time with texture_revert_time (approximately 1 second before) ---

# Collect all texture_revert times from all trials (needed for both probe matching and unpaired analysis)
all_revert_times = []
for trial_idx in range(len(trial_log_df)):
    revert_list = safe_literal_eval(trial_log_df.iloc[trial_idx]['texture_revert'])
    for revert_time in revert_list:
        if not pd.isna(revert_time) and revert_time != '':
            try:
                all_revert_times.append(float(revert_time))
            except (ValueError, TypeError):
                continue
all_revert_times = np.array(all_revert_times)

# Check if probe_time column exists and has data
if 'probe_time' not in trial_log_df.columns:
    print("Warning: No 'probe_time' column found in trial_log_df. Skipping probe-revert matching analysis.")
    probe_revert_array = None
else:
    # Extract probe times
    probe_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna().values
    
    if len(probe_times) == 0:
        print("Warning: No valid probe times found in the data. Skipping probe-revert matching analysis.")
        probe_revert_array = None
    else:
        if len(all_revert_times) == 0:
            print("Warning: No valid texture revert times found in the data. Cannot match with probe times.")
            probe_revert_array = None
        else:
            # Match each probe_time with the closest texture_revert_time that occurs ~1 second before
            probe_revert_matches = []
            tolerance = 0.5  # Allow ±0.5 seconds around the 1-second target
            
            for probe_time in probe_times:
                # Find revert times that occur before the probe time
                candidate_reverts = all_revert_times[all_revert_times < probe_time]
                
                if len(candidate_reverts) > 0:
                    # Calculate time differences (probe_time - revert_time)
                    time_diffs = probe_time - candidate_reverts
                    
                    # Find revert times that are approximately 1 second before (within tolerance)
                    target_diff = 1.0  # 1 second
                    valid_matches = np.abs(time_diffs - target_diff) <= tolerance
                    
                    if np.any(valid_matches):
                        # Get the closest match to exactly 1 second before
                        valid_diffs = time_diffs[valid_matches]
                        valid_reverts = candidate_reverts[valid_matches]
                        closest_idx = np.argmin(np.abs(valid_diffs - target_diff))
                        matched_revert = valid_reverts[closest_idx]
                        actual_diff = time_diffs[valid_matches][closest_idx]
                    else:
                        # If no matches within tolerance, find the closest revert time before probe
                        closest_idx = np.argmin(time_diffs)
                        matched_revert = candidate_reverts[closest_idx]
                        actual_diff = time_diffs[closest_idx]
                else:
                    # No revert times before this probe
                    matched_revert = np.nan
                    actual_diff = np.nan
                
                probe_revert_matches.append([probe_time, matched_revert, actual_diff])
            
            # Convert to 2D numpy array
            probe_revert_array = np.array(probe_revert_matches)
            
            # Print results only if we have matches
            successful_matches = np.sum(~np.isnan(probe_revert_array[:, 1]))
            
            if successful_matches == 0:
                print("Warning: No successful matches found between probe times and revert times.")
                print("This could indicate:")
                print("- Probe times occur before any texture revert events")
                print("- Time tolerance (±0.5s around 1s) is too strict")
                print("- Data timing issues or missing events")

# --- Find unpaired revert times (not matched with any probe event) ---

if probe_revert_array is not None and len(probe_revert_array) > 0:
    # Get all matched revert times (excluding NaN values)
    matched_revert_times = probe_revert_array[:, 1]  # Column 1 contains matched revert times
    matched_revert_times = matched_revert_times[~np.isnan(matched_revert_times)]
    
    # Find unpaired revert times by comparing with all revert times
    unpaired_revert_times = []
    for revert_time in all_revert_times:
        # Check if this revert time was matched with any probe
        # Use a small tolerance for floating point comparison
        tolerance_match = 1e-6
        is_matched = np.any(np.abs(matched_revert_times - revert_time) < tolerance_match)
        
        if not is_matched:
            unpaired_revert_times.append(revert_time)
    unpaired_revert_times = np.array(unpaired_revert_times)
    
    if len(unpaired_revert_times) == 0:
        print("All revert times have been matched with probe events.")
        unpaired_revert_times = np.array([])

elif len(all_revert_times) > 0:
    # If no probe analysis was performed but we have revert times
    unpaired_revert_times = all_revert_times.copy()
else:
    print("No revert times found in the data.")
    unpaired_revert_times = np.array([])

# --- Analysis of unpaired revert times: Simulate probe events 1 second after revert ---

# Only perform simulated probe analysis if there were actual probe events in the data
if 'probe_time' in trial_log_df.columns and len(unpaired_revert_times) > 0 and probe_revert_array is not None:
    # Create simulated probe times (1 second after each unpaired revert time)
    simulated_probe_times = unpaired_revert_times + 1.0
    
    window_sim = 5  # seconds before and after simulated probe
    
    # --- Capacitive Value aligned to simulated probe events ---
    cap_sim_windows = []
    for sim_probe_time in simulated_probe_times:
        mask = (cap_time >= sim_probe_time - window_sim) & (cap_time <= sim_probe_time + window_sim)
        cap_segment = cap_val[mask]
        cap_sim_windows.append(cap_segment)
    
    # Check if we have any valid windows
    if cap_sim_windows and max(len(seg) for seg in cap_sim_windows) > 0:
        # Pad all segments to the same length
        max_sim_len = max(len(seg) for seg in cap_sim_windows)
        cap_sim_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_sim_len - len(seg)), constant_values=np.nan)
            for seg in cap_sim_windows
        ])
        
        # Apply moving median filter (21-point window) to simulated probe capacitive data
        cap_sim_windows_filtered = apply_moving_median_21point(cap_sim_windows_padded)
        
        # --- Treadmill Speed aligned to simulated probe events ---
        speed_sim_windows = []
        for sim_probe_time in simulated_probe_times:
            mask = (cap_time >= sim_probe_time - window_sim) & (cap_time <= sim_probe_time + window_sim)
            speed_segment = speed_val[mask]
            speed_sim_windows.append(speed_segment)
        
        # Pad speed segments to the same length
        speed_sim_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_sim_len - len(seg)), constant_values=np.nan)
            for seg in speed_sim_windows
        ])
        
        # Create a common time axis centered at 0 (simulated probe at t=0)
        aligned_time_sim = np.linspace(-window_sim, window_sim, max_sim_len)
        
        # --- Combined Subplots: Treadmill Speed and Capacitive Value aligned to simulated probe events ---
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        n_sim_probes = cap_sim_windows_padded.shape[0]
        
        # --- Plot 1: Treadmill Speed aligned to simulated probe events ---
        mean_speed_sim = np.nanmean(speed_sim_windows_padded, axis=0)
        sem_speed_sim = np.nanstd(speed_sim_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_sim_windows_padded), axis=0))
        axs[0].plot(aligned_time_sim, mean_speed_sim, color='purple', label=f'Mean Speed (n={n_sim_probes})')
        axs[0].fill_between(aligned_time_sim, mean_speed_sim - sem_speed_sim, mean_speed_sim + sem_speed_sim, color='purple', alpha=0.2, label='SEM')
        axs[0].axvline(0, color='black', linestyle='--', label='Simulated Probe (t=0)')
        #axs[0].axvline(-1, color='red', linestyle=':', alpha=0.7, label='Revert Time (t=-1s)')
        axs[0].set_ylabel('Treadmill Speed (interpolated)')
        axs[0].set_title('Treadmill Speed Aligned to Simulated Probe Events (1s after unpaired reverts)')
        axs[0].legend()
        axs[0].set_xlim(-5, 5)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].tick_params(axis='both', direction='out')
        
        # --- Plot 2: Capacitive Value aligned to simulated probe events ---
        mean_cap_sim = np.nanmean(cap_sim_windows_filtered, axis=0)
        sem_cap_sim = np.nanstd(cap_sim_windows_filtered, axis=0) / np.sqrt(np.sum(~np.isnan(cap_sim_windows_filtered), axis=0))
        axs[1].plot(aligned_time_sim, mean_cap_sim, color='green', label=f'Mean (n={n_sim_probes})')
        axs[1].fill_between(aligned_time_sim, mean_cap_sim - sem_cap_sim, mean_cap_sim + sem_cap_sim, color='green', alpha=0.2, label='SEM')
        axs[1].axvline(0, color='black', linestyle='--', label='Simulated Probe (t=0)')
        #axs[1].axvline(-1, color='red', linestyle=':', alpha=0.7, label='Revert Time (t=-1s)')
        axs[1].set_xlabel('Time from Simulated Probe Event (s)')
        axs[1].set_ylabel('Capacitive Value')
        axs[1].set_title('Capacitive Value Aligned to Simulated Probe Events (1s after unpaired reverts)')
        axs[1].legend()
        axs[1].set_xlim(-5, 5)
        axs[1].set_ylim(bottom=0)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].tick_params(axis='both', direction='out')
        axs[1].set_xticks(np.arange(-5, 6, 1))
        
        plt.tight_layout()
        #plt.show()
        
    else:
        print("Warning: No valid data windows found for simulated probe analysis.")
        print("This could be due to unpaired revert times being too close to data boundaries.")

elif len(unpaired_revert_times) > 0:
    # If no probe events exist but we have unpaired revert times, skip simulated analysis
    pass
else:
    # No unpaired revert times available
    pass

plt.show()