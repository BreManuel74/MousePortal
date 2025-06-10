import pandas as pd
import ast
import numpy as np
import os

# File paths (update these if your files are in a different location)
trial_log_path = r'Kaufman_Project\Algernon\Session2\beh\1749576021trial_log.csv'
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

reward_times = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()

lick_cutoff = (capacitive_df['capacitive_value'].quantile(0.90))/2
print(lick_cutoff)
lick_bouts = capacitive_df[capacitive_df['capacitive_value'] > lick_cutoff].values

# Get lick bout times (assuming capacitive_df has a 'time' column)
lick_bout_times = capacitive_df.loc[capacitive_df['capacitive_value'] > lick_cutoff, 'elapsed_time'].values

# Flatten reward_texture_change_time and convert to numeric, removing NaNs
reward_change_times_flat = reward_texture_change_time.flatten()
reward_change_times_flat = pd.to_numeric(reward_change_times_flat, errors='coerce')
reward_change_times_flat = reward_change_times_flat[~np.isnan(reward_change_times_flat)]

reward_times_flat = reward_times.values
if len(reward_change_times_flat) != len(reward_times_flat):
    print("Warning: reward_texture_change_time and reward_times have different lengths!")

licks_before_reward = []
for t_change, t_reward in zip(reward_change_times_flat, reward_times_flat):
    # Count licks between t_change (inclusive) and t_reward (exclusive)
    count = int(np.sum((lick_bout_times >= t_change) & (lick_bout_times < t_reward)))
    licks_before_reward.append(count)

#print("Number of lick bouts between each reward zone entry and reward delivery:", licks_before_reward)
average_licks_before_reward = np.mean(licks_before_reward)
print("Average number of lick bouts between each reward zone entry and reward delivery:", average_licks_before_reward)

licks_before_reward_zone = []
for reward_time in reward_change_times_flat:
    # Count licks in the 1 second before reward_time
    count = int(np.sum((lick_bout_times >= (reward_time - 1)) & (lick_bout_times < reward_time)))
    licks_before_reward_zone.append(count)

#print("Number of lick bouts in 1s before each reward zone:", licks_before_reward_zone)
average_licks_before_reward_zone = np.mean(licks_before_reward_zone)
print("Average number of lick bouts in 1s before each reward zone:", average_licks_before_reward_zone)

licks_after_reward = []
for reward_time in reward_times:
    # Count licks in the 1 second after reward_time
    count = int(np.sum((lick_bout_times >= reward_time) & (lick_bout_times < (reward_time + 1))))
    licks_after_reward.append(count)

#print("Number of lick bouts in 1s after each reward_time:", licks_after_reward)
average_licks_after_reward = np.mean(licks_after_reward)
print("Average number of lick bouts in 1s after each reward_time:", average_licks_after_reward)

ratio_licks_before_reward_to_before_zone = average_licks_before_reward / average_licks_before_reward_zone
print("Ratio of licks before reward delivery to licks before reward zone:", ratio_licks_before_reward_to_before_zone)

def append_lick_metrics_to_csv(csv_path, row_index, avg_before_reward, avg_before_zone, avg_after_reward, ratio):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Add columns if they don't exist
    for col in [
        'AvgLicks_BeforeZone', 
        'AvgLicks_BeforeReward', 
        'AvgLicks_AfterReward', 
        'Ratio_BeforeReward_to_BeforeZone'
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Update the specified row
    df.at[row_index, 'AvgLicks_BeforeZone'] = avg_before_zone
    df.at[row_index, 'AvgLicks_BeforeReward'] = avg_before_reward
    df.at[row_index, 'AvgLicks_AfterReward'] = avg_after_reward
    df.at[row_index, 'Ratio_BeforeReward_to_BeforeZone'] = ratio

    # Save back to CSV (overwrite)
    df.to_csv(csv_path, index=False)
    print(f"Updated row {row_index} in {os.path.basename(csv_path)}.")

if __name__ == "__main__":
    # Prompt the user for the row index
    row_index = int(input("Enter the row index (0-based) to update in the CSV: "))

    append_lick_metrics_to_csv(
        r'Progress_Reports/buddy_log.csv',
        row_index=row_index,
        avg_before_reward=average_licks_before_reward,
        avg_before_zone=average_licks_before_reward_zone,
        avg_after_reward=average_licks_after_reward,
        ratio=ratio_licks_before_reward_to_before_zone
    )