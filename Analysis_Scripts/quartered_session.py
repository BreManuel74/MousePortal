import pandas as pd
import ast
import numpy as np
import os
import matplotlib.pyplot as plt

class LickAnalysis:
    def __init__(self, trial_log_path, treadmill_path, capacitive_path):
        self.trial_log_path = trial_log_path
        self.treadmill_path = treadmill_path
        self.capacitive_path = capacitive_path

        self.trial_log_df = pd.read_csv(trial_log_path, engine='python')
        self.treadmill_df = pd.read_csv(treadmill_path, comment='/', engine='python')
        self.capacitive_df = pd.read_csv(capacitive_path, comment='/', engine='python')

        self.lick_cutoff = (self.capacitive_df['capacitive_value'].quantile(0.90)) / 2
        self.lick_bout_times = self.capacitive_df.loc[
            self.capacitive_df['capacitive_value'] > self.lick_cutoff, 'elapsed_time'
        ].values

    @staticmethod
    def safe_literal_eval(val):
        try:
            if isinstance(val, list):
                return val
            if pd.isna(val) or val == '':
                return []
            if isinstance(val, (int, float)):
                return [val]
            if isinstance(val, str) and not (val.strip().startswith("[") and val.strip().endswith("]")):
                try:
                    return [float(val)]
                except Exception:
                    return [val]
            return ast.literal_eval(val)
        except Exception:
            return []

    def prepare_arrays(self):
        texture_history = self.trial_log_df['texture_history'].apply(self.safe_literal_eval)
        texture_change_time = self.trial_log_df['texture_change_time'].apply(self.safe_literal_eval)
        revert_time = self.trial_log_df['texture_revert'].apply(self.safe_literal_eval)

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

        is_reward = texture_history_padded[:, 0] == "assets/reward_mean100.jpg"
        reward_array = combined_array[is_reward]
        reward_texture_change_time = reward_array[:, 1, :]
        return reward_texture_change_time

    def compute_metrics(self):
        reward_texture_change_time = self.prepare_arrays()
        reward_times = pd.to_numeric(self.trial_log_df['reward_event'], errors='coerce').dropna()
        lick_bout_times = self.lick_bout_times

        reward_change_times_flat = reward_texture_change_time.flatten()
        reward_change_times_flat = pd.to_numeric(reward_change_times_flat, errors='coerce')
        reward_change_times_flat = reward_change_times_flat[~np.isnan(reward_change_times_flat)]

        reward_times_flat = reward_times.values
        if len(reward_change_times_flat) != len(reward_times_flat):
            print("Warning: reward_texture_change_time and reward_times have different lengths!")
            
        reward_delays = [t_reward - t_change for t_change, t_reward in zip(reward_change_times_flat, reward_times_flat)]
        print(f"Reward delays: {reward_delays}")

        licks_before_reward = [
            int(np.sum((lick_bout_times >= t_change) & (lick_bout_times < t_reward)))
            for t_change, t_reward in zip(reward_change_times_flat, reward_times_flat)
        ]
        average_licks_before_reward = np.mean(licks_before_reward)

        licks_before_reward_zone = [
            int(np.sum((lick_bout_times >= (reward_time - reward_delay)) & (lick_bout_times < reward_time)))
            for reward_time in reward_change_times_flat
            for reward_delay in reward_delays
        ]
        average_licks_before_reward_zone = np.mean(licks_before_reward_zone)

        licks_after_reward = [
            int(np.sum((lick_bout_times >= reward_time) & (lick_bout_times < (reward_time + reward_delay))))
            for reward_time in reward_times
            for reward_delay in reward_delays
        ]
        average_licks_after_reward = np.mean(licks_after_reward)

        ratio_licks_before_reward_to_before_zone = (
            average_licks_before_reward / average_licks_before_reward_zone
            if average_licks_before_reward_zone != 0 else np.nan
        )

        return {
            "average_licks_before_reward": average_licks_before_reward,
            "average_licks_before_reward_zone": average_licks_before_reward_zone,
            "average_licks_after_reward": average_licks_after_reward,
            "ratio_licks_before_reward_to_before_zone": ratio_licks_before_reward_to_before_zone
        }

    def get_session_quarters(self):
        """Return a list of dicts for each quarter with (start, end, reward_delays)."""
        min_time = self.capacitive_df['elapsed_time'].min()
        max_time = self.capacitive_df['elapsed_time'].max()
        quarter_length = (max_time - min_time) / 4

        # Prepare arrays for reward zone and reward times
        reward_texture_change_time = self.prepare_arrays()
        reward_change_times_flat = reward_texture_change_time.flatten()
        reward_change_times_flat = pd.to_numeric(reward_change_times_flat, errors='coerce')
        reward_change_times_flat = reward_change_times_flat[~np.isnan(reward_change_times_flat)]
        reward_times = pd.to_numeric(self.trial_log_df['reward_event'], errors='coerce').dropna().values

        quarters = []
        for i in range(4):
            start = min_time + i * quarter_length
            end = min_time + (i + 1) * quarter_length
            mask = (reward_times >= start) & (reward_times < end)
            reward_times_q = reward_times[mask]
            # Do NOT filter reward_change_times_flat to the quarter!
            # Use all reward zone times up to each reward
            matched_zone_times_q = self.get_reward_zone_times_for_rewards(reward_times_q, reward_change_times_flat)
            valid = ~np.isnan(matched_zone_times_q)
            reward_times_q_valid = reward_times_q[valid]
            matched_zone_times_q_valid = matched_zone_times_q[valid]
            reward_delays_q = reward_times_q_valid - matched_zone_times_q_valid
            # print(f"\nDEBUG Quarter {i+1}: {start:.2f} to {end:.2f}")
            # print(f"  reward_times_q: {reward_times_q_valid}")
            # print(f"  matched_zone_times_q: {matched_zone_times_q_valid}")
            # print(f"  reward_delays_q: {reward_delays_q}")
            quarters.append({
                "start": start,
                "end": end,
                "reward_delays": reward_delays_q
            })
        return quarters

    def compute_metrics_for_window(self, start_time, end_time):
        lick_bout_times = self.lick_bout_times
        lick_bout_times_window = lick_bout_times[(lick_bout_times >= start_time) & (lick_bout_times < end_time)]

        # Filter trial_log_df to window
        trial_log_window = self.trial_log_df[
            (self.trial_log_df['reward_event'] >= start_time) & (self.trial_log_df['reward_event'] < end_time)
        ]
        if trial_log_window.empty:
            return {
                "average_licks_before_reward": np.nan,
                "average_licks_before_reward_zone": np.nan,
                "average_licks_after_reward": np.nan,
                "ratio_licks_before_reward_to_before_zone": np.nan
            }

        # Get reward times for this window
        reward_times = pd.to_numeric(trial_log_window['reward_event'], errors='coerce').dropna().values
        reward_texture_change_time = self.prepare_arrays()
        reward_zone_times_flat = reward_texture_change_time.flatten()
        reward_zone_times_flat = pd.to_numeric(reward_zone_times_flat, errors='coerce')
        reward_zone_times_flat = reward_zone_times_flat[~np.isnan(reward_zone_times_flat)]
        # Only use reward zone times before each reward
        matched_zone_times = self.get_reward_zone_times_for_rewards(reward_times, reward_zone_times_flat)
        valid = ~np.isnan(matched_zone_times)
        reward_times_valid = reward_times[valid]
        matched_zone_times_valid = matched_zone_times[valid]
        reward_delays = reward_times_valid - matched_zone_times_valid

        licks_before_reward = [
            int(np.sum((lick_bout_times_window >= t_change) & (lick_bout_times_window < t_reward)))
            for t_change, t_reward in zip(matched_zone_times_valid, reward_times_valid)
        ]
        average_licks_before_reward = int(np.mean(licks_before_reward)) if licks_before_reward else np.nan

        licks_before_reward_zone = [
            int(np.sum((lick_bout_times_window >= (t_change - delay)) & (lick_bout_times_window < t_change)))
            for t_change, delay in zip(matched_zone_times_valid, reward_delays)
        ]
        average_licks_before_reward_zone = int(np.mean(licks_before_reward_zone)) if licks_before_reward_zone else np.nan

        licks_after_reward = [
            int(np.sum((lick_bout_times_window >= t_reward) & (lick_bout_times_window < (t_reward + delay))))
            for t_reward, delay in zip(reward_times_valid, reward_delays)
        ]
        average_licks_after_reward = int(np.mean(licks_after_reward)) if licks_after_reward else np.nan

        ratio_licks_before_reward_to_before_zone = (
            average_licks_before_reward / average_licks_before_reward_zone
            if average_licks_before_reward_zone and average_licks_before_reward_zone != 0 else np.nan
        )

        return {
            "average_licks_before_reward": average_licks_before_reward,
            "average_licks_before_reward_zone": average_licks_before_reward_zone,
            "average_licks_after_reward": average_licks_after_reward,
            "ratio_licks_before_reward_to_before_zone": ratio_licks_before_reward_to_before_zone
        }
    @staticmethod
    def get_reward_zone_times_for_rewards(reward_times, reward_zone_times):
        """For each reward time, find the most recent reward zone time before it."""
        reward_zone_times = np.sort(reward_zone_times)
        #print(f"DEBUG: reward_zone_times used for pairing: {reward_zone_times}")  # <-- Add this line
        matched_zone_times = []
        for t_reward in reward_times:
            # Only consider reward zone times before the reward
            prior_zones = reward_zone_times[reward_zone_times < t_reward]
            if len(prior_zones) == 0:
                matched_zone_times.append(np.nan)
            else:
                matched_zone_times.append(prior_zones[-1])
        return np.array(matched_zone_times)

class LickMetricsAppender:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    # def append_metrics(self, row_index, avg_before_reward, avg_before_zone, avg_after_reward, ratio):
    #     df = pd.read_csv(self.csv_path)
    #     for col in [
    #         'AvgLicks_BeforeZone',
    #         'AvgLicks_BeforeReward',
    #         'AvgLicks_AfterReward',
    #         'Ratio_BeforeReward_to_BeforeZone'
    #     ]:
    #         if col not in df.columns:
    #             df[col] = np.nan

    #     df.at[row_index, 'AvgLicks_BeforeZone'] = avg_before_zone
    #     df.at[row_index, 'AvgLicks_BeforeReward'] = avg_before_reward
    #     df.at[row_index, 'AvgLicks_AfterReward'] = avg_after_reward
    #     df.at[row_index, 'Ratio_BeforeReward_to_BeforeZone'] = ratio

    #     df.to_csv(self.csv_path, index=False)
    #     print(f"Updated row {row_index} in {os.path.basename(self.csv_path)}.")

    def append_quarter_ratios(self, row_index, ratios):
        df = pd.read_csv(self.csv_path)
        # Ensure columns exist
        for i in range(4):
            col = f'Ratio_Q{i+1}'
            if col not in df.columns:
                df[col] = np.nan

        # Write ratios, replacing nan with 0
        for i, ratio in enumerate(ratios):
            col = f'Ratio_Q{i+1}'
            value = 0 if pd.isna(ratio) else ratio
            df.at[row_index, col] = value

        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} with quarter ratios in {os.path.basename(self.csv_path)}.")

if __name__ == "__main__":
    # File paths
    trial_log_path = r'Kaufman_Project/Algernon/Session50/beh/1749651827trial_log.csv'
    treadmill_path = r'Kaufman_Project/Algernon/Session50/beh/1749651827treadmill.csv'
    capacitive_path = r'Kaufman_Project/Algernon/Session50/beh/1749651827capacitive.csv'
    csv_path = r'Progress_Reports/Algernon_log.csv'

    # Run analysis
    analysis = LickAnalysis(trial_log_path, treadmill_path, capacitive_path)
    #metrics = analysis.compute_metrics()

    # Collect metrics for each quarter
    quarters = analysis.get_session_quarters()
    quarter_data = []
    for i, q in enumerate(quarters):
        start = q['start']
        end = q['end']
        metrics_quarter = analysis.compute_metrics_for_window(start, end)
        metrics_quarter['Quarter'] = f'Q{i+1}'
        quarter_data.append(metrics_quarter)
        # Print metrics for each quarter
        print(f"Quarter {i+1} ({start:.2f} to {end:.2f}):")
        print(f"  Avg licks before reward: {metrics_quarter['average_licks_before_reward']}")
        print(f"  Avg licks before reward zone: {metrics_quarter['average_licks_before_reward_zone']}")
        print(f"  Avg licks after reward: {metrics_quarter['average_licks_after_reward']}")
        print(f"  Ratio before reward / before zone: {metrics_quarter['ratio_licks_before_reward_to_before_zone']}\n")
        print(f"Quarter {i+1}: start={q['start']}, end={q['end']}, reward_delays={q['reward_delays']}")

    # Create DataFrame
    df_quarters = pd.DataFrame(quarter_data)

    # Plot only the averages
    ax = df_quarters.set_index('Quarter')[[
        'average_licks_before_reward',
        'average_licks_before_reward_zone',
        'average_licks_after_reward'
    ]].plot(kind='bar', figsize=(10,6))

    plt.ylabel('Value')
    plt.title('Lick Metrics by Session Quarter')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Annotate the ratio above each quarter
    for idx, row in df_quarters.iterrows():
        xpos = idx
        ymax = max(row['average_licks_before_reward'],
                   row['average_licks_before_reward_zone'],
                   row['average_licks_after_reward'])
        ratio = row['ratio_licks_before_reward_to_before_zone']
        plt.text(xpos, ymax + 0.5, f"Ratio: {ratio}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.show()

    # Collect ratio values for each quarter
    quarter_ratios = [
        0 if pd.isna(row['ratio_licks_before_reward_to_before_zone']) else row['ratio_licks_before_reward_to_before_zone']
        for _, row in df_quarters.iterrows()
    ]

    # Prompt for row index
    row_index = int(input("Enter the row index (0-based) to update in the CSV: "))

    # Append quarter ratios
    appender = LickMetricsAppender(csv_path)
    appender.append_quarter_ratios(row_index, quarter_ratios)