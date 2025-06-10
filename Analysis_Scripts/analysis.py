import pandas as pd
import ast
import numpy as np
import os

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

        licks_before_reward = [
            int(np.sum((lick_bout_times >= t_change) & (lick_bout_times < t_reward)))
            for t_change, t_reward in zip(reward_change_times_flat, reward_times_flat)
        ]
        average_licks_before_reward = np.mean(licks_before_reward)

        licks_before_reward_zone = [
            int(np.sum((lick_bout_times >= (reward_time - 1)) & (lick_bout_times < reward_time)))
            for reward_time in reward_change_times_flat
        ]
        average_licks_before_reward_zone = np.mean(licks_before_reward_zone)

        licks_after_reward = [
            int(np.sum((lick_bout_times >= reward_time) & (lick_bout_times < (reward_time + 1))))
            for reward_time in reward_times
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

class LickMetricsAppender:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def append_metrics(self, row_index, avg_before_reward, avg_before_zone, avg_after_reward, ratio):
        df = pd.read_csv(self.csv_path)
        for col in [
            'AvgLicks_BeforeZone',
            'AvgLicks_BeforeReward',
            'AvgLicks_AfterReward',
            'Ratio_BeforeReward_to_BeforeZone'
        ]:
            if col not in df.columns:
                df[col] = np.nan

        df.at[row_index, 'AvgLicks_BeforeZone'] = avg_before_zone
        df.at[row_index, 'AvgLicks_BeforeReward'] = avg_before_reward
        df.at[row_index, 'AvgLicks_AfterReward'] = avg_after_reward
        df.at[row_index, 'Ratio_BeforeReward_to_BeforeZone'] = ratio

        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} in {os.path.basename(self.csv_path)}.")

if __name__ == "__main__":
    # File paths
    trial_log_path = r'Kaufman_Project\Algernon\Session2\beh\1749576021trial_log.csv'
    treadmill_path = r'Kaufman_Project/Algernon/Session2/beh/1749576021treadmill.csv'
    capacitive_path = r'Kaufman_Project/Algernon/Session2/beh/1749576021capacitive.csv'
    csv_path = r'Progress_Reports/buddy_log.csv'

    # Run analysis
    analysis = LickAnalysis(trial_log_path, treadmill_path, capacitive_path)
    metrics = analysis.compute_metrics()

    # Prompt for row index
    row_index = int(input("Enter the row index (0-based) to update in the CSV: "))

    # Append metrics
    appender = LickMetricsAppender(csv_path)
    appender.append_metrics(
        row_index=row_index,
        avg_before_reward=metrics["average_licks_before_reward"],
        avg_before_zone=metrics["average_licks_before_reward_zone"],
        avg_after_reward=metrics["average_licks_after_reward"],
        ratio=metrics["ratio_licks_before_reward_to_before_zone"]
    )