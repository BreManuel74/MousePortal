import pandas as pd
import ast
import numpy as np
import os
import matplotlib.pyplot as plt

class LickAnalysis:
    def __init__(self, trial_log_path, capacitive_path):
        self.trial_log_path = trial_log_path
        self.capacitive_path = capacitive_path

        self.trial_log_df = pd.read_csv(trial_log_path, engine='python')
        self.capacitive_df = pd.read_csv(capacitive_path, comment='/', engine='python')

        self.lick_cutoff = (self.capacitive_df['capacitive_value'].quantile(0.90)) / 2
        #print(f"Using lick cutoff value: {self.lick_cutoff}")
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
        #print(f"[LickAnalysis] min_time: {min_time}, max_time: {max_time}")
        quarter_length = (max_time - min_time) / 4

        # Prepare arrays for reward zone and reward times
        reward_texture_change_time = self.prepare_arrays()
        reward_change_times_flat = reward_texture_change_time.flatten()
        reward_change_times_flat = pd.to_numeric(reward_change_times_flat, errors='coerce')
        reward_change_times_flat = reward_change_times_flat[~np.isnan(reward_change_times_flat)]
        reward_times = pd.to_numeric(self.trial_log_df['reward_event'], errors='coerce').dropna().values

        # Pair each reward with its most recent reward zone
        matched_zone_times = self.get_reward_zone_times_for_rewards(reward_times, reward_change_times_flat)
        valid = ~np.isnan(matched_zone_times)
        reward_times_valid = reward_times[valid]
        matched_zone_times_valid = matched_zone_times[valid]

        # Assign each reward to the quarter where its zone started
        reward_quarter_indices = ((matched_zone_times_valid - min_time) // quarter_length).astype(int)
        reward_quarter_indices = np.clip(reward_quarter_indices, 0, 3)  # Ensure indices are 0-3

        quarters = []
        for i in range(4):
            start = min_time + i * quarter_length
            end = min_time + (i + 1) * quarter_length
            mask = reward_quarter_indices == i
            reward_times_q = reward_times_valid[mask]
            matched_zone_times_q = matched_zone_times_valid[mask]
            reward_delays_q = reward_times_q - matched_zone_times_q
            quarters.append({
                "start": start,
                "end": end,
                "reward_times": reward_times_q,
                "matched_zone_times": matched_zone_times_q,
                "reward_delays": reward_delays_q
            })
        return quarters

    def compute_metrics_for_window(self, start_time, end_time, reward_times=None, matched_zone_times=None):
        lick_bout_times = self.lick_bout_times
        lick_bout_times_window = lick_bout_times[(lick_bout_times >= start_time) & (lick_bout_times < end_time)]

        # Always prepare reward_zone_times_flat
        reward_texture_change_time = self.prepare_arrays()
        reward_zone_times_flat = reward_texture_change_time.flatten()
        reward_zone_times_flat = pd.to_numeric(reward_zone_times_flat, errors='coerce')
        reward_zone_times_flat = reward_zone_times_flat[~np.isnan(reward_zone_times_flat)]

        # If reward_times and matched_zone_times are provided, use them
        if reward_times is not None and matched_zone_times is not None:
            reward_times_valid = reward_times
            matched_zone_times_valid = matched_zone_times
            reward_delays = reward_times_valid - matched_zone_times_valid
        else:
            # Fallback to old logic (for non-quarter use)
            trial_log_window = self.trial_log_df[
                (self.trial_log_df['reward_event'] >= start_time) & (self.trial_log_df['reward_event'] < end_time)
            ]
            if trial_log_window.empty:
                return {
                    "average_licks_before_reward": np.nan,
                    "average_licks_before_reward_zone": np.nan,
                    "average_licks_after_reward": np.nan,
                    "ratio_licks_before_reward_to_before_zone": np.nan,
                    "no_reward_licks_before": np.nan,
                    "no_reward_licks_after": np.nan
                }
            reward_times = pd.to_numeric(trial_log_window['reward_event'], errors='coerce').dropna().values
            matched_zone_times = self.get_reward_zone_times_for_rewards(reward_times, reward_zone_times_flat)
            valid = ~np.isnan(matched_zone_times)
            reward_times_valid = reward_times[valid]
            matched_zone_times_valid = matched_zone_times[valid]
            reward_delays = reward_times_valid - matched_zone_times_valid

        licks_before_reward = [
            int(np.sum((lick_bout_times_window >= t_change) & (lick_bout_times_window < t_reward)))
            for t_change, t_reward in zip(matched_zone_times_valid, reward_times_valid)
        ]
        #print(f"DEBUG: licks_before_reward for {start_time}-{end_time}: {licks_before_reward}")
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

        # --- New logic for reward zones with NO reward delivery ---
        # Find reward zones in window that do not have a reward in window after them
        # (i.e., no reward in [zone_time, end_time))
        reward_zones_in_window = reward_zone_times_flat[(reward_zone_times_flat >= start_time) & (reward_zone_times_flat < end_time)]
        #print(f"\nDEBUG: reward_zones_in_window for {start_time}-{end_time}: {reward_zones_in_window}")
        #print(f"DEBUG: reward_times in window: {reward_times}")

        no_reward_zones = []
        # For each reward zone in the window, check if it is the most recent zone before any reward in the window
        for t_zone in reward_zones_in_window:
            # For each reward in the window, find the most recent zone before it
            is_zone_used = False
            for t_reward in reward_times:
                prior_zones = reward_zone_times_flat[reward_zone_times_flat < t_reward]
                if len(prior_zones) > 0 and prior_zones[-1] == t_zone:
                    is_zone_used = True
                    break
            #print(f"  Checking zone {t_zone}: is_zone_used_for_reward = {is_zone_used}")
            if not is_zone_used:
                no_reward_zones.append(t_zone)
        #print(f"DEBUG: no_reward_zones identified: {no_reward_zones}")
        # Calculate licks before and after for these zones
        no_reward_licks_before = [
            int(np.sum((lick_bout_times_window >= (t_zone - 2)) & (lick_bout_times_window < t_zone)))
            for t_zone in no_reward_zones
        ]
        no_reward_licks_after = [
            int(np.sum((lick_bout_times_window >= t_zone) & (lick_bout_times_window < t_zone + 2)))
            for t_zone in no_reward_zones
        ]
        avg_no_reward_licks_before = np.mean(no_reward_licks_before) if no_reward_licks_before else np.nan
        avg_no_reward_licks_after = np.mean(no_reward_licks_after) if no_reward_licks_after else np.nan

        return {
            "average_licks_before_reward": average_licks_before_reward,
            "average_licks_before_reward_zone": average_licks_before_reward_zone,
            "average_licks_after_reward": average_licks_after_reward,
            "ratio_licks_before_reward_to_before_zone": ratio_licks_before_reward_to_before_zone,
            "no_reward_licks_before": avg_no_reward_licks_before,
            "no_reward_licks_after": avg_no_reward_licks_after
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

    def analyze_zones_without_rewards(self):
        min_time = self.capacitive_df['elapsed_time'].min()
        max_time = self.capacitive_df['elapsed_time'].max()
        quarter_length = (max_time - min_time) / 4

        reward_texture_change_time = self.prepare_arrays()
        reward_change_times_flat = reward_texture_change_time.flatten()
        reward_change_times_flat = pd.to_numeric(reward_change_times_flat, errors='coerce')
        reward_change_times_flat = reward_change_times_flat[~np.isnan(reward_change_times_flat)]
        reward_times = pd.to_numeric(self.trial_log_df['reward_event'], errors='coerce').dropna().values
        lick_bout_times = self.lick_bout_times

        results = []
        for i in range(4):
            start = min_time + i * quarter_length
            end = min_time + (i + 1) * quarter_length

            # Reward zones in this quarter
            zones_in_quarter = reward_change_times_flat[(reward_change_times_flat >= start) & (reward_change_times_flat < end)]
            # Rewards in this quarter
            rewards_in_quarter = reward_times[(reward_times >= start) & (reward_times < end)]

            if len(zones_in_quarter) > 0 and len(rewards_in_quarter) == 0:
                licks_before = []
                licks_after = []
                for t_zone in zones_in_quarter:
                    licks_before.append(int(np.sum((lick_bout_times >= (t_zone - 2)) & (lick_bout_times < t_zone))))
                    licks_after.append(int(np.sum((lick_bout_times >= t_zone) & (lick_bout_times < t_zone + 2))))
                avg_licks_before = np.mean(licks_before) if licks_before else np.nan
                avg_licks_after = np.mean(licks_after) if licks_after else np.nan
                results.append({
                    "quarter": i + 1,
                    "start": start,
                    "end": end,
                    "avg_licks_2s_before_zone": avg_licks_before,
                    "avg_licks_2s_after_zone": avg_licks_after,
                    "n_zones": len(zones_in_quarter)
                })
        return results

class SpeedAnalysis:
    def __init__(self, trial_log_path, capacitive_path, treadmill_path):
        self.trial_log_path = trial_log_path
        self.capacitive_path = capacitive_path
        self.treadmill_path = treadmill_path

        self.trial_log_df = pd.read_csv(trial_log_path, engine='python')
        self.capacitive_df = pd.read_csv(capacitive_path, comment='/', engine='python')
        self.treadmill_df = pd.read_csv(treadmill_path, comment='/', engine='python')

        # Interpolate treadmill speed to capacitive elapsed_time
        self.treadmill_interp = pd.Series(
            data=np.interp(
                self.capacitive_df['elapsed_time'],
                self.treadmill_df['global_time'],
                self.treadmill_df['speed']
            ),
            index=self.capacitive_df['elapsed_time']
        )

    def get_session_quarters(self):
        min_time = self.capacitive_df['elapsed_time'].min()
        max_time = self.capacitive_df['elapsed_time'].max()
        quarter_length = (max_time - min_time) / 4
        #print(f"[SpeedAnalysis] min_time: {min_time}, max_time: {max_time}")

        quarters = []
        for i in range(4):
            start = min_time + i * quarter_length
            end = min_time + (i + 1) * quarter_length
            quarters.append({"start": start, "end": end})
        return quarters

    def compute_metrics_for_window(self, start_time, end_time):
        # Get the interpolated speed values in this window
        mask = (self.treadmill_interp.index >= start_time) & (self.treadmill_interp.index < end_time)
        speeds_in_window = self.treadmill_interp[mask]
        avg_speed = speeds_in_window.mean() if not speeds_in_window.empty else np.nan

        return {
            "average_speed": avg_speed
        }

    def compute_speed_metrics_for_window(self, start_time, end_time, reward_times=None, matched_zone_times=None):
        # Restrict to the current quarter
        speed_times = self.treadmill_interp.index
        speeds = self.treadmill_interp.values

        # Always prepare reward_zone_times_flat
        reward_texture_change_time = LickAnalysis(self.trial_log_path, self.capacitive_path).prepare_arrays()
        reward_zone_times_flat = reward_texture_change_time.flatten()
        reward_zone_times_flat = pd.to_numeric(reward_zone_times_flat, errors='coerce')
        reward_zone_times_flat = reward_zone_times_flat[~np.isnan(reward_zone_times_flat)]

        # If reward_times and matched_zone_times are provided, use them
        if reward_times is not None and matched_zone_times is not None:
            reward_times_valid = reward_times
            matched_zone_times_valid = matched_zone_times
            reward_delays = reward_times_valid - matched_zone_times_valid
        else:
            # Fallback to old logic (for non-quarter use)
            trial_log_window = self.trial_log_df[
                (self.trial_log_df['reward_event'] >= start_time) & (self.trial_log_df['reward_event'] < end_time)
            ]
            if trial_log_window.empty:
                return {
                    "average_speed_before_reward": np.nan,
                    "average_speed_before_reward_zone": np.nan,
                    "average_speed_after_reward": np.nan,
                    "no_reward_speed_before": np.nan,
                    "no_reward_speed_after": np.nan
                }
            reward_times = pd.to_numeric(trial_log_window['reward_event'], errors='coerce').dropna().values
            matched_zone_times = LickAnalysis.get_reward_zone_times_for_rewards(reward_times, reward_zone_times_flat)
            valid = ~np.isnan(matched_zone_times)
            reward_times_valid = reward_times[valid]
            matched_zone_times_valid = matched_zone_times[valid]
            reward_delays = reward_times_valid - matched_zone_times_valid

        # Average speed before reward (between zone and reward)
        speeds_before_reward = [
            self.treadmill_interp[(speed_times >= t_change) & (speed_times < t_reward)].mean()
            for t_change, t_reward in zip(matched_zone_times_valid, reward_times_valid)
            if np.any((speed_times >= t_change) & (speed_times < t_reward))
        ]
        avg_speed_before_reward = np.nanmean(speeds_before_reward) if speeds_before_reward else np.nan

        # Average speed before reward zone (same logic as licks)
        speeds_before_reward_zone = [
            self.treadmill_interp[(speed_times >= (t_change - delay)) & (speed_times < t_change)].mean()
            for t_change, delay in zip(matched_zone_times_valid, reward_delays)
            if np.any((speed_times >= (t_change - delay)) & (speed_times < t_change))
        ]
        avg_speed_before_reward_zone = np.nanmean(speeds_before_reward_zone) if speeds_before_reward_zone else np.nan

        # Average speed after reward (from reward to reward+delay)
        speeds_after_reward = [
            self.treadmill_interp[(speed_times >= t_reward) & (speed_times < (t_reward + delay))].mean()
            for t_reward, delay in zip(reward_times_valid, reward_delays)
            if np.any((speed_times >= t_reward) & (speed_times < (t_reward + delay)))
        ]
        avg_speed_after_reward = np.nanmean(speeds_after_reward) if speeds_after_reward else np.nan

        # --- New logic for reward zones with NO reward delivery ---
        reward_zones_in_window = reward_zone_times_flat[(reward_zone_times_flat >= start_time) & (reward_zone_times_flat < end_time)]

        no_reward_zones = []
        for t_zone in reward_zones_in_window:
            is_zone_used = False
            for t_reward in reward_times:
                prior_zones = reward_zone_times_flat[reward_zone_times_flat < t_reward]
                if len(prior_zones) > 0 and prior_zones[-1] == t_zone:
                    is_zone_used = True
                    break
            if not is_zone_used:
                no_reward_zones.append(t_zone)

        # Calculate average speed 2s before and after for these zones
        no_reward_speed_before = [
            self.treadmill_interp[(speed_times >= (t_zone - 2)) & (speed_times < t_zone)].mean()
            for t_zone in no_reward_zones
        ]
        no_reward_speed_after = [
            self.treadmill_interp[(speed_times >= t_zone) & (speed_times < t_zone + 2)].mean()
            for t_zone in no_reward_zones
        ]
        avg_no_reward_speed_before = np.nanmean(no_reward_speed_before) if no_reward_speed_before else np.nan
        avg_no_reward_speed_after = np.nanmean(no_reward_speed_after) if no_reward_speed_after else np.nan

        # Calculate ratio, mirroring lick analysis
        ratio_speed_before_reward_to_before_zone = (
            avg_speed_before_reward / avg_speed_before_reward_zone
            if avg_speed_before_reward_zone and avg_speed_before_reward_zone != 0 else np.nan
        )

        return {
            "average_speed_before_reward": avg_speed_before_reward,
            "average_speed_before_reward_zone": avg_speed_before_reward_zone,
            "average_speed_after_reward": avg_speed_after_reward,
            "no_reward_speed_before": avg_no_reward_speed_before,
            "no_reward_speed_after": avg_no_reward_speed_after,
            "ratio_speed_before_reward_to_before_zone": ratio_speed_before_reward_to_before_zone
        }
    
    def analyze_zones_without_rewards(self):
        min_time = self.capacitive_df['elapsed_time'].min()
        max_time = self.capacitive_df['elapsed_time'].max()
        quarter_length = (max_time - min_time) / 4

        # Prepare reward zone and reward times
        reward_texture_change_time = LickAnalysis(self.trial_log_path, self.capacitive_path).prepare_arrays()
        reward_change_times_flat = reward_texture_change_time.flatten()
        reward_change_times_flat = pd.to_numeric(reward_change_times_flat, errors='coerce')
        reward_change_times_flat = reward_change_times_flat[~np.isnan(reward_change_times_flat)]
        reward_times = pd.to_numeric(self.trial_log_df['reward_event'], errors='coerce').dropna().values

        results = []
        for i in range(4):
            start = min_time + i * quarter_length
            end = min_time + (i + 1) * quarter_length

            # Reward zones in this quarter
            zones_in_quarter = reward_change_times_flat[(reward_change_times_flat >= start) & (reward_change_times_flat < end)]
            # Rewards in this quarter
            rewards_in_quarter = reward_times[(reward_times >= start) & (reward_times < end)]

            if len(zones_in_quarter) > 0 and len(rewards_in_quarter) == 0:
                speeds_before = []
                speeds_after = []
                for t_zone in zones_in_quarter:
                    # Average speed 2s before the zone
                    mask_before = (self.treadmill_interp.index >= (t_zone - 2)) & (self.treadmill_interp.index < t_zone)
                    speeds_before.append(self.treadmill_interp[mask_before].mean())
                    # Average speed 2s after the zone
                    mask_after = (self.treadmill_interp.index >= t_zone) & (self.treadmill_interp.index < t_zone + 2)
                    speeds_after.append(self.treadmill_interp[mask_after].mean())
                avg_speed_before = np.nanmean(speeds_before) if speeds_before else np.nan
                avg_speed_after = np.nanmean(speeds_after) if speeds_after else np.nan
                results.append({
                    "quarter": i + 1,
                    "start": start,
                    "end": end,
                    "avg_speed_2s_before_zone": avg_speed_before,
                    "avg_speed_2s_after_zone": avg_speed_after,
                    "n_zones": len(zones_in_quarter)
                })
        return results

class LickMetricsAppender:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
    def append_quarter_ratios(self, row_index, ratios):
        df = pd.read_csv(self.csv_path)
        # Ensure columns exist
        for i in range(4):
            col = f'LickRatio_Q{i+1}'
            if col not in df.columns:
                df[col] = np.nan

        # Write ratios, replacing nan with 0 and rounding to 2 decimals
        for i, ratio in enumerate(ratios):
            col = f'LickRatio_Q{i+1}'
            value = 0 if pd.isna(ratio) else round(ratio, 2)
            df.at[row_index, col] = value

        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} with quarter ratios in {os.path.basename(self.csv_path)}.")

    def append_hits_to_misses_ratios(self, row_index, hits_to_misses_ratios):
        df = pd.read_csv(self.csv_path)
        for i in range(4):
            col = f'HitsToMisses_Q{i+1}'
            if col not in df.columns:
                df[col] = np.nan
            # Ensure dtype is object so we can store strings
            df[col] = df[col].astype(object)
            df.at[row_index, col] = hits_to_misses_ratios[i]
        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} with hits/misses ratios in {os.path.basename(self.csv_path)}.")

class SpeedMetricsAppender:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def append_quarter_speed_ratios(self, row_index, speed_ratios):
        df = pd.read_csv(self.csv_path)
        # Ensure columns exist
        for i in range(4):
            col = f'SpeedRatio_Q{i+1}'
            if col not in df.columns:
                df[col] = np.nan

        # Write ratios, replacing nan with 0 and rounding to 2 decimals
        for i, ratio in enumerate(speed_ratios):
            col = f'SpeedRatio_Q{i+1}'
            value = 0 if pd.isna(ratio) else round(ratio, 2)
            df.at[row_index, col] = value

        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} with speed quarter ratios in {os.path.basename(self.csv_path)}.")

class LickPlotter:
    @staticmethod
    def plot_lick_metrics(df_quarters):
        fig, ax = plt.subplots(figsize=(12, 7))
        quarters = df_quarters['Quarter']
        x = np.arange(len(quarters))
        width = 0.13

        ax.bar(x - 2*width, df_quarters['average_licks_before_reward'], width, label='Licks Before Reward')
        ax.bar(x - width, df_quarters['average_licks_before_reward_zone'], width, label='Licks Before Reward Zone')
        ax.bar(x, df_quarters['average_licks_after_reward'], width, label='Licks After Reward')
        ax.bar(x + width, df_quarters['no_reward_licks_before'], width, label='Licks 2s Before No-Reward Zone')
        ax.bar(x + 2*width, df_quarters['no_reward_licks_after'], width, label='Licks 2s After No-Reward Zone')

        for idx, row in df_quarters.iterrows():
            xpos = x[idx]
            ymax = max(
                row['average_licks_before_reward'],
                row['average_licks_before_reward_zone'],
                row['average_licks_after_reward'],
                0 if np.isnan(row['no_reward_licks_before']) else row['no_reward_licks_before'],
                0 if np.isnan(row['no_reward_licks_after']) else row['no_reward_licks_after']
            )
            ratio = row['ratio_licks_before_reward_to_before_zone']
            ax.text(xpos, ymax + 1, f"Ratio: {ratio:.2f}", ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')
            n_no_reward = row['n_no_reward_zones']
            if n_no_reward > 0:
                ax.text(xpos, ymax + 5, f"No-reward zones: {int(n_no_reward)}", ha='center', va='bottom', fontsize=9, color='purple')

        ax.set_xticks(x)
        ax.set_xticklabels(quarters)
        ax.set_ylabel('Licks / Value')
        ax.set_title('Lick Metrics by Session Quarter')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_lick_metrics_table(df_quarters, table_columns):
        table_data = df_quarters[table_columns].copy()
        table_data = table_data.round(2)
        table_data = table_data.fillna('')

        fig2, ax2 = plt.subplots(figsize=(14, 2 + 0.5 * len(df_quarters)))
        ax2.axis('off')
        mpl_table = ax2.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            loc='center',
            cellLoc='center'
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(10)
        mpl_table.auto_set_column_width(col=list(range(len(table_data.columns))))
        plt.title("Lick Metrics Table by Quarter")
        plt.tight_layout()
        return fig2, ax2

    @staticmethod
    def plot_hits_misses_bar(df_quarters):
        hits = [0 if pd.isna(df_quarters[f'Q{i+1}_hits'].iloc[i]) else df_quarters[f'Q{i+1}_hits'].iloc[i] for i in range(4)]
        misses = [0 if pd.isna(df_quarters[f'Q{i+1}_misses'].iloc[i]) else df_quarters[f'Q{i+1}_misses'].iloc[i] for i in range(4)]
        quarters_labels = [f'Q{i+1}' for i in range(4)]
        x = np.arange(len(quarters_labels))
        width = 0.35

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        rects1 = ax3.bar(x - width/2, hits, width, label='Hits', color='green')
        rects2 = ax3.bar(x + width/2, misses, width, label='Misses', color='red')

        ax3.set_ylabel('Count')
        ax3.set_title('Hits and Misses by Quarter')
        ax3.set_xticks(x)
        ax3.set_xticklabels(quarters_labels)
        ax3.legend()

        # Annotate bars with values
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax3.annotate(f'{int(height)}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        return fig3, ax3

class SpeedPlotter:
    @staticmethod
    def plot_speed_metrics(df_speed_quarters):
        # Fill NaNs with 0 for plotting
        df_plot = df_speed_quarters.fillna(0)

        fig, ax = plt.subplots(figsize=(12, 7))
        quarters = df_plot['Quarter']
        x = np.arange(len(quarters))
        width = 0.13

        ax.bar(x - 2*width, df_plot['average_speed_before_reward'], width, label='Speed Before Reward')
        ax.bar(x - width, df_plot['average_speed_before_reward_zone'], width, label='Speed Before Reward Zone')
        ax.bar(x, df_plot['average_speed_after_reward'], width, label='Speed After Reward')
        ax.bar(x + width, df_plot['no_reward_speed_before'], width, label='Speed 2s Before No-Reward Zone')
        ax.bar(x + 2*width, df_plot['no_reward_speed_after'], width, label='Speed 2s After No-Reward Zone')

        for idx, row in df_plot.iterrows():
            xpos = x[idx]
            ymax = max(
                row['average_speed_before_reward'],
                row['average_speed_before_reward_zone'],
                row['average_speed_after_reward'],
                row['no_reward_speed_before'],
                row['no_reward_speed_after']
            )
            # Annotate with ratio
            ratio = row.get('ratio_speed_before_reward_to_before_zone', 0)
            ax.text(xpos, ymax + 0.5, f"Ratio: {ratio:.2f}", ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')
            if 'n_no_reward_zones' in row:
                n_no_reward = row['n_no_reward_zones'] if not pd.isna(row['n_no_reward_zones']) else 0
                if n_no_reward > 0:
                    ax.text(xpos, ymax + 1.5, f"No-reward zones: {int(n_no_reward)}", ha='center', va='bottom', fontsize=9, color='purple')

        ax.set_xticks(x)
        ax.set_xticklabels(quarters)
        ax.set_ylabel('Speed (units)')
        ax.set_title('Speed Metrics by Session Quarter')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_speed_metrics_table(df_speed_quarters, table_columns):
        table_data = df_speed_quarters[table_columns].copy()
        table_data = table_data.round(2)
        table_data = table_data.fillna(0)  # Fill NaNs with 0

        fig2, ax2 = plt.subplots(figsize=(14, 2 + 0.5 * len(df_speed_quarters)))
        ax2.axis('off')
        mpl_table = ax2.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            loc='center',
            cellLoc='center'
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(10)
        mpl_table.auto_set_column_width(col=list(range(len(table_data.columns))))
        plt.title("Speed Metrics Table by Quarter")
        plt.tight_layout()
        return fig2, ax2

if __name__ == "__main__":
    # File paths
    trial_log_path = r'Kaufman_Project/Algernon/Session52/beh/1749662752trial_log.csv'
    treadmill_path = r'Kaufman_Project/Algernon/Session52/beh/1749662752treadmill.csv'
    capacitive_path = r'Kaufman_Project/Algernon/Session52/beh/1749662752capacitive.csv'
    csv_path = r'Progress_Reports/Algernon_log.csv'

    # Run lick analysis
    analysis = LickAnalysis(trial_log_path, capacitive_path)
  
    # Speed analysis
    speed_analysis = SpeedAnalysis(trial_log_path, capacitive_path, treadmill_path)
    quarters = speed_analysis.get_session_quarters()

    # Collect lick metrics for each quarter
    quarters = analysis.get_session_quarters()
    quarter_data = []
    reward_texture_change_time = analysis.prepare_arrays()
    reward_zone_times_flat = reward_texture_change_time.flatten()
    reward_zone_times_flat = pd.to_numeric(reward_zone_times_flat, errors='coerce')
    reward_zone_times_flat = reward_zone_times_flat[~np.isnan(reward_zone_times_flat)]

    for i, q in enumerate(quarters):
        start = q['start']
        end = q['end']
        metrics_quarter = analysis.compute_metrics_for_window(
            start, end,
            reward_times=q['reward_times'],
            matched_zone_times=q['matched_zone_times']
        )
        metrics_quarter['Quarter'] = f'Q{i+1}'
        metrics_quarter[f'Q{i+1}_hits'] = len(q['reward_times'])
        reward_zones_in_quarter = reward_zone_times_flat[(reward_zone_times_flat >= start) & (reward_zone_times_flat < end)]
        metrics_quarter[f'Q{i+1}_reward_zones'] = len(reward_zones_in_quarter)
        # Add Q#_misses: reward_zones - hits
        metrics_quarter[f'Q{i+1}_misses'] = len(reward_zones_in_quarter) - len(q['reward_times'])
        quarter_data.append(metrics_quarter)
        # Print metrics for each quarter
        # print(f"Quarter {i+1} ({start:.2f} to {end:.2f}):")
        # print(f"  Avg licks before reward: {metrics_quarter['average_licks_before_reward']}")
        # print(f"  Avg licks before reward zone: {metrics_quarter['average_licks_before_reward_zone']}")
        # print(f"  Avg licks after reward: {metrics_quarter['average_licks_after_reward']}")
        # print(f"  Ratio before reward / before zone: {metrics_quarter['ratio_licks_before_reward_to_before_zone']}\n")
        #print(f"Quarter {i+1}: start={q['start']}, end={q['end']}, reward_delays={q['reward_delays']}")
        # # Print if no-reward zone data exists
        # if not np.isnan(metrics_quarter['no_reward_licks_before']) or not np.isnan(metrics_quarter['no_reward_licks_after']):
        #     print(f"  No-reward zones present in this quarter:")
        #     print(f"    Avg licks 2s before zone: {metrics_quarter['no_reward_licks_before']}")
        #     print(f"    Avg licks 2s after zone: {metrics_quarter['no_reward_licks_after']}")
        # else:
        #     print(f"  No-reward zones: None in this quarter")
        # print()

    # Analyze zones without rewards
    zone_analysis_results = analysis.analyze_zones_without_rewards()
    for result in zone_analysis_results:
        print(f"Quarter {result['quarter']}: Avg licks 2s before zone = {result['avg_licks_2s_before_zone']}, "
              f"Avg licks 2s after zone = {result['avg_licks_2s_after_zone']}, "
              f"Number of zones = {result['n_zones']}")

    # Analyze speed for reward zones with no reward delivery (per quarter)
    for i, q in enumerate(quarters):
        start = q['start']
        end = q['end']
        speed_metrics = speed_analysis.compute_speed_metrics_for_window(
            start, end,
            reward_times=q['reward_times'],
            matched_zone_times=q['matched_zone_times']
        )
        # print(f"Quarter {i+1}: Avg speed before reward: {speed_metrics['average_speed_before_reward']}, "
        #       f"before reward zone: {speed_metrics['average_speed_before_reward_zone']}, "
        #       f"after reward: {speed_metrics['average_speed_after_reward']}")
        # # Print speed for reward zones with no reward delivery
        # print(f"  No-reward zones present in this quarter (speed):")
        # print(f"    Avg speed 2s before zone: {speed_metrics['no_reward_speed_before']}")
        # print(f"    Avg speed 2s after zone: {speed_metrics['no_reward_speed_after']}")

    # Create DataFrame
    df_quarters = pd.DataFrame(quarter_data)

    # Optionally, add Q#_hits columns for clarity (not strictly necessary if already in DataFrame)
    for i in range(4):
        col_hits = f'Q{i+1}_hits'
        col_reward_zones = f'Q{i+1}_reward_zones'
        col_misses = f'Q{i+1}_misses'
        if col_hits not in df_quarters.columns:
            df_quarters[col_hits] = [row.get(col_hits, 0) for row in quarter_data]
        if col_reward_zones not in df_quarters.columns:
            df_quarters[col_reward_zones] = [row.get(col_reward_zones, 0) for row in quarter_data]
        if col_misses not in df_quarters.columns:
            df_quarters[col_misses] = [row.get(col_misses, 0) for row in quarter_data]

    # Collect ratio values for each quarter
    quarter_ratios = [
        0 if pd.isna(row['ratio_licks_before_reward_to_before_zone']) else row['ratio_licks_before_reward_to_before_zone']
        for _, row in df_quarters.iterrows()
    ]

    # Add no-reward zone counts and averages to the DataFrame
    df_quarters['no_reward_licks_before'] = df_quarters['no_reward_licks_before'].astype(float)
    df_quarters['no_reward_licks_after'] = df_quarters['no_reward_licks_after'].astype(float)
    df_quarters['n_no_reward_zones'] = [
        np.sum(~np.isnan([
            metrics_quarter['no_reward_licks_before'],
            metrics_quarter['no_reward_licks_after']
        ])) if not (np.isnan(metrics_quarter['no_reward_licks_before']) and np.isnan(metrics_quarter['no_reward_licks_after'])) else 0
        for metrics_quarter in quarter_data
    ]

    # Plotting
    LickPlotter.plot_lick_metrics(df_quarters)
    LickPlotter.plot_lick_metrics_table(df_quarters, [
        'Quarter',
        'average_licks_before_reward',
        'average_licks_before_reward_zone',
        'average_licks_after_reward',
        'ratio_licks_before_reward_to_before_zone',
        'no_reward_licks_before',
        'no_reward_licks_after',
        'n_no_reward_zones',
    ])
    LickPlotter.plot_hits_misses_bar(df_quarters)
    #plt.show()

    # Calculate hits to misses ratios for each quarter with explicit indication
    hits_to_misses_ratios = []
    for i in range(4):
        hits_val = df_quarters[f'Q{i+1}_hits'].iloc[i]
        misses_val = df_quarters[f'Q{i+1}_misses'].iloc[i]
        if pd.isna(hits_val) and pd.isna(misses_val):
            hits_to_misses_ratios.append("no_data")
        elif (hits_val == 0 or pd.isna(hits_val)) and (misses_val != 0 and not pd.isna(misses_val)):
            hits_to_misses_ratios.append("no_hits")
        elif (misses_val == 0 or pd.isna(misses_val)) and (hits_val != 0 and not pd.isna(hits_val)):
            hits_to_misses_ratios.append("no_misses")
        elif misses_val == 0 or pd.isna(misses_val):
            hits_to_misses_ratios.append("no_misses")
        else:
            hits_to_misses_ratios.append(hits_val / misses_val)

    # Collect speed metrics for each quarter
    speed_quarter_data = []
    for i, q in enumerate(quarters):
        start = q['start']
        end = q['end']
        speed_metrics_quarter = speed_analysis.compute_speed_metrics_for_window(
            start, end,
            reward_times=q['reward_times'],
            matched_zone_times=q['matched_zone_times']
        )
        speed_metrics_quarter['Quarter'] = f'Q{i+1}'
        speed_quarter_data.append(speed_metrics_quarter)

    # Create DataFrame
    df_speed_quarters = pd.DataFrame(speed_quarter_data)

    # After collecting your speed metrics for each quarter into a DataFrame:
    SpeedPlotter.plot_speed_metrics(df_speed_quarters)
    SpeedPlotter.plot_speed_metrics_table(df_speed_quarters, [
        'Quarter',
        'average_speed_before_reward',
        'average_speed_before_reward_zone',
        'average_speed_after_reward',
        'ratio_speed_before_reward_to_before_zone',
        'no_reward_speed_before',
        'no_reward_speed_after'
    ])
    # Collect speed ratios for each quarter
    speed_quarter_ratios = [
        0 if pd.isna(row['ratio_speed_before_reward_to_before_zone']) else row['ratio_speed_before_reward_to_before_zone']
        for _, row in df_speed_quarters.iterrows()
    ]

    plt.show()

    # Prompt for row index
    row_index = int(input("Enter the row index (0-based) to update in the CSV: "))

    # Append quarter ratios
    appender = LickMetricsAppender(csv_path)
    appender.append_quarter_ratios(row_index, quarter_ratios)
    # Append hits to misses ratios
    appender.append_hits_to_misses_ratios(row_index, hits_to_misses_ratios)
    speed_appender = SpeedMetricsAppender(csv_path)
    speed_appender.append_quarter_speed_ratios(row_index, speed_quarter_ratios)