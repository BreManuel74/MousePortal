import pandas as pd
import ast
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class LickAnalysis:
    def __init__(self, trial_log_path, capacitive_path):
        self.trial_log_path = trial_log_path
        self.capacitive_path = capacitive_path

        self.trial_log_df = pd.read_csv(trial_log_path, engine='python')
        self.capacitive_df = pd.read_csv(capacitive_path, comment='/', engine='python')

        self.lick_cutoff = (self.capacitive_df['capacitive_value'].quantile(0.99)) / 2
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

        # Enforce minimum reward delay of 1 second
        reward_delays = np.maximum(reward_times_valid - matched_zone_times_valid, 1.0)

        reward_quarter_indices = ((matched_zone_times_valid - min_time) // quarter_length).astype(int)
        reward_quarter_indices = np.clip(reward_quarter_indices, 0, 3)

        quarters = []
        for i in range(4):
            start = min_time + i * quarter_length
            end = min_time + (i + 1) * quarter_length
            mask = reward_quarter_indices == i
            reward_times_q = reward_times_valid[mask]
            matched_zone_times_q = matched_zone_times_valid[mask]
            reward_delays_q = reward_delays[mask]
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
            reward_delays = np.maximum(reward_times_valid - matched_zone_times_valid, 1.0)
            #print(reward_delays)
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
            reward_delays = np.maximum(reward_times_valid - matched_zone_times_valid, 1.0)

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
                if np.isclose(t_zone, t_reward):
                    is_zone_used = True
                    break
                prior_zones = reward_zone_times_flat[reward_zone_times_flat < t_reward]
                if len(prior_zones) > 0 and prior_zones[-1] == t_zone:
                    is_zone_used = True
                    break
            #print(f"  Checking zone {t_zone}: is_zone_used_for_reward = {is_zone_used}")
            #print(f"  Reward times in window: {reward_times}")
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
            "no_reward_licks_after": avg_no_reward_licks_after,
            "n_no_reward_zones": len(no_reward_zones)
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

        is_punish = texture_history_padded[:, 0] == "assets/punish_mean100.jpg"
        punish_array = combined_array[is_punish]
        punish_texture_change_time = punish_array[:, 1, :]
        return punish_texture_change_time

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
    
    def get_puff_quarters(self):
        """Return a list of dicts for each quarter with puff events and puff zones assigned by zone start."""
        min_time = self.capacitive_df['elapsed_time'].min()
        max_time = self.capacitive_df['elapsed_time'].max()
        quarter_length = (max_time - min_time) / 4

        # Prepare arrays for puff zone and puff times
        punish_texture_change_time = self.prepare_arrays()
        punish_zone_times_flat = punish_texture_change_time.flatten()
        punish_zone_times_flat = pd.to_numeric(punish_zone_times_flat, errors='coerce')
        punish_zone_times_flat = punish_zone_times_flat[~np.isnan(punish_zone_times_flat)]
        puff_times = pd.to_numeric(self.trial_log_df['puff_event'], errors='coerce').dropna().values

        # Pair each puff with its most recent puff zone
        matched_puff_zone_times = self.get_puff_zone_times_for_puffs(puff_times, punish_zone_times_flat)
        valid = ~np.isnan(matched_puff_zone_times)
        puff_times_valid = puff_times[valid]
        matched_puff_zone_times_valid = matched_puff_zone_times[valid]

        # Assign each puff to the quarter where its zone started
        puff_quarter_indices = ((matched_puff_zone_times_valid - min_time) // quarter_length).astype(int)
        puff_quarter_indices = np.clip(puff_quarter_indices, 0, 3)  # Ensure indices are 0-3

        quarters = []
        for i in range(4):
            start = min_time + i * quarter_length
            end = min_time + (i + 1) * quarter_length
            mask = puff_quarter_indices == i
            puff_times_q = puff_times_valid[mask]
            matched_puff_zone_times_q = matched_puff_zone_times_valid[mask]
            puff_delays_q = puff_times_q - matched_puff_zone_times_q
            # Puff zones in this quarter
            puff_zones_in_quarter = punish_zone_times_flat[(punish_zone_times_flat >= start) & (punish_zone_times_flat < end)]
            quarters.append({
                "start": start,
                "end": end,
                "puff_times": puff_times_q,
                "matched_puff_zone_times": matched_puff_zone_times_q,
                "puff_delays": puff_delays_q,
                "puff_zones": puff_zones_in_quarter
            })
        return quarters

    def compute_metrics_for_window(self, start_time, end_time):
        # Get the interpolated speed values in this window
        mask = (self.treadmill_interp.index >= start_time) & (self.treadmill_interp.index < end_time)
        speeds_in_window = self.treadmill_interp[mask]
        avg_speed = speeds_in_window.mean() if not speeds_in_window.empty else np.nan

        return {
            "average_speed": avg_speed
        }

    def compute_reward_speed_metrics_for_window(self, start_time, end_time, reward_times=None, matched_zone_times=None):
        speed_times = self.treadmill_interp.index
        speeds = self.treadmill_interp.values

        quarter_mask = (speed_times >= start_time) & (speed_times < end_time)
        speed_times_window = speed_times[quarter_mask]
        treadmill_interp_window = self.treadmill_interp[quarter_mask]

        reward_texture_change_time = LickAnalysis(self.trial_log_path, self.capacitive_path).prepare_arrays()
        reward_zone_times_flat = reward_texture_change_time.flatten()
        reward_zone_times_flat = pd.to_numeric(reward_zone_times_flat, errors='coerce')
        reward_zone_times_flat = reward_zone_times_flat[~np.isnan(reward_zone_times_flat)]

        if reward_times is not None and matched_zone_times is not None:
            reward_times_valid = reward_times
            matched_zone_times_valid = matched_zone_times
            reward_delays = np.maximum(reward_times_valid - matched_zone_times_valid, 1.0)
        else:
            trial_log_window = self.trial_log_df[
                (self.trial_log_df['reward_event'] >= start_time) & (self.trial_log_df['reward_event'] < end_time)
            ]
            if trial_log_window.empty:
                return {
                    "average_speed_before_reward": np.nan,
                    "average_speed_before_reward_zone": np.nan,
                    "average_speed_after_reward": np.nan,
                    "no_reward_speed_before": np.nan,
                    "no_reward_speed_after": np.nan,
                    "ratio_speed_before_reward_to_before_zone": np.nan,
                    "n_no_reward_zones": np.nan
                }
            reward_times = pd.to_numeric(trial_log_window['reward_event'], errors='coerce').dropna().values
            matched_zone_times = LickAnalysis.get_reward_zone_times_for_rewards(reward_times, reward_zone_times_flat)
            valid = ~np.isnan(matched_zone_times)
            reward_times_valid = reward_times[valid]
            matched_zone_times_valid = matched_zone_times[valid]
            reward_delays = np.maximum(reward_times_valid - matched_zone_times_valid, 1.0)

        speeds_before_reward = [
            treadmill_interp_window[(speed_times_window >= t_change) & (speed_times_window < t_reward)].mean()
            for t_change, t_reward in zip(matched_zone_times_valid, reward_times_valid)
            if np.any((speed_times_window >= t_change) & (speed_times_window < t_reward))
        ]
        avg_speed_before_reward = np.nanmean(speeds_before_reward) if speeds_before_reward else np.nan

        speeds_before_reward_zone = [
            treadmill_interp_window[(speed_times_window >= (t_change - delay)) & (speed_times_window < t_change)].mean()
            for t_change, delay in zip(matched_zone_times_valid, reward_delays)
            if np.any((speed_times_window >= (t_change - delay)) & (speed_times_window < t_change))
        ]
        avg_speed_before_reward_zone = np.nanmean(speeds_before_reward_zone) if speeds_before_reward_zone else np.nan

        speeds_after_reward = [
            treadmill_interp_window[(speed_times_window >= t_reward) & (speed_times_window < (t_reward + delay))].mean()
            for t_reward, delay in zip(reward_times_valid, reward_delays)
            if np.any((speed_times_window >= t_reward) & (speed_times_window < (t_reward + delay)))
        ]
        avg_speed_after_reward = np.nanmean(speeds_after_reward) if speeds_after_reward else np.nan

        reward_zones_in_window = reward_zone_times_flat[(reward_zone_times_flat >= start_time) & (reward_zone_times_flat < end_time)]

        no_reward_zones = []
        for t_zone in reward_zones_in_window:
            is_zone_used = False
            for t_reward in reward_times:
                if np.isclose(t_zone, t_reward):
                    is_zone_used = True
                    break
                prior_zones = reward_zone_times_flat[reward_zone_times_flat < t_reward]
                if len(prior_zones) > 0 and prior_zones[-1] == t_zone:
                    is_zone_used = True
                    break
            if not is_zone_used:
                no_reward_zones.append(t_zone)

        no_reward_speed_before = [
            treadmill_interp_window[(speed_times_window >= (t_zone - 2)) & (speed_times_window < t_zone)].mean()
            for t_zone in no_reward_zones
        ]
        no_reward_speed_after = [
            treadmill_interp_window[(speed_times_window >= t_zone) & (speed_times_window < t_zone + 2)].mean()
            for t_zone in no_reward_zones
        ]
        avg_no_reward_speed_before = np.nanmean(no_reward_speed_before) if no_reward_speed_before else np.nan
        avg_no_reward_speed_after = np.nanmean(no_reward_speed_after) if no_reward_speed_after else np.nan

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
            "ratio_speed_before_reward_to_before_zone": ratio_speed_before_reward_to_before_zone,
            "n_no_reward_zones": len(no_reward_zones)
        }

    def compute_puff_speed_metrics_for_window(self, start_time, end_time, puff_times=None, matched_puff_zone_times=None):
        speed_times = self.treadmill_interp.index
        speeds = self.treadmill_interp.values

        quarter_mask = (speed_times >= start_time) & (speed_times < end_time)
        speed_times_window = speed_times[quarter_mask]
        treadmill_interp_window = self.treadmill_interp[quarter_mask]

        punish_texture_change_time = self.prepare_arrays()
        punish_zone_times_flat = punish_texture_change_time.flatten()
        punish_zone_times_flat = pd.to_numeric(punish_zone_times_flat, errors='coerce')
        punish_zone_times_flat = punish_zone_times_flat[~np.isnan(punish_zone_times_flat)]

        if puff_times is not None and matched_puff_zone_times is not None:
            puff_times_valid = puff_times
            matched_puff_zone_times_valid = matched_puff_zone_times
            puff_delays = np.maximum(puff_times_valid - matched_puff_zone_times_valid, 1.0)
        else:
            trial_log_window = self.trial_log_df[
                (self.trial_log_df['puff_event'] >= start_time) & (self.trial_log_df['puff_event'] < end_time)
            ]
            if trial_log_window.empty:
                return {
                    "average_speed_before_puff": np.nan,
                    "average_speed_before_puff_zone": np.nan,
                    "average_speed_after_puff": np.nan,
                    "no_puff_speed_before": np.nan,
                    "no_puff_speed_after": np.nan,
                    "n_no_puff_zones": np.nan,
                    "n_no_reward_zones": np.nan
                }
            puff_times = pd.to_numeric(self.trial_log_df['puff_event'], errors='coerce').dropna().values
            matched_puff_zone_times = self.get_puff_zone_times_for_puffs(puff_times, punish_zone_times_flat)
            valid = ~np.isnan(matched_puff_zone_times)
            puff_times_valid = puff_times[valid]
            matched_puff_zone_times_valid = matched_puff_zone_times[valid]
            puff_delays = np.maximum(puff_times_valid - matched_puff_zone_times_valid, 1.0)

        speeds_before_puff = [
            treadmill_interp_window[(speed_times_window >= t_change) & (speed_times_window < t_puff)].mean()
            for t_change, t_puff in zip(matched_puff_zone_times_valid, puff_times_valid)
            if np.any((speed_times_window >= t_change) & (speed_times_window < t_puff))
        ]
        avg_speed_before_puff = np.nanmean(speeds_before_puff) if speeds_before_puff else np.nan
        #print(f"DEBUG: speeds_before_puff for {start_time}-{end_time}: {avg_speed_before_puff}")

        speeds_before_puff_zone = [
            treadmill_interp_window[(speed_times_window >= (t_change - delay)) & (speed_times_window < t_change)].mean()
            for t_change, delay in zip(matched_puff_zone_times_valid, puff_delays)
            if np.any((speed_times_window >= (t_change - delay)) & (speed_times_window < t_change))
        ]
        avg_speed_before_puff_zone = np.nanmean(speeds_before_puff_zone) if speeds_before_puff_zone else np.nan

        speeds_after_puff = [
            treadmill_interp_window[(speed_times_window >= t_puff) & (speed_times_window < (t_puff + delay))].mean()
            for t_puff, delay in zip(puff_times_valid, puff_delays)
            if np.any((speed_times_window >= t_puff) & (speed_times_window < (t_puff + delay)))
        ]
        avg_speed_after_puff = np.nanmean(speeds_after_puff) if speeds_after_puff else np.nan

        puff_zones_in_window = punish_zone_times_flat[(punish_zone_times_flat >= start_time) & (punish_zone_times_flat < end_time)]

        no_puff_zones = []
        for t_zone in puff_zones_in_window:
            is_zone_used = False
            for t_puff in puff_times_valid:
                if np.isclose(t_zone, t_puff):
                    is_zone_used = True
                    break
                prior_zones = punish_zone_times_flat[punish_zone_times_flat < t_puff]
                if len(prior_zones) > 0 and prior_zones[-1] == t_zone:
                    is_zone_used = True
                    break
            if not is_zone_used:
                no_puff_zones.append(t_zone)

        no_puff_speed_before = [
            treadmill_interp_window[(speed_times_window >= (t_zone - 2)) & (speed_times_window < t_zone)].mean()
            for t_zone in no_puff_zones
        ]
        no_puff_speed_after = [
            treadmill_interp_window[(speed_times_window >= t_zone) & (speed_times_window < t_zone + 2)].mean()
            for t_zone in no_puff_zones
        ]
        avg_no_puff_speed_before = np.nanmean(no_puff_speed_before) if no_puff_speed_before else np.nan
        avg_no_puff_speed_after = np.nanmean(no_puff_speed_after) if no_puff_speed_after else np.nan

        ratio_speed_puffs = (avg_no_puff_speed_before / avg_no_puff_speed_after
            if avg_no_puff_speed_after and avg_no_puff_speed_after != 0 else np.nan
        )

        return {
            "average_speed_before_puff": avg_speed_before_puff,
            "average_speed_before_puff_zone": avg_speed_before_puff_zone,
            "average_speed_after_puff": avg_speed_after_puff,
            "no_puff_speed_before": avg_no_puff_speed_before,
            "no_puff_speed_after": avg_no_puff_speed_after,
            "n_no_puff_zones": len(no_puff_zones),
            "n_no_reward_zones": np.nan,  # Not relevant here, but for merge compatibility
            "ratio_speed_puffs": ratio_speed_puffs
        }
    @staticmethod
    def get_puff_zone_times_for_puffs(puff_times, punish_zone_times):
        """For each puff time, find the most recent punish zone time before it."""
        punish_zone_times = np.sort(punish_zone_times)
        matched_puff_zone_times = []
        for t_puff in puff_times:
            # Only consider punish zone times before the puff
            prior_zones = punish_zone_times[punish_zone_times < t_puff]
            if len(prior_zones) == 0:
                matched_puff_zone_times.append(np.nan)
            else:
                matched_puff_zone_times.append(prior_zones[-1])
        #print(f"DEBUG: matched_puff_zone_times for puffs {puff_times}: {matched_puff_zone_times}")
        return np.array(matched_puff_zone_times)

class LickMetricsAppender:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
    def append_quarter_ratios(self, row_index, ratios):
        df = pd.read_csv(self.csv_path)
        # Ensure columns exist
        for i in range(4):
            col = f'LickRewardRatio_Q{i+1}'
            if col not in df.columns:
                df[col] = np.nan

        # Write ratios, replacing nan with 0 and rounding to 2 decimals
        for i, ratio in enumerate(ratios):
            col = f'LickRewardRatio_Q{i+1}'
            value = 0 if pd.isna(ratio) else round(ratio, 2)
            df.at[row_index, col] = value

        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} with lick reward quarter ratios in {os.path.basename(self.csv_path)}.")

class DPrimeAppender:
    def __init__(self, csv_path):
        self.csv_path = csv_path

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

    def append_correct_rejections_to_false_alarms_ratios(self, row_index, correct_rejections_to_false_alarms_ratios):
        df = pd.read_csv(self.csv_path)
        for i in range(4):
            col = f'CorrectRejectionsToFalseAlarms_Q{i+1}'
            if col not in df.columns:
                df[col] = np.nan
            # Ensure dtype is object so we can store strings
            df[col] = df[col].astype(object)
            df.at[row_index, col] = correct_rejections_to_false_alarms_ratios[i]
        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} with correct rejections/false alarms ratios in {os.path.basename(self.csv_path)}.")

class SpeedMetricsAppender:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def append_quarter_speed_ratios(self, row_index, speed_ratios):
        df = pd.read_csv(self.csv_path)
        # Ensure columns exist
        for i in range(4):
            col = f'SpeedRewardRatio_Q{i+1}'
            if col not in df.columns:
                df[col] = np.nan

        # Write ratios, replacing nan with 0 and rounding to 2 decimals
        for i, ratio in enumerate(speed_ratios):
            col = f'SpeedRewardRatio_Q{i+1}'
            value = 0 if pd.isna(ratio) else round(ratio, 2)
            df.at[row_index, col] = value

        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} with speed reward quarter ratios in {os.path.basename(self.csv_path)}.")

    def append_puff_speed_ratios(self, row_index, puff_speed_ratios):
        df = pd.read_csv(self.csv_path)
        # Ensure columns exist
        for i in range(4):
            col = f'SpeedPuffRatio_Q{i+1}'
            if col not in df.columns:
                df[col] = np.nan

        # Write ratios, replacing nan with 0 and rounding to 2 decimals
        for i, ratio in enumerate(puff_speed_ratios):
            col = f'SpeedPuffRatio_Q{i+1}'
            value = 0 if pd.isna(ratio) else round(ratio, 2)
            df.at[row_index, col] = value

        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} with speed puff quarter ratios in {os.path.basename(self.csv_path)}.")

class SessionLengthAppender:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def append_session_length(self, row_index, session_length_minutes):
        df = pd.read_csv(self.csv_path)
        if 'session_length' not in df.columns:
            df['session_length'] = np.nan
        df.at[row_index, 'session_length'] = round(session_length_minutes, 2)
        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} with session length ({session_length_minutes:.2f} min) in {os.path.basename(self.csv_path)}.")

class TrialNumberAppender:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def append_trial_number(self, row_index, n_trials):
        df = pd.read_csv(self.csv_path)
        if 'session_trials' not in df.columns:
            df['session_trials'] = np.nan
        df.at[row_index, 'session_trials'] = int(n_trials)
        df.to_csv(self.csv_path, index=False)
        print(f"Updated row {row_index} with session_trials = {n_trials} in {os.path.basename(self.csv_path)}.")

class LickPlotter:
    @staticmethod
    def plot_lick_metrics(df_quarters, ax=None):
        if ax is None:
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
    
        ax.set_xticks(x)
        ax.set_xticklabels(quarters)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylabel('Licks Bouts')
        ax.set_title('Lick Metrics by Session Quarter')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    @staticmethod
    def plot_lick_metrics_table(df_quarters, table_columns, ax=None):
        table_data = df_quarters[table_columns].copy()
        table_data = table_data.round(2)
        table_data = table_data.fillna(0)

        if ax is None:
            fig2, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(df_quarters)))
        ax.axis('off')
        mpl_table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            loc='center',
            cellLoc='center'
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(10)
        mpl_table.auto_set_column_width(col=list(range(len(table_data.columns))))
        ax.set_title("Lick Metrics Table by Quarter")
        return ax
    
class DPrimePlotter:
    @staticmethod
    def plot_hits_misses_cr_fa_bar(df_quarters, df_puff_quarters, ax=None):
        hits = [0 if pd.isna(df_quarters[f'Q{i+1}_hits'].iloc[i]) else df_quarters[f'Q{i+1}_hits'].iloc[i] for i in range(4)]
        misses = [0 if pd.isna(df_quarters[f'Q{i+1}_misses'].iloc[i]) else df_quarters[f'Q{i+1}_misses'].iloc[i] for i in range(4)]
        correct_rejections = [0 if pd.isna(df_puff_quarters[f'Q{i+1}_correct_rejections'].iloc[i]) else df_puff_quarters[f'Q{i+1}_correct_rejections'].iloc[i] for i in range(4)]
        false_alarms = [0 if pd.isna(df_puff_quarters[f'Q{i+1}_false_alarms'].iloc[i]) else df_puff_quarters[f'Q{i+1}_false_alarms'].iloc[i] for i in range(4)]
        quarters_labels = [f'Q{i+1}' for i in range(4)]
        x = np.arange(len(quarters_labels))
        width = 0.18  # Make bars a bit thinner for four per group

        if ax is None:
            fig3, ax = plt.subplots(figsize=(10, 5))
        rects1 = ax.bar(x - 1.5*width, hits, width, label='Hits', color='green')
        rects2 = ax.bar(x - 0.5*width, misses, width, label='Misses', color='red')
        rects3 = ax.bar(x + 0.5*width, correct_rejections, width, label='Correct Rejections', color='blue')
        rects4 = ax.bar(x + 1.5*width, false_alarms, width, label='False Alarms', color='orange')

        ax.set_ylabel('Count')
        ax.set_title('Hits, Misses, Correct Rejections, False Alarms by Quarter')
        ax.set_xticks(x)
        ax.set_xticklabels(quarters_labels)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Annotate bars with values
        for rect in rects1 + rects2 + rects3 + rects4:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=10)
        return ax

class SpeedPlotter:
    @staticmethod
    def plot_speed_metrics(df_speed_quarters, ax=None):
        df_plot = df_speed_quarters.fillna(0)
        if ax is None:
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
            ratio = row.get('ratio_speed_before_reward_to_before_zone', 0)
            ax.text(xpos, ymax + 0.5, f"Ratio: {ratio:.2f}", ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(quarters)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylabel('Speed')
        ax.set_title('Speed Metrics by Session Quarter')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax
    
    @staticmethod
    def plot_speed_puff_metrics(df_speed_quarters, ax=None):
        df_plot = df_speed_quarters.fillna(0)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
        quarters = df_plot['Quarter']
        x = np.arange(len(quarters))
        width = 0.13

        ax.bar(x - 2*width, df_plot['average_speed_before_puff_puff'], width, label='Speed Before Puff')
        ax.bar(x - width, df_plot['average_speed_before_puff_zone_puff'], width, label='Speed Before Puff Zone')
        ax.bar(x, df_plot['average_speed_after_puff_puff'], width, label='Speed After Puff')
        ax.bar(x + width, df_plot['no_puff_speed_before_puff'], width, label='Speed 2s Before No-Puff Zone')
        ax.bar(x + 2*width, df_plot['no_puff_speed_after_puff'], width, label='Speed 2s After No-Puff Zone')

        for idx, row in df_plot.iterrows():
            xpos = x[idx]
            ymax = max(
                row['average_speed_before_puff_puff'],
                row['average_speed_before_puff_zone_puff'],
                row['average_speed_after_puff_puff'],
                row['no_puff_speed_before_puff'],
                row['no_puff_speed_after_puff']
            )

            ratio = row.get('ratio_speed_puffs_puff', 0)
            ax.text(xpos, ymax + 0.5, f"Ratio: {ratio:.2f}", ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(quarters)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylabel('Speed')
        ax.set_title('Puff Speed Metrics by Session Quarter')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    @staticmethod
    def plot_speed_metrics_table(df_speed_quarters, table_columns, ax=None):
        table_data = df_speed_quarters[table_columns].copy()
        table_data = table_data.round(2)
        table_data = table_data.fillna(0)  # Fill NaNs with 0

        if ax is None:
            fig2, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(df_speed_quarters)))
        ax.axis('off')
        mpl_table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            loc='center',
            cellLoc='center'
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(10)
        mpl_table.auto_set_column_width(col=list(range(len(table_data.columns))))
        ax.set_title("Speed Metrics Table by Quarter")
        return ax

    @staticmethod
    def plot_puff_speed_metrics_table(df_speed_quarters, ax=None):
        """Plot a table of all puff-based speed metrics by quarter."""
        puff_columns = [
            'Quarter',
            'average_speed_before_puff_puff',
            'average_speed_before_puff_zone_puff',
            'average_speed_after_puff_puff',
            'no_puff_speed_before_puff',
            'no_puff_speed_after_puff',
            'n_no_puff_zones_puff',
            'ratio_speed_puffs_puff'
        ]
        table_data = df_speed_quarters[puff_columns].copy()
        table_data = table_data.round(2)
        table_data = table_data.fillna(0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(df_speed_quarters)))
        ax.axis('off')
        mpl_table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            loc='center',
            cellLoc='center'
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(10)
        mpl_table.auto_set_column_width(col=list(range(len(table_data.columns))))
        ax.set_title("Puff-Based Speed Metrics Table by Quarter")
        return ax

if __name__ == "__main__":
    # File paths
    trial_log_path = r'Kaufman_Project/BM14/Session 1/beh/1750699917trial_log.csv'
    treadmill_path = r'Kaufman_Project/BM14/Session 1/beh/1750699918treadmill.csv'
    capacitive_path = r'Kaufman_Project/BM14/Session 1/beh/1750699918capacitive.csv'
    csv_path = r'Progress_Reports/Algernon_log.csv'

    # Prepare the analysis objects
    analysis = LickAnalysis(trial_log_path, capacitive_path)
    max_time = analysis.capacitive_df['elapsed_time'].max()
    session_length_minutes = max_time / 60

    # ---------------- SPEED ANALYSIS SECTION ----------------
    speed_analysis = SpeedAnalysis(trial_log_path, capacitive_path, treadmill_path)

    # Get session quarters for reward-based speed analysis
    quarters = analysis.get_session_quarters()

    # Prepare reward zone times for later use
    reward_texture_change_time = analysis.prepare_arrays()
    reward_zone_times_flat = reward_texture_change_time.flatten()
    reward_zone_times_flat = pd.to_numeric(reward_zone_times_flat, errors='coerce')
    reward_zone_times_flat = reward_zone_times_flat[~np.isnan(reward_zone_times_flat)]

    # Collect speed metrics for each quarter (reward-based)
    speed_quarter_data = []
    for i, q in enumerate(quarters):
        start = q['start']
        end = q['end']
        speed_metrics_quarter = speed_analysis.compute_reward_speed_metrics_for_window(
            start, end,
            reward_times=q['reward_times'],
            matched_zone_times=q['matched_zone_times']
        )
        speed_metrics_quarter['Quarter'] = f'Q{i+1}'
        speed_quarter_data.append(speed_metrics_quarter)

    # Create DataFrame for speed metrics
    df_speed_quarters = pd.DataFrame(speed_quarter_data)
    #print("DEBUG: df_speed_quarters before adding columns:", df_speed_quarters)

    # --- Puff-based speed analysis per quarter ---
    puff_quarters = speed_analysis.get_puff_quarters()
    punish_texture_change_time = speed_analysis.prepare_arrays()
    punish_zone_times_flat = punish_texture_change_time.flatten()
    punish_zone_times_flat = pd.to_numeric(punish_zone_times_flat, errors='coerce')
    punish_zone_times_flat = punish_zone_times_flat[~np.isnan(punish_zone_times_flat)]
    puff_quarter_data = []
    for i, q in enumerate(puff_quarters):
        start = q['start']
        end = q['end']
        metrics_quarter = speed_analysis.compute_puff_speed_metrics_for_window(
            start, end,
            puff_times=q['puff_times'],
            matched_puff_zone_times=q['matched_puff_zone_times']
        )
        metrics_quarter['Quarter'] = f'Q{i+1}'
        metrics_quarter[f'Q{i+1}_false_alarms'] = len(q['puff_times'])
        puff_zones_in_quarter = punish_zone_times_flat[(punish_zone_times_flat >= start) & (punish_zone_times_flat < end)]
        metrics_quarter[f'Q{i+1}_puff_zones'] = len(puff_zones_in_quarter)
        metrics_quarter[f'Q{i+1}_correct_rejections'] = len(puff_zones_in_quarter) - len(q['puff_times'])
        puff_quarter_data.append(metrics_quarter)
    #     print(f"Quarter {i+1} ({start:.2f} to {end:.2f}): Puff times: {q['puff_times']}")
    #     print(f"  Avg speed before puff: {metrics_quarter['average_speed_before_puff']}")
    #     print(f"  Avg speed before puff zone: {metrics_quarter['average_speed_before_puff_zone']}")
    #     print(f"  Avg speed after puff: {metrics_quarter['average_speed_after_puff']}")
    #     print(f"  No-puff zones present in this quarter (speed):")
    #     print(f"    Avg speed 2s before zone: {metrics_quarter['no_puff_speed_before']}")
    #     print(f"    Avg speed 2s after zone: {metrics_quarter['no_puff_speed_after']}")
    #     print(metrics_quarter.get('n_no_puff_zones', 0), "no-puff zones in this quarter")

    # Convert puff quarter data to DataFrame
    df_puff_quarters = pd.DataFrame(puff_quarter_data)
    # Before merging, rename puff columns
    puff_cols = [
        'average_speed_before_puff',
        'average_speed_before_puff_zone',
        'average_speed_after_puff',
        'no_puff_speed_before',
        'no_puff_speed_after',
        'n_no_reward_zones',
        'n_no_puff_zones',
        'ratio_speed_puffs'
    ]
    df_puff_quarters = df_puff_quarters.rename(columns={col: f"{col}_puff" for col in puff_cols})

    # Now merge
    df_speed_quarters = pd.merge(
        df_speed_quarters,
        df_puff_quarters,
        on='Quarter',
        how='left'
    )

    # Ensure all required speed columns are present
    required_speed_columns = [
        'average_speed_before_reward',
        'average_speed_before_reward_zone',
        'average_speed_after_reward',
        'ratio_speed_before_reward_to_before_zone',
        'no_reward_speed_before',
        'no_reward_speed_after',
        'average_speed_before_puff_puff',
        'average_speed_before_puff_zone_puff',
        'average_speed_after_puff_puff',
        'no_puff_speed_before_puff',
        'no_puff_speed_after_puff',
        'n_no_reward_zones_puff',
        'n_no_puff_zones_puff',
        'ratio_speed_puffs_puff'
    ]
    for col in required_speed_columns:
        if col not in df_speed_quarters.columns:
            df_speed_quarters[col] = np.nan


    # # ---------------- LICK ANALYSIS SECTION ----------------

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
        metrics_quarter[f'Q{i+1}_misses'] = len(reward_zones_in_quarter) - len(q['reward_times'])
        quarter_data.append(metrics_quarter)
        #Print metrics for each quarter
        # print(f"Quarter {i+1} ({start:.2f} to {end:.2f}):")
        # print(f"  Avg licks before reward: {metrics_quarter['average_licks_before_reward']}")
        # print(f"  Avg licks before reward zone: {metrics_quarter['average_licks_before_reward_zone']}")
        # print(f"  Avg licks after reward: {metrics_quarter['average_licks_after_reward']}")
        # print(f"  Ratio before reward / before zone: {metrics_quarter['ratio_licks_before_reward_to_before_zone']}\n")
        # print(f"Quarter {i+1}: start={q['start']}, end={q['end']}, reward_delays={q['reward_delays']}")
        # # Print if no-reward zone data exists
        # if not np.isnan(metrics_quarter['no_reward_licks_before']) or not np.isnan(metrics_quarter['no_reward_licks_after']):
        #     print(f"  No-reward zones present in this quarter:")
        #     print(f"    Avg licks 2s before zone: {metrics_quarter['no_reward_licks_before']}")
        #     print(f"    Avg licks 2s after zone: {metrics_quarter['no_reward_licks_after']}")
        # else:
        #     print(f"  No-reward zones: None in this quarter")
        # print(metrics_quarter['n_no_reward_zones'], "no-reward zones in this quarter for licks")

    # Create DataFrame
    df_quarters = pd.DataFrame(quarter_data)



                     ##########D PRIME ANALYSIS SECTION##########



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

    correct_rejections_to_false_alarms_ratios = []
    for i in range(4):
        false_alarms_vals = df_puff_quarters[f'Q{i+1}_false_alarms'].iloc[i]
        correct_rejections_vals = df_puff_quarters[f'Q{i+1}_correct_rejections'].iloc[i]
        if pd.isna(false_alarms_vals) and pd.isna(correct_rejections_vals):
            correct_rejections_to_false_alarms_ratios.append("no_data")
        elif (false_alarms_vals == 0 or pd.isna(false_alarms_vals)) and (correct_rejections_vals != 0 and not pd.isna(correct_rejections_vals)):
            correct_rejections_to_false_alarms_ratios.append("no_false_alarms")
        elif (correct_rejections_vals == 0 or pd.isna(correct_rejections_vals)) and (false_alarms_vals != 0 and not pd.isna(false_alarms_vals)):
            correct_rejections_to_false_alarms_ratios.append("no_correct_rejections")
        elif correct_rejections_vals == 0 or pd.isna(correct_rejections_vals):
            correct_rejections_to_false_alarms_ratios.append("no_correct_rejections")
        else:
            correct_rejections_to_false_alarms_ratios.append(correct_rejections_vals / false_alarms_vals)


              
              
              
              
              
              
               ################ Create subplots ####################



    fig, axs = plt.subplots(4, 1, figsize=(14, 18))
    
    fig_tables, axs_tables = plt.subplots(3, 1, figsize=(14, 9))

    # After collecting your speed metrics for each quarter into a DataFrame:
    SpeedPlotter.plot_speed_metrics_table(df_speed_quarters, [
        'Quarter',
        'average_speed_before_reward',
        'average_speed_before_reward_zone',
        'average_speed_after_reward',
        'ratio_speed_before_reward_to_before_zone',
        'no_reward_speed_before',
        'no_reward_speed_after',
        'n_no_reward_zones',
    ], ax=axs_tables[1])

    SpeedPlotter.plot_puff_speed_metrics_table(df_speed_quarters, ax=axs_tables[2])

    # Plot each metric on its own subplot
    LickPlotter.plot_lick_metrics(df_quarters, ax=axs[0])
    DPrimePlotter.plot_hits_misses_cr_fa_bar(df_quarters, df_puff_quarters, ax=axs[1])
    SpeedPlotter.plot_speed_metrics(df_speed_quarters, ax=axs[2])
    SpeedPlotter.plot_speed_puff_metrics(df_speed_quarters, ax=axs[3])

    
    LickPlotter.plot_lick_metrics_table(df_quarters, [
        'Quarter',
        'average_licks_before_reward',
        'average_licks_before_reward_zone',
        'average_licks_after_reward',
        'ratio_licks_before_reward_to_before_zone',
        'no_reward_licks_before',
        'no_reward_licks_after',
        'n_no_reward_zones',
    ], ax=axs_tables[0])

    plt.tight_layout()
    plt.show()




                       ########Append metrics to CSV#########


    # Prompt for row index
    row_index = int(input("Enter the row index (0-based) to update in the CSV: "))

    # Calculate the number of trials as the total number of objects in the texture_history column
    def count_total_trials(trial_log_df):
        def safe_count(val):
            try:
                if pd.isna(val) or val == '':
                    return 0
                if isinstance(val, list):
                    return len(val)
                if isinstance(val, str):
                    # Try to parse as list
                    import ast
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return len(parsed)
                    return 1
                return 1
            except Exception:
                return 1
        return trial_log_df['texture_history'].apply(safe_count).sum()

    trial_log_df = pd.read_csv(trial_log_path, engine='python')
    n_trials = count_total_trials(trial_log_df)

    # Collect speed ratios for each quarter
    speed_quarter_ratios = [
        0 if pd.isna(row['ratio_speed_before_reward_to_before_zone']) else row['ratio_speed_before_reward_to_before_zone']
        for _, row in df_speed_quarters.iterrows()
    ]

    # Collect ratio values for each quarter
    quarter_ratios = [
        0 if pd.isna(row['ratio_licks_before_reward_to_before_zone']) else row['ratio_licks_before_reward_to_before_zone']
        for _, row in df_quarters.iterrows()
    ]

    # Collect speed puff ratios for each quarter
    speed_puff_ratios = [
        0 if pd.isna(row['ratio_speed_puffs_puff']) else row['ratio_speed_puffs_puff']
        for _, row in df_speed_quarters.iterrows()
    ]

    # Append metrics to CSV
    appender = LickMetricsAppender(csv_path)
    appender.append_quarter_ratios(row_index, quarter_ratios)
    speed_appender = SpeedMetricsAppender(csv_path)
    speed_appender.append_quarter_speed_ratios(row_index, speed_quarter_ratios)
    speed_appender.append_puff_speed_ratios(row_index, speed_puff_ratios)
    session_appender = SessionLengthAppender(csv_path)
    session_appender.append_session_length(row_index, session_length_minutes)
    dprime_appender = DPrimeAppender(csv_path)
    dprime_appender.append_hits_to_misses_ratios(row_index, hits_to_misses_ratios)
    dprime_appender.append_correct_rejections_to_false_alarms_ratios(row_index, correct_rejections_to_false_alarms_ratios)
    trial_appender = TrialNumberAppender(csv_path)
    trial_appender.append_trial_number(row_index, n_trials)