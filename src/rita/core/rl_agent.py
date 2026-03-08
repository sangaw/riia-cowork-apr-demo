"""
RITA Core — Double DQN Agent
NiftyTradingEnv (gymnasium) + Double DQN (stable-baselines3 DQN).

Double DQN reduces Q-value overestimation by:
  - Online network: selects the best action
  - Target network: evaluates that action's Q-value
stable-baselines3 DQN implements this natively.

State (7 features):
  [daily_return, rsi_norm, macd_norm, bb_pct_b, trend_score,
   current_allocation, days_remaining_norm]

Action (Discrete 3):
  0 → 0%  invested (Cash)
  1 → 50% invested (Half)
  2 → 100% invested (Full)

Reward:
  daily_portfolio_return - 10.0 if cumulative_drawdown > 10%
"""

import math
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from .performance import sharpe_ratio, max_drawdown, compute_all_metrics


# ─── Reward hyper-params ──────────────────────────────────────────────────────
DRAWDOWN_PENALTY = 2.0       # applied per-step when drawdown > 10%
DRAWDOWN_THRESHOLD = -0.10   # -10%
SHARPE_BONUS = 0.001         # small bonus for positive daily return contribution


class NiftyTradingEnv(gym.Env):
    """
    Custom gymnasium environment for Nifty 50 index trading.

    Each episode covers a random 252-day (≈1 year) window from the training data.
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, episode_length: int = 252):
        super().__init__()

        # Features required in df (produced by calculate_indicators)
        self._required_cols = [
            "daily_return", "rsi_14", "macd", "macd_signal",
            "bb_pct_b", "trend_score", "Close",
        ]
        self.df = df.dropna(subset=self._required_cols).copy()
        self.episode_length = min(episode_length, len(self.df) - 1)

        # Normalisation stats (computed once)
        self._macd_std = float(self.df["macd"].std()) or 1.0

        # Spaces
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=cash, 1=half, 2=full

        self._reset_state()

    def _reset_state(self):
        self._step_idx = 0
        self._start_idx = 0
        self._portfolio_value = 1.0
        self._peak_value = 1.0
        self._current_allocation = 0.0
        self._portfolio_history = []

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self._start_idx + self._step_idx]
        obs = np.array([
            float(np.clip(row["daily_return"] * 10, -3, 3)),         # scaled daily return
            float(np.clip(row["rsi_14"] / 100.0, 0, 1)),             # RSI normalised 0-1
            float(np.clip(row["macd"] / (self._macd_std * 3), -3, 3)),  # MACD z-score
            float(np.clip(row["bb_pct_b"], -0.5, 1.5)),              # %B (can exceed 0-1)
            float(np.clip(row["trend_score"], -1, 1)),                # trend score
            float(self._current_allocation),                          # 0, 0.5, or 1.0
            float(1.0 - self._step_idx / self.episode_length),        # days remaining
        ], dtype=np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(0, len(self.df) - self.episode_length - 1)
        self._start_idx = int(self.np_random.integers(0, max_start + 1))
        self._step_idx = 0
        self._portfolio_value = 1.0
        self._peak_value = 1.0
        self._current_allocation = 0.0
        self._portfolio_history = [1.0]
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        alloc_map = {0: 0.0, 1: 0.5, 2: 1.0}
        self._current_allocation = alloc_map[int(action)]

        row = self.df.iloc[self._start_idx + self._step_idx]
        daily_ret = float(row["daily_return"])

        # Portfolio return for this step
        portfolio_ret = self._current_allocation * daily_ret  # cash portion earns 0
        self._portfolio_value *= (1 + portfolio_ret)
        self._portfolio_history.append(self._portfolio_value)

        # Update peak
        self._peak_value = max(self._peak_value, self._portfolio_value)

        # Drawdown check
        current_dd = (self._portfolio_value - self._peak_value) / self._peak_value
        drawdown_exceeded = current_dd < DRAWDOWN_THRESHOLD

        # Reward
        reward = portfolio_ret
        if drawdown_exceeded:
            reward -= DRAWDOWN_PENALTY
        if portfolio_ret > 0:
            reward += SHARPE_BONUS

        self._step_idx += 1
        terminated = self._step_idx >= self.episode_length
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(7, dtype=np.float32)
        info = {
            "portfolio_value": self._portfolio_value,
            "allocation": self._current_allocation,
            "drawdown": current_dd,
        }
        return obs, reward, terminated, truncated, info


def train_agent(
    train_df: pd.DataFrame,
    output_dir: str,
    timesteps: int = 200_000,
    verbose: int = 1,
) -> Tuple[DQN, dict]:
    """
    Train a Double DQN agent on the training DataFrame.

    Args:
        train_df:    Feature-enriched DataFrame (2010-2022)
        output_dir:  Directory to save model zip
        timesteps:   Total training timesteps (default 200,000)
        verbose:     0=silent, 1=progress

    Returns:
        (trained DQN model, training metrics dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "rita_ddqn_model")

    env = NiftyTradingEnv(train_df)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,   # Double DQN target network update
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=verbose,
    )

    model.learn(total_timesteps=timesteps)
    model.save(model_path)

    training_metrics = {
        "model_path": model_path + ".zip",
        "timesteps_trained": timesteps,
        "algorithm": "Double DQN (stable-baselines3)",
        "training_period": f"{train_df.index.min().date()} to {train_df.index.max().date()}",
        "training_days": len(train_df),
    }
    return model, training_metrics


def load_agent(model_path: str) -> DQN:
    """Load a saved DDQN model from disk."""
    return DQN.load(model_path)


def run_episode(model: DQN, test_df: pd.DataFrame) -> dict:
    """
    Run the trained DDQN model through the full test_df sequentially (no random start).
    Used for validation (2023-2024) and backtest (2025).

    Returns a dict with:
        portfolio_values    : list of daily portfolio values
        benchmark_values    : list of Buy-and-Hold values
        allocations         : list of daily allocation (0, 0.5, or 1.0)
        daily_returns       : list of daily portfolio returns
        dates               : list of dates
        close_prices        : list of close prices
        q_values_by_feature : dict for interpretability plots
        performance         : performance metrics dict
    """
    from .performance import compute_all_metrics

    required = ["daily_return", "rsi_14", "macd", "macd_signal", "bb_pct_b", "trend_score", "Close"]
    df = test_df.dropna(subset=required).copy()

    if len(df) == 0:
        raise ValueError("test_df has no valid rows after dropping NaN indicators")

    macd_std = float(df["macd"].std()) or 1.0

    portfolio_value = 1.0
    peak_value = 1.0
    portfolio_values = [1.0]
    benchmark_values = [1.0]
    allocations = []
    dates = [df.index[0]]
    close_prices = [float(df["Close"].iloc[0])]

    # For interpretability: collect (obs, action, q_values) at each step
    obs_log = []
    q_confidence_list: list = []   # per-step Q-value spread (max Q − min Q)

    import torch  # local import — already a dep via stable-baselines3

    for i in range(len(df) - 1):
        row = df.iloc[i]
        obs = np.array([
            float(np.clip(row["daily_return"] * 10, -3, 3)),
            float(np.clip(row["rsi_14"] / 100.0, 0, 1)),
            float(np.clip(row["macd"] / (macd_std * 3), -3, 3)),
            float(np.clip(row["bb_pct_b"], -0.5, 1.5)),
            float(np.clip(row["trend_score"], -1, 1)),
            float(allocations[-1] if allocations else 0.0),
            float(1.0 - i / len(df)),
        ], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)

        # Capture per-step Q-value confidence for the Risk View
        try:
            obs_t = torch.FloatTensor(obs.reshape(1, -1)).to(model.device)
            with torch.no_grad():
                q_vals = model.policy.q_net(obs_t).cpu().numpy()[0]
            q_confidence_list.append(float(q_vals.max() - q_vals.min()))
        except Exception:
            q_confidence_list.append(float("nan"))
        alloc_map = {0: 0.0, 1: 0.5, 2: 1.0}
        allocation = alloc_map[int(action)]

        next_row = df.iloc[i + 1]
        daily_ret = float(next_row["daily_return"])
        bench_ret = daily_ret

        portfolio_ret = allocation * daily_ret
        portfolio_value *= (1 + portfolio_ret)
        peak_value = max(peak_value, portfolio_value)

        bench_value = benchmark_values[-1] * (1 + bench_ret)

        portfolio_values.append(portfolio_value)
        benchmark_values.append(bench_value)
        allocations.append(allocation)
        dates.append(df.index[i + 1])
        close_prices.append(float(next_row["Close"]))
        obs_log.append((obs, int(action), row))

    # Build Q-value interpretability data
    q_values_by_feature = _build_q_value_interpretability(model, obs_log)

    # Raw observation matrix — used by SHAPExplainer
    obs_array = np.array([o for o, _, _ in obs_log], dtype=np.float32)  # shape (N, 7)

    port_arr = np.array(portfolio_values)
    bench_arr = np.array(benchmark_values)
    perf = compute_all_metrics(port_arr, bench_arr)

    return {
        "portfolio_values": portfolio_values,
        "benchmark_values": benchmark_values,
        "allocations": allocations,
        "daily_returns": list(np.diff(port_arr) / port_arr[:-1]),
        "dates": pd.DatetimeIndex(dates),
        "close_prices": close_prices,
        "q_values_by_feature": q_values_by_feature,
        "q_confidence_series": q_confidence_list,   # per-step Q-value spread for Risk View
        "observations": obs_array,                  # raw obs matrix for SHAP
        "performance": perf,
    }


def _build_q_value_interpretability(model: DQN, obs_log: list) -> dict:
    """
    For interpretability: bucket observations by feature quintile,
    collect Q-values per action, return structured dict for plotting.
    """
    import torch

    feature_names = ["RSI", "MACD", "BB %B", "Trend Score"]
    feature_indices = [1, 2, 3, 4]  # indices in the 7-dim obs
    action_labels = ["Cash (0%)", "Half (50%)", "Full (100%)"]

    result = {feat: {} for feat in feature_names}

    if len(obs_log) == 0:
        return result

    obs_array = np.array([o for o, _, _ in obs_log], dtype=np.float32)

    # Compute Q-values for all observations in batch
    obs_tensor = torch.FloatTensor(obs_array).to(model.device)
    with torch.no_grad():
        q_vals = model.policy.q_net(obs_tensor).cpu().numpy()  # shape (N, 3)

    for feat, feat_idx in zip(feature_names, feature_indices):
        feat_values = obs_array[:, feat_idx]
        # Split into 3 buckets by tertile
        low_t, high_t = np.percentile(feat_values, [33, 67])
        buckets = {
            "Low": feat_values < low_t,
            "Mid": (feat_values >= low_t) & (feat_values < high_t),
            "High": feat_values >= high_t,
        }
        result[feat] = {}
        for bucket_name, mask in buckets.items():
            if mask.sum() == 0:
                continue
            bucket_q = q_vals[mask]  # shape (n, 3)
            result[feat][bucket_name] = {
                action_labels[i]: list(bucket_q[:, i]) for i in range(3)
            }

    return result


def validate_agent(model: DQN, validation_df: pd.DataFrame) -> dict:
    """
    Run model on validation data (2023-2024) and check constraints.
    Returns a summary with pass/fail for Sharpe > 1 and MDD < 10%.
    """
    result = run_episode(model, validation_df)
    perf = result["performance"]
    return {
        "validation_period": f"{validation_df.index.min().date()} to {validation_df.index.max().date()}",
        "sharpe_ratio": perf["sharpe_ratio"],
        "max_drawdown_pct": perf["max_drawdown_pct"],
        "portfolio_cagr_pct": perf["portfolio_cagr_pct"],
        "benchmark_cagr_pct": perf["benchmark_cagr_pct"],
        "sharpe_constraint_met": perf["sharpe_constraint_met"],
        "drawdown_constraint_met": perf["drawdown_constraint_met"],
        "constraints_met": perf["constraints_met"],
    }
