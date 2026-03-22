"""
RITA Core — Double DQN Agent
NiftyTradingEnv (gymnasium) + Double DQN (stable-baselines3 DQN).

Double DQN reduces Q-value overestimation by:
  - Online network: selects the best action
  - Target network: evaluates that action's Q-value
stable-baselines3 DQN implements this natively.

State (9 features — bull model):
  [daily_return, rsi_norm, macd_norm, bb_pct_b, trend_score,
   current_allocation, days_remaining_norm, atr_norm, ema_ratio_norm]

  atr_norm      = atr_14 / atr_mean      (ratio to historical avg ATR)
  ema_ratio_norm = clip((ema_26/ema_50 - 1.0) * 20, -3, 3)  (Feature 9 — regime signal)

Action (Discrete 3):
  0 → 0%  invested (Cash)
  1 → 50% invested (Half)
  2 → 100% invested (Full)

Reward:
  Bull mode (v1.4):
    reward = portfolio_return
           - 0.005 if cumulative_drawdown < -10%  (flat per-step penalty)

  Bear mode:
    reward = portfolio_return
           - max(0, (|drawdown| - 0.03) * 1.0)   (proportional penalty beyond -3%)

    Teaches aggressive capital protection: the deeper the drawdown below -3%,
    the heavier the per-step penalty (at -5%: 0.02/step ≈ 5× typical daily return).

Backward compatibility:
  Existing 7- and 8-feature models load and run without retraining.
  run_episode() detects model obs-space shape and builds obs accordingly.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from .performance import compute_all_metrics


# ─── Training progress callback ───────────────────────────────────────────────

class TrainingProgressCallback(BaseCallback):
    """
    Records TD loss and mean episode reward at regular intervals during training.
    Saves to {output_dir}/training_progress.csv on training end.

    Columns: timestep, loss, ep_rew_mean
    """

    def __init__(self, log_interval: int = 1_000, output_dir: str = ""):
        super().__init__(verbose=0)
        self.log_interval = log_interval
        self.output_dir = output_dir
        self.records: list = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            vals = self.model.logger.name_to_value
            self.records.append({
                "timestep":    self.num_timesteps,
                "loss":        vals.get("train/loss", float("nan")),
                "ep_rew_mean": vals.get("rollout/ep_rew_mean", float("nan")),
            })
        return True

    def _on_training_end(self) -> None:
        if self.records and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            pd.DataFrame(self.records).to_csv(
                os.path.join(self.output_dir, "training_progress.csv"), index=False
            )


# ─── Reward hyper-params (Bull model) ────────────────────────────────────────
MARKET_PENALTY     = 0.5     # unused in v1.4 but kept for reference
DRAWDOWN_SCALE     = 0.3     # unused in v1.4 but kept for reference
DRAWDOWN_THRESHOLD = -0.10   # -10% — flat penalty threshold for bull model

# ─── Reward hyper-params (Bear model) ────────────────────────────────────────
BEAR_DRAWDOWN_THRESHOLD = -0.03   # -3% — tighter threshold for capital preservation
BEAR_DRAWDOWN_SCALE     = 1.0     # proportional penalty: at -5% → 0.02/step ≈ 5× daily return


class NiftyTradingEnv(gym.Env):
    """
    Custom gymnasium environment for Nifty 50 index trading.

    Each episode covers a random 252-day (≈1 year) window from the training data.

    bear_mode=True:
        Uses tighter drawdown threshold (-3%) with proportional penalty.
        Designed for bear-market episodes to teach aggressive capital protection.
        Requires `ema_ratio` column in df (Feature 9).
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, episode_length: int = 252, bear_mode: bool = False):
        super().__init__()

        self._bear_mode = bear_mode

        # Base required cols; ema_ratio added when available (Feature 9)
        self._base_cols = [
            "daily_return", "rsi_14", "macd", "macd_signal",
            "bb_pct_b", "trend_score", "Close", "atr_14",
        ]
        has_ema_ratio = "ema_ratio" in df.columns and not df["ema_ratio"].isna().all()
        self._use_ema_ratio = has_ema_ratio
        self._n_features = 9 if has_ema_ratio else 8

        required_cols = self._base_cols + (["ema_ratio"] if has_ema_ratio else [])
        self.df = df.dropna(subset=required_cols).copy()
        self.episode_length = min(episode_length, len(self.df) - 1)

        # Normalisation stats (computed once)
        self._macd_std = float(self.df["macd"].std()) or 1.0
        self._atr_mean = float(self.df["atr_14"].mean()) or 1.0

        # Observation space — 8 or 9 features
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self._n_features,), dtype=np.float32
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
        obs_list = [
            float(np.clip(row["daily_return"] * 10, -3, 3)),            # scaled daily return
            float(np.clip(row["rsi_14"] / 100.0, 0, 1)),                # RSI normalised 0-1
            float(np.clip(row["macd"] / (self._macd_std * 3), -3, 3)),  # MACD z-score
            float(np.clip(row["bb_pct_b"], -0.5, 1.5)),                 # %B (can exceed 0-1)
            float(np.clip(row["trend_score"], -1, 1)),                   # trend score
            float(self._current_allocation),                             # 0, 0.5, or 1.0
            float(1.0 - self._step_idx / self.episode_length),           # days remaining
            float(np.clip(row["atr_14"] / self._atr_mean, 0, 3)),       # ATR ratio (1=avg vol)
        ]
        if self._use_ema_ratio:
            # Feature 9: EMA-26/EMA-50 regime signal
            # ema_ratio stored as raw ratio [0.5, 1.5]; normalize: (ratio-1)*20, clip [-3,3]
            ema_ratio_norm = float(np.clip((row["ema_ratio"] - 1.0) * 20, -3, 3))
            obs_list.append(ema_ratio_norm)
        return np.array(obs_list, dtype=np.float32)

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

        # Reward — bull vs bear mode
        if self._bear_mode:
            # Proportional penalty beyond -3%: teaches aggressive capital protection
            excess_dd = max(0.0, abs(current_dd) - abs(BEAR_DRAWDOWN_THRESHOLD))
            reward = portfolio_ret - excess_dd * BEAR_DRAWDOWN_SCALE
        else:
            # Bull mode v1.4 — flat penalty when drawdown exceeds -10%
            reward = portfolio_ret
            if drawdown_exceeded:
                reward -= 0.005  # ~0.5% penalty per step; same scale as a bad day

        self._step_idx += 1
        terminated = self._step_idx >= self.episode_length
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self._n_features, dtype=np.float32)
        info = {
            "portfolio_value": self._portfolio_value,
            "allocation": self._current_allocation,
            "drawdown": current_dd,
        }
        return obs, reward, terminated, truncated, info


def train_agent(
    train_df: pd.DataFrame,
    output_dir: str,
    timesteps: int = 500_000,
    seed: int = 42,
    verbose: int = 1,
    bear_mode: bool = False,
    model_name: str = "rita_ddqn_model",
) -> Tuple[DQN, dict]:
    """
    Train a Double DQN agent on the training DataFrame.

    Key fixes vs original:
    - Monitor wrapper  → ep_rew_mean now logged correctly
    - Soft target updates (tau=0.005, interval=1) → prevents early Q-value collapse
    - Extended exploration (fraction=0.5) → agent explores for half the budget
    - seed param → reproducible runs; used by train_best_of_n for multi-seed selection
    - bear_mode=True → uses bear reward function (proportional penalty from -3%)
    - model_name → allows saving bull and bear models to different files
    """
    from stable_baselines3.common.monitor import Monitor

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)

    env = Monitor(NiftyTradingEnv(train_df, bear_mode=bear_mode))

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=64,
        tau=0.005,                       # soft target updates every step → prevents early Q-value lock-in
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1,        # apply soft update every step (tau controls the blend)
        exploration_fraction=0.5,        # explore for 50% of budget → model won't exploit too early
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [256, 256]},
        seed=seed,
        verbose=verbose,
    )

    progress_cb = TrainingProgressCallback(log_interval=1_000, output_dir=output_dir)
    model.learn(total_timesteps=timesteps, callback=progress_cb)
    model.save(model_path)

    training_metrics = {
        "model_path": model_path + ".zip",
        "timesteps_trained": timesteps,
        "algorithm": "Double DQN (stable-baselines3)",
        "training_period": f"{train_df.index.min().date()} to {train_df.index.max().date()}",
        "training_days": len(train_df),
        "seed": seed,
        "bear_mode": bear_mode,
    }
    return model, training_metrics


def train_best_of_n(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    timesteps: int = 500_000,
    n_seeds: int = 5,
    verbose: int = 1,
    bear_mode: bool = False,
    model_name: str = "rita_ddqn_model",
) -> Tuple[DQN, dict]:
    """
    Train n_seeds models with different random seeds, return the one with
    the best validation Sharpe ratio.  The winner is saved as the main model.

    This addresses the seed-sensitivity of DQN: rather than hoping for a lucky
    single run, we systematically search across seeds and keep the best.

    bear_mode=True: uses bear reward + saves to model_name (default: rita_ddqn_bear_model).
    """
    best_sharpe = -float("inf")
    best_model  = None
    best_seed   = -1
    seed_results = []

    for seed in range(n_seeds):
        print(f"\n[RITA] Training seed {seed + 1}/{n_seeds} (bear_mode={bear_mode}) ...")
        model, metrics = train_agent(
            train_df, output_dir, timesteps=timesteps, seed=seed, verbose=0,
            bear_mode=bear_mode, model_name=model_name,
        )
        result = validate_agent(model, val_df)
        val_sharpe = result["sharpe_ratio"]
        seed_results.append({"seed": seed, "val_sharpe": round(val_sharpe, 4)})
        print(f"[RITA]   seed={seed}  val_sharpe={val_sharpe:.4f}")

        if val_sharpe > best_sharpe:
            best_sharpe = val_sharpe
            best_model  = model
            best_seed   = seed

    print(f"\n[RITA] Best seed: {best_seed}  val_sharpe={best_sharpe:.4f}")
    print(f"[RITA] All results: {seed_results}")

    # Save the best model as the canonical model
    model_path = os.path.join(output_dir, model_name)
    best_model.save(model_path)

    training_metrics = {
        "model_path": model_path + ".zip",
        "timesteps_trained": timesteps,
        "algorithm": "Double DQN best-of-N (stable-baselines3)",
        "training_period": f"{train_df.index.min().date()} to {train_df.index.max().date()}",
        "training_days": len(train_df),
        "seed": best_seed,
        "n_seeds_tried": n_seeds,
        "seed_results": seed_results,
        "bear_mode": bear_mode,
    }
    return best_model, training_metrics


def train_bear_model(
    bear_episodes_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    timesteps: int = 200_000,
    n_seeds: int = 3,
    verbose: int = 1,
) -> Tuple[DQN, dict]:
    """
    Train a bear-market specialist model on extracted correction episodes.

    Uses:
    - bear_mode=True reward (proportional penalty beyond -3% drawdown)
    - 300k timesteps (smaller training set, simpler optimal policy)
    - Saves as rita_ddqn_bear_model.zip

    Args:
        bear_episodes_df: Output of get_bear_episodes() — correction-period rows only
        val_df: Full validation set (2023-2024) for Sharpe scoring
        output_dir: Where to save the model
        timesteps: Default 300k (bear episodes are ~600 days total)
        n_seeds: Number of seeds to try (default 5)
    """
    print(f"\n[RITA] Training bear model on {len(bear_episodes_df)} correction-episode rows ...")
    return train_best_of_n(
        train_df=bear_episodes_df,
        val_df=val_df,
        output_dir=output_dir,
        timesteps=timesteps,
        n_seeds=n_seeds,
        verbose=verbose,
        bear_mode=True,
        model_name="rita_ddqn_bear_model",
    )


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

    # Detect obs-space size first so we know whether to drop ema_ratio NaN rows
    n_obs_features = model.observation_space.shape[0]

    required = ["daily_return", "rsi_14", "macd", "macd_signal", "bb_pct_b", "trend_score", "Close", "atr_14"]
    # For 9-feature models, also drop rows where ema_ratio is NaN
    # (first ~50 rows of any slice have NaN because EMAs need warmup)
    if n_obs_features >= 9 and "ema_ratio" in test_df.columns:
        required.append("ema_ratio")
    df = test_df.dropna(subset=required).copy()

    if len(df) == 0:
        raise ValueError("test_df has no valid rows after dropping NaN indicators")

    macd_std = float(df["macd"].std()) or 1.0
    atr_mean  = float(df["atr_14"].mean()) or 1.0

    has_ema_ratio = "ema_ratio" in df.columns and not df["ema_ratio"].isna().all()

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
        obs_list = [
            float(np.clip(row["daily_return"] * 10, -3, 3)),
            float(np.clip(row["rsi_14"] / 100.0, 0, 1)),
            float(np.clip(row["macd"] / (macd_std * 3), -3, 3)),
            float(np.clip(row["bb_pct_b"], -0.5, 1.5)),
            float(np.clip(row["trend_score"], -1, 1)),
            float(allocations[-1] if allocations else 0.0),
            float(1.0 - i / len(df)),
        ]
        if n_obs_features >= 8:
            obs_list.append(float(np.clip(row["atr_14"] / atr_mean, 0, 3)))
        if n_obs_features >= 9 and has_ema_ratio:
            obs_list.append(float(np.clip((row["ema_ratio"] - 1.0) * 20, -3, 3)))
        obs = np.array(obs_list, dtype=np.float32)

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


def run_regime_episode(
    bull_model: DQN,
    test_df: pd.DataFrame,
    bear_model: Optional[DQN] = None,
    consecutive_bear_days: int = 3,
) -> dict:
    """
    Run a regime-aware backtest that switches between bull and bear models.

    At each step, checks whether ema_ratio has been below 0.99 for
    `consecutive_bear_days` or more. If yes, uses bear_model; otherwise bull_model.

    If bear_model is None, falls back to run_episode() (bull only).
    Requires `ema_ratio` in test_df (i.e., calculate_indicators already called).

    Returns same structure as run_episode(), with an additional key:
        "regime_series": list of "BULL"/"BEAR" per step
    """
    if bear_model is None or "ema_ratio" not in test_df.columns:
        result = run_episode(bull_model, test_df)
        result["regime_series"] = ["BULL"] * len(result["allocations"])
        return result

    import torch  # local import

    required = ["daily_return", "rsi_14", "macd", "macd_signal", "bb_pct_b", "trend_score",
                "Close", "atr_14", "ema_ratio"]
    df = test_df.dropna(subset=required).copy()
    if len(df) == 0:
        raise ValueError("test_df has no valid rows after dropping NaN indicators")

    # Normalization stats (computed from full test_df for consistency)
    bull_n  = bull_model.observation_space.shape[0]
    bear_n  = bear_model.observation_space.shape[0]
    macd_std = float(df["macd"].std()) or 1.0
    atr_mean = float(df["atr_14"].mean()) or 1.0

    portfolio_value = 1.0
    peak_value = 1.0
    portfolio_values = [1.0]
    benchmark_values = [1.0]
    allocations: list = []
    regime_series: list = []
    dates = [df.index[0]]
    close_prices = [float(df["Close"].iloc[0])]
    q_confidence_list: list = []
    obs_log: list = []

    # Pre-compute ema_ratio bear mask for regime detection
    ratio_series = df["ema_ratio"].values  # shape (N,)

    def _build_obs(row, n_features: int) -> np.ndarray:
        obs_list = [
            float(np.clip(row["daily_return"] * 10, -3, 3)),
            float(np.clip(row["rsi_14"] / 100.0, 0, 1)),
            float(np.clip(row["macd"] / (macd_std * 3), -3, 3)),
            float(np.clip(row["bb_pct_b"], -0.5, 1.5)),
            float(np.clip(row["trend_score"], -1, 1)),
            float(allocations[-1] if allocations else 0.0),
            float(1.0 - len(allocations) / len(df)),
        ]
        if n_features >= 8:
            obs_list.append(float(np.clip(row["atr_14"] / atr_mean, 0, 3)))
        if n_features >= 9:
            obs_list.append(float(np.clip((row["ema_ratio"] - 1.0) * 20, -3, 3)))
        return np.array(obs_list, dtype=np.float32)

    alloc_map = {0: 0.0, 1: 0.5, 2: 1.0}

    for i in range(len(df) - 1):
        # Detect regime at current step: count consecutive bear days up to i
        count = 0
        for j in range(i, max(-1, i - consecutive_bear_days - 1), -1):
            if ratio_series[j] < 0.99:
                count += 1
            else:
                break
        regime = "BEAR" if count >= consecutive_bear_days else "BULL"
        regime_series.append(regime)

        active_model = bear_model if regime == "BEAR" else bull_model
        n_features = bear_n if regime == "BEAR" else bull_n

        row = df.iloc[i]
        obs = _build_obs(row, n_features)
        action, _ = active_model.predict(obs, deterministic=True)

        try:
            obs_t = torch.FloatTensor(obs.reshape(1, -1)).to(active_model.device)
            with torch.no_grad():
                q_vals = active_model.policy.q_net(obs_t).cpu().numpy()[0]
            q_confidence_list.append(float(q_vals.max() - q_vals.min()))
        except Exception:
            q_confidence_list.append(float("nan"))

        allocation = alloc_map[int(action)]
        next_row = df.iloc[i + 1]
        daily_ret = float(next_row["daily_return"])

        portfolio_ret = allocation * daily_ret
        portfolio_value *= (1 + portfolio_ret)
        peak_value = max(peak_value, portfolio_value)
        bench_value = benchmark_values[-1] * (1 + daily_ret)

        portfolio_values.append(portfolio_value)
        benchmark_values.append(bench_value)
        allocations.append(allocation)
        dates.append(df.index[i + 1])
        close_prices.append(float(next_row["Close"]))
        obs_log.append((obs, int(action), row))

    # Use bull model obs for interpretability (always 9 features if available)
    q_values_by_feature = _build_q_value_interpretability(bull_model, obs_log)
    obs_array = np.array([o for o, _, _ in obs_log], dtype=np.float32)
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
        "q_confidence_series": q_confidence_list,
        "observations": obs_array,
        "performance": perf,
        "regime_series": regime_series,
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
