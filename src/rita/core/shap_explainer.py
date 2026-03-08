"""
RITA Core — SHAP Explainability
Explains the Double DQN agent's allocation decisions using SHAP values.

For each observation (7 features) the Q-network outputs 3 Q-values (Cash / Half / Full).
SHAPExplainer uses shap.DeepExplainer on model.policy.q_net to compute:

  shap_values : np.ndarray  shape (N, 7, 3)
                N samples × 7 features × 3 actions

This answers:
  • Which features matter most overall?       → feature_importance (mean |SHAP|)
  • Which features drive each action?         → per-action importance
  • How does a feature's value affect Q?      → dependence data
  • What drove this specific trade decision?  → per-sample waterfall data

Output files saved to output_dir:
  shap_values.npz          — full SHAP array + raw obs
  shap_importance.csv      — mean |SHAP| per feature per action
  shap_phase_importance.csv — same, broken out by phase
"""

import json
import os

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "Daily Return",
    "RSI (norm)",
    "MACD (z-score)",
    "Bollinger %B",
    "Trend Score",
    "Allocation",
    "Days Remaining",
]
ACTION_NAMES = ["Cash (0%)", "Half (50%)", "Full (100%)"]

N_BACKGROUND = 150    # background samples for DeepExplainer
N_EXPLAIN_MAX = 500   # cap on explain samples (backtest is ~180 so usually all used)


class SHAPExplainer:
    """
    Wraps shap.DeepExplainer around the DDQN Q-network.

    Usage:
        explainer = SHAPExplainer(model)
        explainer.fit(background_obs)             # train-data sample
        shap_vals = explainer.explain(explain_obs) # backtest obs → (N,7,3)
        importance = explainer.feature_importance(shap_vals)
        explainer.save(shap_vals, explain_obs, phases, output_dir)
    """

    def __init__(self, model):
        self.model = model
        self._explainer = None

    # ─── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, background_obs: np.ndarray) -> None:
        """
        Initialise DeepExplainer using a random sample of background observations.
        background_obs: shape (N, 7) — typically from the training episode.
        """
        import shap
        import torch

        n = min(N_BACKGROUND, len(background_obs))
        idx = np.random.choice(len(background_obs), n, replace=False)
        bg = background_obs[idx].astype(np.float32)
        bg_tensor = torch.FloatTensor(bg).to(self.model.device)

        self._explainer = shap.DeepExplainer(self.model.policy.q_net, bg_tensor)

    # ─── Explain ──────────────────────────────────────────────────────────────

    def explain(self, explain_obs: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for explain_obs.

        Args:
            explain_obs: shape (N, 7)

        Returns:
            shap_values: shape (N, 7, 3)  — N samples × 7 features × 3 actions
        """
        import torch

        if self._explainer is None:
            raise RuntimeError("Call fit() before explain().")

        n = min(N_EXPLAIN_MAX, len(explain_obs))
        obs = explain_obs[:n].astype(np.float32)
        obs_tensor = torch.FloatTensor(obs).to(self.model.device)

        raw = self._explainer.shap_values(obs_tensor)

        # Normalise to (N, 7, 3) regardless of shap version:
        #   shap <0.46  → list of 3 arrays each (N, 7)
        #   shap ≥0.46  → single ndarray (3, N, 7)  or  (N, 7, 3)
        if isinstance(raw, list):
            shap_3d = np.stack(raw, axis=-1)             # list[(N,7)] → (N,7,3)
        elif isinstance(raw, np.ndarray):
            if raw.ndim == 3 and raw.shape[0] == 3:
                shap_3d = raw.transpose(1, 2, 0)         # (3,N,7) → (N,7,3)
            elif raw.ndim == 3 and raw.shape[-1] == 3:
                shap_3d = raw                            # already (N,7,3)
            else:
                raise ValueError(f"Unexpected SHAP ndarray shape: {raw.shape}")
        else:
            raise TypeError(f"Unexpected SHAP return type: {type(raw)}")

        return shap_3d

    # ─── Aggregations ─────────────────────────────────────────────────────────

    def feature_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        Mean absolute SHAP per feature per action.

        Returns DataFrame shape (7, 4): features × [Cash, Half, Full, Overall]
        Sorted by Overall importance descending.
        """
        importance = np.abs(shap_values).mean(axis=0)   # (7, 3)
        df = pd.DataFrame(importance, index=FEATURE_NAMES, columns=ACTION_NAMES)
        df["Overall"] = df.mean(axis=1)
        return df.sort_values("Overall", ascending=False)

    def phase_importance(
        self, shap_values: np.ndarray, phases: np.ndarray
    ) -> dict:
        """
        Feature importance broken down by phase (Train / Validation / Backtest).

        phases: 1-D string array aligned with shap_values axis-0.
        Returns dict {phase_name: DataFrame(7,4)}.
        """
        result = {}
        for phase in np.unique(phases):
            mask = phases == phase
            if mask.sum() == 0:
                continue
            result[phase] = self.feature_importance(shap_values[mask])
        return result

    def dependence_data(
        self, shap_values: np.ndarray, obs: np.ndarray
    ) -> dict:
        """
        For each feature, return (feature_value, shap_value_per_action) arrays
        suitable for dependence scatter plots.

        Returns dict:
          {feature_name: {"values": array(N,),
                          "shap_cash": array(N,),
                          "shap_half": array(N,),
                          "shap_full": array(N,)}}
        """
        result = {}
        for fi, fname in enumerate(FEATURE_NAMES):
            result[fname] = {
                "values":     obs[:, fi],
                "shap_cash":  shap_values[:, fi, 0],
                "shap_half":  shap_values[:, fi, 1],
                "shap_full":  shap_values[:, fi, 2],
            }
        return result

    def top_trade_shap(
        self,
        shap_values: np.ndarray,
        obs: np.ndarray,
        allocations: list,
        dates,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        For the N samples with the highest total |SHAP| magnitude (most
        "explained" decisions), return a tidy DataFrame with per-feature
        SHAP contributions and the chosen action.

        Columns: date, action, feature_0…feature_6 SHAP for chosen action.
        """
        n_samples = min(len(shap_values), len(allocations))
        alloc_arr = np.array(allocations[:n_samples])

        # Map allocation → action index
        alloc_map = {0.0: 0, 0.5: 1, 1.0: 2}
        action_idx = np.array([alloc_map.get(float(a), 2) for a in alloc_arr])

        # |SHAP| for chosen action at each step
        shap_chosen = np.array([
            shap_values[i, :, action_idx[i]] for i in range(n_samples)
        ])   # (N, 7)
        magnitude = np.abs(shap_chosen).sum(axis=1)   # (N,)

        top_idx = np.argsort(magnitude)[::-1][:n]

        rows = []
        date_list = list(dates)
        for i in top_idx:
            row = {
                "date": date_list[i + 1] if i + 1 < len(date_list) else date_list[-1],
                "action": ACTION_NAMES[action_idx[i]],
                "total_|shap|": round(float(magnitude[i]), 4),
            }
            for fi, fname in enumerate(FEATURE_NAMES):
                row[fname] = round(float(shap_chosen[i, fi]), 4)
            rows.append(row)

        return pd.DataFrame(rows)

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(
        self,
        shap_values: np.ndarray,
        obs: np.ndarray,
        phases: np.ndarray,
        output_dir: str,
    ) -> None:
        """Persist SHAP arrays and aggregated importance CSVs."""
        os.makedirs(output_dir, exist_ok=True)
        # Align all arrays to the number of explained samples (explain() may cap at N_EXPLAIN_MAX)
        n = len(shap_values)
        obs    = obs[:n]
        phases = phases[:n]

        # Raw arrays
        np.savez_compressed(
            os.path.join(output_dir, "shap_values.npz"),
            shap_values=shap_values,
            observations=obs,
            phases=phases,
            feature_names=np.array(FEATURE_NAMES),
            action_names=np.array(ACTION_NAMES),
        )

        # Global importance CSV
        imp = self.feature_importance(shap_values)
        imp.to_csv(os.path.join(output_dir, "shap_importance.csv"))

        # Per-phase importance CSVs
        phase_imp = self.phase_importance(shap_values, phases)
        rows = []
        for phase, df in phase_imp.items():
            dfp = df.copy()
            dfp.insert(0, "phase", phase)
            rows.append(dfp)
        if rows:
            pd.concat(rows).to_csv(
                os.path.join(output_dir, "shap_phase_importance.csv")
            )

    @staticmethod
    def load(output_dir: str) -> dict | None:
        """
        Load saved SHAP data.  Returns dict or None if files missing.

        Keys:
            shap_values   : (N, 7, 3)
            observations  : (N, 7)
            phases        : (N,) string array
            feature_names : list[str]
            action_names  : list[str]
            importance    : pd.DataFrame   (global)
            phase_importance : dict{phase: pd.DataFrame}
        """
        npz_path = os.path.join(output_dir, "shap_values.npz")
        imp_path = os.path.join(output_dir, "shap_importance.csv")
        if not os.path.exists(npz_path):
            return None

        data = np.load(npz_path, allow_pickle=True)
        result = {
            "shap_values":   data["shap_values"],
            "observations":  data["observations"],
            "phases":        data["phases"],
            "feature_names": list(data["feature_names"]),
            "action_names":  list(data["action_names"]),
        }

        if os.path.exists(imp_path):
            result["importance"] = pd.read_csv(imp_path, index_col=0)

        phase_imp_path = os.path.join(output_dir, "shap_phase_importance.csv")
        if os.path.exists(phase_imp_path):
            df_all = pd.read_csv(phase_imp_path, index_col=0)
            result["phase_importance"] = {
                ph: grp.drop(columns="phase")
                for ph, grp in df_all.groupby("phase")
            }
        return result
