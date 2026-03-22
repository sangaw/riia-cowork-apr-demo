"""
Core layer unit tests — no RL training, runs in seconds.
Uses a small synthetic DataFrame so no CSV file is needed.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """200 days of synthetic Nifty-like data with a DatetimeIndex."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    close = 10000 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    df = pd.DataFrame({
        "Open":   close * np.random.uniform(0.99, 1.0, n),
        "High":   close * np.random.uniform(1.0, 1.015, n),
        "Low":    close * np.random.uniform(0.985, 1.0, n),
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)
    return df


@pytest.fixture
def feature_df(synthetic_ohlcv) -> pd.DataFrame:
    """synthetic_ohlcv with all technical indicators added."""
    from rita.core.technical_analyzer import calculate_indicators
    return calculate_indicators(synthetic_ohlcv)


# ─── Performance metrics ──────────────────────────────────────────────────────

class TestPerformanceMetrics:

    def test_sharpe_positive_returns(self):
        from rita.core.performance import sharpe_ratio
        # Steady 0.1% daily return → strong Sharpe
        rets = np.full(252, 0.001)
        sr = sharpe_ratio(rets)
        assert sr > 1.0

    def test_sharpe_zero_returns(self):
        from rita.core.performance import sharpe_ratio
        rets = np.zeros(252)
        assert sharpe_ratio(rets) == 0.0

    def test_sharpe_flat_std(self):
        from rita.core.performance import sharpe_ratio
        # Single return value → std = 0 → should return 0 not raise
        assert sharpe_ratio(np.array([0.01])) == 0.0

    def test_max_drawdown_no_drawdown(self):
        from rita.core.performance import max_drawdown
        vals = np.linspace(1.0, 2.0, 100)  # monotonically increasing
        assert max_drawdown(vals) == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_known_value(self):
        from rita.core.performance import max_drawdown
        vals = np.array([1.0, 1.2, 0.9, 1.1])
        dd = max_drawdown(vals)
        assert dd == pytest.approx((0.9 - 1.2) / 1.2, rel=1e-6)

    def test_cagr_doubling(self):
        from rita.core.performance import cagr
        result = cagr(1.0, 2.0, 1.0)
        assert result == pytest.approx(1.0, rel=1e-6)  # 100% in 1 year

    def test_cagr_zero_years(self):
        from rita.core.performance import cagr
        assert cagr(1.0, 2.0, 0) == 0.0

    def test_compute_all_metrics_keys(self):
        from rita.core.performance import compute_all_metrics
        port = np.linspace(1.0, 1.15, 252)
        bench = np.linspace(1.0, 1.10, 252)
        metrics = compute_all_metrics(port, bench)
        required = {
            "sharpe_ratio", "max_drawdown_pct", "portfolio_cagr_pct",
            "benchmark_cagr_pct", "sharpe_constraint_met", "drawdown_constraint_met",
            "constraints_met", "win_rate_pct",
        }
        assert required.issubset(metrics.keys())

    def test_compute_all_metrics_constraint_logic(self):
        from rita.core.performance import compute_all_metrics
        # Flat portfolio → low Sharpe, no drawdown
        port = np.ones(252)
        bench = np.ones(252)
        metrics = compute_all_metrics(port, bench)
        assert metrics["sharpe_constraint_met"] is False
        assert metrics["drawdown_constraint_met"] is True


# ─── Technical indicators ─────────────────────────────────────────────────────

class TestTechnicalAnalyzer:

    def test_calculate_indicators_columns(self, feature_df):
        expected = {
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_mid", "bb_lower", "bb_pct_b",
            "atr_14", "ema_50", "ema_200", "trend_score", "daily_return",
        }
        assert expected.issubset(feature_df.columns)

    def test_rsi_range(self, feature_df):
        rsi = feature_df["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_trend_score_range(self, feature_df):
        ts = feature_df["trend_score"].dropna()
        assert (ts >= -1).all() and (ts <= 1).all()

    def test_bb_pct_b_exists(self, feature_df):
        assert "bb_pct_b" in feature_df.columns
        assert feature_df["bb_pct_b"].notna().sum() > 0

    def test_market_summary_keys(self, feature_df):
        from rita.core.technical_analyzer import get_market_summary
        summary = get_market_summary(feature_df)
        for key in ["trend", "rsi_14", "macd", "sentiment_proxy"]:
            assert key in summary

    def test_market_summary_trend_valid(self, feature_df):
        from rita.core.technical_analyzer import get_market_summary
        summary = get_market_summary(feature_df)
        assert summary["trend"] in ("uptrend", "downtrend", "sideways")


# ─── Goal engine ──────────────────────────────────────────────────────────────

class TestGoalEngine:

    @pytest.fixture
    def hist_stats(self):
        return {
            "cagr_pct": 12.56, "sharpe_ratio": 0.331,
            "max_drawdown_pct": -59.86, "annual_volatility_pct": 22.5,
            "best_year_pct": 75.76, "worst_year_pct": -51.79,
        }

    def test_set_goal_feasibility_aggressive(self, hist_stats):
        from rita.core.goal_engine import set_goal
        goal = set_goal(50.0, 365, "aggressive", hist_stats)
        # 50% is well above historical CAGR of 12.56% → classified as ambitious
        assert goal["feasibility"] in ("aggressive", "ambitious")

    def test_set_goal_feasibility_conservative(self, hist_stats):
        from rita.core.goal_engine import set_goal
        goal = set_goal(5.0, 365, "conservative", hist_stats)
        assert goal["feasibility"] in ("conservative", "realistic")

    def test_set_goal_required_keys(self, hist_stats):
        from rita.core.goal_engine import set_goal
        goal = set_goal(15.0, 365, "moderate", hist_stats)
        for key in ["target_return_pct", "feasibility", "feasibility_note",
                    "suggested_realistic_target_pct"]:
            assert key in goal

    def test_update_goal_met(self, hist_stats):
        from rita.core.goal_engine import set_goal, update_goal_from_results
        goal = set_goal(15.0, 365, "moderate", hist_stats)
        backtest = {
            "performance": {
                "portfolio_total_return_pct": 16.0,
                "sharpe_ratio": 1.2,
                "max_drawdown_pct": -5.0,
                "constraints_met": True,
            }
        }
        updated = update_goal_from_results(goal, backtest)
        assert updated["assessment"] in ("met", "exceeded")
        assert "revised_target_pct" in updated


# ─── Strategy engine ──────────────────────────────────────────────────────────

class TestStrategyEngine:

    @pytest.fixture
    def research(self):
        return {
            "trend": "uptrend", "rsi_14": 55.0, "macd": 10.0,
            "sentiment_proxy": "neutral", "atr_14": 80.0,
        }

    @pytest.fixture
    def goal(self):
        return {
            "target_return_pct": 15.0, "risk_tolerance": "moderate",
            "feasibility": "realistic",
        }

    def test_design_strategy_keys(self, research, goal):
        from rita.core.strategy_engine import design_strategy
        strategy = design_strategy(research, goal)
        for key in ["name", "base_allocation_pct", "rebalancing_frequency"]:
            assert key in strategy

    def test_design_strategy_allocation_range(self, research, goal):
        from rita.core.strategy_engine import design_strategy
        strategy = design_strategy(research, goal)
        assert 0 <= strategy["base_allocation_pct"] <= 100

    def test_validate_constraints_pass(self):
        from rita.core.strategy_engine import validate_strategy_constraints
        result = validate_strategy_constraints({
            "performance": {
                "sharpe_ratio": 1.2, "max_drawdown_pct": -7.0,
                "portfolio_total_return_pct": 15.0, "benchmark_total_return_pct": 12.0,
            }
        })
        assert result["sharpe_constraint_met"] is True
        assert result["drawdown_constraint_met"] is True
        assert result["all_constraints_met"] is True

    def test_validate_constraints_fail_sharpe(self):
        from rita.core.strategy_engine import validate_strategy_constraints
        result = validate_strategy_constraints({
            "performance": {
                "sharpe_ratio": 0.7, "max_drawdown_pct": -5.0,
                "portfolio_total_return_pct": 8.0, "benchmark_total_return_pct": 12.0,
            }
        })
        assert result["sharpe_constraint_met"] is False
        assert result["all_constraints_met"] is False


# ─── RL environment ───────────────────────────────────────────────────────────

class TestNiftyTradingEnv:

    @pytest.fixture
    def env(self, feature_df):
        from rita.core.rl_agent import NiftyTradingEnv
        return NiftyTradingEnv(feature_df, episode_length=50)

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (7,)

    def test_action_space_size(self, env):
        assert env.action_space.n == 3

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset()
        assert obs.shape == (7,)
        assert env.observation_space.contains(obs)

    def test_step_returns_correct_types(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "portfolio_value" in info

    def test_episode_terminates(self, env):
        env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < 200:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert done, "Episode should terminate within episode_length steps"

    def test_allocation_map(self, env):
        env.reset(seed=0)
        for action, expected_alloc in [(0, 0.0), (1, 0.5), (2, 1.0)]:
            env.reset(seed=0)
            _, _, _, _, info = env.step(action)
            assert info["allocation"] == expected_alloc


# ─── API endpoints (integration, no training) ────────────────────────────────

@pytest.fixture(scope="module")
def api_client():
    """TestClient for the FastAPI app — uses real CSV if available, skips if not."""
    csv_path = os.getenv(
        "NIFTY_CSV_PATH",
        "",
    )
    if not os.path.exists(csv_path):
        pytest.skip("NIFTY_CSV_PATH not found — skipping API integration tests")

    from fastapi.testclient import TestClient
    from rita.interfaces.rest_api import app
    import rita.interfaces.rest_api as api_module
    # Reset orchestrator for clean test state
    api_module._orchestrator = None
    return TestClient(app)


class TestAPIEndpoints:

    def test_health(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_step1_set_goal(self, api_client):
        r = api_client.post("/api/v1/goal", json={
            "target_return_pct": 15.0,
            "time_horizon_days": 365,
            "risk_tolerance": "moderate",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["step"] == 1
        assert "feasibility" in data["result"]

    def test_step2_analyze_market(self, api_client):
        r = api_client.post("/api/v1/market")
        assert r.status_code == 200
        data = r.json()
        assert data["step"] == 2
        assert "trend" in data["result"]

    def test_step3_design_strategy(self, api_client):
        r = api_client.post("/api/v1/strategy")
        assert r.status_code == 200
        data = r.json()
        assert data["step"] == 3
        assert "name" in data["result"]

    def test_step4_load_existing_model(self, api_client):
        """Loads existing model — does not retrain."""
        r = api_client.post("/api/v1/train", json={
            "timesteps": 200000,
            "force_retrain": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["step"] == 4
        assert "validation" in data["result"]

    def test_step5_set_period(self, api_client):
        r = api_client.post("/api/v1/period", json={"start": "2025-01-01"})
        assert r.status_code == 200
        assert r.json()["step"] == 5

    def test_step6_run_backtest(self, api_client):
        r = api_client.post("/api/v1/backtest")
        assert r.status_code == 200
        data = r.json()
        assert data["step"] == 6
        assert "sharpe_ratio" in data["result"]["performance"]

    def test_step7_get_results(self, api_client):
        r = api_client.get("/api/v1/results")
        assert r.status_code == 200
        data = r.json()
        assert data["step"] == 7
        assert "plots" in data["result"]

    def test_step8_update_goal(self, api_client):
        r = api_client.post("/api/v1/goal/update")
        assert r.status_code == 200
        data = r.json()
        assert data["step"] == 8
        assert "assessment" in data["result"]

    def test_progress(self, api_client):
        r = api_client.get("/progress")
        assert r.status_code == 200
        assert "steps_completed" in r.json()
