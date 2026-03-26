"""Tests for performance metrics, Kelly criterion, and optimisation functions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.performance import (
    calmar_ratio,
    compute_metrics,
    kelly_fraction,
    half_kelly,
    max_drawdown,
    optimize_bet_size,
    optimize_confidence,
    optimize_stop_loss,
    sharpe_ratio,
    sortino_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trades(
    n_wins: int = 6,
    n_losses: int = 4,
    avg_win: float = 0.03,
    avg_loss: float = -0.02,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    confidence: float = 0.70,
) -> pd.DataFrame:
    rows = []
    for i in range(n_wins):
        rows.append({
            "outcome": "WIN", "realized_pnl": avg_win,
            "signal_direction": "BUY", "signal_ticker": "SPY",
            "holding_period_min": 60, "confidence": confidence,
            "entry_price": 500.0, "exit_price": 500.0 * (1 + avg_win),
            "stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct,
            "created_at": f"2025-06-0{i+1}T14:00:00+00:00",
        })
    for i in range(n_losses):
        rows.append({
            "outcome": "LOSS", "realized_pnl": avg_loss,
            "signal_direction": "SHORT", "signal_ticker": "QQQ",
            "holding_period_min": 30, "confidence": confidence - 0.1,
            "entry_price": 450.0, "exit_price": 450.0 * (1 + abs(avg_loss)),
            "stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct,
            "created_at": f"2025-07-0{i+1}T15:00:00+00:00",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sharpe / Sortino / Drawdown
# ---------------------------------------------------------------------------

class TestMetricFunctions:
    def test_sharpe_positive_edge(self):
        returns = np.array([0.01] * 100)   # constant positive return
        assert sharpe_ratio(returns) > 0

    def test_sharpe_zero_variance(self):
        returns = np.zeros(10)
        assert sharpe_ratio(returns) == 0.0

    def test_sharpe_single_sample(self):
        assert sharpe_ratio(np.array([0.01])) == 0.0

    def test_sortino_better_than_sharpe_when_upside_only(self):
        # Returns that are positive or zero should have sortino > sharpe
        returns = np.array([0.01, 0.02, 0.01, 0.00, 0.03])
        s = sharpe_ratio(returns)
        so = sortino_ratio(returns)
        assert so >= s

    def test_max_drawdown_flat(self):
        equity = np.ones(10)
        assert max_drawdown(equity) == 0.0

    def test_max_drawdown_declining(self):
        equity = np.array([1.0, 0.9, 0.8, 0.7])
        dd = max_drawdown(equity)
        assert dd == pytest.approx(-0.3, abs=0.01)

    def test_max_drawdown_recovery(self):
        equity = np.array([1.0, 0.8, 1.2, 1.0])
        dd = max_drawdown(equity)
        assert dd == pytest.approx(-0.2, abs=0.01)

    def test_calmar_positive(self):
        assert calmar_ratio(0.20, -0.10) == pytest.approx(2.0)

    def test_calmar_zero_drawdown(self):
        assert calmar_ratio(0.10, 0.0) == float("inf")


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

class TestKelly:
    def test_positive_edge(self):
        f = kelly_fraction(win_rate=0.60, avg_win=0.03, avg_loss=0.02)
        assert 0 < f <= 1

    def test_no_edge(self):
        # win_rate=0.40 with 1:1 odds → negative edge → 0
        f = kelly_fraction(win_rate=0.40, avg_win=0.02, avg_loss=0.02)
        assert f == 0.0

    def test_half_kelly_is_half(self):
        f = kelly_fraction(win_rate=0.65, avg_win=0.03, avg_loss=0.02)
        hf = half_kelly(win_rate=0.65, avg_win=0.03, avg_loss=0.02)
        assert hf == pytest.approx(f / 2.0)

    def test_zero_loss(self):
        assert kelly_fraction(0.6, 0.03, 0.0) == 0.0

    def test_zero_win_rate(self):
        assert kelly_fraction(0.0, 0.03, 0.02) == 0.0

    def test_clipped_to_one(self):
        # Very high win rate with high odds
        f = kelly_fraction(win_rate=0.99, avg_win=0.10, avg_loss=0.01)
        assert f <= 1.0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_basic_metrics(self):
        df = _make_trades(n_wins=6, n_losses=4)
        m = compute_metrics(df)
        assert m.total_trades == 10
        assert m.winning_trades == 6
        assert m.win_rate == pytest.approx(0.6)
        assert m.total_pnl_pct == pytest.approx(6 * 0.03 + 4 * (-0.02))

    def test_sharpe_computed(self):
        df = _make_trades(n_wins=8, n_losses=2, avg_win=0.03, avg_loss=-0.01)
        m = compute_metrics(df)
        assert m.sharpe > 0

    def test_kelly_positive_edge(self):
        df = _make_trades(n_wins=7, n_losses=3, avg_win=0.04, avg_loss=-0.02)
        m = compute_metrics(df)
        assert m.kelly_f > 0

    def test_by_ticker_breakdown(self):
        df = _make_trades()
        m = compute_metrics(df)
        assert "SPY" in m.by_ticker
        assert "QQQ" in m.by_ticker

    def test_empty_df(self):
        m = compute_metrics(pd.DataFrame())
        assert m.total_trades == 0

    def test_profit_factor(self):
        df = _make_trades(n_wins=6, n_losses=4, avg_win=0.03, avg_loss=-0.02)
        m = compute_metrics(df)
        expected_pf = (6 * 0.03) / (4 * 0.02)
        assert m.profit_factor == pytest.approx(expected_pf, rel=0.01)


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------

class TestOptimisers:
    def test_sl_sweep_returns_dataframe(self):
        df = _make_trades(n_wins=6, n_losses=4)
        result = optimize_stop_loss(df)
        assert isinstance(result, pd.DataFrame)
        assert "stop_loss_pct" in result.columns
        assert "sharpe" in result.columns
        assert len(result) > 0

    def test_sl_sweep_sorted_descending(self):
        df = _make_trades(n_wins=6, n_losses=4)
        result = optimize_stop_loss(df)
        # Should be sorted by sharpe descending
        assert result["sharpe"].iloc[0] >= result["sharpe"].iloc[-1]

    def test_confidence_sweep_returns_dataframe(self):
        df = _make_trades(n_wins=7, n_losses=3, confidence=0.75)
        result = optimize_confidence(df)
        assert isinstance(result, pd.DataFrame)
        assert "min_confidence" in result.columns

    def test_confidence_sweep_trade_count_decreasing(self):
        df = _make_trades(n_wins=7, n_losses=3, confidence=0.75)
        result = optimize_confidence(df, conf_grid=[0.50, 0.60, 0.70, 0.80])
        result_sorted_by_conf = result.sort_values("min_confidence")
        assert list(result_sorted_by_conf["trades"]) == sorted(
            result_sorted_by_conf["trades"].tolist(), reverse=True
        )

    def test_bet_size_sweep(self):
        df = _make_trades(n_wins=6, n_losses=4)
        result = optimize_bet_size(df, kelly_f=0.15)
        assert isinstance(result, pd.DataFrame)
        assert "kelly_fraction" in result.columns
        assert "bet_size_pct" in result.columns

    def test_bet_size_pnl_scales_with_fraction(self):
        df = _make_trades(n_wins=8, n_losses=2, avg_win=0.03, avg_loss=-0.01)
        result = optimize_bet_size(df, kelly_f=0.20, fractions=[0.25, 0.50, 1.0])
        # Higher fraction → higher total P&L (positive edge case)
        pnls = result.sort_values("kelly_fraction")["total_pnl"].values
        assert pnls[0] < pnls[1] < pnls[2]
