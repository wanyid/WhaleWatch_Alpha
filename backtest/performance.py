"""performance.py — Strategy performance metrics and optimisation tools.

Functions:
  metrics(trades)          → full performance dict (Sharpe, Sortino, win rate, etc.)
  sharpe_ratio(returns)    → annualised Sharpe
  sortino_ratio(returns)   → annualised Sortino (downside deviation only)
  max_drawdown(equity)     → maximum peak-to-trough drawdown
  kelly_fraction(win_rate, avg_win, avg_loss) → optimal Kelly bet size
  half_kelly(win_rate, avg_win, avg_loss)     → conservative half-Kelly
  optimize_stop_loss(trades, sl_grid)         → sweep SL thresholds → best Sharpe
  optimize_confidence(trades, conf_grid)      → sweep min-confidence → best Sharpe
  optimize_bet_size(trades, kelly_fraction)   → position size vs return trade-off
  print_report(metrics_dict)                  → human-readable performance report
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_YEAR = TRADING_DAYS_PER_YEAR * 390   # US market minutes


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Per-trade return series (not cumulative).
        periods_per_year: Scaling factor — use TRADING_DAYS_PER_YEAR for daily,
                          MINUTES_PER_YEAR for minute-level P&L.
        risk_free: Annual risk-free rate (default 0).
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free: float = 0.0,
    target: float = 0.0,
) -> float:
    """Annualised Sortino ratio — penalises only downside volatility."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < target]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return float("inf")
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a fraction (e.g. 0.15 = 15%)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / np.where(peak == 0, 1, peak)
    return float(drawdowns.min())


def calmar_ratio(total_return: float, max_dd: float) -> float:
    """Calmar ratio: annualised return / |max drawdown|."""
    if max_dd == 0:
        return float("inf") if total_return > 0 else 0.0
    return float(total_return / abs(max_dd))


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Full Kelly fraction: f* = W/L - (1-W)/W  (discrete Kelly formula).

    Args:
        win_rate: Fraction of trades that win [0, 1].
        avg_win:  Average gain on winning trades (positive, e.g. 0.03 = 3%).
        avg_loss: Average loss on losing trades (positive, e.g. 0.02 = 2%).

    Returns:
        Optimal fraction of capital to risk per trade.
        Clipped to [0, 1].
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    lose_rate = 1.0 - win_rate
    b = avg_win / avg_loss   # odds ratio
    f = (b * win_rate - lose_rate) / b
    return float(np.clip(f, 0.0, 1.0))


def half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Half-Kelly: standard conservative position-sizing recommendation."""
    return kelly_fraction(win_rate, avg_win, avg_loss) / 2.0


# ---------------------------------------------------------------------------
# Full metrics computation
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    # Counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    stop_out_trades: int = 0

    # Win stats
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0     # gross wins / gross losses

    # Return stats
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    std_pnl_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0

    # Risk-adjusted
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar: float = 0.0

    # Position sizing
    kelly_f: float = 0.0
    half_kelly_f: float = 0.0

    # Holding periods
    avg_hold_minutes: float = 0.0
    median_hold_minutes: float = 0.0

    # Breakdown
    by_ticker: dict = field(default_factory=dict)
    by_direction: dict = field(default_factory=dict)
    by_hour: dict = field(default_factory=dict)


def compute_metrics(df: pd.DataFrame) -> PerformanceMetrics:
    """Compute full performance metrics from a trades DataFrame.

    Expected columns: outcome, realized_pnl, holding_period_min,
                      signal_ticker, signal_direction, created_at (optional).
    """
    m = PerformanceMetrics()
    if df.empty:
        return m

    m.total_trades = len(df)
    m.winning_trades = int((df["outcome"] == "WIN").sum())
    m.losing_trades = int((df["outcome"] == "LOSS").sum())
    m.stop_out_trades = int((df["outcome"] == "STOP_OUT").sum())
    m.win_rate = m.winning_trades / m.total_trades

    pnl = df["realized_pnl"].fillna(0.0).values
    m.total_pnl_pct = float(pnl.sum())
    m.avg_pnl_pct = float(pnl.mean())
    m.std_pnl_pct = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    m.best_trade_pct = float(pnl.max())
    m.worst_trade_pct = float(pnl.min())

    wins_pnl = df.loc[df["outcome"] == "WIN", "realized_pnl"].values
    losses_pnl = df.loc[df["outcome"] != "WIN", "realized_pnl"].values

    m.avg_win_pct = float(wins_pnl.mean()) if len(wins_pnl) else 0.0
    m.avg_loss_pct = float(losses_pnl.mean()) if len(losses_pnl) else 0.0

    gross_wins = wins_pnl.sum() if len(wins_pnl) else 0.0
    gross_losses = abs(losses_pnl.sum()) if len(losses_pnl) else 0.0
    m.profit_factor = float(gross_wins / gross_losses) if gross_losses > 0 else float("inf")

    # Risk-adjusted metrics (treat each trade as one "period")
    m.sharpe = sharpe_ratio(pnl, periods_per_year=m.total_trades)
    m.sortino = sortino_ratio(pnl, periods_per_year=m.total_trades)

    equity = np.cumsum(np.insert(pnl, 0, 1.0))   # starting at 1.0
    m.max_drawdown_pct = max_drawdown(equity)
    m.calmar = calmar_ratio(m.total_pnl_pct, m.max_drawdown_pct)

    # Kelly
    m.kelly_f = kelly_fraction(m.win_rate, abs(m.avg_win_pct), abs(m.avg_loss_pct))
    m.half_kelly_f = m.kelly_f / 2.0

    # Holding periods
    if "holding_period_min" in df.columns:
        hp = df["holding_period_min"].dropna()
        m.avg_hold_minutes = float(hp.mean()) if len(hp) else 0.0
        m.median_hold_minutes = float(hp.median()) if len(hp) else 0.0

    # Breakdown by ticker
    for ticker in df["signal_ticker"].unique():
        sub = df[df["signal_ticker"] == ticker]
        sub_pnl = sub["realized_pnl"].fillna(0.0).values
        wr = (sub["outcome"] == "WIN").mean()
        m.by_ticker[ticker] = {
            "trades": len(sub), "win_rate": round(float(wr), 3),
            "total_pnl": round(float(sub_pnl.sum()), 4),
            "sharpe": round(sharpe_ratio(sub_pnl, len(sub_pnl)), 3),
        }

    # Breakdown by direction
    for direction in df["signal_direction"].unique():
        sub = df[df["signal_direction"] == direction]
        sub_pnl = sub["realized_pnl"].fillna(0.0).values
        wr = (sub["outcome"] == "WIN").mean()
        m.by_direction[direction] = {
            "trades": len(sub), "win_rate": round(float(wr), 3),
            "total_pnl": round(float(sub_pnl.sum()), 4),
        }

    # Breakdown by hour-of-day (if created_at available)
    if "created_at" in df.columns:
        df2 = df.copy()
        df2["hour"] = pd.to_datetime(df2["created_at"], utc=True).dt.hour
        for hour in sorted(df2["hour"].unique()):
            sub = df2[df2["hour"] == hour]
            sub_pnl = sub["realized_pnl"].fillna(0.0).values
            wr = (sub["outcome"] == "WIN").mean()
            m.by_hour[int(hour)] = {
                "trades": len(sub), "win_rate": round(float(wr), 3),
                "total_pnl": round(float(sub_pnl.sum()), 4),
            }

    return m


# ---------------------------------------------------------------------------
# Parameter optimisers
# ---------------------------------------------------------------------------

def optimize_stop_loss(
    df: pd.DataFrame,
    sl_grid: Optional[list[float]] = None,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Sweep stop-loss thresholds and report metrics for each.

    Simulates applying each SL threshold to the raw P&L column.
    Returns a DataFrame sorted by the chosen metric (descending).

    Args:
        df: Trades DataFrame with 'entry_price', 'exit_price',
            'signal_direction', 'realized_pnl', 'outcome' columns.
        sl_grid: List of SL fractions to test (default 0.5% – 5%).
        metric: Column to optimise on ('sharpe', 'win_rate', 'total_pnl').
    """
    if sl_grid is None:
        sl_grid = [round(x, 3) for x in np.arange(0.005, 0.055, 0.005)]

    rows = []
    for sl in sl_grid:
        # Apply SL to raw returns
        raw_ret = _compute_raw_returns(df)
        tp_col = df.get("take_profit_pct", pd.Series([0.04] * len(df)))
        tp = tp_col.fillna(0.04).values if hasattr(tp_col, "values") else np.full(len(df), 0.04)
        clamped = np.clip(raw_ret, -sl, tp)

        outcome = np.where(raw_ret <= -sl, "STOP_OUT",
                  np.where(clamped > 0, "WIN", "LOSS"))
        sim = df.copy()
        sim["realized_pnl"] = clamped
        sim["outcome"] = outcome

        m = compute_metrics(sim)
        rows.append({
            "stop_loss_pct": sl,
            "win_rate": round(m.win_rate, 4),
            "total_pnl": round(m.total_pnl_pct, 4),
            "sharpe": round(m.sharpe, 4),
            "sortino": round(m.sortino, 4),
            "max_drawdown": round(m.max_drawdown_pct, 4),
            "profit_factor": round(m.profit_factor, 4),
            "stop_out_rate": round(m.stop_out_trades / max(m.total_trades, 1), 4),
        })

    result = pd.DataFrame(rows).sort_values(metric, ascending=False)
    return result


def optimize_confidence(
    df: pd.DataFrame,
    conf_grid: Optional[list[float]] = None,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Sweep min-confidence thresholds and report metrics for each.

    Requires 'confidence' column in df (from L2 predictor output).
    Shows trade-off between selectivity and performance.

    Args:
        df: Trades DataFrame including 'confidence' column.
        conf_grid: Min-confidence thresholds to test (default 0.50 – 0.80).
        metric: Column to optimise on ('sharpe', 'win_rate', 'total_pnl').
    """
    if "confidence" not in df.columns:
        logger.warning("No 'confidence' column — cannot optimise confidence threshold")
        return pd.DataFrame()

    if conf_grid is None:
        conf_grid = [round(x, 2) for x in np.arange(0.50, 0.82, 0.02)]

    rows = []
    for threshold in conf_grid:
        subset = df[df["confidence"] >= threshold]
        if subset.empty:
            rows.append({"min_confidence": threshold, "trades": 0,
                         "win_rate": 0, "sharpe": 0, "total_pnl": 0,
                         "sortino": 0, "max_drawdown": 0})
            continue
        m = compute_metrics(subset)
        rows.append({
            "min_confidence": threshold,
            "trades": m.total_trades,
            "win_rate": round(m.win_rate, 4),
            "total_pnl": round(m.total_pnl_pct, 4),
            "sharpe": round(m.sharpe, 4),
            "sortino": round(m.sortino, 4),
            "max_drawdown": round(m.max_drawdown_pct, 4),
            "kelly_f": round(m.kelly_f, 4),
        })

    result = pd.DataFrame(rows).sort_values(metric, ascending=False)
    return result


def optimize_bet_size(
    df: pd.DataFrame,
    kelly_f: float,
    fractions: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Show risk-return profile at different fractions of the Kelly bet size.

    At fraction=1.0 → full Kelly (maximum geometric growth, high variance).
    At fraction=0.5 → half Kelly (standard conservative recommendation).
    At fraction=0.25 → quarter Kelly (very conservative).

    Returns a DataFrame showing how total P&L and volatility scale with
    bet fraction, so you can choose based on risk tolerance.
    """
    if fractions is None:
        fractions = [round(x, 2) for x in np.arange(0.1, 1.05, 0.1)]

    pnl = df["realized_pnl"].fillna(0.0).values
    rows = []
    for frac in fractions:
        scaled = pnl * (kelly_f * frac)
        eq = np.cumsum(np.insert(scaled, 0, 1.0))
        rows.append({
            "kelly_fraction": round(frac, 2),
            "bet_size_pct": round(kelly_f * frac * 100, 2),
            "total_pnl": round(float(scaled.sum()), 4),
            "sharpe": round(sharpe_ratio(scaled, len(scaled)), 4),
            "max_drawdown": round(max_drawdown(eq), 4),
            "std_per_trade": round(float(np.std(scaled, ddof=1)), 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_raw_returns(df: pd.DataFrame) -> np.ndarray:
    """Recompute raw (unclamped) return from entry/exit prices."""
    ep = df["entry_price"].fillna(0).values
    xp = df["exit_price"].fillna(0).values
    dir_ = df["signal_direction"].values
    raw = np.where(
        dir_ == "BUY",
        np.where(ep > 0, (xp - ep) / ep, 0.0),
        np.where(ep > 0, (ep - xp) / ep, 0.0),
    )
    # Fall back to stored realized_pnl where price data is missing
    has_price = (ep > 0) & (xp > 0)
    stored = df["realized_pnl"].fillna(0.0).values
    return np.where(has_price, raw, stored)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(m: PerformanceMetrics) -> None:
    """Print a human-readable performance report."""
    print("\n" + "=" * 60)
    print("STRATEGY PERFORMANCE REPORT")
    print("=" * 60)
    print(f"  Total trades          : {m.total_trades}")
    print(f"  Win / Loss / StopOut  : {m.winning_trades} / {m.losing_trades} / {m.stop_out_trades}")
    print(f"  Win rate              : {m.win_rate:.1%}")
    print(f"  Profit factor         : {m.profit_factor:.2f}")
    print()
    print(f"  Total P&L             : {m.total_pnl_pct:+.4f}  ({m.total_pnl_pct*100:+.2f}%)")
    print(f"  Avg P&L / trade       : {m.avg_pnl_pct:+.4f}  ({m.avg_pnl_pct*100:+.2f}%)")
    print(f"  Avg win               : {m.avg_win_pct:+.4f}  ({m.avg_win_pct*100:+.2f}%)")
    print(f"  Avg loss              : {m.avg_loss_pct:+.4f}  ({m.avg_loss_pct*100:+.2f}%)")
    print(f"  Best trade            : {m.best_trade_pct:+.4f}")
    print(f"  Worst trade           : {m.worst_trade_pct:+.4f}")
    print()
    print(f"  Sharpe ratio          : {m.sharpe:.3f}")
    print(f"  Sortino ratio         : {m.sortino:.3f}")
    print(f"  Max drawdown          : {m.max_drawdown_pct:.1%}")
    print(f"  Calmar ratio          : {m.calmar:.3f}")
    print()
    print(f"  Full Kelly f*         : {m.kelly_f:.1%}  (risk {m.kelly_f*100:.1f}% of capital/trade)")
    print(f"  Half Kelly f*         : {m.half_kelly_f:.1%}  (recommended starting point)")
    print()
    print(f"  Avg holding period    : {m.avg_hold_minutes:.0f} min  ({m.avg_hold_minutes/60:.1f} hrs)")
    print(f"  Median holding period : {m.median_hold_minutes:.0f} min")

    if m.by_ticker:
        print()
        print("  By ticker:")
        for tkr, stats in sorted(m.by_ticker.items()):
            print(f"    {tkr:6s}: trades={stats['trades']:4d}  win_rate={stats['win_rate']:.1%}"
                  f"  total_pnl={stats['total_pnl']:+.4f}  sharpe={stats['sharpe']:.2f}")

    if m.by_direction:
        print()
        print("  By direction:")
        for d, stats in sorted(m.by_direction.items()):
            print(f"    {d:7s}: trades={stats['trades']:4d}  win_rate={stats['win_rate']:.1%}"
                  f"  total_pnl={stats['total_pnl']:+.4f}")

    print("=" * 60 + "\n")
