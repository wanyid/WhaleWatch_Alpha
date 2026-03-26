"""transaction_costs.py — Realistic trading cost model.

Applies two cost components to each trade's gross P&L:

  1. Bid-ask spread (SPY / QQQ only)
     0.5 bps per side → 1 bp round-trip.
     Both entry and exit incur half the spread, so total = 2 × 0.5 bps = 0.01%.
     VIX is a volatility index and not directly traded; its vehicles (VXX, UVXY)
     have wider spreads but are excluded unless VIX trading is added.

  2. Short-sell borrowing fee (all tickers, SHORT direction only)
     8% per annum, prorated to the actual holding period in minutes.
     Formula: cost = 0.08 × (holding_minutes / 525_600)
     where 525_600 = 365 × 24 × 60.
     For a 1-day hold: 0.08 / 365 ≈ 0.022%.

Cost parameters are configurable at construction time so they can be swept
during optimisation without modifying call sites.

Usage:
    from backtest.transaction_costs import CostModel

    costs = CostModel()                          # default parameters
    df_net = costs.apply(df)                     # adds 'net_pnl', 'trade_cost' columns
    summary = costs.summary(df)                  # dict with aggregate cost stats
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


MINUTES_PER_YEAR = 365 * 24 * 60   # 525,600 — calendar minutes, not trading minutes


@dataclass
class CostModel:
    """Parameters for the transaction cost model.

    Attributes:
        bid_ask_bps:      One-way bid-ask cost in basis points for spread-bearing
                          tickers. Round-trip = 2 × bid_ask_bps.
        spread_tickers:   Tickers subject to the bid-ask cost (others assumed free
                          or modelled separately).
        short_borrow_annual: Annual borrowing rate for short positions (0.08 = 8%).
        apply_to_shorts:  Whether to apply borrow fee at all (can disable for
                          sensitivity analysis).
    """
    bid_ask_bps: float = 0.5              # per side; round-trip = 2× = 1 bp
    spread_tickers: tuple = ("SPY", "QQQ")
    short_borrow_annual: float = 0.08     # 8% per annum
    apply_to_shorts: bool = True

    # ------------------------------------------------------------------
    # Per-trade cost
    # ------------------------------------------------------------------

    def cost_for_trade(
        self,
        ticker: str,
        direction: str,
        holding_minutes: float,
    ) -> float:
        """Return total cost as a decimal fraction of notional (e.g. 0.0003 = 3 bps).

        Args:
            ticker:          e.g. "SPY", "QQQ", "VIX"
            direction:       "BUY" or "SHORT"
            holding_minutes: Actual holding period in minutes.
        """
        cost = 0.0

        # 1. Bid-ask spread (round-trip: entry + exit)
        if ticker.upper() in self.spread_tickers:
            cost += 2.0 * self.bid_ask_bps / 10_000  # bps → fraction

        # 2. Short borrowing fee (prorated by holding period)
        if self.apply_to_shorts and direction.upper() == "SHORT":
            cost += self.short_borrow_annual * (holding_minutes / MINUTES_PER_YEAR)

        return cost

    # ------------------------------------------------------------------
    # DataFrame-level operations
    # ------------------------------------------------------------------

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'trade_cost' and 'net_pnl' columns to a trades DataFrame.

        Expects columns: signal_ticker, signal_direction, holding_period_min,
                         realized_pnl.
        Returns a copy (does not modify in place).
        """
        df = df.copy()

        ticker_col = df["signal_ticker"].str.upper()
        dir_col = df["signal_direction"].str.upper()
        hold_col = df["holding_period_min"].fillna(1440).astype(float)

        costs = np.array([
            self.cost_for_trade(t, d, h)
            for t, d, h in zip(ticker_col, dir_col, hold_col)
        ])

        df["trade_cost"] = np.round(costs, 8)
        df["net_pnl"] = np.round(df["realized_pnl"].fillna(0.0) - costs, 8)

        # Re-derive net outcome from net P&L and stop-loss threshold
        sl = df.get("stop_loss_pct", pd.Series(0.02, index=df.index)).fillna(0.02)
        net = df["net_pnl"]
        df["net_outcome"] = np.where(
            net <= -sl,   "STOP_OUT",
            np.where(net > 0, "WIN", "LOSS"),
        )

        return df

    def summary(self, df: pd.DataFrame) -> dict:
        """Return aggregate cost statistics for a trades DataFrame.

        Call apply() first to ensure 'trade_cost' and 'net_pnl' columns exist.
        If not present, calls apply() internally.
        """
        if "trade_cost" not in df.columns:
            df = self.apply(df)

        total_trades = len(df)
        total_gross_pnl = float(df["realized_pnl"].fillna(0.0).sum())
        total_cost = float(df["trade_cost"].sum())
        total_net_pnl = float(df["net_pnl"].sum())

        # Gross vs net win rate
        gross_win_rate = float((df["outcome"] == "WIN").mean()) if "outcome" in df.columns else float("nan")
        net_win_rate = float((df["net_outcome"] == "WIN").mean())

        # Cost breakdown by component
        spread_mask = df["signal_ticker"].str.upper().isin(self.spread_tickers)
        short_mask = df["signal_direction"].str.upper() == "SHORT"
        spread_cost = float(df.loc[spread_mask, "trade_cost"].sum())
        borrow_cost = float(
            df.loc[short_mask, "trade_cost"].sum()
            - (2.0 * self.bid_ask_bps / 10_000) * spread_mask[short_mask].sum()
        )

        return {
            "total_trades": total_trades,
            "gross_pnl": round(total_gross_pnl, 4),
            "total_cost": round(total_cost, 4),
            "net_pnl": round(total_net_pnl, 4),
            "cost_drag_pct": round(total_cost / total_trades * 100, 4),  # avg cost per trade %
            "spread_cost": round(spread_cost, 4),
            "borrow_cost": round(borrow_cost, 4),
            "gross_win_rate": round(gross_win_rate, 4),
            "net_win_rate": round(net_win_rate, 4),
            "trades_turned_loss": int(
                ((df.get("outcome", "WIN") == "WIN") & (df["net_outcome"] != "WIN")).sum()
            ),
        }


# Convenience singleton with default parameters
DEFAULT_COSTS = CostModel()
