"""WhaleWatch_Alpha — main orchestration loop.

Pipeline (runs continuously):
  Scanners (A + B, independent threads)
        ↓  raw events queued
  Reasoner L1 (LLM) → signal_direction + signal_ticker
        ↓
  Reasoner L2 (Predictor) → confidence + holding_period_minutes
        ↓
  RiskManager → approve / reject
        ↓
  PaperExecutor → open position (SQLite)
        ↓  background sweep
  PaperExecutor.close_expired_positions() → close + P&L

Run:
    python main.py                    # live paper-trade mode
    python main.py --once             # single scan cycle (useful for testing)
    python main.py --backtest         # replay historical SignalEvents
"""

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Union

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env before any module that reads environment variables
# ---------------------------------------------------------------------------
load_dotenv()

from backtest.backtester import Backtester
from executor.paper_executor import PaperExecutor
from models.raw_events import PolymarketRawEvent, TruthSocialRawEvent
from models.signal_event import SignalEvent
from reasoner.layer1_llm.claude_llm import ClaudeLLM
from reasoner.layer2_predictor.stat_predictor import StatPredictor
from risk.risk_manager import RiskManager
from scanners.market_data.yfinance_provider import YFinanceProvider
from scanners.polymarket_scanner import PolymarketScanner
from scanners.truthsocial_scanner import TruthSocialScanner

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SETTINGS = "config/settings.yaml"

with open(_SETTINGS) as f:
    _CFG = yaml.safe_load(f)


def _l2_min_confidence() -> float:
    return _CFG.get("reasoner", {}).get("layer2", {}).get("min_confidence", 0.60)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _build_signal_event(
    raw: Union[PolymarketRawEvent, TruthSocialRawEvent],
) -> SignalEvent:
    """Wrap a raw scanner event in a fresh SignalEvent."""
    now = datetime.now(tz=timezone.utc)
    ev = SignalEvent(
        event_id=str(uuid.uuid4()),
        created_at=now,
    )
    if isinstance(raw, PolymarketRawEvent):
        ev.poly_market_id = raw.market_id
        ev.poly_market_slug = raw.market_slug
        ev.poly_market_question = raw.market_question
        ev.poly_outcome_token = raw.outcome_token
        ev.poly_price_before = raw.price_before
        ev.poly_price_after = raw.price_after
        ev.poly_price_delta = raw.price_delta
        ev.poly_volume_24h = raw.volume_24h
        ev.poly_volume_spike_pct = raw.volume_spike_pct
    elif isinstance(raw, TruthSocialRawEvent):
        ev.ts_post_id = raw.post_id
        ev.ts_post_content = raw.content
        ev.ts_post_timestamp = raw.posted_at
        ev.ts_post_keywords = raw.keywords
    return ev


def run_pipeline(
    raw: Union[PolymarketRawEvent, TruthSocialRawEvent],
    l1: ClaudeLLM,
    l2: StatPredictor,
    risk: RiskManager,
    executor: PaperExecutor,
) -> Optional[str]:
    """Run one raw event through the full pipeline. Returns order_id or None."""
    # Build SignalEvent
    event = _build_signal_event(raw)

    # --- Layer 1: LLM direction + ticker ---
    try:
        l1_signal = l1.get_signal(raw)
    except Exception as exc:
        logger.warning("L1 failed for event %s: %s", event.event_id[:8], exc)
        return None

    if l1_signal.direction == "HOLD":
        logger.debug("L1 → HOLD, discarding event %s", event.event_id[:8])
        return None

    event.signal_direction = l1_signal.direction
    event.signal_ticker = l1_signal.ticker
    event.llm_model = l1_signal.llm_model

    # --- Layer 2: confidence + holding period ---
    try:
        l2.predict(event)
    except Exception as exc:
        logger.warning("L2 failed for event %s: %s — using fallback 0.50", event.event_id[:8], exc)
        event.confidence = 0.50
        event.holding_period_minutes = 60

    # --- Risk check ---
    if not risk.approve(event):
        return None

    # --- Execute ---
    order_id = executor.submit_signal(event)
    return order_id


# ---------------------------------------------------------------------------
# Scanner threads
# ---------------------------------------------------------------------------

def _scanner_thread(
    scanner,
    raw_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    name = scanner.name()
    logger.info("Scanner thread started: %s", name)
    try:
        for raw_event in scanner.scan():
            if stop_event.is_set():
                break
            raw_queue.put(raw_event)
    except Exception as exc:
        logger.error("Scanner %s crashed: %s", name, exc, exc_info=True)
    logger.info("Scanner thread exiting: %s", name)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_live(once: bool = False) -> None:
    """Live paper-trading mode."""
    market_provider = YFinanceProvider()
    l1 = ClaudeLLM()
    l2 = StatPredictor()
    risk = RiskManager()
    executor = PaperExecutor(market_provider)

    # Build scanners
    ts_user = os.getenv("TRUTHSOCIAL_USERNAME", "")
    ts_pass = os.getenv("TRUTHSOCIAL_PASSWORD", "")

    scanners = []

    try:
        poly_scanner = PolymarketScanner()
        scanners.append(poly_scanner)
        logger.info("Polymarket scanner initialised")
    except Exception as exc:
        logger.warning("Polymarket scanner failed to init: %s — skipping", exc)

    if ts_user and ts_pass:
        try:
            ts_scanner = TruthSocialScanner(username=ts_user, password=ts_pass)
            scanners.append(ts_scanner)
            logger.info("Truth Social scanner initialised for @%s", ts_user)
        except Exception as exc:
            logger.warning("Truth Social scanner failed to init: %s — skipping", exc)
    else:
        logger.warning("No Truth Social credentials — scanner disabled")

    if not scanners:
        logger.error("No scanners available — exiting")
        sys.exit(1)

    raw_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    # Graceful shutdown on Ctrl-C / SIGTERM
    def _handle_signal(sig, frame):
        logger.info("Shutdown signal received — stopping scanners")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Start scanner threads
    threads = []
    for sc in scanners:
        t = threading.Thread(target=_scanner_thread, args=(sc, raw_queue, stop_event), daemon=True)
        t.start()
        threads.append(t)

    logger.info("WhaleWatch_Alpha running — %d scanner(s) active", len(scanners))

    last_sweep = time.time()
    sweep_interval = 60  # seconds between expired-position sweeps

    while not stop_event.is_set():
        # Process queued events
        try:
            raw = raw_queue.get(timeout=1.0)
            order_id = run_pipeline(raw, l1, l2, risk, executor)
            if order_id:
                logger.info("New position opened: order=%s", order_id[:8])
        except queue.Empty:
            pass
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)

        # Periodic sweep of expired positions
        now = time.time()
        if now - last_sweep >= sweep_interval:
            closed = executor.close_expired_positions()
            if closed:
                logger.info("Swept %d expired position(s)", closed)
            # Log session summary periodically
            summary = executor.session_summary()
            logger.info(
                "Session P&L: %.4f  trades=%d  wins=%d  win_rate=%.1f%%",
                summary["total_pnl"],
                summary["trade_count"],
                summary["win_count"],
                summary["win_rate"] * 100,
            )
            last_sweep = now

        if once:
            stop_event.set()

    logger.info("Main loop exited — waiting for scanner threads")
    for t in threads:
        t.join(timeout=5)
    logger.info("WhaleWatch_Alpha stopped cleanly")


def run_backtest(start: str = "2025-01-20", end: Optional[str] = None) -> None:
    """Replay mode — backtests resolved positions from the paper DB."""
    bt = Backtester()
    results = bt.run(start_date=start, end_date=end)
    summary = results.summary()

    print("\n=== Backtest Results ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")

    if results.trades:
        out = bt.save_results(results)
        print(f"\nTrade-level CSV saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhaleWatch_Alpha trading bot")
    parser.add_argument("--once", action="store_true",
                        help="Run one scan cycle then exit (testing)")
    parser.add_argument("--backtest", action="store_true",
                        help="Backtest mode: replay historical SignalEvents")
    parser.add_argument("--start", default="2025-01-20",
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="Backtest end date (YYYY-MM-DD), default=today")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(start=args.start, end=args.end)
    else:
        run_live(once=args.once)
