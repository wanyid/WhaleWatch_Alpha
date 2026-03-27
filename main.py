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
  Executor → open position  (paper: SQLite sim | alpaca: real broker)
        ↓  background sweep
  Executor.close_expired_positions() → close + P&L

Executor is selected via settings.yaml → executor.provider:
  "paper"  — PaperExecutor  (default, safe)
  "alpaca" — AlpacaExecutor (requires ALPACA_API_KEY + ALPACA_SECRET_KEY)

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
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env before any module that reads environment variables
# ---------------------------------------------------------------------------
load_dotenv()

from backtest.backtester import Backtester
from executor.base_executor import BaseExecutor
from executor.paper_executor import PaperExecutor
from models.raw_events import PolymarketRawEvent, TruthSocialRawEvent
from models.signal_event import SignalEvent
from reasoner.layer1_llm.claude_llm import ClaudeLLM
from reasoner.layer2_predictor.stat_predictor import StatPredictor
from risk.risk_manager import RiskManager
from scanners.market_data.yfinance_provider import YFinanceProvider
from scanners.polymarket_scanner import PolymarketScanner, PolymarketSessionEvent
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


def _dual_signal_bonus() -> float:
    return _CFG.get("reasoner", {}).get("layer2", {}).get("dual_signal_confidence_bonus", 0.0)


# ---------------------------------------------------------------------------
# Signal deduplication — prevent re-processing the same event twice
# ---------------------------------------------------------------------------

class SignalDeduper:
    """Tracks recently processed signals to prevent duplicate pipeline runs.

    Polymarket: dedups by (market_id, direction) — the same market moving in
    the same direction within `ttl_minutes` is considered a duplicate.
    Truth Social: dedups by post_id.
    """

    def __init__(self, ttl_minutes: int = 30) -> None:
        self._ttl = timedelta(minutes=ttl_minutes)
        self._poly: dict[str, datetime] = {}  # key: f"{market_id}:{direction}"
        self._ts: set[str] = set()
        self._ts_times: dict[str, datetime] = {}

    def _prune(self) -> None:
        now = datetime.now(tz=timezone.utc)
        stale = [k for k, t in self._poly.items() if now - t > self._ttl]
        for k in stale:
            del self._poly[k]
        stale_ts = [pid for pid, t in self._ts_times.items() if now - t > self._ttl]
        for pid in stale_ts:
            self._ts.discard(pid)
            del self._ts_times[pid]

    def is_duplicate(self, raw: Union["PolymarketRawEvent", "TruthSocialRawEvent"]) -> bool:
        self._prune()
        now = datetime.now(tz=timezone.utc)
        from models.raw_events import PolymarketRawEvent, TruthSocialRawEvent
        if isinstance(raw, PolymarketRawEvent):
            key = f"{raw.market_id}:{getattr(raw, 'direction', '')}"
            if key in self._poly:
                return True
            self._poly[key] = now
            return False
        elif isinstance(raw, TruthSocialRawEvent):
            if raw.post_id in self._ts:
                return True
            self._ts.add(raw.post_id)
            self._ts_times[raw.post_id] = now
            return False
        return False


# ---------------------------------------------------------------------------
# Dual-signal matcher — detect when Poly + TS fire on the same theme
# ---------------------------------------------------------------------------

class DualSignalMatcher:
    """Buffer of recent raw events from each scanner.

    When an event arrives, checks if the opposite scanner recently emitted an
    event sharing ≥ 1 keyword. If so, marks the current SignalEvent as
    dual_signal=True and enriches it with the other signal's data.

    Buffer TTL: 15 minutes — if both signals fired within that window, they
    likely reflect the same news event.
    """

    _TTL = timedelta(minutes=15)

    def __init__(self) -> None:
        self._poly_buf: deque = deque(maxlen=50)  # (event, timestamp)
        self._ts_buf: deque = deque(maxlen=50)

    def _prune_buf(self, buf: deque) -> None:
        now = datetime.now(tz=timezone.utc)
        while buf and now - buf[0][1] > self._TTL:
            buf.popleft()

    def _keywords(self, raw) -> set[str]:
        from models.raw_events import PolymarketRawEvent, TruthSocialRawEvent
        if isinstance(raw, TruthSocialRawEvent):
            return {kw.lower() for kw in (raw.keywords or [])}
        if isinstance(raw, PolymarketRawEvent):
            # Extract keywords from market question
            q = (raw.market_question or "").lower()
            return {w for w in q.split() if len(w) > 4}
        return set()

    def record_and_match(
        self, raw
    ) -> tuple[bool, Optional[object]]:
        """Record a raw event and return (dual_signal, matching_other_event).

        Returns (True, matching_event) if there's a recent cross-signal match,
        (False, None) otherwise.
        """
        from models.raw_events import PolymarketRawEvent, TruthSocialRawEvent
        now = datetime.now(tz=timezone.utc)
        kws = self._keywords(raw)
        match = None

        if isinstance(raw, PolymarketRawEvent):
            self._prune_buf(self._ts_buf)
            for other, _ in self._ts_buf:
                if kws & self._keywords(other):
                    match = other
                    break
            self._poly_buf.append((raw, now))
        elif isinstance(raw, TruthSocialRawEvent):
            self._prune_buf(self._poly_buf)
            for other, _ in self._poly_buf:
                if kws & self._keywords(other):
                    match = other
                    break
            self._ts_buf.append((raw, now))

        return (match is not None), match


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _build_signal_event(
    raw: Union[PolymarketRawEvent, TruthSocialRawEvent],
    dual_companion: Optional[Union[PolymarketRawEvent, TruthSocialRawEvent]] = None,
) -> SignalEvent:
    """Wrap a raw scanner event in a fresh SignalEvent.

    If dual_companion is supplied (the other scanner's matching event), the
    SignalEvent is populated with both A and B signal data and dual_signal=True.
    """
    now = datetime.now(tz=timezone.utc)
    ev = SignalEvent(
        event_id=str(uuid.uuid4()),
        created_at=now,
    )

    def _fill_poly(e: PolymarketRawEvent) -> None:
        ev.poly_market_id = e.market_id
        ev.poly_market_slug = e.market_slug
        ev.poly_market_question = e.market_question
        ev.poly_outcome_token = e.outcome_token
        ev.poly_price_before = e.price_before
        ev.poly_price_after = e.price_after
        ev.poly_price_delta = e.price_delta
        ev.poly_volume_24h = e.volume_24h
        ev.poly_volume_spike_pct = e.volume_spike_pct

    def _fill_ts(e: TruthSocialRawEvent) -> None:
        ev.ts_post_id = e.post_id
        ev.ts_post_content = e.content
        ev.ts_post_timestamp = e.posted_at
        ev.ts_post_keywords = e.keywords

    if isinstance(raw, PolymarketRawEvent):
        _fill_poly(raw)
        if isinstance(dual_companion, TruthSocialRawEvent):
            _fill_ts(dual_companion)
    elif isinstance(raw, TruthSocialRawEvent):
        _fill_ts(raw)
        if isinstance(dual_companion, PolymarketRawEvent):
            _fill_poly(dual_companion)

    ev.dual_signal = dual_companion is not None
    return ev


def _run_poly_session_pipeline(
    session: "PolymarketSessionEvent",
    risk: RiskManager,
    executor: BaseExecutor,
) -> Optional[str]:
    """Fast path for Polymarket session events.

    The PolymarketScanner's SessionManager has already run the L2 model and
    set direction/confidence/holding_period — no LLM call needed here.
    We use the strongest raw event's market data to populate the SignalEvent.
    """
    if not session.raw_events:
        return None

    # Pick the raw event with the largest price delta as the "anchor"
    anchor = max(session.raw_events, key=lambda e: abs(e.price_delta))
    event = _build_signal_event(anchor)

    event.signal_direction = session.signal_direction
    event.signal_ticker = "SPY"           # Poly models predict SPY direction
    event.llm_model = "poly_l2_model"
    event.confidence = session.confidence
    event.holding_period_minutes = session.holding_period_minutes

    # Dual-signal confidence bonus: when both Poly and TS signal the same
    # direction, apply a configurable additive boost (settings.yaml).
    if event.dual_signal:
        bonus = _dual_signal_bonus()
        if bonus > 0:
            event.confidence = min(event.confidence + bonus, 1.0)
            logger.debug("Dual-signal bonus +%.2f → confidence=%.3f", bonus, event.confidence)

    if not risk.approve(event):
        return None

    return executor.submit_signal(event)


def run_pipeline(
    raw: Union[PolymarketRawEvent, TruthSocialRawEvent],
    l1: ClaudeLLM,
    l2: StatPredictor,
    risk: RiskManager,
    executor: BaseExecutor,
    dual_companion: Optional[Union[PolymarketRawEvent, TruthSocialRawEvent]] = None,
) -> Optional[str]:
    """Run one Truth Social raw event through L1 → L2 → risk → execute.

    Polymarket session events are handled by _run_poly_session_pipeline()
    and do not go through this function.
    """
    # Build SignalEvent (with dual-signal enrichment if companion present)
    event = _build_signal_event(raw, dual_companion)

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

    # Dual-signal confidence bonus (Truth Social path)
    if event.dual_signal:
        bonus = _dual_signal_bonus()
        if bonus > 0:
            event.confidence = min(event.confidence + bonus, 1.0)
            logger.debug("Dual-signal bonus +%.2f → confidence=%.3f", bonus, event.confidence)

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

def _run_true_news_check(executor: BaseExecutor, market_provider: "YFinanceProvider") -> None:
    """Fetch current Polymarket prices for open positions and apply True News Stop.

    Queries the CLOB API for the latest price on each open position's poly_market_id.
    If the probability has moved further in the signal direction (fade invalidated), exit.
    """
    try:
        import requests

        open_positions = executor.open_positions()
        market_ids = list({p.get("poly_market_id") for p in open_positions
                           if p.get("poly_market_id")})
        if not market_ids:
            return

        session = requests.Session()
        session.headers.update({"Accept": "application/json"})
        CLOB_BASE = "https://clob.polymarket.com"

        for market_id in market_ids:
            try:
                resp = session.get(
                    f"{CLOB_BASE}/last-trade-price",
                    params={"token_id": market_id},
                    timeout=5,
                )
                if resp.ok:
                    price_data = resp.json()
                    current_prob = float(price_data.get("price", 0))
                    closed = executor.check_true_news_stop(market_id, current_prob)
                    if closed:
                        logger.info(
                            "True News Stop: closed %d position(s) for market %s (prob=%.3f)",
                            len(closed), market_id[:12], current_prob,
                        )
            except Exception as exc:
                logger.debug("True news check failed for %s: %s", market_id[:12], exc)
    except Exception as exc:
        logger.debug("_run_true_news_check error: %s", exc)


def _build_executor(market_provider) -> BaseExecutor:
    """Instantiate the configured executor (paper or alpaca)."""
    try:
        with open(_SETTINGS) as f:
            cfg = yaml.safe_load(f) or {}
        provider = cfg.get("executor", {}).get("provider", "paper").lower()
    except Exception:
        provider = "paper"

    if provider == "alpaca":
        from executor.alpaca_executor import AlpacaExecutor
        logger.info("Executor: AlpacaExecutor (LIVE)")
        return AlpacaExecutor()

    logger.info("Executor: PaperExecutor")
    return PaperExecutor(market_provider)


def run_live(once: bool = False) -> None:
    """Live paper-trading mode."""
    market_provider = YFinanceProvider()
    l1 = ClaudeLLM()
    l2 = StatPredictor()
    risk = RiskManager()
    executor = _build_executor(market_provider)

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
    deduper = SignalDeduper(ttl_minutes=30)
    dual_matcher = DualSignalMatcher()

    # Graceful shutdown on Ctrl-C / SIGTERM
    def _handle_signal(sig, frame):
        logger.info("Shutdown signal received — stopping scanners")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Track scanner→thread mapping for watchdog restarts
    scanner_threads: list[dict] = []
    for sc in scanners:
        t = threading.Thread(target=_scanner_thread, args=(sc, raw_queue, stop_event), daemon=True)
        t.start()
        scanner_threads.append({"scanner": sc, "thread": t, "started_at": time.time()})

    logger.info("WhaleWatch_Alpha running — %d scanner(s) active", len(scanners))

    last_sweep = time.time()
    sweep_interval = 60  # seconds between expired-position sweeps

    while not stop_event.is_set():
        # Process queued events
        try:
            raw = raw_queue.get(timeout=1.0)

            # Polymarket session events have already been scored by the scanner's
            # internal L2 model — route them directly to risk + execute.
            if isinstance(raw, PolymarketSessionEvent):
                order_id = _run_poly_session_pipeline(raw, risk, executor)
                if order_id:
                    logger.info("Poly session position opened: order=%s", order_id[:8])
                continue

            # Deduplication check (Truth Social raw events only)
            if deduper.is_duplicate(raw):
                logger.debug("Duplicate signal skipped: %s", type(raw).__name__)
                continue

            # Dual-signal detection
            is_dual, companion = dual_matcher.record_and_match(raw)
            if is_dual:
                logger.info("Dual signal detected — enriching event with both A + B data")

            order_id = run_pipeline(raw, l1, l2, risk, executor,
                                    dual_companion=companion)
            if order_id:
                logger.info("New position opened: order=%s", order_id[:8])
        except queue.Empty:
            pass
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)

        # Periodic sweep
        now = time.time()
        if now - last_sweep >= sweep_interval:
            # 1. Close expired positions (time-based); feed P&L to circuit breaker
            expired_pnls = executor.close_expired_positions()
            if expired_pnls:
                logger.info("Swept %d expired position(s)", len(expired_pnls))
                for pnl in expired_pnls:
                    risk.record_pnl(pnl)

            # 2. True News Stop — check open Poly positions against latest prices
            _run_true_news_check(executor, market_provider)

            # 3. Session summary
            summary = executor.session_summary()
            logger.info(
                "Session P&L: %.4f  trades=%d  wins=%d  win_rate=%.1f%%",
                summary["total_pnl"],
                summary["trade_count"],
                summary["win_count"],
                summary["win_rate"] * 100,
            )

            # 4. Scanner watchdog — restart crashed threads
            for entry in scanner_threads:
                t = entry["thread"]
                if not t.is_alive() and not stop_event.is_set():
                    sc = entry["scanner"]
                    logger.warning("Scanner %s thread died — restarting", sc.name())
                    new_t = threading.Thread(
                        target=_scanner_thread,
                        args=(sc, raw_queue, stop_event),
                        daemon=True,
                    )
                    new_t.start()
                    entry["thread"] = new_t
                    entry["started_at"] = time.time()

            last_sweep = now

        if once:
            stop_event.set()

    logger.info("Main loop exited — waiting for scanner threads")
    for entry in scanner_threads:
        entry["thread"].join(timeout=5)
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
