"""One-time historical data pull — no API key required.

Downloads and stores:
  1. SPY / QQQ / VIX  — daily OHLCV since 2024-01-01 via yfinance (unlimited lookback)
                       + last 60 days of 5-minute bars via yfinance (intraday detail)
  2. Polymarket        — hourly price history for all politically-relevant
                         markets (including closed) via CLOB + Gamma APIs
  3. Truth Social      — all @realDonaldTrump posts since 2024-01-01
                         via truthbrush (requires Truth Social credentials)

yfinance resolution limits (hard Yahoo Finance API restrictions):
  1m  → last 7 days only
  5m  → last 60 days only
  1h  → last 730 days only  (but 2024-01-01 is ~815 days ago — too far)
  1d  → unlimited ✓

Strategy: store daily bars for full history (backtesting) + 5m bars for the
last 60 days (intraday signal validation). For true 1-minute history back to
2024, upgrade to Polygon.io free tier via the existing base_provider interface.

Output directory: D:/WhaleWatch_Data/
  equity/
    SPY_1d.parquet       (daily, 2024-01-01 → today)
    QQQ_1d.parquet
    VIX_1d.parquet
    SPY_5m_recent.parquet   (5-minute, last 60 days)
    QQQ_5m_recent.parquet
    VIX_5m_recent.parquet
  polymarket/
    markets_catalog.parquet
    prices/{condition_id}_YES.parquet
  truth_social/
    trump_posts.jsonl

Prerequisites:
  - Python env with requirements.txt installed
  - TRUTHSOCIAL_USERNAME + TRUTHSOCIAL_PASSWORD in .env  (Truth Social only)
  - No other API keys needed

Run once:
    python scripts/pull_historical_data.py

Re-running is safe:
  - Equity files are overwritten (full re-download)
  - Polymarket price files skip markets already on disk
  - Truth Social appends only posts newer than the last stored post_id
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when running as a script
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_START = "2024-01-01"
DATA_START_DT = datetime(2024, 1, 1, tzinfo=timezone.utc)
DATA_ROOT = Path("D:/WhaleWatch_Data")

EQUITY_DIR = DATA_ROOT / "equity"
POLY_DIR = DATA_ROOT / "polymarket"
POLY_PRICES_DIR = POLY_DIR / "prices"
TS_DIR = DATA_ROOT / "truth_social"

TICKERS = ["SPY", "QQQ", "VIX"]

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

WATCHLIST_KEYWORDS = [
    "tariff", "tariffs", "trade war", "trade deal",
    "ukraine", "russia", "china", "iran", "israel", "nato", "north korea",
    "ceasefire", "sanctions",
    "trump", "executive order", "cabinet", "fired", "resign",
    "federal reserve", "fed rate", "interest rate", "inflation", "recession",
    "oil price", "opec",
]

TRUTH_SOCIAL_USER = "realDonaldTrump"


# ===========================================================================
# 1. Equity data — yfinance 5-minute, chunked
# ===========================================================================

def pull_equity() -> None:
    """Download equity data for SPY, QQQ, VIX via yfinance.

    Two passes per ticker:
      - Daily bars from DATA_START → today (unlimited yfinance lookback)
      - 5-minute bars for last 60 days (finest intraday resolution available free)
    """
    import yfinance as yf

    EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    for ticker in TICKERS:
        yf_ticker = f"^{ticker}" if ticker == "VIX" else ticker

        # --- Daily bars: full history ---
        out_daily = EQUITY_DIR / f"{ticker}_1d.parquet"
        logger.info("--- %s daily ---", ticker)
        try:
            df = yf.download(yf_ticker, start=DATA_START, end=today,
                             interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                logger.warning("No daily data for %s.", ticker)
            else:
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                df.index = pd.to_datetime(df.index, utc=True)
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                df.to_parquet(out_daily)
                logger.info("Saved %s daily → %s rows, %.1f MB",
                            ticker, len(df), out_daily.stat().st_size / 1e6)
        except Exception as exc:
            logger.error("Daily pull failed for %s: %s", ticker, exc)

        # --- 5-minute bars: last 60 days ---
        out_5m = EQUITY_DIR / f"{ticker}_5m_recent.parquet"
        logger.info("--- %s 5m (last 60 days) ---", ticker)
        try:
            df5 = yf.download(yf_ticker, period="60d",
                              interval="5m", auto_adjust=True, progress=False)
            if df5.empty:
                logger.warning("No 5m data for %s.", ticker)
            else:
                df5.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df5.columns]
                df5.index = pd.to_datetime(df5.index, utc=True)
                df5 = df5[["open", "high", "low", "close", "volume"]].dropna()
                df5.to_parquet(out_5m)
                logger.info("Saved %s 5m → %s rows, %.1f MB",
                            ticker, len(df5), out_5m.stat().st_size / 1e6)
        except Exception as exc:
            logger.error("5m pull failed for %s: %s", ticker, exc)


# ===========================================================================
# 2. Polymarket historical data
# ===========================================================================

def _is_relevant(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in WATCHLIST_KEYWORDS)


def _extract_yes_token_id(market: dict) -> str | None:
    """Extract YES token ID from Gamma API market dict.

    Gamma returns clobTokenIds as a JSON string e.g. '["123...", "456..."]'
    where index 0 = YES token, index 1 = NO token (matches outcomes order).
    """
    clob_ids_raw = market.get("clobTokenIds", "")
    if clob_ids_raw:
        try:
            ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
            if ids:
                return str(ids[0])
        except (json.JSONDecodeError, IndexError):
            pass
    return None


def discover_polymarket_markets(session: requests.Session) -> list[dict]:
    """Fetch all relevant markets (active + closed) from Gamma API.

    Server-side filters for closed markets:
      - end_date_min=2025-01-20: only markets that closed after Trump inauguration
        (matches our training data floor; avoids pre-inauguration regime)
      - volume_num_min=50000: only whale-scale markets ($50k+ total volume)

    At volume_num_min=1000 there were still 150k+ pages to crawl (hours).
    Raising to 50k cuts to the markets whales actually trade.
    Active markets: no volume filter (all politically-relevant open markets kept).
    """
    relevant = []
    limit = 100

    for closed in (False, True):
        offset = 0
        pages = 0
        base_params: dict = {"closed": str(closed).lower(), "limit": limit}
        if closed:
            base_params["end_date_min"] = "2025-01-20"  # Trump inauguration = training floor
            base_params["volume_num_min"] = 50000        # whale-scale only

        while True:
            try:
                resp = session.get(
                    f"{GAMMA_BASE}/markets",
                    params={**base_params, "offset": offset},
                    timeout=15,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Gamma API error (closed=%s, offset=%d): %s", closed, offset, exc)
                break

            items = resp.json()
            if isinstance(items, dict):
                items = items.get("markets", [])
            if not items:
                break

            pages += 1
            for m in items:
                if _is_relevant(m.get("question", "")):
                    relevant.append(m)

            if pages % 20 == 0:
                logger.info("  Gamma page %d (closed=%s, offset=%d) → %d relevant so far",
                            pages, closed, offset, len(relevant))

            if len(items) < limit:
                break
            offset += limit
            time.sleep(0.2)

        logger.info("Gamma closed=%s done: %d pages, %d relevant", closed, pages, len(relevant))

    logger.info("Discovered %d relevant Polymarket markets total", len(relevant))
    return relevant


def pull_polymarket_prices(session: requests.Session, markets: list[dict]) -> None:
    POLY_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    start_ts = int(DATA_START_DT.timestamp())
    end_ts = int(datetime.now(tz=timezone.utc).timestamp())

    for i, market in enumerate(markets, 1):
        cid = market.get("conditionId", "")
        question = market.get("question", "")[:60]
        yes_token = _extract_yes_token_id(market)

        if not yes_token:
            continue

        out_path = POLY_PRICES_DIR / f"{cid}_YES.parquet"
        if out_path.exists():
            logger.info("[%d/%d] Skip (exists): %s", i, len(markets), question)
            continue

        logger.info("[%d/%d] Fetching: %s", i, len(markets), question)
        try:
            resp = session.get(
                f"{CLOB_BASE}/prices-history",
                params={
                    "market": yes_token,
                    "interval": "max",
                    "fidelity": 1440,   # daily — hourly (60) only returns ~28 days
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                },
                timeout=20,
            )
            resp.raise_for_status()
            history = resp.json().get("history", [])
        except requests.RequestException as exc:
            logger.warning("  Failed: %s", exc)
            time.sleep(1)
            continue

        if not history:
            continue

        df = pd.DataFrame(history)
        if "t" in df.columns:
            df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
            df = df.rename(columns={"p": "yes_price"})
            df = df[["timestamp", "yes_price"]].set_index("timestamp").sort_index()

        df.to_parquet(out_path)
        logger.info("  Saved %d rows → %s", len(df), out_path.name)
        time.sleep(0.3)


def pull_polymarket() -> None:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    markets = discover_polymarket_markets(session)

    POLY_DIR.mkdir(parents=True, exist_ok=True)
    catalog = pd.DataFrame([{
        "condition_id": m.get("conditionId", ""),
        "question":     m.get("question", ""),
        "slug":         m.get("slug", ""),
        "closed":       m.get("closed", False),
        "end_date":     m.get("endDateIso", m.get("endDate", "")),
        "volume_24h":   m.get("volume24hr", 0),
        "yes_token_id": _extract_yes_token_id(m) or "",
    } for m in markets])
    catalog_path = POLY_DIR / "markets_catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)
    logger.info("Markets catalog → %s (%d markets)", catalog_path, len(catalog))

    pull_polymarket_prices(session, markets)


# ===========================================================================
# 3. Truth Social historical posts
# ===========================================================================

def pull_truth_social() -> None:
    try:
        from truthbrush.api import Api  # type: ignore
    except ImportError as exc:
        raise ImportError("truthbrush not installed. Run: pip install truthbrush") from exc

    username = os.environ.get("TRUTHSOCIAL_USERNAME")
    password = os.environ.get("TRUTHSOCIAL_PASSWORD")
    if not username or not password:
        logger.warning("TRUTHSOCIAL_USERNAME / TRUTHSOCIAL_PASSWORD not set — skipping Truth Social pull.")
        return

    TS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TS_DIR / "trump_posts.jsonl"

    last_stored_id: str | None = None
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines:
            try:
                last_stored_id = json.loads(lines[-1]).get("id")
                logger.info("Resuming from last stored post_id=%s", last_stored_id)
            except json.JSONDecodeError:
                pass

    api = Api(username, password)
    new_posts: list[dict] = []

    logger.info("Fetching @%s posts since %s ...", TRUTH_SOCIAL_USER, DATA_START)
    try:
        for post in api.pull_statuses(
            TRUTH_SOCIAL_USER,
            replies=False,
            created_after=DATA_START_DT,
        ):
            pid = str(post.get("id", ""))
            if last_stored_id and pid == last_stored_id:
                logger.info("Reached previously stored post — done.")
                break
            new_posts.append(post)
            if len(new_posts) % 40 == 0:
                logger.info("  %d posts fetched, pausing 25s for rate limits...", len(new_posts))
                time.sleep(25)
    except Exception as exc:
        logger.warning("Truth Social fetch interrupted: %s", exc)

    if not new_posts:
        logger.info("No new posts to store.")
        return

    new_posts.reverse()
    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for post in new_posts:
            f.write(json.dumps(post, default=str) + "\n")
    logger.info("Saved %d posts → %s", len(new_posts), out_path)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("WhaleWatch_Alpha — Historical Data Pull")
    logger.info("Output : %s", DATA_ROOT)
    logger.info("Start  : %s", DATA_START)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Equity: daily bars (full 2024-present) + 5m bars (last 60 days) via yfinance")
    logger.info("")

    logger.info("[1/3] Equity (SPY 5m / QQQ 5m / VIX 5m)...")
    pull_equity()

    logger.info("\n[2/3] Polymarket historical prices...")
    pull_polymarket()

    logger.info("\n[3/3] Truth Social posts...")
    pull_truth_social()

    logger.info("\nDone. Files in %s:", DATA_ROOT)
    for p in sorted(DATA_ROOT.rglob("*.parquet"), key=lambda x: x.stat().st_size, reverse=True):
        logger.info("  %-60s  %.1f MB", str(p.relative_to(DATA_ROOT)), p.stat().st_size / 1e6)
    jsonl = DATA_ROOT / "truth_social" / "trump_posts.jsonl"
    if jsonl.exists():
        logger.info("  %-60s  %.1f MB", "truth_social/trump_posts.jsonl", jsonl.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
