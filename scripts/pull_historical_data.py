"""One-time historical data pull.

Downloads and stores:
  1. SPY / QQQ / VIX  — hourly OHLCV since 2024-01-01 via yfinance
  2. Polymarket        — price history for all politically-relevant markets
                         (including closed ones) via CLOB + Gamma APIs
  3. Truth Social      — all @realDonaldTrump posts since 2024-01-01 via truthbrush

Output directory: D:/WhaleWatch_Data/
  equity/
    SPY_1h.parquet
    QQQ_1h.parquet
    VIX_1h.parquet
  polymarket/
    markets_catalog.parquet       (metadata for all discovered markets)
    prices/{condition_id}_YES.parquet   (hourly price history per market)
  truth_social/
    trump_posts.jsonl             (one JSON object per line)

Run once:
    python scripts/pull_historical_data.py

Re-running is safe — equity and Polymarket files are overwritten, while
Truth Social posts are appended only if newer than the last stored post_id.
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
CLOB_BASE = "https://clob.polymarket.com"

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
# 1. Equity data (yfinance)
# ===========================================================================

def pull_equity() -> None:
    """Download hourly OHLCV for SPY, QQQ, VIX and save as parquet."""
    import yfinance as yf

    EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    for ticker in TICKERS:
        yf_ticker = f"^{ticker}" if ticker == "VIX" else ticker
        out_path = EQUITY_DIR / f"{ticker}_1h.parquet"

        logger.info("Downloading %s hourly bars %s → %s ...", ticker, DATA_START, today)
        df = yf.download(
            yf_ticker,
            start=DATA_START,
            end=today,
            interval="1h",
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            logger.warning("No data returned for %s — skipping.", ticker)
            continue

        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.to_parquet(out_path)
        logger.info("Saved %s rows → %s", len(df), out_path)


# ===========================================================================
# 2. Polymarket historical data
# ===========================================================================

def _is_relevant(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in WATCHLIST_KEYWORDS)


def _extract_yes_token_id(market: dict) -> str | None:
    for token in market.get("tokens", []):
        if str(token.get("outcome", "")).upper() == "YES":
            return str(token.get("token_id", ""))
    ids = market.get("clobTokenIds", [])
    return str(ids[0]) if ids else None


def discover_polymarket_markets(session: requests.Session) -> list[dict]:
    """Fetch all relevant markets (active + closed) from Gamma API."""
    relevant = []
    offset = 0
    limit = 100

    for closed in (False, True):
        offset = 0
        while True:
            try:
                resp = session.get(
                    f"{GAMMA_BASE}/markets",
                    params={"closed": str(closed).lower(), "limit": limit, "offset": offset},
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

            for m in items:
                if _is_relevant(m.get("question", "")):
                    relevant.append(m)

            if len(items) < limit:
                break
            offset += limit
            time.sleep(0.2)  # be polite to Gamma API

    logger.info("Discovered %d relevant Polymarket markets", len(relevant))
    return relevant


def pull_polymarket_prices(
    session: requests.Session,
    markets: list[dict],
) -> None:
    """Fetch hourly YES price history for each market and save as parquet."""
    POLY_PRICES_DIR.mkdir(parents=True, exist_ok=True)

    start_ts = int(DATA_START_DT.timestamp())
    end_ts = int(datetime.now(tz=timezone.utc).timestamp())

    for i, market in enumerate(markets, 1):
        cid = market.get("condition_id", "")
        question = market.get("question", "")[:60]
        yes_token = _extract_yes_token_id(market)

        if not yes_token:
            logger.debug("No YES token for market %s — skipping.", cid)
            continue

        out_path = POLY_PRICES_DIR / f"{cid}_YES.parquet"
        if out_path.exists():
            logger.info("[%d/%d] Already exists: %s — skipping.", i, len(markets), question)
            continue

        logger.info("[%d/%d] Fetching price history: %s", i, len(markets), question)
        try:
            resp = session.get(
                f"{CLOB_BASE}/prices-history",
                params={
                    "token_id": yes_token,
                    "interval": "max",
                    "fidelity": 60,     # 60-minute resolution
                    "start_time": start_ts,
                    "end_time": end_ts,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("Price history failed for %s: %s", cid, exc)
            time.sleep(1)
            continue

        history = data.get("history", [])
        if not history:
            logger.debug("Empty price history for %s.", cid)
            continue

        df = pd.DataFrame(history)
        # Normalize: history items are typically {"t": unix_ts, "p": price}
        if "t" in df.columns:
            df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
            df = df.rename(columns={"p": "yes_price"})
            df = df[["timestamp", "yes_price"]].set_index("timestamp").sort_index()

        df.to_parquet(out_path)
        logger.info("  Saved %d rows → %s", len(df), out_path.name)

        time.sleep(0.3)  # stay within rate limits


def pull_polymarket() -> None:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    markets = discover_polymarket_markets(session)

    # Save catalog
    POLY_DIR.mkdir(parents=True, exist_ok=True)
    catalog_rows = []
    for m in markets:
        catalog_rows.append({
            "condition_id": m.get("condition_id", ""),
            "question": m.get("question", ""),
            "slug": m.get("market_slug", ""),
            "closed": m.get("closed", False),
            "end_date": m.get("end_date_iso", ""),
            "yes_token_id": _extract_yes_token_id(m) or "",
        })
    catalog_df = pd.DataFrame(catalog_rows)
    catalog_path = POLY_DIR / "markets_catalog.parquet"
    catalog_df.to_parquet(catalog_path, index=False)
    logger.info("Saved markets catalog → %s (%d markets)", catalog_path, len(catalog_df))

    pull_polymarket_prices(session, markets)


# ===========================================================================
# 3. Truth Social historical posts
# ===========================================================================

def pull_truth_social() -> None:
    """Download all @realDonaldTrump posts since DATA_START_DT via truthbrush."""
    try:
        from truthbrush.api import Api  # type: ignore
    except ImportError as exc:
        raise ImportError("truthbrush not installed. Run: pip install truthbrush") from exc

    username = os.environ.get("TRUTHSOCIAL_USERNAME")
    password = os.environ.get("TRUTHSOCIAL_PASSWORD")
    if not username or not password:
        logger.error("TRUTHSOCIAL_USERNAME / TRUTHSOCIAL_PASSWORD not set — skipping Truth Social pull.")
        return

    TS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TS_DIR / "trump_posts.jsonl"

    # Determine the most-recently stored post_id to avoid duplicate appends
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
    new_posts = []

    logger.info(
        "Fetching @%s posts since %s (this may take several minutes due to rate limits)...",
        TRUTH_SOCIAL_USER, DATA_START,
    )

    try:
        for post in api.pull_statuses(
            TRUTH_SOCIAL_USER,
            replies=False,
            created_after=DATA_START_DT,
        ):
            pid = str(post.get("id", ""))
            if last_stored_id and pid == last_stored_id:
                logger.info("Reached previously stored post — stopping.")
                break
            new_posts.append(post)

            # Brief pause every 40 posts to avoid Cloudflare rate limiting
            if len(new_posts) % 40 == 0:
                logger.info("  Fetched %d posts so far, pausing briefly...", len(new_posts))
                time.sleep(25)

    except Exception as exc:
        logger.warning("Truth Social fetch interrupted: %s", exc)

    if not new_posts:
        logger.info("No new posts found.")
        return

    # Append new posts to JSONL (newest-first from API → reverse for chronological order)
    new_posts.reverse()
    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for post in new_posts:
            f.write(json.dumps(post, default=str) + "\n")

    logger.info("Saved %d new posts → %s", len(new_posts), out_path)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("WhaleWatch_Alpha — Historical Data Pull")
    logger.info("Output directory: %s", DATA_ROOT)
    logger.info("Start date: %s", DATA_START)
    logger.info("=" * 60)

    logger.info("\n[1/3] Pulling equity data (SPY / QQQ / VIX)...")
    pull_equity()

    logger.info("\n[2/3] Pulling Polymarket historical prices...")
    pull_polymarket()

    logger.info("\n[3/3] Pulling Truth Social posts...")
    pull_truth_social()

    logger.info("\nDone. Data stored in %s", DATA_ROOT)


if __name__ == "__main__":
    main()
