"""setup_schwab_auth.py — One-time (and weekly) Schwab OAuth token setup.

Run this script once before using SchwabExecutor, and again every 7 days
when Schwab's refresh token expires.

It will open a browser window for you to log in to Schwab, then save
the token file to SCHWAB_TOKEN_PATH (default: D:/WhaleWatch_Data/schwab_token.json).

Requirements in .env:
  SCHWAB_API_KEY=your_app_key
  SCHWAB_APP_SECRET=your_app_secret
  SCHWAB_TOKEN_PATH=D:/WhaleWatch_Data/schwab_token.json   (optional override)

Usage:
    python scripts/setup_schwab_auth.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

try:
    from schwab import auth
except ImportError:
    print("schwab-py not installed. Run: pip install schwab-py>=0.4.0")
    sys.exit(1)

api_key    = os.getenv("SCHWAB_API_KEY", "")
app_secret = os.getenv("SCHWAB_APP_SECRET", "")
token_path = os.getenv("SCHWAB_TOKEN_PATH", "D:/WhaleWatch_Data/schwab_token.json")
callback_url = "https://127.0.0.1:8182"

if not api_key or not app_secret:
    print("ERROR: SCHWAB_API_KEY and SCHWAB_APP_SECRET must be set in .env")
    sys.exit(1)

Path(token_path).parent.mkdir(parents=True, exist_ok=True)

print(f"Opening browser for Schwab OAuth login...")
print(f"Token will be saved to: {token_path}")
print(f"Callback URL: {callback_url}")
print()
print("After logging in, paste the full redirect URL from the browser address bar.")
print()

client = auth.easy_client(
    api_key=api_key,
    app_secret=app_secret,
    callback_url=callback_url,
    token_path=token_path,
)

# Verify by fetching account numbers
try:
    resp     = client.get_account_numbers()
    accounts = resp.json()
    print("\nAuthentication successful!")
    print("\nYour account hashes (add one to .env as SCHWAB_ACCOUNT_HASH):")
    for acct in accounts:
        num  = acct.get("accountNumber", "?")
        hash_ = acct.get("hashValue", "?")
        print(f"  Account ...{str(num)[-4:]}  →  SCHWAB_ACCOUNT_HASH={hash_}")
    print(f"\nToken saved to: {token_path}")
    print("NOTE: Re-run this script every 7 days — Schwab refresh tokens expire weekly.")
except Exception as exc:
    print(f"\nAuth succeeded but account fetch failed: {exc}")
    print("Token saved. Set SCHWAB_ACCOUNT_HASH manually from the Schwab portal.")
