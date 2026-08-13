"""
collect_cboe_snapshots.py
-------------------------
Collects synchronized SPX option-chain snapshots from CBOE's free,
public, delayed-quotes endpoint. Each snapshot contains the ENTIRE
option chain AND the underlying index level in a single payload with
a single timestamp -- i.e., options and underlying at the same moment.

Data is 15-minute delayed. That is irrelevant for density estimation:
each snapshot is internally consistent, which is what matters.

Usage (run on a trading day, e.g. 9:45am-3:45pm ET):
    python collect_cboe_snapshots.py --ticker _SPX --interval 10 --count 36

This takes one snapshot every 10 minutes, 36 times (one full session).
A single run with --count 1 works too (e.g., to grab Friday's close).

Output:
    snapshots/raw/_SPX_<UTC timestamp>.json   (raw payload, untouched)
    snapshots/csv/_SPX_<UTC timestamp>.csv    (flattened, one row per option)

Requires: Python 3.8+. Standard library only.
Please keep the interval >= 5 minutes -- this is a courtesy endpoint.
"""

import argparse
import csv
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ENDPOINT = "https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
OSI_RE = re.compile(r"^(?P<root>.+?)(?P<exp>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")

CSV_FIELDS = [
    "capture_utc", "cboe_timestamp", "underlying_ticker", "underlying_price",
    "option_symbol", "type", "expiry", "strike",
    "bid", "ask", "last_trade_price", "volume", "open_interest",
    "iv", "delta", "gamma", "theta", "vega", "rho",
]


def fetch_snapshot(ticker: str) -> dict:
    url = ENDPOINT.format(ticker=ticker)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (academic research; RND estimation project)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_option_symbol(symbol: str):
    m = OSI_RE.match(symbol.replace(" ", ""))
    if not m:
        return None
    expiry = "20" + m.group("exp")  # YYMMDD -> YYYYMMDD
    expiry = f"{expiry[0:4]}-{expiry[4:6]}-{expiry[6:8]}"
    strike = int(m.group("strike")) / 1000.0
    opt_type = "call" if m.group("cp") == "C" else "put"
    return expiry, opt_type, strike


def flatten(payload: dict, capture_utc: str, ticker: str):
    data = payload.get("data", {})
    underlying_price = data.get("current_price")
    cboe_ts = data.get("timestamp", "")
    rows = []
    for opt in data.get("options", []):
        parsed = parse_option_symbol(opt.get("option", ""))
        if parsed is None:
            continue
        expiry, opt_type, strike = parsed
        rows.append({
            "capture_utc": capture_utc,
            "cboe_timestamp": cboe_ts,
            "underlying_ticker": ticker,
            "underlying_price": underlying_price,
            "option_symbol": opt.get("option"),
            "type": opt_type,
            "expiry": expiry,
            "strike": strike,
            "bid": opt.get("bid"),
            "ask": opt.get("ask"),
            "last_trade_price": opt.get("last_trade_price"),
            "volume": opt.get("volume"),
            "open_interest": opt.get("open_interest"),
            "iv": opt.get("iv"),
            "delta": opt.get("delta"),
            "gamma": opt.get("gamma"),
            "theta": opt.get("theta"),
            "vega": opt.get("vega"),
            "rho": opt.get("rho"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="_SPX", help="_SPX for the S&P 500 index; SPY also works")
    ap.add_argument("--interval", type=float, default=10.0, help="minutes between snapshots (>= 5)")
    ap.add_argument("--count", type=int, default=1, help="number of snapshots to take")
    args = ap.parse_args()

    if args.count > 1 and args.interval < 5:
        raise SystemExit("Please use an interval of at least 5 minutes.")

    raw_dir = Path("snapshots/raw"); raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = Path("snapshots/csv"); csv_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        capture_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        try:
            payload = fetch_snapshot(args.ticker)
            stem = f"{args.ticker}_{capture_utc}"
            (raw_dir / f"{stem}.json").write_text(json.dumps(payload))
            rows = flatten(payload, capture_utc, args.ticker)
            with open(csv_dir / f"{stem}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)
            px = payload.get("data", {}).get("current_price")
            print(f"[{i+1}/{args.count}] {capture_utc}  underlying={px}  options={len(rows)}  saved.")
        except Exception as e:
            print(f"[{i+1}/{args.count}] {capture_utc}  FAILED: {e}  (continuing)")
        if i < args.count - 1:
            time.sleep(args.interval * 60)

    print("Done. Zip the 'snapshots' folder and upload it for QA.")


if __name__ == "__main__":
    main()