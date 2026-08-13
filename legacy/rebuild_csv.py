#!/usr/bin/env python3
"""
rebuild_csv.py
--------------
Repairs the CSVs you have already collected, using the raw JSON you kept.
 
The original collector's CSVs have two defects:
 
  1. `cboe_timestamp` is empty in every row. It was read from
     payload["data"]["timestamp"], but that key is at the top level of the
     payload, so .get() returned "" every time.
 
  2. The option root is missing. SPX (AM-settled) and SPXW (PM-settled) share
     ~3,300 (expiry, type, strike) triples per snapshot on third-Friday
     expiries. Without the root those rows are indistinguishable, and they are
     genuinely different contracts at different prices.
 
It also drops the bid_size / ask_size / theo fields, which matter if you plan
to filter on quote quality or compare against CBOE's own theoretical value.
 
Every one of those is recoverable from the raw JSON, which the original script
wrote untouched. This script re-derives corrected CSVs from raw/ and reports
which captures were duplicates.
 
USAGE
    python3 rebuild_csv.py --dir "'/Users/1off_k/Downloads/options project /snapshots'"
 
    # write .csv.gz instead of .csv (about 4.7x smaller)
    python3 rebuild_csv.py --dir ... --gzip
 
    # just report, change nothing
    python3 rebuild_csv.py --dir ... --dry-run
 
Corrected files go to <dir>/csv_fixed/ so your originals are left alone.
Requires Python 3.8+. Standard library only.
"""
 
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from collect_snapshot import CSV_FIELDS, parse_option_symbol
except ImportError:
    print("ERROR: keep rebuild_csv.py next to collect_snapshot.py", file=sys.stderr)
    raise
 
 
def load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)
 
 
def capture_from_name(path: Path, ticker_hint: str = "_SPX") -> str:
    """'_SPX_2026-08-11T20-47-05Z.json' -> '2026-08-11T20-47-05Z'."""
    stem = path.name
    for suffix in (".json.gz", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    _, _, rest = stem.rpartition("_")
    return rest or stem
 
 
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="the snapshots/ directory containing raw/")
    ap.add_argument("--ticker", default="_SPX")
    ap.add_argument("--gzip", action="store_true", help="write .csv.gz")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
 
    root_dir = Path(args.dir).expanduser().resolve()
    raw_dir = root_dir / "raw"
    if not raw_dir.is_dir():
        print(f"ERROR: {raw_dir} not found", file=sys.stderr)
        return 1
 
    out_dir = root_dir / "csv_fixed"
    if not args.dry_run:
        out_dir.mkdir(exist_ok=True)
 
    files = sorted(list(raw_dir.glob("*.json")) + list(raw_dir.glob("*.json.gz")))
    if not files:
        print(f"ERROR: no raw payloads in {raw_dir}", file=sys.stderr)
        return 1
 
    print(f"{len(files)} raw payloads in {raw_dir}\n")
 
    seen_last_trade = {}
    written = duplicates = 0
    meta_rows = []
 
    for path in files:
        payload = load(path)
        data = payload.get("data", {})
        capture = capture_from_name(path, args.ticker)
        cboe_ts = payload.get("timestamp", "")
        idx_last_trade = data.get("last_trade_time", "")
        seqno = data.get("seqno")
        price = data.get("current_price")
 
        dup_of = seen_last_trade.get(idx_last_trade)
        is_dup = dup_of is not None
        if is_dup:
            duplicates += 1
        else:
            seen_last_trade[idx_last_trade] = capture
 
        flag = f"DUPLICATE of {dup_of}" if is_dup else "unique"
        print(f"  {capture}  feed={cboe_ts}  index_last_print={idx_last_trade} ET  "
              f"px={price}  [{flag}]")
 
        meta_rows.append({
            "capture_utc": capture, "cboe_timestamp": cboe_ts,
            "index_last_trade_et": idx_last_trade, "seqno": seqno,
            "ticker": args.ticker, "current_price": price,
            "n_options": len(data.get("options", [])),
            "duplicate_of": dup_of or "",
        })
 
        if args.dry_run:
            continue
 
        rows, unparsed, roots = [], 0, {}
        for opt in data.get("options", []):
            parsed = parse_option_symbol(opt.get("option", ""))
            if parsed is None:
                unparsed += 1
                continue
            r, expiry, opt_type, strike = parsed
            roots[r] = roots.get(r, 0) + 1
            rows.append({
                "capture_utc": capture, "cboe_timestamp": cboe_ts,
                "index_last_trade_et": idx_last_trade, "seqno": seqno,
                "underlying_ticker": args.ticker, "underlying_price": price,
                "option_symbol": opt.get("option"), "root": r,
                "type": opt_type, "expiry": expiry, "strike": strike,
                "bid": opt.get("bid"), "bid_size": opt.get("bid_size"),
                "ask": opt.get("ask"), "ask_size": opt.get("ask_size"),
                "last_trade_price": opt.get("last_trade_price"),
                "last_trade_time": opt.get("last_trade_time"),
                "volume": opt.get("volume"),
                "open_interest": opt.get("open_interest"),
                "iv": opt.get("iv"), "theo": opt.get("theo"),
                "delta": opt.get("delta"), "gamma": opt.get("gamma"),
                "theta": opt.get("theta"), "vega": opt.get("vega"),
                "rho": opt.get("rho"),
                "prev_day_close": opt.get("prev_day_close"),
            })
 
        name = f"{args.ticker}_{capture}.csv" + (".gz" if args.gzip else "")
        target = out_dir / name
        opener = gzip.open if args.gzip else open
        with opener(target, "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        written += 1
        detail = ", ".join(f"{k}={v}" for k, v in sorted(roots.items()))
        extra = f", {unparsed} unparsed" if unparsed else ""
        print(f"      -> {name}  ({len(rows)} rows: {detail}{extra})")
 
    if not args.dry_run:
        with open(root_dir / "meta_rebuilt.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            w.writeheader()
            w.writerows(meta_rows)
 
    unique = len(files) - duplicates
    print(f"\n{len(files)} captures -> {unique} with distinct index prints, "
          f"{duplicates} duplicates")
    if args.dry_run:
        print("(dry run -- nothing written)")
    else:
        print(f"corrected CSVs in {out_dir}")
        print(f"manifest in {root_dir / 'meta_rebuilt.csv'} "
              f"(the duplicate_of column tells you which to exclude)")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
 
