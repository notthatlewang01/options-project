#!/usr/bin/env python3
"""
collect_snapshot.py
-------------------
Takes ONE synchronized SPX option-chain snapshot from CBOE's free, public,
delayed-quotes endpoint, then exits. Scheduling is the operating system's job
(systemd timer / cron), not this script's.

This is a server-oriented rewrite of collect_cboe_snapshots.py. The important
differences:

  * One snapshot per invocation. No internal sleep loop, so a crash or a reboot
    costs you a single sample instead of the rest of the session, and the
    scheduler can restart it.
  * Output paths come from the environment and are absolute. A cron/systemd
    job does not inherit your shell's working directory.
  * Everything is gzipped. The raw JSON compresses ~8x, the CSV ~4.7x.
  * Duplicate and stale captures are detected and skipped, so a holiday or an
    after-hours run costs one HTTP request and writes nothing.
  * The option root (SPX vs SPXW) is preserved. See NOTE ON ROOTS below.
  * cboe_timestamp is actually populated. In the original it was read from
    payload["data"]["timestamp"], but that key lives at the top level, so the
    column was empty in every row ever written.

Data is 15-minute delayed. That is irrelevant for density estimation: each
snapshot is internally consistent, which is what matters.

NOTE ON ROOTS
    The chain contains both SPX (AM-settled, monthly) and SPXW (PM-settled,
    weeklies). On a third Friday BOTH roots list the same expiry date and the
    same strikes -- ~3,300 (expiry, type, strike) triples per snapshot are
    shared. They are different contracts with different quotes. The original
    script parsed the root out of the OSI symbol and then discarded it, making
    those rows indistinguishable. The `root` column below fixes that.

CONFIGURATION (environment variables, all optional except the first)
    CBOE_SNAPSHOT_DIR   absolute path to write under          (required)
    CBOE_TICKER         _SPX (default), SPY, etc.
    CBOE_WRITE_CSV      1 (default) or 0 to keep only raw JSON
    CBOE_MAX_STALE_MIN  22 (default) -- see freshness gating below
    CBOE_RETRIES        3 (default) HTTP attempts, exponential backoff

FRESHNESS GATING
    payload["data"]["last_trade_time"] is the underlying index's last print,
    in US/Eastern. During regular trading hours it runs ~15 minutes behind the
    wall clock (the advertised delay). After the cash index stops printing at
    16:15 ET it freezes, while the CDN keeps serving the same stale body with a
    fresh top-level timestamp -- which is why the original run wrote six
    identical 7728.2002 snapshots after the close.

    So: if last_trade_time has not moved since the previous capture, or is
    older than CBOE_MAX_STALE_MIN, there is no new data and we skip the write.
    This doubles as holiday handling -- no exchange calendar dependency needed,
    because on a holiday the feed simply never freshens.

EXIT CODES
    0   wrote a snapshot, or deliberately skipped a stale/duplicate one
    1   fetch or write failed (systemd will log and can retry)

Requires Python 3.9+ (zoneinfo). Standard library only.
"""

import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ENDPOINT = "https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
USER_AGENT = "Mozilla/5.0 (academic research; RND estimation project)"
EASTERN = ZoneInfo("America/New_York")

# Occupancy-style OSI symbol: ROOT + YYMMDD + C/P + strike*1000 zero-padded to 8.
OSI_RE = re.compile(r"^(?P<root>[A-Z]+)(?P<exp>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")

CSV_FIELDS = [
    "capture_utc", "cboe_timestamp", "index_last_trade_et", "seqno",
    "underlying_ticker", "underlying_price",
    "option_symbol", "root", "type", "expiry", "strike",
    "bid", "bid_size", "ask", "ask_size",
    "last_trade_price", "last_trade_time", "volume", "open_interest",
    "iv", "theo", "delta", "gamma", "theta", "vega", "rho",
    "prev_day_close",
]

# Appended once per successful capture. Keeps per-snapshot constants out of the
# big per-option file, and doubles as a manifest of what you actually have.
META_FIELDS = [
    "capture_utc", "cboe_timestamp", "index_last_trade_et", "seqno",
    "ticker", "current_price", "bid", "ask", "open", "high", "low",
    "prev_day_close", "iv30", "n_options", "raw_gz_bytes",
]


def log(msg: str) -> None:
    """Single-line stdout logging; journald picks this up verbatim."""
    print(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------

def fetch_snapshot(ticker: str, retries: int = 3, timeout: int = 30) -> dict:
    """GET the chain, with gzip transfer and exponential backoff on failure."""
    url = ENDPOINT.format(ticker=ticker)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"},
    )
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return json.loads(body.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            last_err = e
            if attempt < retries:
                backoff = 2 ** attempt
                log(f"fetch attempt {attempt}/{retries} failed ({type(e).__name__}: {e}); "
                    f"retrying in {backoff}s")
                time.sleep(backoff)
    raise RuntimeError(f"all {retries} fetch attempts failed: {last_err}")


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_option_symbol(symbol: str):
    """'SPXW260821C01400000' -> ('SPXW', '2026-08-21', 'call', 1400.0)."""
    m = OSI_RE.match(symbol.replace(" ", ""))
    if not m:
        return None
    yy, mm, dd = m.group("exp")[0:2], m.group("exp")[2:4], m.group("exp")[4:6]
    expiry = f"20{yy}-{mm}-{dd}"
    strike = int(m.group("strike")) / 1000.0
    opt_type = "call" if m.group("cp") == "C" else "put"
    return m.group("root"), expiry, opt_type, strike


def parse_last_trade(ts: str):
    """'2026-08-11T16:14:59' (US/Eastern, naive) -> aware datetime, or None."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=EASTERN)
    except ValueError:
        return None


def flatten(payload: dict, capture_utc: str, ticker: str):
    """Yield one dict per option. Generator so we never hold 31k dicts at once."""
    data = payload.get("data", {})
    underlying_price = data.get("current_price")
    cboe_ts = payload.get("timestamp", "")          # top level, not under "data"
    idx_last_trade = data.get("last_trade_time", "")
    seqno = data.get("seqno")
    unparsed = 0

    for opt in data.get("options", []):
        parsed = parse_option_symbol(opt.get("option", ""))
        if parsed is None:
            unparsed += 1
            continue
        root, expiry, opt_type, strike = parsed
        yield {
            "capture_utc": capture_utc,
            "cboe_timestamp": cboe_ts,
            "index_last_trade_et": idx_last_trade,
            "seqno": seqno,
            "underlying_ticker": ticker,
            "underlying_price": underlying_price,
            "option_symbol": opt.get("option"),
            "root": root,
            "type": opt_type,
            "expiry": expiry,
            "strike": strike,
            "bid": opt.get("bid"),
            "bid_size": opt.get("bid_size"),
            "ask": opt.get("ask"),
            "ask_size": opt.get("ask_size"),
            "last_trade_price": opt.get("last_trade_price"),
            "last_trade_time": opt.get("last_trade_time"),
            "volume": opt.get("volume"),
            "open_interest": opt.get("open_interest"),
            "iv": opt.get("iv"),
            "theo": opt.get("theo"),
            "delta": opt.get("delta"),
            "gamma": opt.get("gamma"),
            "theta": opt.get("theta"),
            "vega": opt.get("vega"),
            "rho": opt.get("rho"),
            "prev_day_close": opt.get("prev_day_close"),
        }

    if unparsed:
        log(f"WARNING: {unparsed} option symbols did not match the OSI pattern "
            f"and were dropped")


# --------------------------------------------------------------------------
# state (duplicate / staleness detection)
# --------------------------------------------------------------------------

def read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file + rename so a kill mid-write can't leave a partial."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def acquire_lock(path: Path):
    """Refuse to run two collectors at once. Returns the fd, or None if locked."""
    try:
        return os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # Stale lock from a killed process? Clear it if it's older than an hour.
        try:
            if time.time() - path.stat().st_mtime > 3600:
                path.unlink()
                return os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileNotFoundError:
            pass
        return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default=os.environ.get("CBOE_TICKER", "_SPX"))
    ap.add_argument("--dir", default=os.environ.get("CBOE_SNAPSHOT_DIR"),
                    help="absolute output directory (or set CBOE_SNAPSHOT_DIR)")
    ap.add_argument("--max-stale-min", type=float,
                    default=env_int("CBOE_MAX_STALE_MIN", 22),
                    help="skip if the index's last print is older than this")
    ap.add_argument("--retries", type=int, default=env_int("CBOE_RETRIES", 3))
    ap.add_argument("--no-csv", action="store_true",
                    default=os.environ.get("CBOE_WRITE_CSV", "1") == "0")
    ap.add_argument("--force", action="store_true",
                    help="write even if the snapshot looks stale or duplicated")
    ap.add_argument("--from-file", metavar="PATH",
                    help="read the payload from a local .json/.json.gz instead of "
                         "the network (for testing the pipeline offline)")
    args = ap.parse_args()

    if not args.dir:
        log("ERROR: no output directory. Set CBOE_SNAPSHOT_DIR or pass --dir.")
        return 1

    out = Path(args.dir).expanduser().resolve()
    raw_dir = out / "raw"
    csv_dir = out / "csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)

    lock_path = out / ".collect.lock"
    lock_fd = acquire_lock(lock_path)
    if lock_fd is None:
        log("another collector is already running; exiting")
        return 0

    try:
        capture_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        state_path = out / ".state.json"
        state = read_state(state_path)

        # ---- fetch -------------------------------------------------------
        try:
            if args.from_file:
                p = Path(args.from_file)
                opener = gzip.open if p.suffix == ".gz" else open
                with opener(p, "rt") as f:
                    payload = json.load(f)
                log(f"loaded payload from {p} (offline mode)")
            else:
                payload = fetch_snapshot(args.ticker, retries=args.retries)
        except Exception as e:
            log(f"ERROR: {e}")
            return 1

        data = payload.get("data", {})
        cboe_ts = payload.get("timestamp", "")
        idx_last_trade = data.get("last_trade_time", "")
        seqno = data.get("seqno")
        price = data.get("current_price")

        # ---- duplicate check --------------------------------------------
        # last_trade_time is the primary signal. seqno alone is not enough --
        # in the original run it advanced once (16947306680) while the index
        # print stayed frozen at 16:14:59, which would have let a stale
        # snapshot through.
        if not args.force and idx_last_trade and idx_last_trade == state.get("index_last_trade_et"):
            log(f"skip: no new data (index last print still {idx_last_trade}, "
                f"price {price})")
            return 0

        # ---- freshness check --------------------------------------------
        lt = parse_last_trade(idx_last_trade)
        if not args.force:
            if lt is None:
                log(f"skip: could not parse last_trade_time {idx_last_trade!r}")
                return 0
            age = datetime.now(timezone.utc) - lt.astimezone(timezone.utc)
            if age > timedelta(minutes=args.max_stale_min):
                log(f"skip: feed is stale (index last print {idx_last_trade} ET, "
                    f"{age.total_seconds()/60:.1f} min old > "
                    f"{args.max_stale_min:.0f} min) -- market closed or holiday")
                return 0

        # ---- write -------------------------------------------------------
        stem = f"{args.ticker}_{capture_utc}"
        raw_blob = gzip.compress(json.dumps(payload, separators=(",", ":")).encode(),
                                 compresslevel=6)
        write_atomic(raw_dir / f"{stem}.json.gz", raw_blob)

        n_options = 0
        if not args.no_csv:
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
            w.writeheader()
            for row in flatten(payload, capture_utc, args.ticker):
                w.writerow(row)
                n_options += 1
            write_atomic(csv_dir / f"{stem}.csv.gz",
                         gzip.compress(buf.getvalue().encode(), compresslevel=6))
        else:
            n_options = len(data.get("options", []))

        # ---- meta index --------------------------------------------------
        meta_path = out / "meta.csv"
        new_meta = not meta_path.exists()
        with open(meta_path, "a", newline="") as f:
            mw = csv.DictWriter(f, fieldnames=META_FIELDS)
            if new_meta:
                mw.writeheader()
            mw.writerow({
                "capture_utc": capture_utc,
                "cboe_timestamp": cboe_ts,
                "index_last_trade_et": idx_last_trade,
                "seqno": seqno,
                "ticker": args.ticker,
                "current_price": price,
                "bid": data.get("bid"),
                "ask": data.get("ask"),
                "open": data.get("open"),
                "high": data.get("high"),
                "low": data.get("low"),
                "prev_day_close": data.get("prev_day_close"),
                "iv30": data.get("iv30"),
                "n_options": n_options,
                "raw_gz_bytes": len(raw_blob),
            })

        write_atomic(state_path, json.dumps({
            "index_last_trade_et": idx_last_trade,
            "seqno": seqno,
            "capture_utc": capture_utc,
            "current_price": price,
        }).encode())

        log(f"wrote {stem}  price={price}  options={n_options}  "
            f"index_last_print={idx_last_trade} ET  raw={len(raw_blob)/1e6:.2f}MB")
        return 0

    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())