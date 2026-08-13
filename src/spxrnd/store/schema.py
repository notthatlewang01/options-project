"""The canonical curated row: one row per option per capture.

Written explicitly rather than inferred. Inference reads types off whatever
happens to be in the first file, so a capture where every `bid_size` came back
whole becomes an int64 column, the next one is float64, and the two Parquet
files no longer union. Declaring the schema makes that a loud failure at write
time instead of a quiet one at read time.

Column groups
-------------
**Capture** columns repeat identically down every row of a snapshot. That looks
wasteful and is not: Parquet dictionary-encodes and run-length-encodes them to
near nothing, and having them on the row means any single row is
self-describing, with no join required to know when it was taken.

**Contract** columns are the parsed OSI symbol, `root` included -- see
`spxrnd.ingest.osi` for why that column is correctness rather than bookkeeping.

**Quote** columns are the feed's, carried through unchanged. Nothing is
filtered, coerced, or repaired here. `iv == 0.0` sentinels, zero bids, degenerate
deep-ITM vols and years-stale last trades all survive into the curated layer
exactly as served, because the store's job is to make the archive queryable, not
to decide what is usable. That decision belongs to `curate`, where it is visible
and reversible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pyarrow as pa

from spxrnd.ingest.osi import OptionSymbol
from spxrnd.ingest.payload import Snapshot

# Fields lifted straight off each option dict, with the Arrow type they take.
# float64 throughout for numerics: the feed mixes ints and floats in the same
# column across captures, and a size that is sometimes 0 and sometimes 0.0 must
# not change the column's type.
QUOTE_FIELDS: dict[str, pa.DataType] = {
    "bid": pa.float64(),
    "bid_size": pa.float64(),
    "ask": pa.float64(),
    "ask_size": pa.float64(),
    "last_trade_price": pa.float64(),
    "last_trade_time": pa.string(),
    "volume": pa.float64(),
    "open_interest": pa.float64(),
    "iv": pa.float64(),
    "theo": pa.float64(),
    "delta": pa.float64(),
    "gamma": pa.float64(),
    "theta": pa.float64(),
    "vega": pa.float64(),
    "rho": pa.float64(),
    "prev_day_close": pa.float64(),
    "change": pa.float64(),
    "percent_change": pa.float64(),
    "open": pa.float64(),
    "high": pa.float64(),
    "low": pa.float64(),
    "tick": pa.string(),
}

SCHEMA = pa.schema(
    [
        # --- capture ---
        # Millisecond resolution, not second: Parquet's minimum is ms and it
        # silently promotes anything finer-grained, so declaring "s" would
        # leave the declared schema disagreeing with what is on disk.
        pa.field("capture_utc", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("feed_timestamp", pa.string(), nullable=False),
        pa.field("index_last_trade", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("seqno", pa.int64(), nullable=False),
        pa.field("underlying_ticker", pa.string(), nullable=False),
        pa.field("underlying_price", pa.float64(), nullable=False),
        # --- contract ---
        pa.field("option_symbol", pa.string(), nullable=False),
        pa.field("root", pa.string(), nullable=False),
        # `right`, not `option_right`, would collide with SQL's RIGHT JOIN
        # keyword and need quoting in every query that touches it.
        pa.field("option_right", pa.string(), nullable=False),
        pa.field("expiry", pa.date32(), nullable=False),
        pa.field("strike", pa.float64(), nullable=False),
        # --- quote ---
        *[pa.field(name, dtype) for name, dtype in QUOTE_FIELDS.items()],
    ]
)

CAPTURE_COLUMNS = [
    "capture_utc",
    "feed_timestamp",
    "index_last_trade",
    "seqno",
    "underlying_ticker",
    "underlying_price",
]
CONTRACT_COLUMNS = ["option_symbol", "root", "option_right", "expiry", "strike"]


def _number(value: Any) -> float | None:
    """Coerce a feed number, mapping anything unusable to null.

    Null and 0.0 are different claims -- "not reported" versus "reported as
    zero" -- and the difference matters for exactly the fields where it is
    easiest to lose: an absent `iv` is unknown, while `iv == 0.0` is CBOE's
    "not computed" sentinel sitting on 2,160 contracts per capture. Collapsing
    them would erase the evidence that seed regression 11 exists.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def rows(
    snapshot: Snapshot,
    symbols: dict[str, OptionSymbol],
    *,
    captured_at: datetime,
) -> dict[str, list]:
    """Build column-oriented arrays for one snapshot.

    Args:
        snapshot: the validated payload.
        symbols: parsed symbol per contract, keyed by raw symbol string. Passed
            in rather than parsed here so the caller decides what to do with
            unparseable contracts and can report the count.
        captured_at: when we fetched it. Not on the payload -- the feed does not
            know when we asked -- so it comes from the collector at capture time
            or from the archive filename on backfill.

    Returns:
        A dict of column name -> list, ready for `pa.table`. Column-oriented
        because that is what Arrow wants; building 30,692 row dicts and
        transposing them costs several times as much for the same result.
    """
    n = len(symbols)

    columns: dict[str, list] = {
        "capture_utc": [captured_at] * n,
        "feed_timestamp": [snapshot.feed_timestamp] * n,
        "index_last_trade": [snapshot.index_last_trade] * n,
        "seqno": [snapshot.seqno] * n,
        "underlying_ticker": [snapshot.ticker] * n,
        "underlying_price": [snapshot.spot] * n,
        "option_symbol": [],
        "root": [],
        "option_right": [],
        "expiry": [],
        "strike": [],
    }
    for name in QUOTE_FIELDS:
        columns[name] = []

    for opt in snapshot.options:
        sym = symbols.get(opt.get("option", ""))
        if sym is None:
            continue
        columns["option_symbol"].append(sym.raw)
        columns["root"].append(sym.root)
        columns["option_right"].append(sym.right)
        columns["expiry"].append(sym.expiry)
        columns["strike"].append(sym.strike)
        for name, dtype in QUOTE_FIELDS.items():
            value = opt.get(name)
            columns[name].append(
                _text(value) if dtype == pa.string() else _number(value)
            )

    return columns
