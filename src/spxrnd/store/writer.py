"""Turning archived captures into queryable Parquet.

One Parquet file per capture, named for the capture it came from. That is a
deliberate choice over a single partitioned dataset:

  * **Idempotent.** The output path is a pure function of the input, so
    rebuilding overwrites rather than duplicating and a half-finished backfill
    is repaired by running it again.
  * **Diffable.** Raw and curated are matched by name alone, which is what makes
    `catalog.divergence` a set difference rather than a query -- and seed
    regression 6 was precisely a capture that reached raw and never reached the
    curated layer, with nothing to notice.
  * **Cheap to discard.** Everything here is derived. Deleting `data/curated`
    and running `make backfill` must always be safe.

DuckDB globs the directory, so the many-files layout costs nothing at query
time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from spxrnd.ingest import osi
from spxrnd.ingest.payload import Snapshot
from spxrnd.ingest.payload import parse as parse_payload

from . import schema
from .archive import Capture, load

COMPRESSION = "zstd"
"""Better ratio than snappy at similar read speed, and Parquet's own framing
makes it seekable -- unlike the gzip used for the raw archive, which is written
once and read whole."""


@dataclass(frozen=True, slots=True)
class WriteResult:
    path: Path
    rows: int
    unparsed: int
    """Contracts whose symbol did not match the OSI layout, and so are absent
    from the Parquet. Zero for every capture in the archive; reported rather
    than assumed, because silently dropping contracts is how a chain quietly
    loses an expiry."""

    bytes_written: int


def curated_path(curated_dir: Path, capture: Capture) -> Path:
    """Where a capture's Parquet lives. Mirrors the raw archive's naming."""
    return Path(curated_dir) / f"{capture.stem}.parquet"


def build_table(snapshot: Snapshot, *, captured_at) -> tuple[pa.Table, int]:
    """Convert a validated snapshot into an Arrow table under the fixed schema.

    Returns the table and the number of contracts dropped as unparseable.
    """
    symbols = {}
    unparsed = 0
    for opt in snapshot.options:
        raw = opt.get("option", "")
        sym = osi.try_parse(raw)
        if sym is None:
            unparsed += 1
        else:
            symbols[raw] = sym

    columns = schema.rows(snapshot, symbols, captured_at=captured_at)
    # Passing the schema explicitly, rather than letting Arrow infer: inference
    # would type a column from whatever this capture happens to contain, and two
    # captures that inferred differently will not union at read time.
    return pa.table(columns, schema=schema.SCHEMA), unparsed


def write_capture(
    capture: Capture, curated_dir: Path, *, overwrite: bool = True
) -> WriteResult | None:
    """Convert one archived capture to Parquet.

    Args:
        capture: the archived capture, from `archive.scan`.
        curated_dir: output directory, created if absent.
        overwrite: when False, an existing output is left alone and None is
            returned. Makes an interrupted backfill cheap to resume.

    Returns:
        A :class:`WriteResult`, or None if the output existed and `overwrite`
        was False.
    """
    target = curated_path(curated_dir, capture)
    if not overwrite and target.exists():
        return None

    snapshot = parse_payload(load(capture.path), ticker=capture.ticker)
    table, unparsed = build_table(snapshot, captured_at=capture.captured_at)

    target.parent.mkdir(parents=True, exist_ok=True)
    # Temp-then-rename, same reasoning as the raw archive: a reader must never
    # see a half-written Parquet, and a killed backfill must not leave one.
    tmp = target.with_name(target.name + ".tmp")
    try:
        pq.write_table(table, tmp, compression=COMPRESSION)
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    return WriteResult(
        path=target,
        rows=table.num_rows,
        unparsed=unparsed,
        bytes_written=target.stat().st_size,
    )
