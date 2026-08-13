"""Backfill, divergence detection, and SQL over the curated layer.

The catalog is what makes the archive answerable. Everything here treats
`data/raw` as authoritative and `data/curated` as a cache that can be deleted
and rebuilt at any time.

Divergence
----------
Seed regression 6 was a capture that reached the raw archive and never reached
the curated layer -- `_SPX_2026-08-11T23-57-57Z` had raw JSON and no CSV, and
nothing noticed. Because raw and curated share a filename stem, catching that is
a set difference rather than a query, and `divergence` is cheap enough to run on
every backfill.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from spxrnd.ingest.errors import IngestError

from . import writer
from .archive import Capture, scan


@dataclass(frozen=True, slots=True)
class Divergence:
    """How raw and curated differ, by capture stem."""

    missing_curated: list[str]
    """Archived but never converted. The seed-6 case."""

    orphan_curated: list[str]
    """Converted but no longer archived -- a raw file was deleted or renamed.
    Alarming: the raw archive is the source of truth and should only grow."""

    @property
    def aligned(self) -> bool:
        return not self.missing_curated and not self.orphan_curated


def divergence(raw_dir: Path, curated_dir: Path) -> Divergence:
    raw_stems = {c.stem for c in scan(raw_dir)}
    curated_stems = {
        p.name.removesuffix(".parquet")
        for p in Path(curated_dir).glob("*.parquet")
        if p.is_file()
    }
    return Divergence(
        missing_curated=sorted(raw_stems - curated_stems),
        orphan_curated=sorted(curated_stems - raw_stems),
    )


@dataclass
class BackfillReport:
    written: list[writer.WriteResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    """(stem, error). A capture that will not parse must not abort the run --
    the other 30 are still worth converting, and the failure is reported."""

    @property
    def total_rows(self) -> int:
        return sum(w.rows for w in self.written)

    @property
    def total_bytes(self) -> int:
        return sum(w.bytes_written for w in self.written)

    @property
    def total_unparsed(self) -> int:
        return sum(w.unparsed for w in self.written)


def backfill(
    raw_dir: Path,
    curated_dir: Path,
    *,
    overwrite: bool = True,
    captures: Iterable[Capture] | None = None,
) -> BackfillReport:
    """Rebuild the curated layer from the raw archive.

    Idempotent: output paths are a pure function of their inputs, so running
    twice produces the same files rather than duplicates.

    Args:
        raw_dir: the immutable archive.
        curated_dir: Parquet output, created if absent.
        overwrite: when False, captures already converted are skipped. Resuming
            an interrupted run.
        captures: convert only these, instead of everything found.

    Returns:
        A :class:`BackfillReport`. Check `.failed` -- it is never raised.
    """
    report = BackfillReport()
    for capture in captures if captures is not None else scan(raw_dir):
        try:
            result = writer.write_capture(capture, curated_dir, overwrite=overwrite)
        except (IngestError, OSError, ValueError) as exc:
            report.failed.append((capture.stem, f"{type(exc).__name__}: {exc}"))
            continue
        if result is None:
            report.skipped.append(capture.stem)
        else:
            report.written.append(result)
    return report


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

QUOTES_VIEW = "quotes"
CAPTURES_VIEW = "captures"


def connect(curated_dir: Path) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with views over the curated Parquet.

    In-memory on purpose. A persisted database would be a second thing that can
    drift from the archive; here the Parquet files are the only state, and the
    connection is free to recreate.

    Views:
        quotes    -- one row per option per capture, the full curated schema
        captures  -- one row per capture: time, spot, contract count

    Raises:
        FileNotFoundError: no Parquet found. Almost always means backfill has
            not run, so say that rather than surfacing a DuckDB glob error.
    """
    curated_dir = Path(curated_dir)
    files = sorted(curated_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no curated Parquet in {curated_dir} -- run `make backfill` first"
        )

    con = duckdb.connect()
    # Render every timestamp in UTC regardless of the host's timezone.
    # Without this the same query prints different wall-clock times on a
    # laptop in Chicago and a VM in Frankfurt, and a reader has no way to
    # tell which they are looking at.
    con.execute("SET TimeZone='UTC'")
    pattern = str(curated_dir / "*.parquet")
    con.execute(f"CREATE VIEW {QUOTES_VIEW} AS SELECT * FROM read_parquet('{pattern}')")
    con.execute(f"""
        CREATE VIEW {CAPTURES_VIEW} AS
        SELECT
            capture_utc,
            any_value(feed_timestamp)    AS feed_timestamp,
            any_value(index_last_trade)  AS index_last_trade,
            any_value(seqno)             AS seqno,
            any_value(underlying_ticker) AS underlying_ticker,
            any_value(underlying_price)  AS underlying_price,
            count(*)                     AS n_options,
            count(DISTINCT expiry)       AS n_expiries,
            count(DISTINCT root)         AS n_roots
        FROM {QUOTES_VIEW}
        GROUP BY capture_utc
        ORDER BY capture_utc
    """)
    return con
