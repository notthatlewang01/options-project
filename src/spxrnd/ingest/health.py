"""A durable record of how collection is going.

A scheduled collector fails silently by default. The job stops, the log scrolls,
and you find out weeks later when an analysis has a hole in it -- by which point
the captures are permanently gone, because a delayed-quote endpoint serves the
present and never the past.

So every invocation records its outcome here, and `spxrnd status` reads it. One
small file, no external service, and enough state for a launchd job to alert on.

Three distinct clocks, because they fail in different ways and only one of them
means "something is broken":

    last_attempt   the collector ran at all -- if this is old, the SCHEDULER
                   is broken, not the collector
    last_ok        fetch and parse succeeded, whatever the gate then decided --
                   if this is old, the network or the feed is broken
    last_write     a capture actually reached the archive -- this is legitimately
                   old every weekend, so it alone proves nothing

A skip is not a failure. Weekends, holidays and off-hours ticks all end in a
skip, and counting those as failures would make the counter meaningless exactly
when you need to trust it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from .state import write_atomic

HEALTH_FILENAME = "health.json"


@dataclass(frozen=True, slots=True)
class Health:
    last_attempt: str | None = None
    """UTC ISO-8601. The collector process ran."""

    last_ok: str | None = None
    """UTC ISO-8601. Fetch and parse both succeeded, gate verdict irrelevant."""

    last_write: str | None = None
    """UTC ISO-8601. A capture was archived."""

    last_verdict: str | None = None
    consecutive_failures: int = 0
    """Errors in a row. Reset by any successful fetch+parse, including one that
    is then skipped -- a skipped capture still proves the pipeline works."""

    last_error: str | None = None
    total_writes: int = 0
    total_skips: int = 0
    total_failures: int = 0

    def stale_by(self, now: datetime, *, field: str = "last_ok") -> timedelta | None:
        """How long since `field`, or None if it has never happened."""
        value = getattr(self, field)
        if value is None:
            return None
        return now - datetime.fromisoformat(value)


# Declared types, keyed by field. Needed because most fields default to None,
# so the default's type says nothing about what a valid value looks like.
_FIELD_TYPES: dict[str, type] = {
    "last_attempt": str,
    "last_ok": str,
    "last_write": str,
    "last_verdict": str,
    "consecutive_failures": int,
    "last_error": str,
    "total_writes": int,
    "total_skips": int,
    "total_failures": int,
}


def read_health(path: Path) -> Health:
    """Read the health record, returning a blank one if absent or corrupt.

    Never raises. Health reporting must not be able to stop a capture -- the
    telemetry is worth strictly less than the data it reports on.
    """
    try:
        raw = json.loads(path.read_text())
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError, OSError):
        return Health()
    if not isinstance(raw, dict):
        return Health()

    blank = Health()
    fields = {}
    for name, expected in _FIELD_TYPES.items():
        default = getattr(blank, name)
        if name not in raw:
            fields[name] = default
            continue
        value = raw[name]
        if value is None:
            fields[name] = None if default is None else default
        elif isinstance(value, expected) and not (
            expected is int and isinstance(value, bool)
        ):
            # bool passes isinstance(_, int); a JSON `true` in a counter is
            # corruption, not a count of one.
            fields[name] = value
        else:
            fields[name] = default
    return Health(**fields)


def record(
    path: Path,
    *,
    now: datetime,
    verdict: str,
    written: bool = False,
    error: str | None = None,
) -> Health:
    """Fold one invocation's outcome into the health record and persist it.

    Args:
        path: the health file.
        now: timezone-aware; injected so tests are deterministic.
        verdict: the freshness verdict, or an error label if the run never got
            far enough to have one.
        written: a capture reached the archive.
        error: the failure, if the run raised.

    Returns:
        The updated record, already written to disk.
    """
    stamp = now.isoformat(timespec="seconds")
    current = read_health(path)

    if error is not None:
        updated = replace(
            current,
            last_attempt=stamp,
            last_verdict=verdict,
            last_error=error,
            consecutive_failures=current.consecutive_failures + 1,
            total_failures=current.total_failures + 1,
        )
    else:
        updated = replace(
            current,
            last_attempt=stamp,
            last_ok=stamp,
            last_verdict=verdict,
            last_error=None,
            consecutive_failures=0,
            last_write=stamp if written else current.last_write,
            total_writes=current.total_writes + (1 if written else 0),
            total_skips=current.total_skips + (0 if written else 1),
        )

    write_atomic(path, json.dumps(asdict(updated), indent=1, sort_keys=True).encode())
    return updated
