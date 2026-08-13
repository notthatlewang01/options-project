"""Durable collector state, atomic writes, and the single-collector lock.

Everything here exists because the collector runs unattended on a schedule and
can be killed at any instant -- by a reboot, a laptop lid, or an overlapping run
the scheduler started because the previous one was slow.

Atomicity is not decoration. A partially written state file makes the *next* run
misjudge freshness, and a partially written archive entry is a corrupt capture
that cannot be re-fetched.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

DEFAULT_LOCK_TIMEOUT = timedelta(hours=1)
"""Age past which a lock file is treated as abandoned.

A collector that has held the lock for an hour is not slow, it is dead: a
capture is one HTTP request and a few hundred milliseconds of writing. Without
this, one SIGKILL at the wrong moment stops collection permanently and silently.
"""


@dataclass(frozen=True, slots=True)
class CollectorState:
    """What the previous accepted capture recorded.

    All fields optional so the first ever run has something valid to compare
    against instead of a special case threaded through the gate.
    """

    index_last_trade: str | None = None
    """The index print of the last ACCEPTED capture, as the feed's own naive
    string. Stored verbatim rather than normalized so a state file written by an
    older version still compares correctly."""

    seqno: int | None = None
    capture_utc: str | None = None
    spot: float | None = None

    @property
    def is_empty(self) -> bool:
        return self.index_last_trade is None


def read_state(path: Path) -> CollectorState:
    """Read collector state, returning an empty state if absent or corrupt.

    A missing state file is the first run. A *corrupt* one is also treated as
    empty rather than raising: the cost of re-capturing one snapshot is a
    duplicate row, while the cost of refusing to run is a permanent gap in a
    series that cannot be backfilled.
    """
    raise NotImplementedError


def write_state(path: Path, state: CollectorState) -> None:
    """Persist collector state atomically."""
    raise NotImplementedError


def write_atomic(path: Path, data: bytes) -> None:
    """Write `data` to `path` so that readers see all of it or none of it.

    Writes to a sibling temp file, flushes, fsyncs, then renames. `os.replace`
    is atomic within a filesystem, so a kill at any point leaves either the old
    file or the new one -- never a truncated body that a later run would parse as
    a valid but wrong capture.
    """
    raise NotImplementedError


@contextmanager
def collector_lock(
    path: Path, *, stale_after: timedelta = DEFAULT_LOCK_TIMEOUT
) -> Iterator[bool]:
    """Hold an exclusive collector lock for the duration of the block.

    Yields True if the lock was acquired and False if another collector holds
    it. Yielding rather than raising keeps "someone else is already collecting"
    a normal, non-error outcome: the scheduler overlapping two runs is expected,
    and exiting 0 keeps it out of the failure logs.

    A lock older than `stale_after` is reclaimed -- see
    :data:`DEFAULT_LOCK_TIMEOUT`.

    The lock is always released on the way out, including on exception.
    """
    raise NotImplementedError


def _pid_payload() -> bytes:
    """Contents written into the lock file: pid and start time.

    A bare lock file says "someone is running" but not who. When a stale lock
    turns up in three months, this is the difference between diagnosing it and
    guessing.
    """
    return f"{os.getpid()}\n".encode()
