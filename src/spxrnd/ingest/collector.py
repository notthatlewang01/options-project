"""One snapshot per invocation. Scheduling belongs to the operating system.

The original collector slept between captures inside the process, so a crash or
a reboot cost the rest of the session rather than a single sample. Here a
capture is one process: fetch, validate, gate, write, record, exit. launchd
restarts it on the next tick and a lost run costs exactly one sample.

Order matters. State is recorded **after** the archive write succeeds, so a
crash between the two re-captures a snapshot we already hold -- harmless, the
dedup gate drops it -- rather than advancing past a capture that was never
persisted, which would leave a permanent hole.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .freshness import DEFAULT_MAX_STALE, Decision
from .payload import Snapshot

DEFAULT_TICKER = "_SPX"


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """The outcome of one collector invocation.

    A skip is a success. A holiday, a weekend, and an off-hours tick all cost
    one HTTP request and write nothing; reporting that as failure would train
    whoever reads the logs to ignore them.
    """

    decision: Decision
    snapshot: Snapshot | None = None
    raw_path: Path | None = None
    """Where the capture was archived, or None if nothing was written."""

    @property
    def written(self) -> bool:
        return self.raw_path is not None


def collect_once(
    data_dir: Path,
    *,
    ticker: str = DEFAULT_TICKER,
    now: datetime | None = None,
    max_stale: timedelta = DEFAULT_MAX_STALE,
    retries: int = 3,
    force: bool = False,
    payload: dict | None = None,
) -> CaptureResult:
    """Take exactly one snapshot, or decide not to, then return.

    Args:
        data_dir: root of the data tree. The archive is written under
            ``data_dir/raw``; state and lock live at the root. Absolute --
            a scheduled job does not inherit a working directory.
        ticker: chain to request.
        now: current time, injected for testability. Defaults to the real clock.
        max_stale: tolerated age of the index's last print.
        retries: HTTP attempts.
        force: write even if the gates say the payload is stale or duplicated.
        payload: use this payload instead of fetching. For replaying a captured
            body through the full pipeline offline.

    Returns:
        A :class:`CaptureResult`. Check ``.written``; log ``.decision.detail``.

    Raises:
        FetchError: the endpoint could not be reached.
        PayloadError: the response was not a usable chain payload.
    """
    raise NotImplementedError


def archive_path(data_dir: Path, ticker: str, captured_at: datetime) -> Path:
    """Where a capture taken at `captured_at` is archived.

    ``<data_dir>/raw/<ticker>_<YYYY-MM-DDTHH-MM-SSZ>.json.gz``

    Colons are illegal in filenames on some filesystems and awkward on all of
    them, so the UTC timestamp uses hyphens. The existing 30 captures follow
    this convention uncompressed; the archive writer in `store` is what
    reconciles the two.
    """
    raise NotImplementedError
