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

import gzip
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import endpoint, freshness, health, state
from . import payload as payload_mod
from .errors import IngestError
from .freshness import DEFAULT_MAX_STALE, Decision, Verdict
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
    data_dir = Path(data_dir).expanduser().resolve()
    captured_at = now or datetime.now(UTC)
    health_path = data_dir / health.HEALTH_FILENAME
    state_path = data_dir / "state.json"

    with state.collector_lock(data_dir / ".collect.lock") as acquired:
        if not acquired:
            # Not an error: the scheduler overlapping two runs is expected, and
            # this is not the run that should record an outcome for the tick.
            return CaptureResult(
                decision=Decision(
                    verdict=Verdict.SKIPPED_LOCKED,
                    detail="another collector holds the lock; exiting",
                )
            )

        try:
            body = payload_mod.parse(
                payload
                if payload is not None
                else endpoint.fetch(ticker, retries=retries),
                ticker=ticker,
            )
        except IngestError as exc:
            health.record(
                health_path,
                now=captured_at,
                verdict=type(exc).__name__,
                error=str(exc),
            )
            raise

        decision = freshness.evaluate(
            body,
            state.read_state(state_path),
            now=captured_at,
            max_stale=max_stale,
            force=force,
        )

        if not decision.accepted:
            health.record(
                health_path, now=captured_at, verdict=decision.verdict, written=False
            )
            return CaptureResult(decision=decision, snapshot=body)

        raw_path = archive_path(data_dir, ticker, captured_at)
        state.write_atomic(
            raw_path,
            gzip.compress(
                json.dumps(body.raw, separators=(",", ":")).encode(),
                compresslevel=6,
                mtime=0,
            ),
        )

        # Only now. A crash between the write above and the write below
        # re-captures a snapshot we already hold -- harmless, the gate drops it.
        # The reverse order would advance past a capture that was never
        # persisted, leaving a hole nothing can fill.
        state.write_state(
            state_path,
            state.CollectorState(
                index_last_trade=body.index_last_trade.replace(tzinfo=None).isoformat(
                    timespec="seconds"
                ),
                seqno=body.seqno,
                capture_utc=_stamp(captured_at),
                spot=body.spot,
            ),
        )
        health.record(
            health_path, now=captured_at, verdict=decision.verdict, written=True
        )
        return CaptureResult(decision=decision, snapshot=body, raw_path=raw_path)


def archive_path(data_dir: Path, ticker: str, captured_at: datetime) -> Path:
    """Where a capture taken at `captured_at` is archived.

    ``<data_dir>/raw/<ticker>_<YYYY-MM-DDTHH-MM-SSZ>.json.gz``

    Colons are illegal in filenames on some filesystems and awkward on all of
    them, so the UTC timestamp uses hyphens. The existing 30 captures follow
    this convention uncompressed; the archive writer in `store` is what
    reconciles the two.
    """
    return Path(data_dir) / "raw" / f"{ticker}_{_stamp(captured_at)}.json.gz"


def _stamp(when: datetime) -> str:
    """UTC capture timestamp in filename form: ``2026-08-11T20-37-03Z``.

    Colons are illegal on some filesystems and awkward on all of them, so the
    time separators are hyphens. Always normalised to UTC first -- a filename
    whose meaning depends on the machine's timezone is a filename that sorts
    wrong the moment you move the collector.
    """
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
