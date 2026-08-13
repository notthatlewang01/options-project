"""Deciding whether a fetched payload carries new information.

The CDN always answers. After the cash index stops printing at 16:15 ET it keeps
serving the same body under a fresh top-level timestamp, so "the request
succeeded" says nothing about whether there is anything new in it. Five of the
thirty captures in the archive are post-close repeats admitted for exactly this
reason.

Two independent gates, both keyed on ``data["last_trade_time"]``:

**Duplicate** -- the index print has not moved since the last accepted capture.

**Stale** -- the index print is older than the tolerated age. During regular
hours it trails the wall clock by the advertised ~15 minutes; once it exceeds
that by a margin, the market is closed or it is a holiday. This is also why no
exchange calendar is needed: on a holiday the feed simply never freshens.

Why not ``seqno``
-----------------
It looks like the obvious dedup key and it is not safe as one. Measured across
the post-close freeze in the archive:

    20:37:03Z  print=16:14:59  seqno=16938026517   <- first, legitimate
    20:47:05Z  print=16:14:59  seqno=16938026517   <- repeat
    20:57:15Z  print=16:14:59  seqno=16947306680   <- seqno MOVED, print did not
    21:07:17Z  print=16:14:59  seqno=16947306680
    23:57:57Z  print=16:14:59  seqno=16947306680

A seqno-keyed gate admits the 20:57 capture as new data. It is not new data: the
index last printed at 16:14:59 and the chain is unchanged. `seqno` is recorded
on the snapshot for forensics and is never consulted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from .payload import Snapshot
from .state import CollectorState

CADENCE = timedelta(minutes=10)
"""The scheduled interval between capture attempts."""

FEED_DELAY = timedelta(minutes=15)
"""The endpoint's advertised delay."""

DEFAULT_MAX_STALE = FEED_DELAY + CADENCE + timedelta(minutes=2)
"""Tolerated age of the index's last print: 27 minutes.

Derived, not guessed. A print becomes visible ~15 minutes after it happens, and
the next capture attempt lands at most one cadence later, so **any genuinely new
print is at most FEED_DELAY + CADENCE = 25 minutes old when we first see it**.
Older than that and it is either a repeat -- which the duplicate gate rejects on
its own -- or the market is closed. Two minutes of headroom absorbs scheduler
jitter.

The inherited default was 22 minutes, and it was wrong. Measured over the whole
archive:

    in-session captures      15.12 - 15.92 min   (22 captures, tight cluster)
    the 16:14:59 close        22.07 min          <- REJECTED by a 22 min limit
    first post-close repeat   32.10 min
    off-hours / weekend      222 - 3024 min

A 22-minute limit discards the close capture by four seconds. That is the
settlement print -- the single most valuable snapshot of the session -- and it
would have been dropped silently, forever, with the skip logged as normal. The
close is structurally older than in-session captures because the index stops
printing at 16:14:59 while the collector keeps its cadence; it is not stale.

27 minutes accepts the close with 4.9 minutes of margin and rejects the first
post-close repeat with 5.1 minutes. Tunable per call; this is the default, not a
constant of nature.
"""


class Verdict(StrEnum):
    """Why a payload was accepted or rejected. Machine-readable on purpose --
    these are counted in logs and asserted on in tests."""

    ACCEPT = "accept"
    DUPLICATE_PRINT = "duplicate_print"
    STALE_FEED = "stale_feed"
    FORCED = "forced"


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    detail: str
    """Human-readable, with the numbers that drove the call. Goes straight to
    the log line, so a skipped capture explains itself without a debugger."""

    age: timedelta | None = None

    @property
    def accepted(self) -> bool:
        return self.verdict in (Verdict.ACCEPT, Verdict.FORCED)


def evaluate(
    snapshot: Snapshot,
    previous: CollectorState,
    *,
    now: datetime,
    max_stale: timedelta = DEFAULT_MAX_STALE,
    force: bool = False,
) -> Decision:
    """Decide whether `snapshot` should be written.

    Args:
        snapshot: the freshly parsed payload.
        previous: state from the last accepted capture. An empty state (first
            ever run) must not block acceptance.
        now: current time, timezone-aware. **Injected, never read from the
            clock inside this function** -- a gate whose behaviour depends on
            the wall clock cannot be tested deterministically, and this one
            decides what reaches the permanent archive.
        max_stale: tolerated age of the index's last print.
        force: bypass both gates, recording :attr:`Verdict.FORCED`. For
            deliberate re-capture; never the default.

    Returns:
        A :class:`Decision`. Callers branch on ``.accepted`` and log ``.detail``.
    """
    raise NotImplementedError
