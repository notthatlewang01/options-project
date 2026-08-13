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
    SKIPPED_LOCKED = "skipped_locked"
    """Another collector held the lock. Not produced by :func:`evaluate` -- the
    collector records it when it never got as far as fetching. Lives here so
    every outcome the collector can report is enumerated in one place."""


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
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware")

    print_str = snapshot.index_last_trade.isoformat(timespec="seconds")
    # The state file stores the feed's own naive string. Compare in that form so
    # a state file written by an older version still matches.
    print_naive = snapshot.index_last_trade.replace(tzinfo=None).isoformat(
        timespec="seconds"
    )
    age = now - snapshot.index_last_trade

    if force:
        return Decision(
            verdict=Verdict.FORCED,
            detail=(
                f"forced: index last printed {print_str} "
                f"({age.total_seconds() / 60:.1f} min old), spot {snapshot.spot}"
            ),
            age=age,
        )

    # --- duplicate ------------------------------------------------------
    # Checked first: when we have state, an unmoved print is a more precise
    # diagnosis than "stale", and it is the case that actually recurs.
    if (
        previous.index_last_trade is not None
        and previous.index_last_trade == print_naive
    ):
        return Decision(
            verdict=Verdict.DUPLICATE_PRINT,
            detail=(
                f"no new data: index last print still {print_str}, spot "
                f"{snapshot.spot} (seqno {snapshot.seqno}"
                + (
                    f", advanced from {previous.seqno} -- seqno moves without new "
                    "prints and is not a dedup key"
                    if previous.seqno is not None and previous.seqno != snapshot.seqno
                    else ""
                )
                + ")"
            ),
            age=age,
        )

    # --- staleness ------------------------------------------------------
    # The backstop. Catches holidays, weekends, and off-hours ticks with no
    # exchange calendar, and covers the case where state was lost.
    if age > max_stale:
        return Decision(
            verdict=Verdict.STALE_FEED,
            detail=(
                f"feed is stale: index last printed {print_str} ET, "
                f"{age.total_seconds() / 60:.1f} min ago, over the "
                f"{max_stale.total_seconds() / 60:.0f} min limit -- market closed "
                "or holiday"
            ),
            age=age,
        )

    return Decision(
        verdict=Verdict.ACCEPT,
        detail=(
            f"new data: index last printed {print_str} ET "
            f"({age.total_seconds() / 60:.1f} min ago), spot {snapshot.spot}, "
            f"{snapshot.n_options} options"
        ),
        age=age,
    )
