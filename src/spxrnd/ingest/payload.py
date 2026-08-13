"""Validation of a raw CBOE delayed-quotes payload into a `Snapshot`.

The payload's shape, abbreviated:

    {
      "timestamp": "2026-08-11 20:36:41",     <-- when the CDN served this body
      "symbol": "_SPX",
      "data": {
        "current_price": 7728.2002,
        "last_trade_time": "2026-08-11T16:14:59",   <-- index's last print, ET
        "seqno": 16938026517,
        "iv30": 12.353, "bid": ..., "ask": ..., "open": ..., ...
        "options": [ {...} x 30692 ]
      }
    }

**The two timestamps are different things and neither substitutes for the
other.** `payload["timestamp"]` is when the CDN served the body; it advances
every few minutes forever, including on holidays and at 2am.
`data["last_trade_time"]` is when the cash index last printed; it is the only
field that reflects whether there is new information in the body.

The original collector read `payload["data"]["timestamp"]` -- a key that does not
exist -- so `.get()` returned `""` and the `cboe_timestamp` column was empty in
every row it ever wrote. That is why this module exists rather than callers
reaching into the dict themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A payload that has been validated and had its key fields lifted out.

    Holds `raw` so the archive writer never has to reconstruct the body: what we
    validate is exactly what we persist. Everything derived is recoverable from
    `raw`, which is why the raw archive -- not this object -- is the source of
    truth.
    """

    feed_timestamp: str
    """Top-level ``payload["timestamp"]``. NOT ``data["timestamp"]``, which does
    not exist. Records when the CDN served the body; useless for freshness."""

    index_last_trade: datetime
    """``data["last_trade_time"]``, made timezone-aware in US/Eastern. The only
    field that says whether the body carries new information."""

    seqno: int
    """Feed sequence number. Recorded, but never used for deduplication -- see
    :mod:`spxrnd.ingest.freshness` for why it is not safe for that."""

    ticker: str
    spot: float
    """``data["current_price"]``: the underlying index level, captured in the
    same body as the chain. This synchronicity is the whole point of the feed."""

    options: list[dict[str, Any]]
    raw: dict[str, Any]

    @property
    def n_options(self) -> int:
        return len(self.options)


def parse(payload: dict[str, Any], *, ticker: str) -> Snapshot:
    """Validate a payload and lift out the fields the pipeline depends on.

    Args:
        payload: the decoded JSON body, exactly as served.
        ticker: the ticker requested, e.g. ``"_SPX"``. Recorded on the snapshot;
            the payload's own ``symbol`` field is ``"^SPX"`` and does not
            round-trip to the request form.

    Returns:
        The validated :class:`Snapshot`.

    Raises:
        PayloadError: a required field is missing, or ``last_trade_time`` is not
            a parseable timestamp. Never defaults -- a snapshot that cannot be
            freshness-gated must not reach the archive.
    """
    raise NotImplementedError


def parse_eastern(timestamp: str) -> datetime:
    """Parse a naive US/Eastern timestamp from the feed into an aware datetime.

    The feed emits ``"2026-08-11T16:14:59"`` with no offset, and it is always
    US/Eastern. Attaching the zone at the boundary means nothing downstream has
    to remember that, and DST is handled by the zone rather than by an offset
    someone hardcoded in March.

    Raises:
        PayloadError: the string is empty or not an ISO-8601 timestamp.
    """
    raise NotImplementedError
