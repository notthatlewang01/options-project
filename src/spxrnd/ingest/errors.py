"""Exception types for the ingest component.

One base class so a caller can catch everything from this component with a
single `except IngestError`, and specific subclasses so it can distinguish a
malformed symbol from a dead endpoint when it cares.
"""

from __future__ import annotations


class IngestError(Exception):
    """Base for every error raised by `spxrnd.ingest`."""


class OsiParseError(IngestError):
    """A contract symbol did not match the OSI layout."""


class PayloadError(IngestError):
    """A payload was missing a required field or carried an unusable value.

    Raised rather than defaulted. A snapshot with no `last_trade_time` cannot be
    freshness-gated, and silently substituting a default would let stale data
    into the archive -- the exact failure this component exists to prevent.
    """


class FetchError(IngestError):
    """Every HTTP attempt failed."""


class LockError(IngestError):
    """The collector lock could not be acquired or released."""
