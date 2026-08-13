"""Parsing of OSI contract symbols.

The layout is `ROOT + YYMMDD + C|P + strike x 1000, zero-padded to 8`:

    SPXW260821C01400000
    ^^^^ ^^^^^^ ^^^^^^^^
    root expiry  strike x 1000 -> 1400.0

The root is the reason this module exists as its own unit rather than a regex
inlined at the call site. SPX (AM-settled monthlies) and SPXW (PM-settled
weeklies) list the *same strikes on the same expiry dates* -- 986 (right, strike)
pairs collide on the 2026-08-21 expiry alone. They are different contracts with
different quotes. Parsing the root and then discarding it, as the original
collector did, makes those rows indistinguishable and corrupts any put-call
parity fit that mixes them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

# Anchored, and the root is [A-Z]+ rather than a non-greedy `.+?`. Non-greedy
# matching would happily accept junk before the date and split it in surprising
# places; this rejects it outright.
OSI_RE = re.compile(r"^(?P<root>[A-Z]+)(?P<exp>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")

CALL = "call"
PUT = "put"


@dataclass(frozen=True, slots=True)
class OptionSymbol:
    """A parsed contract symbol.

    Frozen because a parsed symbol is a value, not a record to be edited: two
    symbols with the same fields are the same contract, and it stays hashable so
    it can key a dict of quotes.
    """

    root: str
    expiry: date
    right: str  # CALL or PUT
    strike: float
    raw: str

    @property
    def is_call(self) -> bool:
        return self.right == CALL

    @property
    def is_put(self) -> bool:
        return self.right == PUT


def parse(symbol: str) -> OptionSymbol:
    """Parse an OSI symbol, raising on anything malformed.

    Args:
        symbol: e.g. ``"SPXW260821C01400000"``. Embedded spaces are stripped;
            some feeds pad the root.

    Returns:
        The parsed :class:`OptionSymbol`.

    Raises:
        OsiParseError: the symbol does not match the OSI layout, or encodes a
            date that does not exist (e.g. month 13).
    """
    raise NotImplementedError


def try_parse(symbol: str) -> OptionSymbol | None:
    """Parse an OSI symbol, returning None instead of raising.

    For bulk paths that expect a small number of unparseable symbols and want to
    count them rather than abort. The caller is responsible for logging the
    count -- silently dropping contracts is how a chain quietly loses an expiry.
    """
    raise NotImplementedError
