"""Time to expiry, and where a strike sits relative to the forward.

Small module, two decisions that everything downstream inherits.

Settlement time is not the same for both roots
----------------------------------------------
SPX is AM-settled: its settlement value (SET) is built from Friday's *opening*
prints, so the contract is done at 09:30 ET on the expiry date. SPXW is
PM-settled and runs to 16:00 ET the same day. On a third Friday both roots list
the same expiry date, and treating them alike misprices the shorter one by 6.5
hours -- which on a one-day option is 27% of its remaining life.

This is the same collision that makes the `root` column load-bearing (see
`spxrnd.ingest.osi`), reappearing in the time dimension.

Day count
---------
ACT/365 on calendar time, the standard convention for index options and the one
CBOE's own VIX methodology uses. Business-day counts (ACT/252) exist and are
defensible for realised-vol work; mixing the two silently is how a term
structure develops kinks at holidays.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

DAYS_PER_YEAR = 365.0

AM_SETTLED_ROOTS = frozenset({"SPX"})
"""Roots that settle on the opening print rather than the close."""

AM_SETTLEMENT = time(9, 30)
PM_SETTLEMENT = time(16, 0)


def settlement_time(root: str) -> time:
    """When contracts of this root stop accruing time value, in US/Eastern."""
    return AM_SETTLEMENT if root in AM_SETTLED_ROOTS else PM_SETTLEMENT


def expiry_instant(expiry: date, root: str) -> datetime:
    """The exact moment a contract expires, timezone-aware.

    >>> expiry_instant(date(2026, 8, 21), "SPX").isoformat()
    '2026-08-21T09:30:00-04:00'
    >>> expiry_instant(date(2026, 8, 21), "SPXW").isoformat()
    '2026-08-21T16:00:00-04:00'
    """
    return datetime.combine(expiry, settlement_time(root), tzinfo=EASTERN)


def tenor_years(expiry: date, root: str, *, as_of: datetime) -> float:
    """Time to expiry in years, ACT/365.

    Args:
        expiry: the contract's expiry date.
        root: settlement convention, SPX vs SPXW.
        as_of: the capture instant. Must be timezone-aware -- comparing a naive
            clock against an Eastern settlement time is wrong by the UTC offset,
            and wrong in a way that looks plausible.

    Returns:
        Years to expiry. **May be zero or negative**: a capture taken after the
        settlement instant of a same-day expiry has a genuinely expired
        contract in it, and that is information, not an error. Callers decide
        what to do with it; `curate.quality` filters on it explicitly.
    """
    if as_of.tzinfo is None:
        raise ValueError("`as_of` must be timezone-aware")
    delta = expiry_instant(expiry, root) - as_of
    return delta.total_seconds() / (DAYS_PER_YEAR * 24 * 3600)


def log_moneyness(strike: float, forward: float) -> float:
    """log(K / F).

    Against the **forward**, not spot. The forward is what the option is
    actually struck against once carry is accounted for, so a smile plotted in
    log-forward-moneyness is centred at zero and comparable across expiries;
    plotted against spot it drifts with the rate and dividend term structure and
    the wings stop lining up.
    """
    if strike <= 0 or forward <= 0:
        raise ValueError(f"strike and forward must be positive: {strike}, {forward}")
    from math import log

    return log(strike / forward)


def is_otm(right: str, strike: float, forward: float) -> bool:
    """Is this contract out of the money against the forward?

    The selection rule that matters most in this pipeline. Both
    Breeden-Litzenberger and the BKM moment integrals are built on OTM options,
    for a reason that is worth stating plainly: an ITM option's price is
    dominated by intrinsic value, so the small part that carries volatility
    information is swamped by the large part that does not. Inverting an implied
    vol from it amplifies quoting noise without limit.

    CBOE evidently agrees. In the Aug 11 close, 1,156 of 5,837 quoted deep-ITM
    calls carry `iv == 0.0` -- the feed's "not computed" marker -- against 9 of
    6,157 elsewhere. Selecting OTM subsumes that sentinel entirely rather than
    special-casing it.
    """
    return strike >= forward if right == "call" else strike <= forward
