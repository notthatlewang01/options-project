"""The forward price and discount factor per expiry, implied from the quotes.

Breeden-Litzenberger needs both, and CBOE's payload supplies neither. Rather
than bolt on a Treasury feed and a dividend series -- a second dataset to
collect, align and version, whose every timestamp mismatch injects error -- both
are read out of the option prices themselves.

Put-call parity, for European options:

    C(K) - P(K) = D * (F - K)

so regressing `C - P` on `K` across the strikes of one expiry gives

    slope     = -D          ->  D = -slope
    intercept =  D * F      ->  F = intercept / D

This is self-contained, internally consistent with the very quotes being
differentiated, and it is what CBOE's own VIX methodology does.

**The regression must never mix roots.** SPX and SPXW list the same strikes on
third-Friday expiries -- 986 collisions on 2026-08-21 alone -- at different
prices, because they settle at different times of day. Pairing an SPX call with
an SPXW put at the same strike produces a `C - P` that is not a parity relation
at all, and the fitted forward is silently wrong. This is the concrete reason
the `root` column exists.

Thin expiries are excluded, not extrapolated
--------------------------------------------
An expiry with too few usable pairs, or a poor fit, gets **no forward** and a
recorded reason. A fitted-but-wrong forward poisons every density built on it
while looking exactly like a good one; refusing to produce it is the safe
failure, and the reason is kept so exclusions are auditable rather than
mysterious.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import moneyness

MIN_PAIRS = 8
"""Fewest call/put pairs that can support a two-parameter fit with any
confidence. Two points determine a line exactly and tell you nothing about
whether the relation holds."""

MIN_R2 = 0.99
"""Put-call parity is an arbitrage identity, not a statistical tendency. On
real quotes it fits almost perfectly; anything below this means the pairs are
not what we think they are -- mixed roots, stale quotes, or a corrupted chain."""

MAX_ABS_RATE = 0.25
"""Implied rates outside +/-25% are not rates. A sanity bound that catches a
degenerate fit which happened to have a high R-squared."""


@dataclass(frozen=True, slots=True)
class ForwardFit:
    """One expiry's implied forward, or the reason there isn't one."""

    expiry: date
    root: str
    tenor_years: float

    forward: float | None = None
    discount: float | None = None
    rate: float | None = None
    """Continuously compounded, annualised: -ln(D) / T."""

    n_pairs: int = 0
    r2: float | None = None
    rmse: float | None = None
    excluded: str | None = None
    """Why this expiry has no forward. None means it does."""

    @property
    def usable(self) -> bool:
        return self.excluded is None


def _pairs(chain: pd.DataFrame) -> pd.DataFrame:
    """Join calls to puts at matching strikes within one (expiry, root)."""
    calls = chain.loc[chain["option_right"] == "call", ["strike", "mid"]]
    puts = chain.loc[chain["option_right"] == "put", ["strike", "mid"]]
    return calls.merge(puts, on="strike", suffixes=("_call", "_put")).sort_values(
        "strike"
    )


def fit_one(
    chain: pd.DataFrame,
    *,
    expiry: date,
    root: str,
    tenor_years: float,
    min_pairs: int = MIN_PAIRS,
    min_r2: float = MIN_R2,
) -> ForwardFit:
    """Fit the forward for a single (expiry, root).

    Args:
        chain: usable quotes for exactly one expiry and root, carrying `strike`,
            `option_right` and `mid`. Passing more than one root is a caller
            error and produces a meaningless fit -- see the module docstring.
        expiry, root, tenor_years: identity, carried onto the result.
        min_pairs, min_r2: exclusion thresholds.

    Returns:
        A :class:`ForwardFit`. Check `.usable`; `.excluded` says why not.
    """
    base = ForwardFit(expiry=expiry, root=root, tenor_years=tenor_years)

    if tenor_years <= 0:
        return _excluded(base, "expired")

    pairs = _pairs(chain)
    n = len(pairs)
    if n < min_pairs:
        return _excluded(base, f"only {n} call/put pairs, need {min_pairs}", n_pairs=n)

    k = pairs["strike"].to_numpy(dtype=float)
    y = (pairs["mid_call"] - pairs["mid_put"]).to_numpy(dtype=float)

    if np.ptp(k) == 0:
        return _excluded(base, "all pairs at one strike", n_pairs=n)

    slope, intercept = np.polyfit(k, y, 1)
    fitted = slope * k + intercept
    resid = y - fitted
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / n))

    discount = -slope
    if discount <= 0:
        return _excluded(
            base, f"non-positive discount factor {discount:.4f}", n_pairs=n, r2=r2
        )
    if discount > 1.5:
        return _excluded(
            base, f"implausible discount factor {discount:.4f}", n_pairs=n, r2=r2
        )
    if r2 < min_r2:
        return _excluded(
            base, f"parity fit R2 {r2:.4f} below {min_r2}", n_pairs=n, r2=r2
        )

    forward = intercept / discount
    if not np.isfinite(forward) or forward <= 0:
        return _excluded(base, f"non-positive forward {forward}", n_pairs=n, r2=r2)

    rate = -float(np.log(discount)) / tenor_years
    if abs(rate) > MAX_ABS_RATE:
        return _excluded(
            base,
            f"implied rate {rate:.1%} outside +/-{MAX_ABS_RATE:.0%}",
            n_pairs=n,
            r2=r2,
        )

    return ForwardFit(
        expiry=expiry,
        root=root,
        tenor_years=tenor_years,
        forward=float(forward),
        discount=float(discount),
        rate=rate,
        n_pairs=n,
        r2=r2,
        rmse=rmse,
    )


def _excluded(base: ForwardFit, why: str, *, n_pairs: int = 0, r2=None) -> ForwardFit:
    return ForwardFit(
        expiry=base.expiry,
        root=base.root,
        tenor_years=base.tenor_years,
        n_pairs=n_pairs,
        r2=r2,
        excluded=why,
    )


def fit_all(
    quotes: pd.DataFrame,
    *,
    as_of: datetime,
    min_pairs: int = MIN_PAIRS,
    min_r2: float = MIN_R2,
) -> list[ForwardFit]:
    """Fit a forward for every (expiry, root) in one capture.

    Grouped by root as well as expiry -- never by expiry alone. See the module
    docstring for what mixing them does.

    Args:
        quotes: one capture's quotes with a `mid` column.
        as_of: the capture instant, timezone-aware.

    Returns:
        One :class:`ForwardFit` per (expiry, root), including the excluded ones.
        Callers filter on `.usable`; the excluded entries are the audit trail.
    """
    fits: list[ForwardFit] = []
    for (expiry, root), group in quotes.groupby(["expiry", "root"], observed=True):
        fits.append(
            fit_one(
                group,
                expiry=expiry,
                root=root,
                tenor_years=moneyness.tenor_years(expiry, root, as_of=as_of),
                min_pairs=min_pairs,
                min_r2=min_r2,
            )
        )
    return sorted(fits, key=lambda f: (f.expiry, f.root))


def to_frame(fits: list[ForwardFit]) -> pd.DataFrame:
    """Fits as a DataFrame, for joining back onto the quotes."""
    return pd.DataFrame(
        [
            {
                "expiry": f.expiry,
                "root": f.root,
                "tenor_years": f.tenor_years,
                "forward": f.forward,
                "discount": f.discount,
                "rate": f.rate,
                "n_pairs": f.n_pairs,
                "r2": f.r2,
                "rmse": f.rmse,
                "excluded": f.excluded,
                "usable": f.usable,
            }
            for f in fits
        ]
    )
