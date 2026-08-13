"""The risk-neutral density, by Breeden-Litzenberger.

    q(K) = (1/D) * d2C/dK2

The second derivative of the call price in strike *is* the density. Which is
why every earlier stage cared so much about convexity: `d2C/dK2 >= 0` and
`q(K) >= 0` are the same statement, and a chain that breaches one breaches the
other.

Analytic, not finite-difference
-------------------------------
Given a smile in total variance `w(k)`, the density has a closed form:

    q(K) = g(k) / (K * sqrt(2*pi*w(k))) * exp(-d2^2 / 2)

where `g` is the Durrleman function from `smile` and `d2 = -k/sqrt(w) -
sqrt(w)/2`. Differencing a numerically-priced call twice would throw away about
half the available precision, and -- worse -- it would hide the structure: this
form makes `q >= 0 <=> g >= 0` immediate rather than something to be discovered
by looking at a plot.

The finite-difference route is still implemented, as an independent check on
the algebra rather than as the production path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from . import bs
from .smile import SVIParams

DEFAULT_GRID = 2001
"""Points across the strike grid. Odd, so Simpson's rule uses every interval.

Measured: the mass integral is identical to eight decimal places at 2,001 and
8,001 points. Resolution is not what limits accuracy here -- the grid's *extent*
is."""

TAIL_TOLERANCE = 1e-13
"""Density at the grid edge, relative to its peak, below which the truncated
tail is negligible."""

MAX_LOG_MONEYNESS = 8.0
"""Hard stop on grid half-width, `ln(K/F)`. `e^8` is a 3,000-fold move; beyond
this the expansion is chasing a numerical artefact, not probability mass."""


@dataclass(frozen=True, slots=True)
class Density:
    """One expiry's risk-neutral density, on a strike grid."""

    strikes: np.ndarray
    pdf: np.ndarray
    expiry: object
    root: str
    tenor_years: float
    forward: float
    discount: float

    @property
    def log_moneyness(self) -> np.ndarray:
        return np.log(self.strikes / self.forward)

    @property
    def total_mass(self) -> float:
        """Should integrate to 1. The single most informative diagnostic here:
        it is sensitive to a wrong discount factor, a truncated grid, and an
        algebra error in the density formula, all at once."""
        return float(simpson(self.pdf, x=self.strikes))

    @property
    def min_pdf(self) -> float:
        return float(np.min(self.pdf))

    @property
    def is_proper(self) -> bool:
        """Non-negative everywhere and integrating to 1 within tolerance."""
        return self.min_pdf >= -1e-10 and abs(self.total_mass - 1.0) < 1e-3

    def moment(self, power: int) -> float:
        """`E[S_T^power]` under the density."""
        return float(simpson(self.pdf * self.strikes**power, x=self.strikes))

    def cdf(self) -> np.ndarray:
        from scipy.integrate import cumulative_trapezoid

        return cumulative_trapezoid(self.pdf, self.strikes, initial=0.0)

    def quantile(self, p: float) -> float:
        """Strike below which probability `p` sits. Interpolated on the CDF."""
        c = self.cdf()
        return float(np.interp(p, c / c[-1], self.strikes))

    @property
    def implied_forward(self) -> float:
        """`E[S_T]` under the density.

        Must reproduce the forward the chain was curated with. It is an
        independent route to a number we already measured by put-call parity,
        so disagreement means the density is wrong -- not that the forward is.
        """
        return self.moment(1)


def pdf_at(params: SVIParams, k, forward: float):
    """The risk-neutral density at log-moneyness `k`, per unit strike.

    Factored out because the grid selection needs it before the grid exists.
    """
    k = np.asarray(k, dtype=float)
    w = np.maximum(params.total_variance(k), 1e-14)
    sqrt_w = np.sqrt(w)
    d2 = -k / sqrt_w - sqrt_w / 2.0
    strikes = forward * np.exp(k)
    return (
        params.durrleman(k)
        / (strikes * np.sqrt(2.0 * np.pi * w))
        * np.exp(-0.5 * d2**2)
    )


def grid_half_width(
    params: SVIParams,
    forward: float,
    *,
    k_min_required: float = 0.0,
    tail_tolerance: float = TAIL_TOLERANCE,
    max_k: float = MAX_LOG_MONEYNESS,
) -> float:
    """Half-width in log-moneyness at which the tail is negligible.

    Chosen adaptively rather than as a fixed multiple of ATM volatility, and
    the reason is the wings. SVI's total variance grows linearly in `|k|`, so
    implied volatility grows like `sqrt(|k|)` and the density decays like
    `exp(-|k| / 2b)` -- an exponential whose rate is set by the wing slope `b`,
    not by the level at the money.

    Those are wildly different scales. On the 2026-09-18 expiry the ATM total
    volatility is 0.0422, so a "generous" eight standard deviations spans only
    +/-0.34 in log-moneyness -- narrower than the quoted strikes, which reach
    -0.885. That grid truncated 0.28% of the mass and pulled `E[S_T]` 0.18%
    below the forward, which would have been read as a broken density rather
    than a truncated integral.

    Expansion stops when the density at the edge falls below `tail_tolerance`
    times its peak, so short and long tenors are both handled without tuning.
    """
    peak = float(np.max(pdf_at(params, np.linspace(-0.5, 0.5, 101), forward)))
    if not np.isfinite(peak) or peak <= 0:
        return max(k_min_required, 1.0)

    atm = float(params.total_variance(0.0))
    half = max(k_min_required, 4.0 * float(np.sqrt(max(atm, 0.0))), 0.25)
    while half < max_k:
        edge = float(np.max(np.abs(pdf_at(params, np.array([-half, half]), forward))))
        if edge < tail_tolerance * peak:
            break
        half *= 1.5
    return float(min(half, max_k))


def strike_grid(
    forward: float,
    params: SVIParams,
    *,
    n: int = DEFAULT_GRID,
    half_width: float | None = None,
    k_min_required: float = 0.0,
) -> np.ndarray:
    """A strike grid wide enough that the tails are not truncated."""
    if half_width is None:
        half_width = grid_half_width(params, forward, k_min_required=k_min_required)
    return forward * np.exp(np.linspace(-half_width, half_width, n))


def from_smile(
    params: SVIParams,
    *,
    forward: float,
    discount: float = 1.0,
    tenor_years: float = float("nan"),
    expiry=None,
    root: str = "",
    strikes=None,
    n: int = DEFAULT_GRID,
    half_width: float | None = None,
    k_min_required: float = 0.0,
) -> Density:
    """The density implied by a fitted smile, in closed form."""
    if strikes is None:
        strikes = strike_grid(
            forward,
            params,
            n=n,
            half_width=half_width,
            k_min_required=k_min_required,
        )
    strikes = np.asarray(strikes, dtype=float)
    pdf = pdf_at(params, np.log(strikes / forward), forward)
    return Density(
        strikes=strikes,
        pdf=pdf,
        expiry=expiry,
        root=root,
        tenor_years=tenor_years,
        forward=forward,
        discount=discount,
    )


def from_finite_differences(
    params: SVIParams,
    *,
    forward: float,
    discount: float = 1.0,
    tenor_years: float = float("nan"),
    expiry=None,
    root: str = "",
    n: int = DEFAULT_GRID,
    half_width: float | None = None,
    k_min_required: float = 0.0,
) -> Density:
    """The same density by differencing call prices twice.

    Kept as an **independent check on the algebra**, not as the production
    path. It prices the smile through Black-76 and differences the result, so it
    shares no code with the closed form beyond the smile itself; agreement
    between the two is evidence that neither derivation is wrong.
    """
    strikes = strike_grid(
        forward, params, n=n, half_width=half_width, k_min_required=k_min_required
    )
    k = np.log(strikes / forward)
    total_vol = np.sqrt(np.maximum(params.total_variance(k), 1e-14))
    calls = np.asarray(
        bs.price(forward, strikes, total_vol, discount=discount, right="call")
    )
    # Non-uniform grid, so use the general three-point second derivative.
    h_back = strikes[1:-1] - strikes[:-2]
    h_fwd = strikes[2:] - strikes[1:-1]
    second = (
        2.0
        * (h_back * calls[2:] - (h_back + h_fwd) * calls[1:-1] + h_fwd * calls[:-2])
        / (h_back * h_fwd * (h_back + h_fwd))
    )

    pdf = np.empty_like(calls)
    pdf[1:-1] = second / discount
    pdf[0], pdf[-1] = pdf[1], pdf[-2]
    return Density(
        strikes=strikes,
        pdf=pdf,
        expiry=expiry,
        root=root,
        tenor_years=tenor_years,
        forward=forward,
        discount=discount,
    )
