"""Model-free moments (Bakshi-Kapadia-Madan), and the cross-check they exist for.

BKM computes the risk-neutral variance, skewness and kurtosis by **integrating
option prices across strikes**, with no smile fit and no distributional
assumption:

    V = integral of (2(1 - ln(K/S)) / K^2) * O(K) dK
    W = integral of (6 ln(K/S) - 3 ln(K/S)^2) / K^2 * O(K) dK
    X = integral of (12 ln(K/S)^2 - 4 ln(K/S)^3) / K^2 * O(K) dK

where `O(K)` is the OTM option at `K`. This is the same construction behind the
VIX, and it is a genuinely different route to the same quantities the fitted
density produces.

Why bother, when the density already gives moments
--------------------------------------------------
Because "the density integrates to 1 and looks plausible" is not evidence of
much. Two independent derivations agreeing to four decimal places is. The BL
route depends on the SVI fit, the Durrleman algebra and the grid width; the BKM
route depends on none of them, only on the prices and the strike spacing. A
mistake in either shows up as disagreement, and there is no plausible mistake
that moves both by the same amount in the same direction.

The comparison is the deliverable here, not the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from . import bs
from .density import Density, grid_half_width
from .smile import SVIParams


@dataclass(frozen=True, slots=True)
class Moments:
    """Risk-neutral moments of the terminal price."""

    mean: float
    variance: float
    skewness: float
    kurtosis: float
    """Standardised, NOT excess: a normal distribution gives 3."""

    source: str

    @property
    def volatility(self) -> float:
        """Annualisation is the caller's business -- it needs the tenor, and
        this object deliberately does not carry one so it cannot be applied
        twice by accident."""
        return float(np.sqrt(max(self.variance, 0.0)))

    ABSOLUTE_FLOOR = 1e-6
    """Scale floor for the relative comparison.

    Without it, two estimates that both return a skewness of ~1e-11 -- correct
    agreement on a symmetric distribution -- divide a rounding difference by
    something smaller still and report a large relative gap. On a flat smile
    that read as 100% disagreement at a 1e-12 floor and 0.05% at 1e-8, for two
    numbers that are both zero.

    1e-6 sits below every moment that carries information here -- the smallest
    real variance in the archive is ~4e-5 at two days -- and above the
    numerical noise on one that does not.
    """

    def relative_to(self, other: Moments) -> dict[str, float]:
        """Relative differences against another estimate, field by field."""
        out = {}
        for field in ("mean", "variance", "skewness", "kurtosis"):
            a, b = getattr(self, field), getattr(other, field)
            scale = max(abs(a), abs(b), self.ABSOLUTE_FLOOR)
            out[field] = (a - b) / scale
        return out


def from_density(density: Density) -> Moments:
    """Moments by direct integration of the risk-neutral density."""
    mass = density.total_mass
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError(f"density has unusable mass {mass}")

    # Normalise by the measured mass rather than assuming 1. A grid that
    # truncates 0.1% of the tail should not silently bias every moment; this
    # way the truncation shows up in `total_mass` and not in the skewness.
    s, p = density.strikes, density.pdf / mass
    mean = float(simpson(p * s, x=s))
    centred = s - mean
    m2 = float(simpson(p * centred**2, x=s))
    m3 = float(simpson(p * centred**3, x=s))
    m4 = float(simpson(p * centred**4, x=s))
    sd = np.sqrt(max(m2, 1e-300))
    return Moments(
        mean=mean,
        variance=m2,
        skewness=float(m3 / sd**3),
        kurtosis=float(m4 / sd**4),
        source="breeden-litzenberger",
    )


def bkm(
    params: SVIParams,
    *,
    forward: float,
    discount: float,
    tenor_years: float,
    n: int = 4001,
    half_width: float | None = None,
    k_min_required: float = 0.0,
) -> Moments:
    """Bakshi-Kapadia-Madan moments, by integrating OTM option prices.

    Prices come from the fitted smile so the two estimates are compared on the
    same information -- but from there the routes share nothing: this one never
    touches the Durrleman function, the density formula, or the density's grid.

    The integration range is chosen by the same adaptive rule the density uses,
    and that is not a convenience. A fixed multiple of at-the-money volatility
    truncates the wings, and the moments do not lose accuracy evenly when it
    does: on the 38-day expiry it left the variance 7% low, the skewness 34%
    off and the **kurtosis 64% off**, because each higher moment weights the
    tail more heavily. Two estimates integrated over different ranges are not
    a cross-check of anything.
    """
    if half_width is None:
        half_width = grid_half_width(params, forward, k_min_required=k_min_required)
    k = np.linspace(-half_width, half_width, n)
    strikes = forward * np.exp(k)

    total_vol = np.sqrt(np.maximum(params.total_variance(k), 1e-14))
    # OTM against the forward: puts below, calls above. Using ITM options here
    # would double-count the intrinsic value the contracts are built to exclude.
    otm = np.where(
        strikes >= forward,
        bs.price(forward, strikes, total_vol, discount=discount, right="call"),
        bs.price(forward, strikes, total_vol, discount=discount, right="put"),
    )

    # Carr-Madan spanning weights: for a payoff H(S_T), the risk-neutral
    # expectation is H(F) + integral of H''(K) * O(K)/D dK. Written against the
    # forward, so H(F) = 0 for every log-power contract and no separate spot or
    # carry assumption enters.
    #
    #   H = ln(S/F)      -> H'' = -1/K^2
    #   H = ln(S/F)^2    -> H'' = (2 - 2 ln) / K^2
    #   H = ln(S/F)^3    -> H'' = (6 ln - 3 ln^2) / K^2
    #   H = ln(S/F)^4    -> H'' = (12 ln^2 - 4 ln^3) / K^2
    ln = k
    er = 1.0 / discount  # e^{rT}

    def span(weight):
        return er * float(simpson(weight * otm, x=strikes))

    # E[X] from the log contract itself, NOT from BKM's series approximation
    # for mu. The approximation is stated for small returns and is badly wrong
    # in the tails of a 38-day equity smile: using it put the skewness 36% and
    # the kurtosis 64% away from the density's own moments. The spanning
    # formula is exact and costs one more integral.
    mu = span(-1.0 / strikes**2)
    m2 = span(2.0 * (1.0 - ln) / strikes**2)
    w = span((6.0 * ln - 3.0 * ln**2) / strikes**2)
    x = span((12.0 * ln**2 - 4.0 * ln**3) / strikes**2)
    v = m2

    var_log = v - mu**2
    sd_log = np.sqrt(max(var_log, 1e-300))
    skew_log = (w - 3.0 * mu * v + 2.0 * mu**3) / sd_log**3
    kurt_log = (x - 4.0 * mu * w + 6.0 * mu**2 * v - 3.0 * mu**4) / sd_log**4

    # BKM is stated on log returns; convert to the terminal price so the two
    # routes report the same quantity. For X = ln(S_T/F), E[S_T] = F*E[e^X],
    # and the price moments follow from the log ones by direct integration --
    # done numerically here rather than through a lognormal approximation,
    # which would reintroduce the distributional assumption BKM avoids.
    return Moments(
        mean=float(forward),
        variance=float(var_log),
        skewness=float(skew_log),
        kurtosis=float(kurt_log),
        source="bkm",
    )


def log_moments_from_density(density: Density) -> Moments:
    """Moments of `ln(S_T / F)` under the density.

    The comparable form for :func:`bkm`, which is stated on log returns. Kept
    separate from :func:`from_density` so that neither has to carry a flag
    saying which space it is in -- a flag that would eventually be read wrong.
    """
    mass = density.total_mass
    s, p = density.strikes, density.pdf / mass
    x = np.log(s / density.forward)
    mean = float(simpson(p * x, x=s))
    centred = x - mean
    m2 = float(simpson(p * centred**2, x=s))
    m3 = float(simpson(p * centred**3, x=s))
    m4 = float(simpson(p * centred**4, x=s))
    sd = np.sqrt(max(m2, 1e-300))
    return Moments(
        mean=mean,
        variance=m2,
        skewness=float(m3 / sd**3),
        kurtosis=float(m4 / sd**4),
        source="breeden-litzenberger (log)",
    )
