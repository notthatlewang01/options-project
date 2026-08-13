"""SVI smile fitting, constrained to produce a non-negative density.

The raw SVI parameterisation (Gatheral) writes **total variance** as a function
of log-moneyness `k = ln(K/F)`:

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

Five parameters, and each does one thing: `a` sets the level, `b` the overall
wing slope, `rho` the tilt (equity smiles are strongly negative), `m` the
horizontal shift, `sigma` how rounded the vertex is. It is a hyperbola with two
linear asymptotes, which is the right shape for two reasons -- Lee's moment
formula says total variance must grow at most linearly in `|k|`, and the wings
are where the data is thinnest and a parameterisation's own shape does most of
the work.

Why not a spline
----------------
A penalised spline follows the data more closely and imposes no functional form.
It also has nothing to stop it producing negative density between knots, and it
extrapolates badly past the last strike -- which is exactly where the tail mass
that a density integral needs actually lives. This chain runs from strike 200 to
20,000; the wings are not a detail.

Butterfly arbitrage, and why the fit checks it directly
-------------------------------------------------------
For a smile in total variance, the risk-neutral density is proportional to the
**Durrleman function**:

    g(k) = (1 - k*w'/(2w))^2 - (w'/4)^2 * (1/w + 1/4) + w''/2

`q(K) >= 0` if and only if `g(k) >= 0`. So this is not a diagnostic bolted on
afterwards -- it is the density's non-negativity written in the smile's own
variables, and it is what the fit is penalised on. A smile fitted without it
can match every quote beautifully and still imply negative probability between
them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

MIN_POINTS = 6
"""Fewest quotes that can support a five-parameter fit with a residual left
over. Five points fit exactly and say nothing about whether the shape holds."""

MAX_WING_SLOPE = 2.0
"""Lee's moment-formula bound on the asymptotic slope of total variance.

As `|k| -> infinity`, `w(k)/|k| <= 2`, or the underlying has no finite moment of
any order above one. For raw SVI the asymptotic slopes are `b(1+rho)` on the
right and `b(1-rho)` on the left, so the binding constraint is
`b*(1+|rho|) <= 2`.

This is not decoration. Left unconstrained on the 2027-04-16 expiry the solver
found `b = 76.6`, `rho = +0.996` -- a wing slope of 153, and a *positive* skew on
an equity index. It fit the quoted strikes to 0.002 vol points and was
nonsense everywhere else: the density decayed at a rate of 0.007 per log unit,
spread across strikes from 2.7 to 2.4e7, went negative, and returned
`E[S_T] = -1.4e-06` against a forward of 7923.

The lesson generalises. In-sample residual says nothing about extrapolation,
and the wings are exactly where a density integral spends its time."""


@dataclass(frozen=True, slots=True)
class SVIParams:
    """One expiry's fitted smile, in total variance."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k):
        """`w(k)`, total implied variance at log-moneyness `k`."""
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2)
        )

    def volatility(self, k, tenor: float):
        """Annualised implied volatility, for plotting and comparison."""
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / tenor)

    def derivatives(self, k):
        """`(w, w', w'')` at `k`, analytically.

        Analytic rather than finite-difference: `w''` enters the density
        directly, and differencing it twice from a numerical `w` loses about
        half the available precision for no benefit on a function this smooth.
        """
        k = np.asarray(k, dtype=float)
        x = k - self.m
        root = np.sqrt(x**2 + self.sigma**2)
        w = self.a + self.b * (self.rho * x + root)
        dw = self.b * (self.rho + x / root)
        d2w = self.b * self.sigma**2 / root**3
        return w, dw, d2w

    def durrleman(self, k):
        """`g(k)`, proportional to the density. Negative means negative mass."""
        w, dw, d2w = self.derivatives(k)
        w = np.maximum(w, 1e-12)
        return (
            (1.0 - k * dw / (2.0 * w)) ** 2
            - (dw**2 / 4.0) * (1.0 / w + 0.25)
            + d2w / 2.0
        )

    @property
    def min_variance(self) -> float:
        """The vertex of the hyperbola: the smallest total variance attainable."""
        return self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2)

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)


@dataclass(frozen=True, slots=True)
class SmileFit:
    """A fitted smile with everything needed to judge it."""

    params: SVIParams | None
    expiry: object
    root: str
    tenor_years: float
    forward: float
    discount: float

    n_points: int = 0
    rmse_vol: float = float("nan")
    """Root-mean-square residual in *volatility* points, not variance -- it is
    the unit the data is quoted and argued about in."""

    max_abs_vol_error: float = float("nan")
    min_durrleman: float = float("nan")
    """Most negative `g` on a dense grid. Below zero means the fitted smile
    implies negative density somewhere, even if it matches every quote."""

    k_min: float = float("nan")
    k_max: float = float("nan")
    excluded: str | None = None

    @property
    def usable(self) -> bool:
        return self.excluded is None

    @property
    def arbitrage_free(self) -> bool:
        return self.usable and self.min_durrleman >= -1e-8


def _initial_guess(k, w) -> np.ndarray:
    """A starting point derived from the data rather than hardcoded.

    SVI's objective is not convex and a fixed start fails on the short-dated
    expiries, where the smile is steep and the vertex sits well away from zero.
    """
    w = np.asarray(w, dtype=float)
    k = np.asarray(k, dtype=float)
    w_min = float(np.min(w))
    span = float(np.ptp(k)) or 1.0
    # Slope of a straight line through the wings, as a scale for b.
    slope = float(np.ptp(w) / span)
    return np.array(
        [
            max(w_min * 0.5, 1e-6),  # a: below the observed floor
            max(slope, 1e-3),  # b
            -0.7,  # rho: equity smiles skew hard negative
            float(k[int(np.argmin(w))]),  # m: at the observed vertex
            max(span * 0.1, 1e-3),  # sigma
        ]
    )


def fit(
    k,
    total_variance,
    *,
    expiry=None,
    root: str = "",
    tenor_years: float = float("nan"),
    forward: float = float("nan"),
    discount: float = float("nan"),
    weights=None,
    arb_penalty: float = 10.0,
    min_points: int = MIN_POINTS,
) -> SmileFit:
    """Fit raw SVI to one expiry's total variances.

    Args:
        k: log-moneyness, `ln(K/F)`.
        total_variance: `sigma_implied^2 * T` at each `k`.
        weights: relative importance per point. Pass vega, or 1/spread -- the
            wings are quoted far wider than the body and fitting them equally
            lets the least-liquid quotes set the shape.
        arb_penalty: weight on negative Durrleman values in the objective.
            Zero fits the data alone and will happily produce negative density.

    Returns:
        A :class:`SmileFit`. Check `.usable`, then `.arbitrage_free`.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    good = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[good], w[good]

    base = SmileFit(
        params=None,
        expiry=expiry,
        root=root,
        tenor_years=tenor_years,
        forward=forward,
        discount=discount,
        n_points=int(good.sum()),
    )
    if len(k) < min_points:
        return _excluded(base, f"only {len(k)} usable points, need {min_points}")
    if np.ptp(k) <= 0:
        return _excluded(base, "all points at one strike")

    weights = (
        np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)[good]
    )
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(k)
    weights = weights / weights.mean()

    # Grid for the arbitrage penalty. Generously wider than the data, because a
    # smile that is arbitrage-free only where there happen to be quotes is not
    # arbitrage-free where the density integral will evaluate it -- and the
    # density integrates far past the last strike. A narrow penalty grid was
    # what let a fit with negative density in the wings report a clean
    # Durrleman minimum.
    span = float(np.ptp(k)) or 1.0
    lo = min(k.min() - span, -3.0)
    hi = max(k.max() + span, 3.0)
    grid = np.linspace(lo, hi, 400)

    def residuals(theta):
        params = SVIParams(*theta)
        model = params.total_variance(k)
        fit_error = np.sqrt(weights) * (model - w)
        if arb_penalty <= 0:
            return fit_error
        g = params.durrleman(grid)
        breach = np.minimum(g, 0.0) * arb_penalty
        # Lee's bound on the wing slope, as a one-sided penalty. Enforced here
        # rather than as a box constraint because it couples b and rho.
        slope = params.b * (1.0 + abs(params.rho))
        wing = np.array([max(slope - MAX_WING_SLOPE, 0.0) * arb_penalty * 100.0])
        return np.concatenate([fit_error, breach, wing])

    try:
        result = least_squares(
            residuals,
            _initial_guess(k, w),
            bounds=(
                [-np.inf, 0.0, -0.999, -np.inf, 1e-6],
                [np.inf, np.inf, 0.999, np.inf, np.inf],
            ),
            max_nfev=8000,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _excluded(base, f"solver failed: {type(exc).__name__}: {exc}")

    params = SVIParams(*result.x)
    slope = params.b * (1.0 + abs(params.rho))
    if slope > MAX_WING_SLOPE * 1.01:
        return _excluded(
            base,
            f"wing slope b(1+|rho|) = {slope:.3f} exceeds Lee's bound "
            f"{MAX_WING_SLOPE}; the fit extrapolates to a degenerate density",
        )
    if params.min_variance < 0:
        return _excluded(base, f"fitted minimum variance {params.min_variance:.3e} < 0")

    model = params.total_variance(k)
    if np.any(model <= 0):
        return _excluded(base, "fitted total variance is non-positive at a quote")

    # Compare in volatility points: variance residuals are quadratic in the
    # quantity anyone actually reasons about.
    vol_model = np.sqrt(model / tenor_years) if tenor_years > 0 else np.sqrt(model)
    vol_data = np.sqrt(w / tenor_years) if tenor_years > 0 else np.sqrt(w)
    err = vol_model - vol_data

    return SmileFit(
        params=params,
        expiry=expiry,
        root=root,
        tenor_years=tenor_years,
        forward=forward,
        discount=discount,
        n_points=len(k),
        rmse_vol=float(np.sqrt(np.mean(err**2))),
        max_abs_vol_error=float(np.max(np.abs(err))),
        min_durrleman=float(np.min(params.durrleman(grid))),
        k_min=float(k.min()),
        k_max=float(k.max()),
    )


def _excluded(base: SmileFit, why: str) -> SmileFit:
    return SmileFit(
        params=None,
        expiry=base.expiry,
        root=base.root,
        tenor_years=base.tenor_years,
        forward=base.forward,
        discount=base.discount,
        n_points=base.n_points,
        excluded=why,
    )
