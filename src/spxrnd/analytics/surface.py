"""Estimating every expiry in a capture, and saying which ones to trust.

Runs the full chain per (expiry, root) -- smile fit, density, both moment
routes -- and attaches a quality verdict. Every expiry is estimated; none is
silently dropped. What varies is how much the result is worth, and that is
reported rather than guessed at.

The verdict
-----------
Four checks, each catching a different failure:

  fit         RMSE of the smile against the quotes, in volatility points.
              Large means SVI's shape does not describe this expiry.

  arbitrage   Minimum of the Durrleman function. Negative means the fitted
              smile implies negative probability somewhere.

  mass        How far the density's integral is from 1. Sensitive to a
              truncated grid, a wrong discount, and an algebra error at once.

  agreement   Largest relative gap between the Breeden-Litzenberger and BKM
              moments. **The most informative of the four**, because the two
              routes share only the smile: BL goes through the Durrleman
              function and a density grid, BKM through Carr-Madan spanning
              integrals of option prices. There is no plausible mistake that
              moves both by the same amount in the same direction.

An expiry passing all four is not thereby *true* -- the quotes could be wrong --
but it is internally consistent by four independent measures, which is the
most a pipeline can honestly claim about itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import bs, density, moments, smile


@dataclass(frozen=True, slots=True)
class Thresholds:
    """What counts as trustworthy. Deliberately explicit and adjustable."""

    max_rmse_vol: float = 0.01
    """One volatility point of RMS residual."""

    min_durrleman: float = -1e-6
    max_mass_error: float = 1e-4
    max_moment_disagreement: float = 1e-2
    """1% between BL and BKM. Dense expiries reach 1e-7; this tolerates the
    thin long end without waving through a genuine inconsistency."""

    max_forward_error: float = 1e-4
    """|E[S_T]/F - 1|. The single strongest check available, and it was
    computed and displayed for a while before being gated on -- which let six
    expiries through reporting E[S_T] near zero against a forward of ~7900.
    The density and the parity forward are entirely independent computations;
    if they disagree, the density is wrong."""

    min_pdf_relative: float = -1e-5
    """Most negative the density may go, **as a fraction of its own peak**.

    Absolute would be meaningless: the peak density is ~5e-4 on a 7900 index
    and ~1e-2 on a short-dated one, so a fixed floor either waves through a
    real negative region at long tenors or flags rounding noise at short ones.
    An absolute -1e-9 flagged two expiries whose peak was 4.6e-4 -- a relative
    -2e-6, which is arithmetic, not negative probability."""

    min_quotes: int = 8


@dataclass
class ExpiryEstimate:
    """One expiry, estimated end to end."""

    expiry: object
    root: str
    tenor_years: float
    forward: float
    discount: float
    n_quotes: int

    fit: smile.SmileFit | None = None
    density: density.Density | None = None
    bl_moments: moments.Moments | None = None
    bkm_moments: moments.Moments | None = None

    failures: list[str] = field(default_factory=list)
    """Which checks did not pass, named. Empty means all four did."""

    excluded: str | None = None
    """Set when estimation could not be attempted at all."""

    @property
    def estimated(self) -> bool:
        return self.density is not None

    @property
    def trustworthy(self) -> bool:
        return self.estimated and not self.failures

    @property
    def moment_disagreement(self) -> float:
        if self.bl_moments is None or self.bkm_moments is None:
            return float("nan")
        rel = self.bl_moments.relative_to(self.bkm_moments)
        return max(abs(rel[f]) for f in ("variance", "skewness", "kurtosis"))

    @property
    def annualised_vol(self) -> float:
        """Volatility implied by the density's own variance, annualised.

        Not the at-the-money implied vol: this integrates the whole smile, so
        it sits above ATM for an equity index in the usual way, and it is the
        quantity comparable with VIX-style measures.
        """
        if self.bl_moments is None or not self.tenor_years > 0:
            return float("nan")
        return float(np.sqrt(max(self.bl_moments.variance, 0.0) / self.tenor_years))


def estimate_expiry(
    group: pd.DataFrame,
    *,
    expiry,
    root: str,
    thresholds: Thresholds | None = None,
    arb_penalty: float = 10.0,
) -> ExpiryEstimate:
    """Fit, build the density, compute both moment routes, and judge the result.

    Args:
        group: one expiry and root's curated quotes, carrying `log_moneyness`,
            `total_variance`, `tenor_years`, `forward`, `discount` and `vega`.
    """
    thresholds = thresholds or Thresholds()
    g = group.dropna(subset=["log_moneyness", "total_variance"])
    tenor = float(g["tenor_years"].iloc[0]) if len(g) else float("nan")
    forward = float(g["forward"].iloc[0]) if len(g) else float("nan")
    discount = float(g["discount"].iloc[0]) if len(g) else float("nan")

    base = ExpiryEstimate(
        expiry=expiry,
        root=root,
        tenor_years=tenor,
        forward=forward,
        discount=discount,
        n_quotes=len(g),
    )
    if len(g) < thresholds.min_quotes:
        base.excluded = f"only {len(g)} usable quotes, need {thresholds.min_quotes}"
        return base

    fit = smile.fit(
        g["log_moneyness"],
        g["total_variance"],
        expiry=expiry,
        root=root,
        tenor_years=tenor,
        forward=forward,
        discount=discount,
        weights=g.get("vega"),
        arb_penalty=arb_penalty,
    )
    base.fit = fit
    if not fit.usable:
        base.excluded = fit.excluded
        return base

    # The density grid must at least span the quoted strikes: a grid narrower
    # than the data would be extrapolating inward, which is absurd.
    k_required = float(
        max(abs(g["log_moneyness"].min()), abs(g["log_moneyness"].max()))
    )
    dens = density.from_smile(
        fit.params,
        forward=forward,
        discount=discount,
        tenor_years=tenor,
        expiry=expiry,
        root=root,
        k_min_required=k_required,
    )
    base.density = dens
    base.bl_moments = moments.log_moments_from_density(dens)
    base.bkm_moments = moments.bkm(
        fit.params,
        forward=forward,
        discount=discount,
        tenor_years=tenor,
        k_min_required=k_required,
    )

    if fit.rmse_vol > thresholds.max_rmse_vol:
        base.failures.append(f"fit rmse {fit.rmse_vol:.4f} vol pts")
    if fit.min_durrleman < thresholds.min_durrleman:
        base.failures.append(f"durrleman {fit.min_durrleman:.2e}")
    if abs(dens.total_mass - 1.0) > thresholds.max_mass_error:
        base.failures.append(f"mass {dens.total_mass:.6f}")
    peak = float(np.max(dens.pdf))
    if peak > 0 and dens.min_pdf / peak < thresholds.min_pdf_relative:
        base.failures.append(
            f"negative density {dens.min_pdf:.2e} ({dens.min_pdf / peak:.1e} of peak)"
        )
    forward_error = abs(dens.implied_forward / forward - 1.0)
    if not np.isfinite(forward_error) or forward_error > thresholds.max_forward_error:
        base.failures.append(f"E[S]/F - 1 = {forward_error:.2e}")
    if base.moment_disagreement > thresholds.max_moment_disagreement:
        base.failures.append(f"BL/BKM disagree {base.moment_disagreement:.2e}")
    return base


def prepare(quotes: pd.DataFrame) -> pd.DataFrame:
    """Add the implied vol, total variance and vega an estimate needs.

    Splitting calls from puts is not optional -- the inversion needs to know
    which payoff it is inverting, and a chain curated to OTM has both.
    """
    out = quotes.copy()
    iv = np.full(len(out), np.nan)
    for right in ("call", "put"):
        mask = (out["option_right"] == right).to_numpy()
        if not mask.any():
            continue
        sigma, _ok = bs.implied_vol(
            out.loc[mask, "mid"].to_numpy(),
            out.loc[mask, "forward"].to_numpy(),
            out.loc[mask, "strike"].to_numpy(),
            out.loc[mask, "tenor_years"].to_numpy(),
            discount=out.loc[mask, "discount"].to_numpy(),
            right=right,
        )
        iv[mask] = sigma
    out["implied_vol"] = iv
    out["total_variance"] = iv**2 * out["tenor_years"]
    # Vega as the fit weight: the wings quote far wider than the body, and
    # weighting equally lets the least reliable quotes set the smile's shape.
    out["vega"] = bs.vega(
        out["forward"],
        out["strike"],
        iv * np.sqrt(out["tenor_years"]),
        discount=out["discount"],
    )
    return out


def estimate_all(
    quotes: pd.DataFrame,
    *,
    thresholds: Thresholds | None = None,
    arb_penalty: float = 10.0,
) -> list[ExpiryEstimate]:
    """Estimate every (expiry, root) in a curated capture."""
    prepared = prepare(quotes) if "total_variance" not in quotes else quotes
    return [
        estimate_expiry(
            g,
            expiry=expiry,
            root=root,
            thresholds=thresholds,
            arb_penalty=arb_penalty,
        )
        for (expiry, root), g in prepared.groupby(["expiry", "root"], observed=True)
    ]


COLUMNS = [
    "expiry",
    "root",
    "tenor_years",
    "days",
    "forward",
    "n_quotes",
    "rmse_vol",
    "min_durrleman",
    "mass",
    "min_pdf",
    "forward_error",
    "annualised_vol",
    "skewness",
    "kurtosis",
    "bl_bkm_gap",
    "trustworthy",
    "failures",
    "excluded",
]


def to_frame(estimates: list[ExpiryEstimate]) -> pd.DataFrame:
    """The term structure as a table, including the failures.

    Returns an empty frame *with the columns present* when given nothing, so a
    caller filtering on `trustworthy` gets an empty result rather than a
    KeyError. An empty capture is a real state -- a holiday, a lost session --
    and it should not need special-casing at every call site.
    """
    if not estimates:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
    return pd.DataFrame(
        [
            {
                "expiry": e.expiry,
                "root": e.root,
                "tenor_years": e.tenor_years,
                "days": e.tenor_years * 365 if np.isfinite(e.tenor_years) else np.nan,
                "forward": e.forward,
                "n_quotes": e.n_quotes,
                "rmse_vol": e.fit.rmse_vol if e.fit else np.nan,
                "min_durrleman": e.fit.min_durrleman if e.fit else np.nan,
                "mass": e.density.total_mass if e.density else np.nan,
                "min_pdf": e.density.min_pdf if e.density else np.nan,
                "forward_error": (
                    e.density.implied_forward / e.forward - 1.0 if e.density else np.nan
                ),
                "annualised_vol": e.annualised_vol,
                "skewness": e.bl_moments.skewness if e.bl_moments else np.nan,
                "kurtosis": e.bl_moments.kurtosis if e.bl_moments else np.nan,
                "bl_bkm_gap": e.moment_disagreement,
                "trustworthy": e.trustworthy,
                "failures": "; ".join(e.failures),
                "excluded": e.excluded,
            }
            for e in estimates
        ]
    ).sort_values("tenor_years", ignore_index=True)


def summary(estimates: list[ExpiryEstimate]) -> str:
    frame = to_frame(estimates)
    ok = frame[frame["trustworthy"]]
    lines = [
        f"{len(frame)} (expiry, root) pairs   "
        f"{len(ok)} trustworthy   "
        f"{int(frame['excluded'].notna().sum())} could not be estimated",
    ]
    if len(ok):
        lines += [
            "",
            f"  fit rmse        max {ok['rmse_vol'].max():.5f} vol pts",
            f"  density mass    worst |1 - m| {np.abs(ok['mass'] - 1).max():.2e}",
            f"  E[S] vs forward worst {np.abs(ok['forward_error']).max():.2e}",
            f"  BL vs BKM       worst {ok['bl_bkm_gap'].max():.2e}",
            f"  annualised vol  {ok['annualised_vol'].min():.2%} "
            f"to {ok['annualised_vol'].max():.2%}",
        ]
    bad = frame[~frame["trustworthy"] & frame["excluded"].isna()]
    if len(bad):
        lines += ["", f"  {len(bad)} estimated but flagged:"]
        lines += [
            f"    {str(r.expiry)[:10]} {r.root:<5} n={r.n_quotes:<4} {r.failures}"
            for r in bad.itertuples()
        ]
    return "\n".join(lines)
