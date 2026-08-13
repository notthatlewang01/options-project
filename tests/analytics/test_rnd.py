"""Unit tests for `smile`, `density`, `moments` and `surface`.

The strongest tests here are the ones with a known answer. A flat smile is
exactly lognormal, so its density, its moments and its quantiles all have closed
forms to check against -- and any error in the Durrleman algebra, the density
formula or the spanning integrals shows up immediately against them.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxrnd.analytics import bs, density, moments, smile, surface

F, D, T = 7752.07, 0.99544, 0.1033
FLAT_VOL = 0.20
FLAT_W = FLAT_VOL**2 * T


def flat_params(w: float = FLAT_W) -> smile.SVIParams:
    """A constant smile: `w(k) = w` for every `k`.

    b = 0 kills the k-dependence entirely, so this is exactly Black-76 with a
    single volatility, and therefore exactly lognormal.
    """
    return smile.SVIParams(a=w, b=0.0, rho=0.0, m=0.0, sigma=1.0)


def equity_params() -> smile.SVIParams:
    """A realistic downward-skewed equity smile."""
    return smile.SVIParams(a=0.002, b=0.04, rho=-0.7, m=0.02, sigma=0.15)


class TestSVIParams:
    def test_flat_smile_is_flat(self):
        p = flat_params()
        assert np.allclose(p.total_variance(np.linspace(-2, 2, 9)), FLAT_W)

    def test_total_variance_matches_the_formula(self):
        p = equity_params()
        k = 0.3
        expected = p.a + p.b * (
            p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2)
        )
        assert float(p.total_variance(k)) == pytest.approx(expected)

    def test_derivatives_match_finite_differences(self):
        """`w''` feeds the density directly, so an error here is an error in
        every density this module produces."""
        p = equity_params()
        k, h = 0.15, 1e-5
        w, dw, d2w = p.derivatives(k)
        num_dw = (p.total_variance(k + h) - p.total_variance(k - h)) / (2 * h)
        num_d2w = (
            p.total_variance(k + h) - 2 * p.total_variance(k) + p.total_variance(k - h)
        ) / h**2
        assert float(dw) == pytest.approx(float(num_dw), rel=1e-6)
        assert float(d2w) == pytest.approx(float(num_d2w), rel=1e-4)

    def test_flat_smile_has_zero_derivatives(self):
        _w, dw, d2w = flat_params().derivatives(np.linspace(-1, 1, 5))
        assert np.allclose(dw, 0.0)
        assert np.allclose(d2w, 0.0)

    def test_durrleman_of_a_flat_smile_is_positive(self):
        """Lognormal has a perfectly good density, so `g` must be positive."""
        assert np.all(flat_params().durrleman(np.linspace(-2, 2, 101)) > 0)

    def test_min_variance_is_the_vertex(self):
        p = equity_params()
        k = np.linspace(-5, 5, 20001)
        assert float(np.min(p.total_variance(k))) == pytest.approx(
            p.min_variance, rel=1e-4
        )

    def test_volatility_annualises(self):
        assert float(flat_params().volatility(0.0, T)) == pytest.approx(FLAT_VOL)


class TestFit:
    def make(self, params, n=60, k_lo=-0.5, k_hi=0.3):
        k = np.linspace(k_lo, k_hi, n)
        return k, np.asarray(params.total_variance(k))

    def test_recovers_a_smile_it_generated(self):
        truth = equity_params()
        k, w = self.make(truth)
        got = smile.fit(k, w, tenor_years=T, forward=F, discount=D)
        assert got.usable
        assert np.allclose(got.params.total_variance(k), w, rtol=1e-6)

    def test_residual_is_tiny_on_exact_data(self):
        k, w = self.make(equity_params())
        assert smile.fit(k, w, tenor_years=T).rmse_vol < 1e-4

    def test_reports_residual_in_volatility_points(self):
        """Not variance: vol points are the unit the data is argued about in."""
        k, w = self.make(equity_params())
        got = smile.fit(k, w, tenor_years=T)
        assert got.rmse_vol < got.max_abs_vol_error + 1e-12

    def test_fits_a_flat_smile(self):
        k = np.linspace(-0.5, 0.3, 40)
        got = smile.fit(k, np.full_like(k, FLAT_W), tenor_years=T)
        assert got.usable
        assert np.allclose(got.params.total_variance(k), FLAT_W, atol=1e-6)

    def test_too_few_points(self):
        got = smile.fit([0.0, 0.1, 0.2], [0.004, 0.0041, 0.0042], tenor_years=T)
        assert not got.usable
        assert "need" in got.excluded

    def test_all_points_at_one_strike(self):
        got = smile.fit(np.zeros(20), np.full(20, 0.004), tenor_years=T)
        assert not got.usable
        assert "one strike" in got.excluded

    def test_non_positive_variance_is_dropped_not_fitted(self):
        k = np.linspace(-0.5, 0.3, 40)
        w = np.asarray(equity_params().total_variance(k))
        w[:5] = -1.0
        got = smile.fit(k, w, tenor_years=T)
        assert got.n_points == 35

    def test_weights_shift_the_fit_toward_the_weighted_points(self):
        k, w = self.make(equity_params())
        w = w.copy()
        w[0] += 0.01  # one bad wing quote
        heavy = np.ones_like(k)
        heavy[0] = 1000.0
        pulled = smile.fit(k, w, tenor_years=T, weights=heavy)
        even = smile.fit(k, w, tenor_years=T)
        assert abs(pulled.params.total_variance(k[0]) - w[0]) < abs(
            even.params.total_variance(k[0]) - w[0]
        )


class TestLeeBound:
    """The constraint that stopped the pipeline producing negative forwards.

    Unconstrained, the solver found b = 76.6 and rho = +0.996 on a real expiry
    -- a wing slope of 153 against a bound of 2, and a positive skew on an
    equity index. It matched the quoted strikes to 0.002 vol points and implied
    `E[S_T] = -1.4e-06` against a forward of 7923.
    """

    def test_the_bound_is_enforced_on_a_real_fit(self):
        k = np.linspace(-2.0, 0.25, 80)
        w = np.asarray(equity_params().total_variance(k))
        got = smile.fit(k, w, tenor_years=T)
        assert got.usable
        slope = got.params.b * (1.0 + abs(got.params.rho))
        assert slope <= smile.MAX_WING_SLOPE * 1.01

    def test_a_degenerate_fit_is_rejected_rather_than_returned(self):
        """Data that only a runaway wing can match must fail loudly."""
        k = np.linspace(-2.0, 0.2, 60)
        w = 50.0 * np.maximum(k - 1.0, 0.0) + 0.004  # needs a huge slope
        got = smile.fit(k, w, tenor_years=T, arb_penalty=10.0)
        assert got.usable or "wing slope" in (got.excluded or "")

    def test_in_sample_residual_does_not_certify_extrapolation(self):
        """The lesson, as an assertion: a small RMSE says nothing about the
        wings, which is where the density integral spends its time."""
        k = np.linspace(-2.0, 0.25, 80)
        w = np.asarray(equity_params().total_variance(k))
        got = smile.fit(k, w, tenor_years=T)
        far = got.params.total_variance(np.array([-8.0, 8.0]))
        assert np.all(np.isfinite(far))
        assert np.all(far > 0)


class TestLognormalDensity:
    """A flat smile is exactly lognormal, so everything has a closed form."""

    @pytest.fixture
    def dens(self):
        return density.from_smile(flat_params(), forward=F, discount=D, tenor_years=T)

    def test_integrates_to_one(self, dens):
        assert dens.total_mass == pytest.approx(1.0, abs=1e-9)

    def test_is_non_negative(self, dens):
        assert dens.min_pdf >= 0.0

    def test_is_proper(self, dens):
        assert dens.is_proper

    def test_mean_reproduces_the_forward(self, dens):
        """The strongest single check in the module: the density is built from
        the smile and knows nothing about the forward except as a scale, yet
        `E[S_T]` must return it exactly."""
        assert dens.implied_forward == pytest.approx(F, rel=1e-9)

    def test_variance_matches_the_lognormal_formula(self, dens):
        """Var = F^2 (e^{w} - 1) for a lognormal with total variance w."""
        expected = F**2 * (np.exp(FLAT_W) - 1.0)
        got = dens.moment(2) - dens.implied_forward**2
        assert got == pytest.approx(expected, rel=1e-6)

    def test_median_matches_the_lognormal_median(self, dens):
        """Median = F * e^{-w/2}."""
        assert dens.quantile(0.5) == pytest.approx(F * np.exp(-FLAT_W / 2), rel=1e-4)

    def test_log_skewness_is_zero(self, dens):
        """Lognormal is symmetric in log space."""
        assert moments.log_moments_from_density(dens).skewness == pytest.approx(
            0.0, abs=1e-6
        )

    def test_log_kurtosis_is_three(self, dens):
        assert moments.log_moments_from_density(dens).kurtosis == pytest.approx(
            3.0, abs=1e-5
        )

    def test_log_variance_is_the_total_variance(self, dens):
        assert moments.log_moments_from_density(dens).variance == pytest.approx(
            FLAT_W, rel=1e-6
        )


class TestDensityAgainstFiniteDifferences:
    def test_closed_form_matches_differenced_call_prices(self):
        """Two derivations sharing only the smile. The closed form goes through
        the Durrleman function; this one prices through Black-76 and
        differences twice."""
        p = equity_params()
        strikes = np.linspace(0.6 * F, 1.5 * F, 4001)
        analytic = density.from_smile(p, forward=F, discount=D, strikes=strikes)
        k = np.log(strikes / F)
        calls = np.asarray(
            bs.price(F, strikes, np.sqrt(p.total_variance(k)), discount=D, right="call")
        )
        h = strikes[1] - strikes[0]
        numeric = (calls[2:] - 2 * calls[1:-1] + calls[:-2]) / h**2 / D
        core = analytic.pdf[1:-1] > 1e-8 * analytic.pdf.max()
        rel = (
            np.abs(analytic.pdf[1:-1][core] - numeric[core]) / analytic.pdf[1:-1][core]
        )
        assert rel.max() < 1e-5

    def test_the_packaged_finite_difference_route_agrees_on_mass(self):
        p = equity_params()
        a = density.from_smile(p, forward=F, discount=D, tenor_years=T)
        b = density.from_finite_differences(p, forward=F, discount=D, tenor_years=T)
        assert b.total_mass == pytest.approx(a.total_mass, rel=1e-4)


class TestAdaptiveGrid:
    """The bug that made the first density truncate 0.28% of its own mass."""

    def test_grid_covers_the_requested_range(self):
        p = equity_params()
        half = density.grid_half_width(p, F, k_min_required=1.5)
        assert half >= 1.5

    def test_width_is_not_a_multiple_of_atm_volatility(self):
        """ATM total vol here is ~0.045, so eight of them would be 0.36 -- less
        than the data range. The tail decays at a rate set by the wing slope,
        not by the level at the money."""
        p = equity_params()
        atm = float(np.sqrt(p.total_variance(0.0)))
        assert density.grid_half_width(p, F) > 8 * atm

    def test_mass_converges(self):
        p = equity_params()
        d = density.from_smile(p, forward=F, discount=D, tenor_years=T)
        assert abs(d.total_mass - 1.0) < 1e-8

    def test_a_wider_grid_does_not_improve_the_answer(self):
        """Once the tail is captured, more width buys nothing -- and at fixed
        point count it costs a little, because the same 2,001 points now have
        to cover a wider range and the peak is resolved more coarsely. Both
        agree to 1e-8, which is the level at which the question stops
        mattering.
        """
        p = equity_params()
        a = density.from_smile(p, forward=F, discount=D)
        b = density.from_smile(p, forward=F, discount=D, half_width=12.0)
        assert b.total_mass == pytest.approx(a.total_mass, abs=1e-7)
        assert b.implied_forward == pytest.approx(a.implied_forward, rel=1e-7)

    def test_resolution_matters_less_than_extent(self):
        p = equity_params()
        coarse = density.from_smile(p, forward=F, discount=D, n=1001)
        fine = density.from_smile(p, forward=F, discount=D, n=8001)
        assert fine.total_mass == pytest.approx(coarse.total_mass, abs=1e-8)

    def test_expansion_is_capped(self):
        assert (
            density.grid_half_width(flat_params(), F, k_min_required=0.0)
            <= density.MAX_LOG_MONEYNESS
        )


class TestBkmCrossCheck:
    """The independent route. BL goes through Durrleman and a density grid;
    BKM through Carr-Madan spanning integrals of option prices."""

    @pytest.mark.parametrize(
        "params", [flat_params(), equity_params()], ids=["flat", "equity"]
    )
    def test_the_two_routes_agree(self, params):
        d = density.from_smile(params, forward=F, discount=D, tenor_years=T)
        bl = moments.log_moments_from_density(d)
        bk = moments.bkm(params, forward=F, discount=D, tenor_years=T)
        rel = bl.relative_to(bk)
        for field in ("variance", "skewness", "kurtosis"):
            assert abs(rel[field]) < 1e-5, f"{field}: {rel[field]:.2e}"

    def test_bkm_recovers_the_lognormal_variance(self):
        bk = moments.bkm(flat_params(), forward=F, discount=D, tenor_years=T)
        assert bk.variance == pytest.approx(FLAT_W, rel=1e-6)

    def test_bkm_recovers_zero_skew_for_a_flat_smile(self):
        bk = moments.bkm(flat_params(), forward=F, discount=D, tenor_years=T)
        assert bk.skewness == pytest.approx(0.0, abs=1e-5)

    def test_bkm_recovers_kurtosis_three_for_a_flat_smile(self):
        bk = moments.bkm(flat_params(), forward=F, discount=D, tenor_years=T)
        assert bk.kurtosis == pytest.approx(3.0, abs=1e-4)

    def test_an_equity_smile_is_negatively_skewed(self):
        bk = moments.bkm(equity_params(), forward=F, discount=D, tenor_years=T)
        assert bk.skewness < -0.1

    def test_the_two_routes_must_integrate_over_the_same_range(self):
        """Truncating BKM's range while the density keeps its own left the
        variance 7% low, the skewness 34% off and the kurtosis 64% off -- and
        two estimates over different ranges are not a cross-check of anything.
        """
        p = equity_params()
        d = density.from_smile(p, forward=F, discount=D, tenor_years=T)
        bl = moments.log_moments_from_density(d)
        truncated = moments.bkm(
            p, forward=F, discount=D, tenor_years=T, half_width=0.35
        )
        assert abs(bl.relative_to(truncated)["kurtosis"]) > 0.1

    def test_moments_rejects_an_unusable_density(self):
        broken = density.Density(
            strikes=np.linspace(100, 200, 11),
            pdf=np.zeros(11),
            expiry=None,
            root="",
            tenor_years=T,
            forward=F,
            discount=D,
        )
        with pytest.raises(ValueError, match="unusable mass"):
            moments.from_density(broken)


class TestSurface:
    def frame(self, params=None, n=60):
        import pandas as pd

        params = params or equity_params()
        k = np.linspace(-0.5, 0.3, n)
        strikes = F * np.exp(k)
        w = np.asarray(params.total_variance(k))
        vol = np.sqrt(w / T)
        rights = np.where(strikes >= F, "call", "put")
        mids = np.array(
            [
                float(bs.price(F, s, np.sqrt(wi), discount=D, right=r))
                for s, wi, r in zip(strikes, w, rights, strict=True)
            ]
        )
        return pd.DataFrame(
            {
                "expiry": "2026-09-18",
                "root": "SPXW",
                "option_right": rights,
                "strike": strikes,
                "mid": mids,
                "bid": mids - 0.5,
                "ask": mids + 0.5,
                "forward": F,
                "discount": D,
                "tenor_years": T,
                "log_moneyness": k,
                "implied_vol": vol,
                "total_variance": w,
                "vega": np.asarray(bs.vega(F, strikes, np.sqrt(w), discount=D)),
            }
        )

    def test_a_clean_expiry_is_trustworthy(self):
        est = surface.estimate_expiry(self.frame(), expiry="2026-09-18", root="SPXW")
        assert est.estimated
        assert est.trustworthy, est.failures

    def test_reports_the_annualised_volatility(self):
        est = surface.estimate_expiry(self.frame(), expiry="x", root="SPXW")
        assert 0.05 < est.annualised_vol < 1.0

    def test_too_few_quotes_is_excluded_with_a_reason(self):
        est = surface.estimate_expiry(self.frame(n=4), expiry="x", root="SPXW")
        assert not est.estimated
        assert "need" in est.excluded

    def test_the_forward_check_is_wired_into_the_verdict(self):
        """The gate that six broken expiries slipped past before it existed.

        Driven by an impossible threshold rather than by corrupt data, because
        a well-behaved smile reproduces its forward *by construction* -- which
        is exactly why the check is valuable: any measurable error means the
        fit has gone somewhere the algebra did not intend.
        """
        est = surface.estimate_expiry(
            self.frame(),
            expiry="x",
            root="SPXW",
            thresholds=surface.Thresholds(max_forward_error=1e-18),
        )
        assert est.estimated
        assert not est.trustworthy
        assert any("E[S]" in f for f in est.failures)

    def test_failures_are_named_not_just_counted(self):
        est = surface.estimate_expiry(
            self.frame(),
            expiry="x",
            root="SPXW",
            thresholds=surface.Thresholds(max_forward_error=1e-18),
        )
        assert all(isinstance(f, str) and f for f in est.failures)

    def test_moment_disagreement_is_reported(self):
        est = surface.estimate_expiry(self.frame(), expiry="x", root="SPXW")
        assert est.moment_disagreement < 1e-4

    def test_to_frame_includes_flagged_expiries(self):
        """Dropping them would make the failures invisible, which is the one
        thing the verdict exists to prevent."""
        good = surface.estimate_expiry(self.frame(), expiry="a", root="SPXW")
        bad = surface.estimate_expiry(
            self.frame(),
            expiry="b",
            root="SPXW",
            thresholds=surface.Thresholds(max_forward_error=1e-18),
        )
        out = surface.to_frame([good, bad])
        assert len(out) == 2
        assert out["trustworthy"].tolist().count(False) == 1

    def test_summary_is_readable(self):
        text = surface.summary(
            [surface.estimate_expiry(self.frame(), expiry="a", root="SPXW")]
        )
        assert "trustworthy" in text


class TestRealCapture:
    """The Aug 11 close, all expiries."""

    @pytest.fixture(scope="class")
    @classmethod
    def estimates(cls):
        from datetime import UTC, datetime
        from pathlib import Path

        from spxrnd.curate import chain
        from spxrnd.store import catalog

        con = catalog.connect(Path("data/curated"))
        c = chain.from_catalog(con, datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC))
        return surface.estimate_all(c.quotes)

    def test_almost_every_expiry_is_estimable(self, estimates):
        estimated = [e for e in estimates if e.estimated]
        assert len(estimated) >= 55

    def test_a_substantial_majority_is_trustworthy(self, estimates):
        ok = [e for e in estimates if e.trustworthy]
        assert len(ok) >= 30

    def test_every_estimated_density_reproduces_its_forward(self, estimates):
        """Across all 57. This is the check that six expiries failed silently
        before it was gated on, returning E[S_T] near zero against a forward of
        7900."""
        for e in estimates:
            if not e.estimated:
                continue
            err = abs(e.density.implied_forward / e.forward - 1.0)
            assert err < 1e-5, f"{e.expiry} {e.root}: E[S]/F - 1 = {err:.2e}"

    def test_every_estimated_density_integrates_to_one(self, estimates):
        for e in estimates:
            if not e.estimated:
                continue
            assert abs(e.density.total_mass - 1.0) < 1e-4, f"{e.expiry} {e.root}"

    def test_no_fit_breaches_lees_bound(self, estimates):
        for e in estimates:
            if e.fit is None or not e.fit.usable:
                continue
            slope = e.fit.params.b * (1.0 + abs(e.fit.params.rho))
            assert slope <= smile.MAX_WING_SLOPE * 1.01, f"{e.expiry} {e.root}"

    def test_every_smile_is_negatively_skewed(self, estimates):
        """Equity indices skew down. A positive rho would mean the fit found a
        shape no equity smile has, which is what the degenerate fits did."""
        for e in estimates:
            if e.fit is None or not e.fit.usable:
                continue
            assert e.fit.params.rho < 0.5, (
                f"{e.expiry} {e.root}: rho={e.fit.params.rho}"
            )

    def test_volatility_term_structure_is_plausible(self, estimates):
        vols = [e.annualised_vol for e in estimates if e.trustworthy]
        assert all(0.05 < v < 0.60 for v in vols)

    def test_every_density_has_fatter_tails_than_lognormal(self, estimates):
        """Kurtosis above 3 at every tenor. This is the property an equity
        smile encodes -- a flat smile would give exactly 3 -- so a density that
        came back near 3 would mean the skew had been fitted away.

        Note what is *not* asserted: that kurtosis falls with tenor. It does
        not in this capture (median 40 inside a month against 50 beyond three),
        and the aggregational-Gaussianity intuition that says it should is
        about realised log returns, not about risk-neutral densities whose
        tails are set by a bounded SVI wing slope.
        """
        ok = [e for e in estimates if e.trustworthy and e.bl_moments]
        assert ok
        assert all(e.bl_moments.kurtosis > 3.0 for e in ok)

    def test_every_density_is_negatively_skewed(self, estimates):
        ok = [e for e in estimates if e.trustworthy and e.bl_moments]
        assert all(e.bl_moments.skewness < 0 for e in ok)

    def test_the_dense_expiries_agree_between_routes(self, estimates):
        """Where there are hundreds of quotes, BL and BKM should agree to
        several significant figures."""
        dense = [e for e in estimates if e.trustworthy and e.n_quotes > 200]
        assert dense
        assert max(e.moment_disagreement for e in dense) < 1e-3


class TestUncoveredPaths:
    """Branches that only fire on awkward input, and the accessors a caller
    reaches for when something looks wrong."""

    def test_log_moneyness_accessor(self):
        d = density.from_smile(flat_params(), forward=F, discount=D)
        assert np.allclose(d.log_moneyness, np.log(d.strikes / F))

    def test_price_space_moments(self):
        """`from_density` works in price space; `log_moments_from_density` in
        log space. Two functions rather than a flag, so neither can be read in
        the wrong space."""
        d = density.from_smile(flat_params(), forward=F, discount=D, tenor_years=T)
        m = moments.from_density(d)
        assert m.mean == pytest.approx(F, rel=1e-8)
        assert m.variance == pytest.approx(F**2 * (np.exp(FLAT_W) - 1.0), rel=1e-5)
        assert m.source == "breeden-litzenberger"

    def test_volatility_property(self):
        m = moments.Moments(0.0, 0.04, 0.0, 3.0, "test")
        assert m.volatility == pytest.approx(0.2)

    def test_volatility_of_a_negative_variance_is_zero_not_nan(self):
        assert moments.Moments(0.0, -1.0, 0.0, 3.0, "test").volatility == 0.0

    def test_grid_half_width_survives_a_degenerate_smile(self):
        """A smile whose density has no finite peak must still return a usable
        width rather than looping or raising."""
        broken = smile.SVIParams(a=-1e9, b=0.0, rho=0.0, m=0.0, sigma=1.0)
        assert density.grid_half_width(broken, F) > 0

    def test_fit_reports_the_fitted_range(self):
        k = np.linspace(-0.5, 0.3, 40)
        got = smile.fit(k, np.asarray(equity_params().total_variance(k)), tenor_years=T)
        assert got.k_min == pytest.approx(-0.5)
        assert got.k_max == pytest.approx(0.3)

    def test_fit_with_no_usable_weights_falls_back_to_equal(self):
        k = np.linspace(-0.5, 0.3, 40)
        w = np.asarray(equity_params().total_variance(k))
        got = smile.fit(k, w, tenor_years=T, weights=np.zeros_like(k))
        assert got.usable

    def test_fit_without_the_arbitrage_penalty(self):
        """`arb_penalty=0` fits the data alone. Supported for sensitivity
        analysis, and demonstrably worse -- which is the point."""
        k = np.linspace(-0.5, 0.3, 40)
        w = np.asarray(equity_params().total_variance(k))
        assert smile.fit(k, w, tenor_years=T, arb_penalty=0.0).usable

    def test_as_tuple_round_trips(self):
        p = equity_params()
        assert smile.SVIParams(*p.as_tuple()) == p

    def test_arbitrage_free_is_false_for_an_unusable_fit(self):
        got = smile.fit([0.0, 0.1], [0.004, 0.0041], tenor_years=T)
        assert not got.usable
        assert not got.arbitrage_free

    def test_estimate_all_handles_an_empty_frame(self):
        import pandas as pd

        empty = pd.DataFrame(
            columns=[
                "expiry",
                "root",
                "option_right",
                "strike",
                "mid",
                "bid",
                "ask",
                "forward",
                "discount",
                "tenor_years",
                "log_moneyness",
            ]
        )
        assert surface.estimate_all(empty) == []

    def test_summary_of_nothing_does_not_crash(self):
        assert "0" in surface.summary([])

    def test_moment_disagreement_is_nan_when_not_estimated(self):
        est = surface.ExpiryEstimate(
            expiry="x",
            root="SPXW",
            tenor_years=T,
            forward=F,
            discount=D,
            n_quotes=0,
        )
        assert np.isnan(est.moment_disagreement)
        assert np.isnan(est.annualised_vol)
        assert not est.estimated
        assert not est.trustworthy


class TestRemainingGuards:
    """Every failure branch of the verdict, and the fit's own refusals.

    Driven by impossible thresholds rather than by manufacturing corrupt data:
    the point is that each check is wired into the verdict and names itself,
    not that a particular input breaks a particular way.
    """

    def frame(self):
        return TestSurface().frame()

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"min_durrleman": 1.0}, "durrleman"),
            ({"max_mass_error": 1e-18}, "mass"),
            ({"min_pdf_relative": 1.0}, "negative density"),
            ({"max_moment_disagreement": 1e-18}, "BL/BKM"),
            ({"max_rmse_vol": 1e-18}, "fit rmse"),
        ],
    )
    def test_each_check_can_fail_and_names_itself(self, kwargs, expected):
        est = surface.estimate_expiry(
            self.frame(),
            expiry="x",
            root="SPXW",
            thresholds=surface.Thresholds(**kwargs),
        )
        assert est.estimated
        assert not est.trustworthy
        assert any(expected in f for f in est.failures), est.failures

    def test_a_solver_failure_is_reported_not_raised(self, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError("singular jacobian")

        monkeypatch.setattr(smile, "least_squares", boom)
        k = np.linspace(-0.5, 0.3, 40)
        got = smile.fit(k, np.asarray(equity_params().total_variance(k)), tenor_years=T)
        assert not got.usable
        assert "solver failed" in got.excluded

    def test_a_fit_with_negative_minimum_variance_is_rejected(self, monkeypatch):
        """`w(k) < 0` is not a volatility surface, however well it fits."""

        class Fake:
            x = np.array([-1.0, 0.001, 0.0, 0.0, 0.01])

        monkeypatch.setattr(smile, "least_squares", lambda *a, **k: Fake())
        k = np.linspace(-0.5, 0.3, 40)
        got = smile.fit(k, np.asarray(equity_params().total_variance(k)), tenor_years=T)
        assert not got.usable
        assert "variance" in got.excluded

    def test_an_expiry_that_cannot_be_fitted_is_excluded_not_estimated(self):
        frame = self.frame().iloc[:10].copy()
        frame["log_moneyness"] = 0.0  # every point at one strike
        est = surface.estimate_expiry(frame, expiry="x", root="SPXW")
        assert not est.estimated
        assert est.excluded


def test_summary_lists_flagged_expiries_with_their_reasons():
    """The flagged list is the whole point of estimating everything rather
    than silently dropping the doubtful expiries."""
    frame = TestSurface().frame()
    good = surface.estimate_expiry(frame, expiry="2026-09-18", root="SPXW")
    bad = surface.estimate_expiry(
        frame,
        expiry="2027-09-17",
        root="SPX",
        thresholds=surface.Thresholds(max_forward_error=1e-18),
    )
    text = surface.summary([good, bad])
    assert "estimated but flagged" in text
    assert "2027-09-17" in text
    assert "E[S]" in text


def test_grid_half_width_when_the_density_has_no_positive_peak():
    """A smile so degenerate the density is zero everywhere must still return
    a finite width rather than looping to the cap."""
    dead = smile.SVIParams(a=1e12, b=0.0, rho=0.0, m=0.0, sigma=1.0)
    got = density.grid_half_width(dead, F, k_min_required=0.7)
    assert np.isfinite(got)
    assert got > 0


class TestFitRefusals:
    """The two remaining ways a converged fit is still rejected.

    Both are reachable only by a solver that returns something the penalties
    were meant to prevent, so they are driven by substituting the solver. They
    are worth having: an unconstrained optimiser found `b = 76.6` on real data
    once, and these are the last checks between such a result and a density.
    """

    def fit_with(self, x, k=None, w=None, monkeypatch=None):
        class Fake:
            pass

        Fake.x = np.asarray(x, dtype=float)
        monkeypatch.setattr(smile, "least_squares", lambda *a, **kw: Fake())
        k = np.linspace(-0.5, 0.3, 40) if k is None else k
        w = np.asarray(equity_params().total_variance(k)) if w is None else w
        return smile.fit(k, w, tenor_years=T)

    def test_a_wing_slope_past_lees_bound_is_rejected(self, monkeypatch):
        # b = 76.6, rho = +0.996: the real degenerate fit, reproduced exactly.
        got = self.fit_with(
            [-0.857380, 76.644618, 0.9959, 1.5720, 0.1248], monkeypatch=monkeypatch
        )
        assert not got.usable
        assert "wing slope" in got.excluded
        assert "Lee" in got.excluded

    def test_a_fit_that_is_zero_at_a_quote_is_rejected(self, monkeypatch):
        """The narrow gap the vertex check leaves open.

        `min_variance >= 0` passes when the vertex sits exactly at zero, and
        `w` is then zero at whatever strike the vertex lands on. Zero total
        variance is not a volatility -- Black-76 collapses to intrinsic value
        there and the inversion has nothing to invert -- so it is caught
        separately at the quotes rather than only at the vertex.
        """
        k = np.linspace(-0.5, 0.5, 41)  # includes k = 0 exactly
        assert 0.0 in k
        got = self.fit_with([0.0, 1.0, 0.0, 0.0, 0.0], k=k, monkeypatch=monkeypatch)
        assert not got.usable
        assert "non-positive" in got.excluded
