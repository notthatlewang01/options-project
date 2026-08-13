"""Unit tests for `spxrnd.analytics.bs`.

Black-76 has closed-form answers, so most of these are exact rather than
approximate: price a known volatility, invert it, and demand the original back
to machine precision. A pricing routine that cannot round-trip is not usable
for an inversion, and an inversion that cannot recover a price it just produced
is not usable for anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxrnd.analytics import bs

F, K, T, D = 7752.07, 7800.0, 0.1033, 0.99544
VOLS = [0.01, 0.05, 0.12, 0.20, 0.35, 0.80, 1.20, 3.0]
RIGHTS = ["call", "put"]


def total(sigma, tenor=T):
    return sigma * np.sqrt(tenor)


class TestPrice:
    @pytest.mark.parametrize("right", RIGHTS)
    def test_positive(self, right):
        assert bs.price(F, K, total(0.12), discount=D, right=right) > 0

    @pytest.mark.parametrize("right", RIGHTS)
    def test_monotone_in_volatility(self, right):
        """More uncertainty is worth more, for both rights. This is what makes
        the inversion well posed."""
        prices = [
            float(bs.price(F, K, total(s), discount=D, right=right)) for s in VOLS
        ]
        assert prices == sorted(prices)

    def test_zero_vol_gives_forward_intrinsic(self):
        assert bs.price(F, 7000.0, 0.0, discount=D, right="call") == pytest.approx(
            D * (F - 7000.0)
        )
        assert bs.price(F, 8500.0, 0.0, discount=D, right="put") == pytest.approx(
            D * (8500.0 - F)
        )

    def test_zero_vol_out_of_the_money_is_worthless(self):
        assert bs.price(F, 8500.0, 0.0, discount=D, right="call") == pytest.approx(0.0)
        assert bs.price(F, 7000.0, 0.0, discount=D, right="put") == pytest.approx(0.0)

    def test_at_the_forward_zero_vol_is_zero(self):
        """The `ln(F/K) == 0` branch, which needs its own limit."""
        assert bs.price(F, F, 0.0, discount=D, right="call") == pytest.approx(0.0)

    def test_approaches_the_upper_bound_at_huge_vol(self):
        """A call is never worth more than the discounted forward."""
        assert bs.price(F, K, total(50.0), discount=D, right="call") == pytest.approx(
            D * F, rel=1e-6
        )

    @pytest.mark.parametrize("sigma", VOLS)
    def test_put_call_parity_holds_in_the_model(self, sigma):
        """C - P = D*(F - K), the same identity `curate.forward` fits.

        If the pricer breached it, the forwards feeding this module and the
        prices coming out would be inconsistent with each other.
        """
        c = bs.price(F, K, total(sigma), discount=D, right="call")
        p = bs.price(F, K, total(sigma), discount=D, right="put")
        assert float(c - p) == pytest.approx(D * (F - K), abs=1e-9)

    def test_discount_scales_linearly(self):
        a = bs.price(F, K, total(0.2), discount=1.0, right="call")
        b = bs.price(F, K, total(0.2), discount=0.5, right="call")
        assert float(b) == pytest.approx(0.5 * float(a))

    def test_vectorises_over_strikes(self):
        strikes = np.array([7000.0, 7500.0, 7752.07, 8000.0, 8500.0])
        got = bs.price(F, strikes, total(0.2), discount=D, right="call")
        assert got.shape == strikes.shape
        assert np.all(np.diff(got) < 0), "calls fall as the strike rises"

    def test_rejects_an_unknown_right(self):
        with pytest.raises(ValueError, match="right must be"):
            bs.price(F, K, total(0.2), right="straddle")


class TestVega:
    def test_positive(self):
        assert bs.vega(F, K, total(0.2), discount=D) > 0

    def test_identical_for_calls_and_puts(self):
        """They differ by a forward, which carries no volatility exposure."""
        eps = 1e-6
        v = total(0.2)
        for right in RIGHTS:
            numeric = float(
                (
                    bs.price(F, K, v + eps, discount=D, right=right)
                    - bs.price(F, K, v - eps, discount=D, right=right)
                )
                / (2 * eps)
            )
            assert numeric == pytest.approx(
                float(bs.vega(F, K, v, discount=D)), rel=1e-5
            )

    def test_matches_a_numerical_derivative(self):
        eps = 1e-7
        v = total(0.2)
        numeric = float(
            (
                bs.price(F, K, v + eps, discount=D, right="call")
                - bs.price(F, K, v - eps, discount=D, right="call")
            )
            / (2 * eps)
        )
        assert numeric == pytest.approx(float(bs.vega(F, K, v, discount=D)), rel=1e-6)

    def test_peaks_near_the_forward(self):
        strikes = np.array([5000.0, 7000.0, 7752.07, 8500.0, 12000.0])
        vegas = bs.vega(F, strikes, total(0.2), discount=D)
        assert int(np.argmax(vegas)) == 2

    def test_tenor_converts_to_conventional_vega(self):
        v = total(0.2)
        assert float(bs.vega(F, K, v, discount=D, tenor=T)) == pytest.approx(
            float(bs.vega(F, K, v, discount=D)) * np.sqrt(T)
        )

    def test_collapses_in_the_far_wings(self):
        """Why Newton alone is not enough: the step is undefined here."""
        assert float(bs.vega(F, 100_000.0, total(0.1), discount=D)) < 1e-100


class TestPriceBounds:
    def test_call_bounds(self):
        lo, hi = bs.price_bounds(F, 7000.0, discount=D, right="call")
        assert float(lo) == pytest.approx(D * (F - 7000.0))
        assert float(hi) == pytest.approx(D * F)

    def test_put_bounds(self):
        lo, hi = bs.price_bounds(F, 8500.0, discount=D, right="put")
        assert float(lo) == pytest.approx(D * (8500.0 - F))
        assert float(hi) == pytest.approx(D * 8500.0)

    def test_out_of_the_money_lower_bound_is_zero(self):
        lo, _ = bs.price_bounds(F, 9000.0, discount=D, right="call")
        assert float(lo) == 0.0

    @pytest.mark.parametrize("right", RIGHTS)
    @pytest.mark.parametrize("sigma", VOLS)
    def test_every_model_price_sits_inside_its_bounds(self, right, sigma):
        lo, hi = bs.price_bounds(F, K, discount=D, right=right)
        p = float(bs.price(F, K, total(sigma), discount=D, right=right))
        assert float(lo) <= p <= float(hi)

    def test_rejects_an_unknown_right(self):
        with pytest.raises(ValueError, match="right must be"):
            bs.price_bounds(F, K, right="straddle")


class TestInversion:
    @pytest.mark.parametrize("right", RIGHTS)
    @pytest.mark.parametrize("sigma", VOLS)
    def test_round_trips_to_machine_precision(self, right, sigma):
        p = bs.price(F, K, total(sigma), discount=D, right=right)
        got, ok = bs.implied_vol(p, F, K, T, discount=D, right=right)
        assert bool(ok)
        assert float(got) == pytest.approx(sigma, rel=1e-9)

    @pytest.mark.parametrize("strike", [5000.0, 7000.0, 7752.07, 8500.0, 11000.0])
    def test_round_trips_across_the_strike_range(self, strike):
        p = bs.price(F, strike, total(0.25), discount=D, right="call")
        got, ok = bs.implied_vol(p, F, strike, T, discount=D, right="call")
        assert bool(ok)
        assert float(got) == pytest.approx(0.25, rel=1e-7)

    @pytest.mark.parametrize("tenor", [0.003, 0.05, 0.5, 2.0, 5.4])
    def test_round_trips_across_the_tenor_range(self, tenor):
        p = bs.price(F, K, total(0.25, tenor), discount=D, right="call")
        got, ok = bs.implied_vol(p, F, K, tenor, discount=D, right="call")
        assert bool(ok)
        assert float(got) == pytest.approx(0.25, rel=1e-7)

    def test_total_vol_and_annualised_vol_agree(self):
        p = bs.price(F, K, total(0.3), discount=D, right="call")
        v, _ = bs.implied_total_vol(p, F, K, discount=D, right="call")
        s, _ = bs.implied_vol(p, F, K, T, discount=D, right="call")
        assert float(v) == pytest.approx(float(s) * np.sqrt(T), rel=1e-12)

    def test_vectorised_inversion(self):
        strikes = np.array([7000.0, 7400.0, 7752.07, 8100.0, 8500.0])
        prices = bs.price(F, strikes, total(0.22), discount=D, right="call")
        got, ok = bs.implied_vol(prices, F, strikes, T, discount=D, right="call")
        assert ok.all()
        assert np.allclose(got, 0.22, rtol=1e-7)

    def test_shape_is_preserved(self):
        strikes = np.array([[7000.0, 7500.0], [8000.0, 8500.0]])
        prices = bs.price(F, strikes, total(0.2), discount=D, right="call")
        got, ok = bs.implied_vol(prices, F, strikes, T, discount=D, right="call")
        assert got.shape == strikes.shape == ok.shape


class TestInversionFailures:
    """A chain of 13,414 quotes contains prices no volatility can produce.
    Every one must return NaN rather than raise -- aborting the run for one
    contract would be the wrong trade."""

    def test_price_below_intrinsic(self):
        lo, _ = bs.price_bounds(F, 7000.0, discount=D, right="call")
        got, ok = bs.implied_vol(float(lo) - 1.0, F, 7000.0, T, discount=D)
        assert not bool(ok)
        assert np.isnan(got)

    def test_price_above_the_upper_bound(self):
        _, hi = bs.price_bounds(F, K, discount=D, right="call")
        got, ok = bs.implied_vol(float(hi) + 1.0, F, K, T, discount=D)
        assert not bool(ok)
        assert np.isnan(got)

    def test_exactly_at_intrinsic_is_rejected(self):
        """The zero-vol boundary. Zero is not a number a smile fit can use."""
        lo, _ = bs.price_bounds(F, 7000.0, discount=D, right="call")
        _, ok = bs.implied_vol(float(lo), F, 7000.0, T, discount=D)
        assert not bool(ok)

    def test_negative_price(self):
        _, ok = bs.implied_vol(-1.0, F, K, T, discount=D)
        assert not bool(ok)

    def test_zero_tenor(self):
        p = bs.price(F, K, total(0.2), discount=D, right="call")
        _, ok = bs.implied_vol(p, F, K, 0.0, discount=D)
        assert not bool(ok)

    def test_negative_tenor(self):
        p = bs.price(F, K, total(0.2), discount=D, right="call")
        _, ok = bs.implied_vol(p, F, K, -0.1, discount=D)
        assert not bool(ok)

    def test_a_bad_price_does_not_poison_its_neighbours(self):
        """Vectorised failure must be local."""
        strikes = np.array([7000.0, 7400.0, 7752.07])
        prices = np.asarray(bs.price(F, strikes, total(0.2), discount=D, right="call"))
        prices[1] = 1e9  # impossible
        got, ok = bs.implied_vol(prices, F, strikes, T, discount=D, right="call")
        assert list(ok) == [True, False, True]
        assert np.allclose(got[[0, 2]], 0.2, rtol=1e-7)

    def test_never_raises_on_a_pathological_batch(self):
        prices = np.array([-5.0, 0.0, 1e12, np.nan, 100.0])
        strikes = np.full(5, K)
        got, ok = bs.implied_vol(prices, F, strikes, T, discount=D, right="call")
        assert got.shape == (5,)
        assert ok.sum() >= 1


class TestNumericalRobustness:
    def test_very_low_volatility_round_trips(self):
        p = bs.price(F, K, total(0.005), discount=D, right="call")
        got, ok = bs.implied_vol(p, F, K, T, discount=D, right="call")
        assert bool(ok)
        assert float(got) == pytest.approx(0.005, rel=1e-4)

    def test_very_high_volatility_round_trips(self):
        p = bs.price(F, K, total(3.0), discount=D, right="call")
        got, ok = bs.implied_vol(p, F, K, T, discount=D, right="call")
        assert float(got) == pytest.approx(3.0, rel=1e-6)

    def test_one_day_tenor_round_trips(self):
        """Where dividing by sqrt(T) magnifies error by 19x."""
        tenor = 1 / 365
        p = bs.price(F, 7760.0, total(0.15, tenor), discount=D, right="call")
        got, ok = bs.implied_vol(p, F, 7760.0, tenor, discount=D, right="call")
        assert bool(ok)
        assert float(got) == pytest.approx(0.15, rel=1e-6)

    def test_five_year_tenor_round_trips(self):
        p = bs.price(9577.0, 6800.0, total(0.24, 5.36), discount=0.78, right="put")
        got, ok = bs.implied_vol(p, 9577.0, 6800.0, 5.36, discount=0.78, right="put")
        assert bool(ok)
        assert float(got) == pytest.approx(0.24, rel=1e-7)

    def test_deep_wing_uses_the_bracketed_fallback(self):
        """Vega underflows here, so Newton cannot step. Brent must catch it."""
        strike = 12_000.0
        p = bs.price(F, strike, total(0.9), discount=D, right="call")
        got, ok = bs.implied_vol(p, F, strike, T, discount=D, right="call")
        assert bool(ok)
        assert float(got) == pytest.approx(0.9, rel=1e-5)

    def test_no_division_or_overflow_warnings_over_a_wide_strike_range(self):
        """The regression behind the `np.divide(..., where=)` in the inversion.

        `np.where(cond, a/b, 0)` evaluates the division for every element before
        selecting, so it still divides by the underflowed vegas it was meant to
        skip -- and in the far wings that is most of the array.

        Underflow is deliberately *not* an error here. `exp(-0.5 * 4880**2)`
        underflowing to zero is the correct answer: the normal density really
        is zero that far out, and demanding otherwise would be demanding a
        number that does not exist in double precision.
        """
        strikes = np.linspace(1000.0, 30_000.0, 400)
        prices = bs.price(F, strikes, total(0.2), discount=D, right="call")
        with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
            bs.implied_vol(prices, F, strikes, T, discount=D, right="call")


class TestBrentFallbackFailure:
    def test_a_failing_bracket_yields_nan_rather_than_raising(self, monkeypatch):
        """The last line of defence.

        If Brent cannot resolve a price that passed the bounds check -- a
        bracket that does not straddle a root, say -- the contract is still
        that the caller gets NaN and a False flag. A raise here would abort a
        13,414-quote run over one contract.
        """

        def always_fails(*args, **kwargs):
            raise ValueError("no sign change in bracket")

        monkeypatch.setattr(bs, "brentq", always_fails)
        # A price Newton alone will not resolve: far wing, vega underflowed.
        strike = 40_000.0
        p = bs.price(F, strike, total(1.5), discount=D, right="call")
        got, ok = bs.implied_vol(p, F, strike, T, discount=D, right="call")
        assert not bool(ok)
        assert np.isnan(got)
