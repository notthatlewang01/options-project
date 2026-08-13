"""Unit tests for `spxrnd.curate.forward`.

Put-call parity is an arbitrage identity, so the strongest tests are the exact
ones: build quotes that satisfy it perfectly for a chosen (F, D) and check the
fit recovers them to machine precision. Everything a real chain adds -- noise,
thin strikes, mixed roots -- is then a deviation from a known answer.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from spxrnd.curate import forward as fwd

EXPIRY = date(2026, 9, 18)
TENOR = 0.1033


def parity_chain(forward, discount, strikes, *, noise=0.0, seed=0):
    """Quotes that satisfy C - P = D*(F - K) exactly, plus optional noise.

    Only the *difference* is pinned; the level of each leg is arbitrary, which
    is precisely what parity says. Splitting D*(F-K) symmetrically keeps both
    legs positive across the strike range.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in strikes:
        diff = discount * (forward - k)
        base = 200.0
        call, put = base + diff / 2, base - diff / 2
        if noise:
            call += rng.normal(0, noise)
            put += rng.normal(0, noise)
        rows += [
            {"strike": float(k), "option_right": "call", "mid": call},
            {"strike": float(k), "option_right": "put", "mid": put},
        ]
    return pd.DataFrame(rows)


def fit(chain, **kw):
    return fwd.fit_one(chain, expiry=EXPIRY, root="SPXW", tenor_years=TENOR, **kw)


class TestExactRecovery:
    F, D = 7752.07, 0.99544

    @pytest.fixture
    def exact(self):
        return parity_chain(self.F, self.D, np.arange(7000, 8500, 25))

    def test_recovers_the_forward(self, exact):
        assert fit(exact).forward == pytest.approx(self.F, rel=1e-10)

    def test_recovers_the_discount_factor(self, exact):
        assert fit(exact).discount == pytest.approx(self.D, rel=1e-10)

    def test_recovers_the_rate(self, exact):
        expected = -math.log(self.D) / TENOR
        assert fit(exact).rate == pytest.approx(expected, rel=1e-9)

    def test_fit_is_perfect(self, exact):
        result = fit(exact)
        assert result.r2 == pytest.approx(1.0, abs=1e-12)
        assert result.rmse == pytest.approx(0.0, abs=1e-9)

    def test_is_usable(self, exact):
        assert fit(exact).usable
        assert fit(exact).excluded is None

    def test_counts_the_pairs(self, exact):
        assert fit(exact).n_pairs == 60

    def test_identity_holds_at_any_level(self):
        """Parity constrains C - P only. Shifting both legs must not move the
        fit, and a test that failed here would mean the regression had picked
        up the arbitrary level."""
        strikes = np.arange(7000, 8500, 25)
        a = fit(parity_chain(self.F, self.D, strikes))
        shifted = parity_chain(self.F, self.D, strikes)
        shifted["mid"] += 500.0
        b = fit(shifted)
        assert b.forward == pytest.approx(a.forward, rel=1e-10)
        assert b.discount == pytest.approx(a.discount, rel=1e-10)


class TestNoise:
    F, D = 7752.07, 0.99544

    def test_small_quote_noise_barely_moves_the_forward(self):
        noisy = parity_chain(self.F, self.D, np.arange(7000, 8500, 25), noise=0.05)
        assert fit(noisy).forward == pytest.approx(self.F, rel=1e-4)

    def test_noise_shows_up_in_rmse_not_in_the_forward(self):
        noisy = parity_chain(self.F, self.D, np.arange(7000, 8500, 25), noise=0.05)
        result = fit(noisy)
        assert result.rmse > 0
        assert result.r2 > 0.999

    def test_heavy_noise_fails_the_r2_gate(self):
        """The gate exists because parity is an identity. A poor fit means the
        pairs are not what we think they are."""
        garbage = parity_chain(self.F, self.D, np.arange(7000, 8500, 25), noise=500.0)
        result = fit(garbage)
        assert not result.usable
        assert "R2" in result.excluded


class TestExclusion:
    """A bad forward poisons every density built on it while looking like a
    good one. Refusing to produce one is the safe failure."""

    def test_too_few_pairs(self):
        thin = parity_chain(7752.0, 0.995, [7700, 7750, 7800])
        result = fit(thin)
        assert not result.usable
        assert "only 3 call/put pairs" in result.excluded
        assert result.n_pairs == 3

    def test_no_pairs_at_all(self):
        calls_only = pd.DataFrame(
            [
                {"strike": float(k), "option_right": "call", "mid": 100.0}
                for k in range(7000, 8000, 25)
            ]
        )
        result = fit(calls_only)
        assert not result.usable
        assert result.n_pairs == 0

    def test_all_pairs_at_one_strike(self):
        """No leverage on the slope; a line through one x is undetermined."""
        same = pd.concat([parity_chain(7752.0, 0.995, [7750])] * 12, ignore_index=True)
        result = fit(same, min_pairs=2)
        assert not result.usable
        assert "one strike" in result.excluded

    def test_expired_expiry(self):
        chain = parity_chain(7752.0, 0.995, np.arange(7000, 8500, 25))
        result = fwd.fit_one(chain, expiry=EXPIRY, root="SPXW", tenor_years=-0.001)
        assert not result.usable
        assert result.excluded == "expired"

    def test_negative_discount_factor(self):
        """An upward-sloping C - P is not a parity relation."""
        inverted = parity_chain(7752.0, -0.995, np.arange(7000, 8500, 25))
        result = fit(inverted)
        assert not result.usable
        assert "discount" in result.excluded

    def test_implausible_discount_factor(self):
        result = fit(parity_chain(7752.0, 2.5, np.arange(7000, 8500, 25)))
        assert not result.usable
        assert "implausible discount" in result.excluded

    def test_negative_forward(self):
        """A valid-looking discount with a negative intercept. Defensive, but
        the alternative is a log-moneyness of NaN propagating silently through
        every downstream comparison."""
        result = fit(parity_chain(-100.0, 0.995, np.arange(7000, 8500, 25)))
        assert not result.usable
        assert "non-positive forward" in result.excluded

    def test_absurd_implied_rate(self):
        """A degenerate fit can still have R2 = 1. The rate bound catches it."""
        chain = parity_chain(7752.0, 0.5, np.arange(7000, 8500, 25))
        result = fwd.fit_one(chain, expiry=EXPIRY, root="SPXW", tenor_years=0.01)
        assert not result.usable
        assert "implied rate" in result.excluded

    def test_every_exclusion_records_a_reason(self):
        """An exclusion with no reason is an unexplained hole in the term
        structure three months from now."""
        for chain, tenor in [
            (parity_chain(7752.0, 0.995, [7700, 7750]), TENOR),
            (parity_chain(7752.0, -0.995, np.arange(7000, 8000, 25)), TENOR),
            (parity_chain(7752.0, 0.995, np.arange(7000, 8000, 25)), -0.1),
        ]:
            result = fwd.fit_one(chain, expiry=EXPIRY, root="SPXW", tenor_years=tenor)
            assert not result.usable
            assert result.excluded


class TestRootSeparation:
    """The concrete payoff of seed regression 2.

    SPX and SPXW list the same strikes on third Fridays at different prices,
    because they settle 6.5 hours apart. Pairing a call from one root with a put
    from the other produces a `C - P` that is not a parity relation, and the
    fitted forward is silently wrong.
    """

    def test_mixing_roots_corrupts_the_fit(self):
        strikes = np.arange(7000, 8500, 25)
        spx = parity_chain(7734.42, 0.99860, strikes)
        spxw = parity_chain(7733.72, 0.99856, strikes)

        # Take calls from one root and puts from the other, as a pipeline
        # without a root column necessarily would.
        mixed = pd.concat(
            [
                spx.loc[spx["option_right"] == "call"],
                spxw.loc[spxw["option_right"] == "put"],
            ],
            ignore_index=True,
        )
        clean = fit(spx)
        corrupted = fit(mixed)
        assert clean.forward == pytest.approx(7734.42, rel=1e-9)
        assert abs(corrupted.forward - 7734.42) > abs(clean.forward - 7734.42)

    def test_fit_all_groups_by_root_not_expiry_alone(self):
        strikes = np.arange(7000, 8500, 25)
        rows = []
        for root, f, d in [("SPX", 7734.42, 0.99860), ("SPXW", 7733.72, 0.99856)]:
            chain = parity_chain(f, d, strikes)
            chain["expiry"] = date(2026, 8, 21)
            chain["root"] = root
            rows.append(chain)
        fits = fwd.fit_all(
            pd.concat(rows, ignore_index=True),
            as_of=datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC),
        )
        assert len(fits) == 2, "one fit per (expiry, root), never per expiry"
        by_root = {f.root: f for f in fits}
        assert by_root["SPX"].forward == pytest.approx(7734.42, rel=1e-6)
        assert by_root["SPXW"].forward == pytest.approx(7733.72, rel=1e-6)

    def test_the_two_roots_get_different_tenors(self):
        """AM vs PM settlement, on the same calendar date."""
        strikes = np.arange(7000, 8500, 25)
        rows = []
        for root in ("SPX", "SPXW"):
            chain = parity_chain(7734.0, 0.9986, strikes)
            chain["expiry"] = date(2026, 8, 21)
            chain["root"] = root
            rows.append(chain)
        fits = {
            f.root: f
            for f in fwd.fit_all(
                pd.concat(rows, ignore_index=True),
                as_of=datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC),
            )
        }
        assert fits["SPXW"].tenor_years > fits["SPX"].tenor_years


class TestToFrame:
    def test_includes_excluded_fits(self):
        """The excluded rows are the audit trail; dropping them here would make
        exclusions invisible."""
        good = fit(parity_chain(7752.0, 0.995, np.arange(7000, 8500, 25)))
        bad = fit(parity_chain(7752.0, 0.995, [7700, 7750]))
        frame = fwd.to_frame([good, bad])
        assert len(frame) == 2
        assert frame["usable"].tolist() == [True, False]
        assert frame.loc[~frame["usable"], "excluded"].notna().all()

    def test_columns_support_a_join_back_onto_quotes(self):
        frame = fwd.to_frame(
            [fit(parity_chain(7752.0, 0.995, np.arange(7000, 8500, 25)))]
        )
        assert {"expiry", "root", "forward", "discount", "tenor_years"} <= set(frame)
