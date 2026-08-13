"""Unit tests for `spxrnd.analytics.arbitrage`.

Constructed chains first -- a Black-76 chain is arbitrage-free by construction,
so anything the checks flag on one is a false positive, and a deliberately
broken chain must be caught. Then the real Aug 11 capture, where the numbers
are measurements rather than constructions.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from spxrnd.analytics import arbitrage, bs

F, D, T = 7752.07, 0.99544, 0.1033
STRIKES = np.arange(6800.0, 8800.0, 50.0)


def model_calls(strikes=STRIKES, sigma=0.20, forward=F, discount=D, tenor=T):
    """Black-76 call prices: arbitrage-free by construction."""
    return np.asarray(
        bs.price(
            forward, strikes, sigma * np.sqrt(tenor), discount=discount, right="call"
        )
    )


def chain_frame(
    strikes=STRIKES,
    sigma=0.20,
    *,
    half_spread=0.0,
    expiry=None,
    root="SPXW",
    forward=F,
    discount=D,
    tenor=T,
):
    """A curated-shaped frame of OTM quotes from a Black-76 chain."""
    expiry = expiry or dt.date(2026, 9, 18)
    rights = np.where(strikes >= forward, "call", "put")
    mids = np.array(
        [
            float(
                bs.price(forward, k, sigma * np.sqrt(tenor), discount=discount, right=r)
            )
            for k, r in zip(strikes, rights, strict=True)
        ]
    )
    return pd.DataFrame(
        {
            "expiry": expiry,
            "root": root,
            "option_right": rights,
            "strike": strikes,
            "mid": mids,
            "bid": mids - half_spread,
            "ask": mids + half_spread,
            "forward": forward,
            "discount": discount,
            "tenor_years": tenor,
        }
    )


class TestVerticalOnCleanChains:
    def test_a_model_chain_has_no_violations(self):
        assert arbitrage.check_vertical(STRIKES, model_calls(), discount=D) == []

    def test_uneven_strike_spacing_is_fine(self):
        strikes = np.array([6800.0, 7000.0, 7500.0, 7752.0, 7760.0, 8000.0, 8800.0])
        assert arbitrage.check_vertical(strikes, model_calls(strikes), discount=D) == []

    def test_unsorted_input_is_sorted_first(self):
        order = np.random.default_rng(0).permutation(len(STRIKES))
        assert (
            arbitrage.check_vertical(STRIKES[order], model_calls()[order], discount=D)
            == []
        )


class TestVerticalCatchesBreaches:
    def test_a_call_price_rising_with_strike(self):
        calls = model_calls().copy()
        calls[10] = calls[9] + 5.0
        hits = arbitrage.check_vertical(STRIKES, calls, discount=D)
        assert any("rises with strike" in h.detail for h in hits)

    def test_a_spread_cheaper_than_its_worst_case(self):
        calls = model_calls().copy()
        calls[10] = calls[9] - 200.0
        hits = arbitrage.check_vertical(STRIKES, calls, discount=D)
        assert any("worst case" in h.detail for h in hits)

    def test_records_the_offending_strikes(self):
        calls = model_calls().copy()
        calls[10] = calls[9] + 5.0
        hit = arbitrage.check_vertical(STRIKES, calls, discount=D)[0]
        assert hit.strikes == (float(STRIKES[9]), float(STRIKES[10]))

    def test_a_breach_inside_the_spread_is_not_executable(self):
        """The distinction the whole report turns on.

        Perturbing the last strike, so exactly one adjacent pair is affected --
        nudging an interior point breaks the step on both sides of it and the
        second breach is a different, larger one.
        """
        calls = model_calls().copy()
        calls[-1] = calls[-2] + 0.02
        wide = 5.0
        hits = arbitrage.check_vertical(
            STRIKES, calls, discount=D, bids=calls - wide, asks=calls + wide
        )
        assert hits, "a rising call price must be flagged"
        assert all("rises with strike" in h.detail for h in hits)
        assert not any(h.executable for h in hits), (
            "0.02 of price against a 10-wide market is not a tradeable edge"
        )

    def test_a_breach_beyond_the_spread_is_executable(self):
        calls = model_calls().copy()
        calls[10] = calls[9] + 50.0
        tight = 0.05
        hits = arbitrage.check_vertical(
            STRIKES, calls, discount=D, bids=calls - tight, asks=calls + tight
        )
        assert any(h.executable for h in hits)


class TestButterflyOnCleanChains:
    def test_a_model_chain_is_convex(self):
        assert arbitrage.check_butterfly(STRIKES, model_calls()) == []

    def test_uneven_spacing_does_not_manufacture_violations(self):
        """The reason the weights are (k3-k2)/(k3-k1), not (1, -2, 1).

        Treating a 200/50 strike gap as if it were even makes a perfectly
        convex chain look broken.
        """
        strikes = np.array([6800.0, 7000.0, 7050.0, 7500.0, 7752.0, 7800.0, 8800.0])
        assert arbitrage.check_butterfly(strikes, model_calls(strikes)) == []

    @pytest.mark.parametrize("sigma", [0.05, 0.20, 0.80, 2.0])
    def test_convex_at_every_volatility(self, sigma):
        assert arbitrage.check_butterfly(STRIKES, model_calls(sigma=sigma)) == []

    def test_fewer_than_three_strikes_is_not_an_error(self):
        assert arbitrage.check_butterfly(STRIKES[:2], model_calls()[:2]) == []


class TestButterflyCatchesBreaches:
    def test_a_dented_body_is_caught(self):
        calls = model_calls().copy()
        calls[10] += 20.0
        hits = arbitrage.check_butterfly(STRIKES, calls)
        assert hits
        assert any(h.strikes[1] == float(STRIKES[10]) for h in hits)

    def test_reports_negative_density(self):
        calls = model_calls().copy()
        calls[10] += 20.0
        assert "negative density" in arbitrage.check_butterfly(STRIKES, calls)[0].detail

    def test_magnitude_is_positive_and_relative_is_scaled(self):
        calls = model_calls().copy()
        calls[10] += 20.0
        hit = arbitrage.check_butterfly(STRIKES, calls, forward=F)[0]
        assert hit.magnitude > 0
        assert hit.relative == pytest.approx(hit.magnitude / F)

    def test_a_dent_smaller_than_the_spread_is_not_executable(self):
        """Calibrated against the real convexity scale.

        The baseline butterfly value on this chain is ~0.63, so a dent must
        exceed that to break convexity at all; a 0.05 "tick-sized" dent simply
        does not, which is itself worth knowing. A dent of 3.0 breaks it on
        mids while a 2.0-wide market still absorbs it.
        """
        calls = model_calls().copy()
        calls[10] += 3.0
        hits = arbitrage.check_butterfly(
            STRIKES, calls, bids=calls - 2.0, asks=calls + 2.0
        )
        assert hits, "a 3.0 dent must break convexity on mids"
        assert not any(h.executable for h in hits)

    def test_a_dent_smaller_than_the_convexity_does_not_break_it(self):
        """The baseline butterfly value is ~0.63 across this chain, so a
        sub-tick perturbation is absorbed entirely. Pinned because it sets the
        scale for reading the real chain's 1,929 mid-level breaches."""
        calls = model_calls().copy()
        calls[10] += 0.05
        assert arbitrage.check_butterfly(STRIKES, calls) == []

    def test_a_large_dent_is_executable(self):
        calls = model_calls().copy()
        calls[10] += 50.0
        hits = arbitrage.check_butterfly(
            STRIKES, calls, bids=calls - 0.05, asks=calls + 0.05
        )
        assert any(h.executable for h in hits)


class TestCalendar:
    def frame(self, tenors_and_vols, root="SPXW"):
        rows = []
        for i, (tenor, sigma) in enumerate(tenors_and_vols):
            f = chain_frame(
                sigma=sigma,
                tenor=tenor,
                root=root,
                expiry=dt.date(2026, 9, 1) + dt.timedelta(days=30 * i),
            )
            f["log_moneyness"] = np.log(f["strike"] / f["forward"])
            f["total_variance"] = sigma**2 * tenor
            rows.append(f)
        return pd.concat(rows, ignore_index=True)

    def test_rising_total_variance_is_clean(self):
        frame = self.frame([(0.1, 0.20), (0.3, 0.20), (0.8, 0.20)])
        assert arbitrage.check_calendar(frame) == []

    def test_falling_total_variance_is_caught(self):
        """A longer option worth less than a shorter one at the same relative
        strike is a calendar arbitrage."""
        frame = self.frame([(0.1, 0.40), (0.3, 0.10)])
        hits = arbitrage.check_calendar(frame)
        assert hits
        assert "total variance falls" in hits[0].detail

    def test_flat_total_variance_is_clean(self):
        """Non-decreasing, so equality passes."""
        frame = self.frame([(0.1, 0.40), (0.4, 0.20)])
        assert arbitrage.check_calendar(frame) == []

    def test_roots_are_never_compared_against_each_other(self):
        """The bug this check had, and the reason it is worth a test.

        SPX and SPXW on a third Friday sit 6.5 hours apart. Sorting every
        (expiry, root) curve into one sequence by tenor puts them adjacent, and
        comparing them is not a calendar condition -- it manufactured four
        violations on a chain that had none.
        """
        spx = self.frame([(0.1033, 0.30)], root="SPX")
        spxw = self.frame([(0.1040, 0.10)], root="SPXW")
        both = pd.concat([spx, spxw], ignore_index=True)
        assert arbitrage.check_calendar(both) == [], (
            "two roots of one expiry must never be compared"
        )

    def test_each_root_is_still_checked_internally(self):
        a = self.frame([(0.1, 0.40), (0.3, 0.10)], root="SPX")
        b = self.frame([(0.1, 0.40), (0.3, 0.10)], root="SPXW")
        hits = arbitrage.check_calendar(pd.concat([a, b], ignore_index=True))
        assert {h.root for h in hits} == {"SPX", "SPXW"}

    def test_missing_columns_raise_rather_than_silently_skip(self):
        with pytest.raises(KeyError, match="total_variance"):
            arbitrage.check_calendar(
                pd.DataFrame({"tenor_years": [1.0], "log_moneyness": [0.0]})
            )


class TestCheckChain:
    def test_a_model_chain_is_clean(self):
        report = arbitrage.check_chain(chain_frame(half_spread=0.5))
        assert report.clean
        assert report.n_checked["butterfly"] > 0

    def test_put_call_parity_converts_the_put_wing(self):
        """The chain holds OTM options only, so the checks need the puts
        expressed as their call equivalents. If the conversion were wrong the
        two wings would not join and every triple at the seam would flag."""
        frame = chain_frame(half_spread=0.5)
        assert (frame["option_right"] == "put").any()
        assert (frame["option_right"] == "call").any()
        assert arbitrage.check_chain(frame).clean

    def test_a_broken_chain_is_caught(self):
        frame = chain_frame(half_spread=0.01)
        frame.loc[20, "mid"] += 100.0
        frame.loc[20, "bid"] += 100.0
        frame.loc[20, "ask"] += 100.0
        assert not arbitrage.check_chain(frame).clean

    def test_summary_separates_mid_breaches_from_executable_ones(self):
        text = arbitrage.check_chain(chain_frame(half_spread=0.5)).summary()
        assert "on mids" in text
        assert "executable" in text

    def test_worst_returns_the_largest_relative_breach(self):
        frame = chain_frame(half_spread=0.01)
        frame.loc[20, ["mid", "bid", "ask"]] += 100.0
        report = arbitrage.check_chain(frame)
        assert report.worst("butterfly").relative > 0


class TestRealChain:
    """The Aug 11 close, curated and inverted."""

    @pytest.fixture(scope="class")
    @classmethod
    def report(cls):
        from datetime import UTC, datetime
        from pathlib import Path

        from spxrnd.curate import chain
        from spxrnd.store import catalog

        con = catalog.connect(Path("data/curated"))
        c = chain.from_catalog(con, datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC))
        q = c.quotes.copy()
        iv = np.full(len(q), np.nan)
        for right in ("call", "put"):
            m = (q["option_right"] == right).to_numpy()
            s, _ = bs.implied_vol(
                q.loc[m, "mid"].to_numpy(),
                q.loc[m, "forward"].to_numpy(),
                q.loc[m, "strike"].to_numpy(),
                q.loc[m, "tenor_years"].to_numpy(),
                discount=q.loc[m, "discount"].to_numpy(),
                right=right,
            )
            iv[m] = s
        q["total_variance"] = iv**2 * q["tenor_years"]
        return arbitrage.check_chain(q)

    def test_thousands_of_triples_are_checked(self, report):
        assert report.n_checked["butterfly"] > 13_000
        assert report.n_checked["vertical"] > 13_000

    def test_no_executable_vertical_breaches(self, report):
        """The ten mid-level breaches are each exactly one tick of quote
        granularity -- SPX quotes in 0.05 increments. You cannot trade a mid."""
        assert report.of_kind("vertical", executable_only=True) == []

    def test_no_calendar_breaches_at_all(self, report):
        assert report.of_kind("calendar") == []

    def test_at_most_one_executable_butterfly(self, report):
        """13,298 triples, one marginal breach worth about nine cents."""
        executable = report.of_kind("butterfly", executable_only=True)
        assert len(executable) <= 1

    def test_the_surviving_breach_is_economically_negligible(self, report):
        executable = report.of_kind("butterfly", executable_only=True)
        if executable:
            assert max(h.relative for h in executable) < 1e-4

    def test_mid_level_breaches_vastly_outnumber_executable_ones(self, report):
        """Which is the point of drawing the distinction. Reporting the mid
        count alone would claim 1,929 arbitrages in a chain that has at most
        one."""
        on_mids = len(report.of_kind("butterfly"))
        executable = len(report.of_kind("butterfly", executable_only=True))
        assert on_mids > 100 * max(executable, 1)


class TestDefensivePaths:
    """Branches that only fire on awkward chains. Untested defensive code is
    where a silent failure hides."""

    def test_expiries_with_no_common_moneyness_are_skipped(self):
        """Two expiries whose strike ranges do not overlap in log-moneyness
        cannot be compared. Interpolating outside either curve's support would
        invent a variance and then flag it."""
        near = chain_frame(
            strikes=np.arange(7700.0, 7800.0, 10.0),
            tenor=0.1,
            expiry=dt.date(2026, 9, 18),
        )
        near["log_moneyness"] = np.log(near["strike"] / near["forward"])
        near["total_variance"] = 0.04
        far = chain_frame(
            strikes=np.arange(3000.0, 3500.0, 50.0),
            tenor=0.5,
            expiry=dt.date(2026, 12, 18),
        )
        far["log_moneyness"] = np.log(far["strike"] / far["forward"])
        far["total_variance"] = 0.001  # would flag loudly if it were compared
        both = pd.concat([near, far], ignore_index=True)
        assert arbitrage.check_calendar(both) == []

    def test_an_expiry_with_too_few_strikes_is_skipped(self):
        """A butterfly needs three strikes. Two is not a violation."""
        thin = chain_frame(strikes=np.array([7700.0, 7800.0]), half_spread=0.5)
        report = arbitrage.check_chain(thin)
        assert report.clean
        assert report.n_checked["butterfly"] == 0

    def test_an_empty_chain_is_clean(self):
        empty = chain_frame().iloc[:0]
        report = arbitrage.check_chain(empty)
        assert report.clean
