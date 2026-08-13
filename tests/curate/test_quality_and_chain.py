"""Unit tests for `spxrnd.curate.quality` and `.chain`.

The end-to-end tests run against the real Aug 11 close capture, so the numbers
asserted here are measurements of a real chain rather than of a construction.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from spxrnd.curate import chain as chain_mod
from spxrnd.curate import quality
from spxrnd.curate.quality import Filters, Flag
from spxrnd.ingest import osi
from spxrnd.ingest.payload import parse as parse_payload

from ..conftest import FIXTURES

AS_OF = datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC)


def quote(**kw):
    base = {
        "option_symbol": "SPXW260918C07750000",
        "root": "SPXW",
        "option_right": "call",
        "expiry": date(2026, 9, 18),
        # OTM against the forward below -- an ITM default would make every
        # "healthy quote" test silently assert the ITM rule instead.
        "strike": 7800.0,
        "bid": 100.0,
        "ask": 102.0,
        "iv": 0.12,
        "tenor_years": 0.1033,
        "forward": 7752.07,
    }
    return {**base, **kw}


def flagged(*rows, filters=None):
    return quality.add_quality_columns(
        pd.DataFrame(list(rows)), filters=filters or Filters()
    )


class TestFlags:
    def test_a_healthy_quote_carries_no_drop_flags(self):
        row = flagged(quote()).iloc[0]
        assert not any(row[f] for f in Filters().drop)

    def test_zero_bid(self):
        assert flagged(quote(bid=0.0)).iloc[0][Flag.ZERO_BID]

    def test_zero_ask(self):
        assert flagged(quote(ask=0.0)).iloc[0][Flag.ZERO_ASK]

    def test_crossed(self):
        assert flagged(quote(bid=102.0, ask=100.0)).iloc[0][Flag.CROSSED]

    def test_wide_spread(self):
        assert flagged(quote(bid=1.0, ask=3.0)).iloc[0][Flag.WIDE_SPREAD]

    def test_spread_exactly_at_the_tolerance_is_not_wide(self):
        """Boundary: the rule is `>`, so equality survives."""
        row = flagged(quote(bid=3.0, ask=5.0)).iloc[0]  # rel spread exactly 0.5
        assert row["relative_spread"] == pytest.approx(0.5)
        assert not row[Flag.WIDE_SPREAD]

    def test_zero_bid_implies_a_200_percent_relative_spread(self):
        """By construction: mid is ask/2, so (ask - 0)/mid is exactly 2."""
        assert flagged(quote(bid=0.0)).iloc[0]["relative_spread"] == pytest.approx(2.0)

    def test_a_zero_mid_gives_an_infinite_relative_spread_not_a_crash(self):
        row = flagged(quote(bid=0.0, ask=0.0)).iloc[0]
        assert row["relative_spread"] == float("inf")
        assert row[Flag.WIDE_SPREAD]

    def test_zero_iv_is_flagged(self):
        assert flagged(quote(iv=0.0)).iloc[0][Flag.ZERO_IV]

    def test_expired(self):
        assert flagged(quote(tenor_years=-0.001)).iloc[0][Flag.EXPIRED]

    def test_zero_dte(self):
        assert flagged(quote(tenor_years=0.0005)).iloc[0][Flag.ZERO_DTE]

    def test_itm_call(self):
        assert flagged(quote(option_right="call", strike=7000.0)).iloc[0][Flag.ITM]

    def test_otm_call(self):
        assert not flagged(quote(option_right="call", strike=8000.0)).iloc[0][Flag.ITM]

    def test_itm_put(self):
        assert flagged(quote(option_right="put", strike=8000.0)).iloc[0][Flag.ITM]

    def test_mid_and_spread(self):
        row = flagged(quote(bid=100.0, ask=102.0)).iloc[0]
        assert row["mid"] == 101.0
        assert row["spread"] == 2.0


class TestZeroIvIsNotADropRule:
    """The deliberate exception, and the reasoning behind it.

    `iv == 0.0` is the feed's "not computed" marker, not a property of the
    quote -- those contracts have live two-sided markets. Measured on the Aug 11
    close it tracks moneyness almost exactly, so the OTM rule removes that
    region for an independent and better reason.
    """

    def test_it_is_not_in_the_default_drop_set(self):
        assert Flag.ZERO_IV not in Filters().drop

    def test_an_otm_quote_with_a_zero_iv_survives(self):
        df = flagged(quote(iv=0.0, strike=8000.0))
        kept, _ = quality.apply(df)
        assert len(kept) == 1

    def test_it_is_still_counted_in_the_attrition_table(self):
        """Not dropping it does not mean not reporting it."""
        _, attrition = quality.apply(flagged(quote(iv=0.0, strike=8000.0)))
        assert attrition.by_rule[str(Flag.ZERO_IV)] == 1

    def test_it_can_be_switched_on_for_a_sensitivity_check(self):
        filters = Filters(drop=frozenset({Flag.ZERO_IV}))
        kept, _ = quality.apply(flagged(quote(iv=0.0, strike=8000.0)), filters=filters)
        assert len(kept) == 0


class TestApply:
    def test_drops_flagged_rows(self):
        df = flagged(quote(), quote(bid=0.0), quote(strike=7000.0))
        kept, _ = quality.apply(df)
        assert len(kept) == 1

    def test_permissive_drops_nothing(self):
        df = flagged(quote(), quote(bid=0.0), quote(strike=7000.0))
        kept, _ = quality.apply(df, filters=quality.PERMISSIVE)
        assert len(kept) == 3

    def test_rules_can_be_disabled_individually(self):
        df = flagged(quote(strike=7000.0))
        kept, _ = quality.apply(df, filters=Filters().without(Flag.ITM))
        assert len(kept) == 1

    def test_attrition_counts_every_rule_even_the_inactive_ones(self):
        df = flagged(quote(iv=0.0), quote(bid=0.0))
        _, attrition = quality.apply(df)
        assert set(attrition.by_rule) >= {str(f) for f in Flag}

    def test_overlapping_rules_do_not_double_count_the_drop(self):
        """A zero-bid quote is usually wide-spread too. The per-rule counts
        overlap; the totals must not."""
        df = flagged(quote(bid=0.0))
        _, attrition = quality.apply(df)
        assert attrition.by_rule[str(Flag.ZERO_BID)] == 1
        assert attrition.by_rule[str(Flag.WIDE_SPREAD)] == 1
        assert attrition.total_dropped == 1

    def test_per_expiry_breakdown(self):
        df = flagged(
            quote(expiry=date(2026, 9, 18)),
            quote(expiry=date(2026, 9, 18), bid=0.0),
            quote(expiry=date(2026, 10, 16)),
        )
        _, attrition = quality.apply(df)
        assert len(attrition.by_expiry) == 2
        assert attrition.by_expiry["n_in"].sum() == 3

    def test_summary_reports_the_headline_numbers(self):
        df = flagged(quote(), quote(bid=0.0))
        _, attrition = quality.apply(df)
        text = attrition.summary()
        assert "2 quotes in, 1 kept" in text
        assert "zero_bid" in text

    def test_input_is_not_mutated(self):
        df = pd.DataFrame([quote()])
        before = df.copy()
        quality.add_quality_columns(df)
        pd.testing.assert_frame_equal(df, before)


class TestRealChain:
    """End to end on the actual Aug 11 close capture."""

    @pytest.fixture(scope="class")
    @classmethod
    def curated(cls):
        with gzip.open(FIXTURES / "chain_full_close.json.gz", "rt") as f:
            snap = parse_payload(json.load(f), ticker="_SPX")
        rows = []
        for opt in snap.options:
            sym = osi.parse(opt["option"])
            rows.append(
                {
                    "option_symbol": sym.raw,
                    "root": sym.root,
                    "option_right": sym.right,
                    "expiry": sym.expiry,
                    "strike": sym.strike,
                    "bid": opt["bid"],
                    "ask": opt["ask"],
                    "iv": opt["iv"],
                }
            )
        return chain_mod.build(pd.DataFrame(rows), as_of=AS_OF, spot=snap.spot)

    def test_starts_from_the_full_chain(self, curated):
        assert curated.attrition.total_in == 30_692

    def test_keeps_a_usable_fraction(self, curated):
        """~44%. Most of the loss is the OTM rule, which removes about half the
        chain by construction."""
        assert 0.40 < curated.attrition.kept_fraction < 0.50

    def test_itm_is_the_dominant_rule(self, curated):
        by_rule = curated.attrition.by_rule
        assert by_rule[str(Flag.ITM)] > 14_000
        assert by_rule[str(Flag.ITM)] > by_rule[str(Flag.WIDE_SPREAD)]

    def test_no_crossed_or_zero_ask_quotes_exist(self, curated):
        """Zero occurrences across the whole chain -- worth asserting rather
        than assuming, since the mid depends on it."""
        assert curated.attrition.by_rule[str(Flag.CROSSED)] == 0
        assert curated.attrition.by_rule[str(Flag.ZERO_ASK)] == 0

    def test_almost_every_expiry_gets_a_forward(self, curated):
        assert len(curated.usable_forwards) == 59
        assert len(curated.forwards) == 60

    def test_the_only_exclusion_is_the_already_settled_expiry(self, curated):
        """The capture is 16:37 ET; the 0-DTE SPXW settled at 16:00."""
        excluded = curated.forwards.loc[~curated.forwards["usable"]]
        assert len(excluded) == 1
        assert excluded.iloc[0]["expiry"] == date(2026, 8, 11)
        assert excluded.iloc[0]["excluded"] == "expired"

    def test_parity_fits_essentially_perfectly(self, curated):
        """It is an arbitrage identity, and it behaves like one."""
        assert curated.usable_forwards["r2"].min() > 0.9999

    def test_the_forward_curve_rises_unambiguously(self, curated):
        """0.03% at one day to 24% at 5.4 years."""
        f = curated.usable_forwards.sort_values("tenor_years")
        assert f["forward"].iloc[0] / curated.spot - 1 < 0.001
        assert f["forward"].iloc[-1] / curated.spot - 1 > 0.20

    def test_within_one_root_the_curve_is_monotonic_to_quote_precision(self, curated):
        """Exact monotonicity does not hold, and should not be asserted.

        Every within-root violation is at most 0.002% of the level -- well
        under a single tick on a 7731 index. Demanding exact monotonicity would
        be demanding that the fit be more precise than the quotes it is fitted
        to.
        """
        spxw = curated.usable_forwards
        spxw = spxw.loc[spxw["root"] == "SPXW"].sort_values("tenor_years")
        steps = spxw["forward"].diff().dropna()
        relative = steps / spxw["forward"].shift(1).dropna()
        assert relative.min() > -1e-4, f"worst within-root step {relative.min():.6%}"
        assert (relative > 0).mean() > 0.9

    def test_shared_expiries_are_the_main_source_of_apparent_kinks(self, curated):
        """Sorting both roots into one curve by tenor produces steps that look
        like kinks and are not: SPX and SPXW on a third Friday are different
        contracts settling 6.5 hours apart, each with its own forward.

        Five of the eight apparent violations in this capture are exactly the
        five third-Friday pairs. Which is the root column doing its job.
        """
        f = curated.usable_forwards.sort_values("tenor_years").reset_index(drop=True)
        drops = f.index[f["forward"].diff() < 0]
        shared = {
            e for e, g in curated.usable_forwards.groupby("expiry") if len(g) == 2
        }
        from_shared = sum(1 for i in drops if f.loc[i, "expiry"] in shared)
        assert from_shared == 5
        assert from_shared / len(drops) > 0.5

    def test_every_forward_is_above_spot(self, curated):
        assert (curated.usable_forwards["forward"] > curated.spot).all()

    def test_implied_rates_are_stable_beyond_the_short_end(self, curated):
        """Past two weeks the curve is tight: 3.9%-6.3%, converging to ~4.6%.

        Inside two weeks it is noisy for a structural reason, not a fit failure:
        rate = -ln(D)/T, and as T goes to zero the division amplifies any error
        in D by 1/T -- 365x at one day.
        """
        f = curated.usable_forwards
        settled = f.loc[f["tenor_years"] > 14 / 365]
        assert settled["rate"].min() > 0.03
        assert settled["rate"].max() < 0.07

    def test_the_long_end_converges(self, curated):
        f = curated.usable_forwards.sort_values("tenor_years")
        assert f.loc[f["tenor_years"] > 1.0, "rate"].std() < 0.005

    def test_shared_expiries_fit_both_roots_separately(self, curated):
        """The third Fridays. Two roots, two forwards, two tenors."""
        f = curated.usable_forwards
        shared = f.groupby("expiry").filter(lambda g: len(g) == 2)
        assert len(shared) == 10, "five third-Friday expiries, both roots each"
        for _expiry, pair in shared.groupby("expiry"):
            spx = pair.loc[pair["root"] == "SPX"].iloc[0]
            spxw = pair.loc[pair["root"] == "SPXW"].iloc[0]
            assert spxw["tenor_years"] > spx["tenor_years"], "PM settles later"

    def test_surviving_quotes_are_all_otm(self, curated):
        assert not curated.quotes[Flag.ITM].any()

    def test_surviving_quotes_all_have_a_live_two_sided_market(self, curated):
        assert (curated.quotes["bid"] > 0).all()
        assert (curated.quotes["ask"] > 0).all()

    def test_log_moneyness_is_centred_near_zero(self, curated):
        """Measured against the forward, so the smile lines up across expiries."""
        lm = curated.quotes["log_moneyness"]
        assert lm.min() < 0 < lm.max()
        assert abs(lm.median()) < 0.5

    def test_zero_dte_contracts_are_gone(self, curated):
        assert (curated.quotes["tenor_years"] >= 1 / 365).all()

    def test_every_expiry_retains_strikes_on_both_sides(self, curated):
        """A density needs both wings. An expiry left with only calls has had
        its put wing eaten by a filter, which the attrition table alone would
        not make obvious."""
        thin = []
        for (expiry, root), group in curated.quotes.groupby(["expiry", "root"]):
            rights = set(group["option_right"])
            if rights != {"call", "put"}:
                thin.append((expiry, root, rights))
        assert thin == []

    def test_summary_is_readable(self, curated):
        text = curated.summary()
        assert "quotes in" in text
        assert "parity R2" in text
        assert "excluded expiries" in text


class TestCuratedChainAccessors:
    @pytest.fixture
    def small(self):
        import numpy as np

        from tests.curate.test_forward import parity_chain

        rows = []
        for root, expiry in [("SPXW", date(2026, 9, 18)), ("SPX", date(2026, 10, 16))]:
            chain = parity_chain(7752.0, 0.9954, np.arange(7000, 8500, 25))
            chain["expiry"] = expiry
            chain["root"] = root
            chain["option_symbol"] = "X"
            chain["iv"] = 0.12
            chain["bid"] = chain["mid"] - 0.5
            chain["ask"] = chain["mid"] + 0.5
            rows.append(chain.drop(columns=["mid"]))
        return chain_mod.build(
            pd.concat(rows, ignore_index=True), as_of=AS_OF, spot=7728.2
        )

    def test_expiries_are_sorted(self, small):
        assert small.expiries == sorted(small.expiries)

    def test_for_expiry_selects_one_root_and_sorts_by_strike(self, small):
        got = small.for_expiry(date(2026, 9, 18), "SPXW")
        assert set(got["root"]) == {"SPXW"}
        assert got["strike"].is_monotonic_increasing

    def test_for_expiry_on_an_absent_key_is_empty_not_an_error(self, small):
        assert len(small.for_expiry(date(2030, 1, 1), "SPXW")) == 0


class TestFromCatalog:
    """The convenience entry point, against the real curated layer."""

    @pytest.fixture(scope="class")
    @classmethod
    def con(cls):
        from pathlib import Path

        from spxrnd.store import catalog

        return catalog.connect(Path("data/curated"))

    def test_curates_a_real_capture(self, con):
        c = chain_mod.from_catalog(con, AS_OF)
        assert c.attrition.total_in == 30_692
        assert len(c.quotes) == 13_414
        assert c.spot == 7728.2002

    def test_reads_spot_from_the_same_payload(self, con):
        """Synchronicity is the property the whole feed is chosen for."""
        c = chain_mod.from_catalog(con, AS_OF)
        assert c.spot == 7728.2002

    def test_filters_are_passed_through(self, con):
        c = chain_mod.from_catalog(con, AS_OF, filters=quality.PERMISSIVE)
        assert len(c.quotes) > 13_414

    def test_an_unknown_capture_raises(self, con):
        with pytest.raises(ValueError, match="no quotes for capture"):
            chain_mod.from_catalog(con, datetime(2020, 1, 1, tzinfo=UTC))

    def test_the_aug_12_settlement_capture_also_curates(self, con):
        """Two trading days held, and both must work."""
        c = chain_mod.from_catalog(con, datetime(2026, 8, 13, 1, 25, 33, tzinfo=UTC))
        assert c.spot == 7748.5
        assert len(c.usable_forwards) > 50
