"""Unit tests for `spxrnd.curate.moneyness`."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time

import pytest

from spxrnd.curate.moneyness import (
    DAYS_PER_YEAR,
    EASTERN,
    expiry_instant,
    is_otm,
    log_moneyness,
    settlement_time,
    tenor_years,
)


class TestSettlementConvention:
    def test_spx_settles_am(self):
        """SET is built from the opening prints, so the contract is done at
        09:30 ET, not at the close."""
        assert settlement_time("SPX") == time(9, 30)

    def test_spxw_settles_pm(self):
        assert settlement_time("SPXW") == time(16, 0)

    def test_unknown_roots_default_to_pm(self):
        """PM is the common case; an unrecognised root is more likely a new
        weekly than a new AM-settled series."""
        assert settlement_time("XSP") == time(16, 0)

    def test_expiry_instants_differ_by_six_and_a_half_hours(self):
        """The whole reason this module knows about roots.

        On a third Friday both list the same date. Treating them alike
        misprices the shorter one by 6.5 hours -- 27% of a one-day option's
        remaining life.
        """
        d = date(2026, 8, 21)
        gap = expiry_instant(d, "SPXW") - expiry_instant(d, "SPX")
        assert gap.total_seconds() == 6.5 * 3600

    def test_expiry_instant_is_eastern_aware(self):
        got = expiry_instant(date(2026, 8, 21), "SPXW")
        assert got.tzinfo is EASTERN
        assert got.isoformat() == "2026-08-21T16:00:00-04:00"

    def test_winter_expiry_uses_standard_time(self):
        """Handled by the zone, not an offset hardcoded in August."""
        assert expiry_instant(date(2027, 1, 15), "SPXW").isoformat() == (
            "2027-01-15T16:00:00-05:00"
        )


class TestTenor:
    AS_OF = datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC)  # 16:37:03 ET

    def test_one_year_is_one(self):
        as_of = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)
        got = tenor_years(date(2027, 8, 21), "SPXW", as_of=as_of)
        assert got == pytest.approx(1.0, abs=0.01)

    def test_matches_the_day_count(self):
        as_of = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)
        got = tenor_years(date(2026, 9, 20), "SPXW", as_of=as_of)
        assert got == pytest.approx(30 / DAYS_PER_YEAR, abs=1e-3)

    def test_root_changes_the_tenor_on_a_shared_expiry(self):
        spx = tenor_years(date(2026, 8, 21), "SPX", as_of=self.AS_OF)
        spxw = tenor_years(date(2026, 8, 21), "SPXW", as_of=self.AS_OF)
        assert spxw > spx
        assert (spxw - spx) * DAYS_PER_YEAR * 24 == pytest.approx(6.5, abs=1e-6)

    def test_a_settled_contract_has_negative_tenor(self):
        """The Aug 11 capture at 16:37 ET holds SPXW contracts that settled at
        16:00. That is information, not an error -- the caller filters."""
        assert tenor_years(date(2026, 8, 11), "SPXW", as_of=self.AS_OF) < 0

    def test_exactly_at_settlement_is_zero(self):
        as_of = expiry_instant(date(2026, 8, 21), "SPXW")
        assert tenor_years(date(2026, 8, 21), "SPXW", as_of=as_of) == 0.0

    def test_naive_as_of_is_rejected(self):
        """Comparing a naive clock against an Eastern settlement time is wrong
        by the UTC offset, and wrong in a way that looks plausible."""
        with pytest.raises(ValueError, match="timezone-aware"):
            tenor_years(date(2026, 8, 21), "SPXW", as_of=datetime(2026, 8, 11))

    def test_tenor_decreases_as_the_capture_advances(self):
        earlier = tenor_years(date(2026, 12, 18), "SPX", as_of=self.AS_OF)
        later = tenor_years(
            date(2026, 12, 18), "SPX", as_of=datetime(2026, 8, 12, 20, 37, tzinfo=UTC)
        )
        assert later < earlier


class TestLogMoneyness:
    def test_at_the_forward_is_zero(self):
        assert log_moneyness(7728.2, 7728.2) == 0.0

    def test_sign_convention(self):
        assert log_moneyness(8000, 7728.2) > 0
        assert log_moneyness(7000, 7728.2) < 0

    def test_matches_the_definition(self):
        assert log_moneyness(8000, 7728.2) == pytest.approx(math.log(8000 / 7728.2))

    def test_symmetric_in_ratio(self):
        assert log_moneyness(2.0, 1.0) == pytest.approx(-log_moneyness(1.0, 2.0))

    @pytest.mark.parametrize(("k", "f"), [(0, 100), (-1, 100), (100, 0), (100, -1)])
    def test_non_positive_inputs_are_rejected(self, k, f):
        with pytest.raises(ValueError, match="must be positive"):
            log_moneyness(k, f)


class TestIsOtm:
    F = 7734.42

    def test_calls_above_the_forward_are_otm(self):
        assert is_otm("call", 8000, self.F)

    def test_calls_below_the_forward_are_itm(self):
        assert not is_otm("call", 7000, self.F)

    def test_puts_below_the_forward_are_otm(self):
        assert is_otm("put", 7000, self.F)

    def test_puts_above_the_forward_are_itm(self):
        assert not is_otm("put", 8000, self.F)

    def test_at_the_forward_both_count_as_otm(self):
        """A boundary that has to go somewhere. At F the two are the same
        contract by parity, so excluding both would leave a hole at exactly the
        strike that matters most."""
        assert is_otm("call", self.F, self.F)
        assert is_otm("put", self.F, self.F)

    def test_the_selection_is_measured_against_the_forward_not_spot(self):
        """They differ by 24% at the 5-year expiry. Using spot would misclassify
        the entire long end."""
        spot, forward = 7728.2, 9577.42
        assert is_otm("call", 9000, spot)
        assert not is_otm("call", 9000, forward)
