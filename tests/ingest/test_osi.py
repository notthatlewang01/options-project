"""Unit tests for `spxrnd.ingest.osi`."""

from __future__ import annotations

from datetime import date

import pytest

from spxrnd.ingest.errors import OsiParseError
from spxrnd.ingest.osi import CALL, PUT, OptionSymbol, parse, try_parse


class TestParseNominal:
    def test_weekly_call(self):
        sym = parse("SPXW260821C01400000")
        assert sym.root == "SPXW"
        assert sym.expiry == date(2026, 8, 21)
        assert sym.right == CALL
        assert sym.strike == 1400.0
        assert sym.raw == "SPXW260821C01400000"

    def test_monthly_put(self):
        sym = parse("SPX260918P07700000")
        assert sym.root == "SPX"
        assert sym.expiry == date(2026, 9, 18)
        assert sym.right == PUT
        assert sym.strike == 7700.0

    def test_right_helpers_agree_with_right(self):
        assert parse("SPX260918C07700000").is_call
        assert not parse("SPX260918C07700000").is_put
        assert parse("SPX260918P07700000").is_put
        assert not parse("SPX260918P07700000").is_call

    def test_embedded_spaces_are_stripped(self):
        """Some feeds pad the root to six characters."""
        assert parse("SPX   260918C07700000") == parse("SPX260918C07700000")

    def test_raw_records_the_cleaned_symbol(self):
        assert parse("SPX   260918C07700000").raw == "SPX260918C07700000"


class TestStrikeScaling:
    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("00000000", 0.0),
            ("00000001", 0.001),
            ("00001000", 1.0),
            ("00200000", 200.0),  # the deep-ITM contract in the archive
            ("07727500", 7727.5),  # half-point strike, exactly representable
            ("07728000", 7728.0),
            ("99999999", 99999.999),
        ],
    )
    def test_strike_is_the_field_over_1000(self, field, expected):
        assert parse(f"SPX260918C{field}").strike == pytest.approx(expected)

    def test_half_point_strikes_are_exact(self):
        """SPX lists 5-point strikes with 2.5 increments near the money.

        Binary floats represent halves exactly, so this is a real equality and
        not an approximation.
        """
        assert parse("SPXW260811C07727500").strike == 7727.5


class TestExpiryDecoding:
    def test_two_digit_year_maps_to_2000s(self):
        assert parse("SPX310821C01400000").expiry == date(2031, 8, 21)

    def test_expiry_is_a_date_object_not_a_string(self):
        assert isinstance(parse("SPX260821C01400000").expiry, date)

    def test_leap_day_parses(self):
        assert parse("SPX280229C01400000").expiry == date(2028, 2, 29)

    def test_non_leap_february_29_is_rejected(self):
        """2027 is not a leap year. A date that does not exist must not parse.

        `date()` catches this; a naive string-slicing parser would happily
        produce "2027-02-29" and fail much later, somewhere less obvious.
        """
        with pytest.raises(OsiParseError, match="impossible expiry"):
            parse("SPX270229C01400000")

    @pytest.mark.parametrize("bad", ["261301", "260832", "260001", "260800"])
    def test_impossible_dates_are_rejected(self, bad):
        with pytest.raises(OsiParseError, match="impossible expiry"):
            parse(f"SPX{bad}C01400000")


class TestParseFailures:
    @pytest.mark.parametrize(
        ("symbol", "why"),
        [
            ("", "empty"),
            ("SPX", "root only"),
            ("260821C01400000", "no root"),
            ("SPX260821C0140000", "strike too short"),
            ("SPX260821C014000000", "strike too long"),
            ("SPX26021C01400000", "expiry too short"),
            ("SPX260821X01400000", "not C or P"),
            ("spxw260821C01400000", "lowercase root"),
            ("SPX260821c01400000", "lowercase right"),
            ("SPX260821C0140000A", "non-digit in strike"),
            ("1SPX260821C01400000", "digit before the root"),
            ("SPX260821C01400000X", "trailing junk"),
        ],
    )
    def test_malformed_symbols_raise(self, symbol, why):
        with pytest.raises(OsiParseError):
            parse(symbol)

    def test_non_greedy_root_regression(self):
        """The original pattern used `(?P<root>.+?)`, which accepts junk.

        A non-greedy dot would match "1SPX" as a root and parse a symbol that
        should have been rejected outright. `[A-Z]+` refuses it.
        """
        with pytest.raises(OsiParseError):
            parse("1SPX260821C01400000")

    def test_error_message_names_the_symbol(self):
        """An operator reading a log needs to know which symbol broke."""
        with pytest.raises(OsiParseError, match=r"NOTASYMBOL"):
            parse("NOTASYMBOL")

    def test_spaces_are_stripped_everywhere_not_only_from_the_root(self):
        """Documented leniency, pinned because it is a judgement call.

        Stripping is unconditional, so a space inside the date is silently
        healed rather than rejected. Chosen because the feed pads the root to a
        fixed width and no observed capture has ever contained an interior
        space -- if one appears, this test is where the decision gets revisited.
        """
        assert parse("SPX2608 21C01400000") == parse("SPX260821C01400000")


class TestTryParse:
    def test_returns_a_symbol_on_success(self):
        assert try_parse("SPX260821C01400000") == parse("SPX260821C01400000")

    @pytest.mark.parametrize("bad", ["", "junk", "SPX270229C01400000"])
    def test_returns_none_instead_of_raising(self, bad):
        assert try_parse(bad) is None


class TestValueSemantics:
    def test_equal_symbols_compare_equal(self):
        assert parse("SPX260821C01400000") == parse("SPX260821C01400000")

    def test_roots_keep_otherwise_identical_contracts_distinct(self):
        spx = parse("SPX260821C01400000")
        spxw = parse("SPXW260821C01400000")
        assert spx != spxw
        assert len({spx, spxw}) == 2

    def test_is_hashable_so_it_can_key_a_quote_dict(self):
        quotes = {parse("SPX260821C01400000"): 6316.7}
        assert quotes[parse("SPX260821C01400000")] == 6316.7

    def test_is_frozen(self):
        sym = parse("SPX260821C01400000")
        with pytest.raises(AttributeError):
            sym.strike = 1.0  # type: ignore[misc]


class TestAgainstTheRealChain:
    def test_every_symbol_in_the_full_chain_parses(self, full_chain):
        """30,692 real symbols. Not one may be dropped silently."""
        options = full_chain["data"]["options"]
        failures = [o["option"] for o in options if try_parse(o["option"]) is None]
        assert failures == [], f"{len(failures)} unparseable symbols: {failures[:5]}"

    def test_roots_observed_are_exactly_spx_and_spxw(self, full_chain):
        roots = {parse(o["option"]).root for o in full_chain["data"]["options"]}
        assert roots == {"SPX", "SPXW"}

    def test_parsed_strikes_are_positive_and_bounded(self, full_chain):
        strikes = [parse(o["option"]).strike for o in full_chain["data"]["options"]]
        assert min(strikes) > 0
        assert max(strikes) < 100_000

    def test_symbols_round_trip_through_their_components(self, full_chain):
        """Reconstructing the symbol from the parsed parts must reproduce it.

        This is the strongest available check that nothing is lost or
        transposed in parsing, and it runs over every real contract.
        """
        for opt in full_chain["data"]["options"]:
            sym: OptionSymbol = parse(opt["option"])
            rebuilt = (
                f"{sym.root}"
                f"{sym.expiry:%y%m%d}"
                f"{'C' if sym.is_call else 'P'}"
                f"{round(sym.strike * 1000):08d}"
            )
            assert rebuilt == sym.raw
