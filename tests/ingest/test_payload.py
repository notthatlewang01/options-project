"""Unit tests for `spxrnd.ingest.payload`."""

from __future__ import annotations

import copy
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from spxrnd.ingest.errors import PayloadError
from spxrnd.ingest.payload import EASTERN, parse, parse_eastern

from ..conftest import load_header


@pytest.fixture
def body():
    """A mutable copy of a real capture, for corrupting one field at a time."""
    return copy.deepcopy(load_header("2026-08-11T16-11-33Z"))


class TestParseEasternNominal:
    def test_attaches_eastern_without_shifting_the_clock(self):
        got = parse_eastern("2026-08-11T16:14:59")
        assert (got.year, got.month, got.day) == (2026, 8, 11)
        assert (got.hour, got.minute, got.second) == (16, 14, 59)
        assert got.tzinfo is EASTERN

    def test_result_is_aware(self):
        assert parse_eastern("2026-08-11T16:14:59").utcoffset() is not None

    def test_summer_timestamp_is_edt(self):
        """August is daylight time: UTC-4."""
        assert (
            parse_eastern("2026-08-11T16:14:59").utcoffset().total_seconds()
            == -4 * 3600
        )

    def test_winter_timestamp_is_est(self):
        """January is standard time: UTC-5. Handled by the zone, not by an
        offset someone hardcoded in August."""
        assert (
            parse_eastern("2026-01-15T16:14:59").utcoffset().total_seconds()
            == -5 * 3600
        )

    def test_space_separated_form_parses(self):
        """The feed's top-level timestamp uses a space, not a T."""
        assert parse_eastern("2026-08-11 20:36:41").hour == 20


class TestParseEasternFailures:
    @pytest.mark.parametrize("bad", ["", None])
    def test_empty_raises(self, bad):
        with pytest.raises(PayloadError, match="empty timestamp"):
            parse_eastern(bad)

    @pytest.mark.parametrize(
        "bad", ["not a time", "2026-13-01T00:00:00", "2026-08-11T25:00:00", "16:14:59"]
    )
    def test_unparseable_raises(self, bad):
        with pytest.raises(PayloadError):
            parse_eastern(bad)

    def test_compact_iso_basic_format_is_accepted(self):
        """`fromisoformat` accepts "20260811" as a bare date, and we allow it.

        Pinned rather than fought: a date with no time would land at midnight,
        which the staleness gate rejects on its own. Tightening here would add
        a second place that decides what counts as usable.
        """
        assert parse_eastern("20260811").hour == 0

    def test_already_aware_timestamp_is_rejected(self):
        """An offset in the string means the feed changed shape.

        Silently accepting it would attach Eastern on top of an existing zone,
        or shift the value. Better to stop and look.
        """
        with pytest.raises(PayloadError, match="aware"):
            parse_eastern("2026-08-11T16:14:59+00:00")


class TestParseNominal:
    def test_lifts_the_documented_fields(self, body):
        snap = parse(body, ticker="_SPX")
        assert snap.ticker == "_SPX"
        assert snap.spot == 7741.3999
        assert snap.seqno == body["data"]["seqno"]
        assert snap.feed_timestamp == body["timestamp"]

    def test_index_last_trade_is_aware_eastern(self, body):
        snap = parse(body, ticker="_SPX")
        assert snap.index_last_trade == datetime(
            2026, 8, 11, 11, 56, 26, tzinfo=ZoneInfo("America/New_York")
        )

    def test_raw_is_the_payload_itself_not_a_copy(self, body):
        """The archive writer persists exactly what was validated.

        If `raw` were a copy, a future change could validate one object and
        write another.
        """
        assert parse(body, ticker="_SPX").raw is body

    def test_ticker_is_taken_from_the_argument_not_the_payload(self, body):
        """The payload says "^SPX"; the request form is "_SPX". They do not
        round-trip, so the requested form is what gets recorded."""
        assert body["data"]["symbol"] == "^SPX"
        assert parse(body, ticker="_SPX").ticker == "_SPX"

    def test_options_are_carried_through(self, trimmed_chain):
        snap = parse(trimmed_chain, ticker="_SPX")
        assert snap.n_options == len(trimmed_chain["data"]["options"])
        assert snap.n_options == 674

    def test_empty_option_list_is_valid(self, body):
        """Header fixtures have none. An empty chain is a data question for the
        curation layer, not a parse error."""
        assert parse(body, ticker="_SPX").n_options == 0


class TestParseFailures:
    def test_non_dict_payload(self):
        with pytest.raises(PayloadError, match="not an object"):
            parse([], ticker="_SPX")

    def test_missing_data_object(self):
        with pytest.raises(PayloadError, match="no 'data' object"):
            parse({"timestamp": "2026-08-11 20:36:41"}, ticker="_SPX")

    def test_missing_top_level_timestamp(self, body):
        del body["timestamp"]
        with pytest.raises(PayloadError, match="no top-level 'timestamp'"):
            parse(body, ticker="_SPX")

    def test_empty_top_level_timestamp(self, body):
        body["timestamp"] = ""
        with pytest.raises(PayloadError, match="no top-level 'timestamp'"):
            parse(body, ticker="_SPX")

    def test_missing_last_trade_time_is_fatal(self, body):
        """The regression that matters most.

        Without this field the freshness gate cannot run. Defaulting it would
        let an ungatable snapshot into a permanent archive -- exactly the
        failure this component exists to prevent.
        """
        del body["data"]["last_trade_time"]
        with pytest.raises(PayloadError, match="empty timestamp"):
            parse(body, ticker="_SPX")

    def test_unparseable_last_trade_time_is_fatal(self, body):
        body["data"]["last_trade_time"] = "yesterday"
        with pytest.raises(PayloadError):
            parse(body, ticker="_SPX")

    @pytest.mark.parametrize("bad", [None, 0, -1, "7728.2", ""])
    def test_unusable_current_price(self, body, bad):
        body["data"]["current_price"] = bad
        with pytest.raises(PayloadError, match="current_price"):
            parse(body, ticker="_SPX")

    @pytest.mark.parametrize("bad", [None, "16938026517", 1.5])
    def test_unusable_seqno(self, body, bad):
        body["data"]["seqno"] = bad
        with pytest.raises(PayloadError, match="seqno"):
            parse(body, ticker="_SPX")

    def test_missing_options_list(self, body):
        del body["data"]["options"]
        with pytest.raises(PayloadError, match="no 'options' list"):
            parse(body, ticker="_SPX")

    def test_options_not_a_list(self, body):
        body["data"]["options"] = {}
        with pytest.raises(PayloadError, match="no 'options' list"):
            parse(body, ticker="_SPX")

    def test_nothing_is_defaulted_silently(self, body):
        """Every required field must fail loudly when absent.

        Enumerated as one test so adding a required field without adding a
        failure path is visible.
        """
        required = ["last_trade_time", "current_price", "seqno", "options"]
        for field in required:
            corrupted = copy.deepcopy(body)
            del corrupted["data"][field]
            with pytest.raises(PayloadError):
                parse(corrupted, ticker="_SPX")


class TestSnapshotValueSemantics:
    def test_snapshot_is_frozen(self, body):
        snap = parse(body, ticker="_SPX")
        with pytest.raises(AttributeError):
            snap.spot = 1.0  # type: ignore[misc]


class TestAgainstEveryCapturedHeader:
    def test_all_nine_headers_parse(self, headers):
        for name, raw in headers.items():
            snap = parse(raw, ticker="_SPX")
            assert snap.feed_timestamp, name
            assert snap.spot > 0, name

    def test_feed_timestamp_is_never_the_index_print(self, headers):
        for name, raw in headers.items():
            snap = parse(raw, ticker="_SPX")
            assert snap.feed_timestamp != raw["data"]["last_trade_time"], name

    def test_full_chain_parses(self, full_chain):
        snap = parse(full_chain, ticker="_SPX")
        assert snap.n_options == 30_692
        assert snap.spot == 7728.2002
