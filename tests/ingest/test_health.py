"""Unit tests for `spxrnd.ingest.health`.

Telemetry is worth strictly less than the data it reports on, so nothing here
may raise. Every corruption case must degrade to a blank record rather than
stopping a capture.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from spxrnd.ingest.health import Health, read_health, record

from ..conftest import utc

T0 = utc("2026-08-11T16:11:33")
T1 = utc("2026-08-11T16:21:36")


@pytest.fixture
def path(tmp_path):
    return tmp_path / "health.json"


class TestBlankAndMissing:
    def test_missing_file_reads_blank(self, path):
        assert read_health(path) == Health()

    def test_blank_record_has_no_timestamps_and_zero_counters(self):
        h = Health()
        assert (h.last_attempt, h.last_ok, h.last_write, h.last_error) == (None,) * 4
        assert (h.total_writes, h.total_skips, h.total_failures) == (0, 0, 0)
        assert h.consecutive_failures == 0


class TestRoundTrip:
    def test_a_written_record_reads_back_identically(self, path):
        """The regression that motivated the declared-type table.

        An earlier implementation compared each value against the type of its
        *default*; since every string field defaults to None, it silently
        discarded every timestamp it had just written.
        """
        written = record(path, now=T0, verdict="accept", written=True)
        assert read_health(path) == written

    def test_all_string_fields_survive_a_round_trip(self, path):
        record(path, now=T0, verdict="accept", written=True)
        got = read_health(path)
        assert got.last_attempt == T0.isoformat(timespec="seconds")
        assert got.last_ok == T0.isoformat(timespec="seconds")
        assert got.last_write == T0.isoformat(timespec="seconds")
        assert got.last_verdict == "accept"

    def test_the_file_is_human_readable(self, path):
        record(path, now=T0, verdict="accept", written=True)
        assert json.loads(path.read_text())["last_verdict"] == "accept"


class TestRecordingSuccess:
    def test_a_write_advances_all_three_clocks(self, path):
        h = record(path, now=T0, verdict="accept", written=True)
        stamp = T0.isoformat(timespec="seconds")
        assert h.last_attempt == h.last_ok == h.last_write == stamp

    def test_a_skip_advances_attempt_and_ok_but_not_write(self, path):
        """The distinction that makes the record diagnosable.

        A skip proves the scheduler ran and the feed answered. Only last_write
        is legitimately old every weekend, so it alone proves nothing.
        """
        record(path, now=T0, verdict="accept", written=True)
        h = record(path, now=T1, verdict="duplicate_print", written=False)
        assert h.last_attempt == h.last_ok == T1.isoformat(timespec="seconds")
        assert h.last_write == T0.isoformat(timespec="seconds")

    def test_counters_accumulate(self, path):
        record(path, now=T0, verdict="accept", written=True)
        record(path, now=T1, verdict="stale_feed", written=False)
        record(path, now=T1, verdict="stale_feed", written=False)
        h = read_health(path)
        assert (h.total_writes, h.total_skips, h.total_failures) == (1, 2, 0)

    def test_success_clears_last_error(self, path):
        record(path, now=T0, verdict="FetchError", error="boom")
        h = record(path, now=T1, verdict="accept", written=True)
        assert h.last_error is None


class TestRecordingFailure:
    def test_failure_increments_both_counters(self, path):
        h = record(path, now=T0, verdict="FetchError", error="boom")
        assert h.consecutive_failures == 1
        assert h.total_failures == 1

    def test_failures_streak(self, path):
        for _ in range(3):
            h = record(path, now=T0, verdict="FetchError", error="boom")
        assert h.consecutive_failures == 3

    def test_failure_records_the_message(self, path):
        h = record(path, now=T0, verdict="FetchError", error="connection reset")
        assert h.last_error == "connection reset"

    def test_failure_advances_attempt_but_not_ok(self, path):
        """If last_attempt is fresh and last_ok is old, the scheduler works and
        the network does not -- a different problem from the reverse."""
        h = record(path, now=T0, verdict="FetchError", error="boom")
        assert h.last_attempt == T0.isoformat(timespec="seconds")
        assert h.last_ok is None

    def test_a_skip_clears_the_failure_streak(self, path):
        """A skipped capture still proves fetch and parse both work."""
        record(path, now=T0, verdict="FetchError", error="boom")
        record(path, now=T0, verdict="FetchError", error="boom")
        h = record(path, now=T1, verdict="stale_feed", written=False)
        assert h.consecutive_failures == 0
        assert h.total_failures == 2, "the running total must not be reset"


class TestCorruption:
    @pytest.mark.parametrize(
        "junk", ["", "{", "not json", "[]", '"a string"', "null", "42"]
    )
    def test_corrupt_files_read_blank_instead_of_raising(self, path, junk):
        path.write_text(junk)
        assert read_health(path) == Health()

    def test_a_directory_in_place_of_the_file(self, path):
        path.mkdir()
        assert read_health(path) == Health()

    def test_unknown_keys_are_ignored(self, path):
        path.write_text(json.dumps({"total_writes": 5, "invented_field": "x"}))
        assert read_health(path).total_writes == 5

    @pytest.mark.parametrize(
        ("field", "junk"),
        [
            ("total_writes", "five"),
            ("consecutive_failures", None),
            ("last_verdict", 42),
            ("last_ok", []),
            ("total_skips", {"a": 1}),
        ],
    )
    def test_wrongly_typed_fields_fall_back_to_the_default(self, path, field, junk):
        path.write_text(json.dumps({field: junk}))
        assert getattr(read_health(path), field) == getattr(Health(), field)

    def test_a_boolean_is_not_a_counter(self, path):
        """`isinstance(True, int)` is True in Python. A JSON `true` in a
        counter is corruption, not a count of one."""
        path.write_text(json.dumps({"total_writes": True}))
        assert read_health(path).total_writes == 0

    def test_recording_over_a_corrupt_file_recovers(self, path):
        path.write_text("garbage")
        h = record(path, now=T0, verdict="accept", written=True)
        assert h.total_writes == 1
        assert read_health(path) == h


class TestStaleBy:
    def test_reports_elapsed_time(self):
        h = Health(last_ok=T0.isoformat(timespec="seconds"))
        assert h.stale_by(T0 + timedelta(minutes=30)) == timedelta(minutes=30)

    def test_none_when_it_has_never_happened(self):
        assert Health().stale_by(T0) is None

    def test_field_is_selectable(self):
        h = Health(last_write=T0.isoformat(timespec="seconds"))
        assert h.stale_by(T0 + timedelta(hours=2), field="last_write") == timedelta(
            hours=2
        )
        assert h.stale_by(T0, field="last_ok") is None
