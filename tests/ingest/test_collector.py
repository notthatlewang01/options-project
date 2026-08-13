"""Unit tests for `spxrnd.ingest.collector`.

Orchestration, driven entirely offline: `collect_once(payload=...)` replays a
real captured body through fetch-free but otherwise complete machinery.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta

import pytest

from spxrnd.ingest import health, state
from spxrnd.ingest.collector import archive_path, collect_once
from spxrnd.ingest.errors import PayloadError
from spxrnd.ingest.freshness import Verdict

from ..conftest import load_header, utc

CLOSE = "2026-08-11T20-37-03Z"
CLOSE_NOW = utc("2026-08-11T20:37:03")
HEALTHY = "2026-08-11T16-11-33Z"
HEALTHY_NOW = utc("2026-08-11T16:11:33")


def capture(tmp_path, name, now, **kw):
    return collect_once(tmp_path, payload=load_header(name), now=now, **kw)


class TestArchivePath:
    def test_layout(self, tmp_path):
        got = archive_path(tmp_path, "_SPX", CLOSE_NOW)
        assert got == tmp_path / "raw" / "_SPX_2026-08-11T20-37-03Z.json.gz"

    def test_no_colons_in_the_name(self, tmp_path):
        assert ":" not in archive_path(tmp_path, "_SPX", CLOSE_NOW).name

    def test_deterministic(self, tmp_path):
        assert archive_path(tmp_path, "_SPX", CLOSE_NOW) == archive_path(
            tmp_path, "_SPX", CLOSE_NOW
        )

    def test_distinct_times_and_tickers_do_not_collide(self, tmp_path):
        a = archive_path(tmp_path, "_SPX", CLOSE_NOW)
        assert a != archive_path(tmp_path, "_SPX", HEALTHY_NOW)
        assert a != archive_path(tmp_path, "SPY", CLOSE_NOW)

    def test_non_utc_input_is_normalised(self):
        """A filename whose meaning depends on the machine's timezone sorts
        wrong the moment the collector moves."""
        from zoneinfo import ZoneInfo

        eastern = CLOSE_NOW.astimezone(ZoneInfo("America/New_York"))
        assert archive_path("/d", "_SPX", eastern) == archive_path(
            "/d", "_SPX", CLOSE_NOW
        )

    def test_names_sort_chronologically(self):
        """ISO-with-hyphens sorts lexicographically in time order, which is what
        makes `sorted(glob(...))` a valid timeline everywhere else."""
        names = [
            archive_path("/d", "_SPX", CLOSE_NOW + timedelta(minutes=m)).name
            for m in (0, 10, 100, 1000)
        ]
        assert names == sorted(names)


class TestAcceptedCapture:
    def test_writes_the_archive_entry(self, tmp_path):
        result = capture(tmp_path, HEALTHY, HEALTHY_NOW)
        assert result.written
        assert result.raw_path.exists()

    def test_archive_entry_is_gzipped_json_matching_the_payload(self, tmp_path):
        result = capture(tmp_path, HEALTHY, HEALTHY_NOW)
        with gzip.open(result.raw_path, "rt") as f:
            assert json.load(f) == load_header(HEALTHY)

    def test_archive_entry_is_reproducible(self, tmp_path):
        """mtime=0 in the gzip header, so identical input gives identical bytes
        and a re-run does not look like a change."""
        a = capture(tmp_path / "a", HEALTHY, HEALTHY_NOW).raw_path.read_bytes()
        b = capture(tmp_path / "b", HEALTHY, HEALTHY_NOW).raw_path.read_bytes()
        assert a == b

    def test_records_state_for_the_next_run(self, tmp_path):
        capture(tmp_path, HEALTHY, HEALTHY_NOW)
        saved = state.read_state(tmp_path / "state.json")
        assert saved.index_last_trade == "2026-08-11T11:56:26"
        assert saved.spot == 7741.3999
        assert saved.capture_utc == "2026-08-11T16-11-33Z"

    def test_decision_is_accept(self, tmp_path):
        assert (
            capture(tmp_path, HEALTHY, HEALTHY_NOW).decision.verdict is Verdict.ACCEPT
        )

    def test_snapshot_is_returned(self, tmp_path):
        assert capture(tmp_path, HEALTHY, HEALTHY_NOW).snapshot.spot == 7741.3999

    def test_leaves_no_lock_behind(self, tmp_path):
        capture(tmp_path, HEALTHY, HEALTHY_NOW)
        assert not (tmp_path / ".collect.lock").exists()


class TestSkippedCapture:
    def test_duplicate_writes_nothing(self, tmp_path):
        capture(tmp_path, CLOSE, CLOSE_NOW)
        second = capture(tmp_path, "2026-08-11T20-47-05Z", utc("2026-08-11T20:47:05"))
        assert not second.written
        assert second.decision.verdict is Verdict.DUPLICATE_PRINT

    def test_only_one_archive_entry_after_a_duplicate(self, tmp_path):
        capture(tmp_path, CLOSE, CLOSE_NOW)
        capture(tmp_path, "2026-08-11T20-47-05Z", utc("2026-08-11T20:47:05"))
        assert len(list((tmp_path / "raw").iterdir())) == 1

    def test_stale_writes_nothing(self, tmp_path):
        result = capture(tmp_path, "2026-08-09T22-38-43Z", utc("2026-08-09T22:38:43"))
        assert not result.written
        assert result.decision.verdict is Verdict.STALE_FEED

    def test_a_skip_does_not_advance_state(self, tmp_path):
        """Advancing on a skip would make the *next* capture look like a
        duplicate of something never archived."""
        capture(tmp_path, CLOSE, CLOSE_NOW)
        before = state.read_state(tmp_path / "state.json")
        capture(tmp_path, "2026-08-11T20-47-05Z", utc("2026-08-11T20:47:05"))
        assert state.read_state(tmp_path / "state.json") == before

    def test_a_skip_still_returns_the_snapshot(self, tmp_path):
        """So a caller can log what it saw and chose not to keep."""
        capture(tmp_path, CLOSE, CLOSE_NOW)
        second = capture(tmp_path, "2026-08-11T20-47-05Z", utc("2026-08-11T20:47:05"))
        assert second.snapshot is not None


class TestForce:
    def test_force_writes_a_duplicate(self, tmp_path):
        capture(tmp_path, CLOSE, CLOSE_NOW)
        forced = capture(
            tmp_path, "2026-08-11T20-47-05Z", utc("2026-08-11T20:47:05"), force=True
        )
        assert forced.written
        assert forced.decision.verdict is Verdict.FORCED

    def test_force_writes_a_stale_capture(self, tmp_path):
        result = capture(
            tmp_path, "2026-08-09T22-38-43Z", utc("2026-08-09T22:38:43"), force=True
        )
        assert result.written


class TestOrdering:
    def test_state_is_written_only_after_the_archive_entry(self, tmp_path, monkeypatch):
        """The invariant that decides what a crash costs.

        Kill the process between the archive write and the state write: the
        capture is on disk and state is untouched, so the next run re-captures
        something we already hold. Harmless -- the gate drops it. The reverse
        order would advance past a capture that was never persisted and leave a
        hole nothing can fill.
        """

        def die(*a, **k):
            raise KeyboardInterrupt("killed between the two writes")

        monkeypatch.setattr("spxrnd.ingest.collector.state.write_state", die)
        with pytest.raises(KeyboardInterrupt):
            capture(tmp_path, HEALTHY, HEALTHY_NOW)

        assert list((tmp_path / "raw").iterdir()), "archive entry must survive"
        assert state.read_state(tmp_path / "state.json").is_empty, (
            "state must not have advanced"
        )

    def test_recovery_after_that_crash_is_a_harmless_duplicate(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "spxrnd.ingest.collector.state.write_state",
            lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            capture(tmp_path, HEALTHY, HEALTHY_NOW)
        monkeypatch.undo()

        # Same capture again: accepted (state never advanced), same path, so it
        # overwrites rather than duplicating.
        again = capture(tmp_path, HEALTHY, HEALTHY_NOW)
        assert again.written
        assert len(list((tmp_path / "raw").iterdir())) == 1

    def test_lock_is_released_after_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "spxrnd.ingest.collector.state.write_state",
            lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            capture(tmp_path, HEALTHY, HEALTHY_NOW)
        assert not (tmp_path / ".collect.lock").exists()


class TestLocking:
    def test_refuses_to_run_while_another_collector_holds_the_lock(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        with state.collector_lock(tmp_path / ".collect.lock"):
            result = capture(tmp_path, HEALTHY, HEALTHY_NOW)
        assert not result.written
        assert result.decision.verdict is Verdict.SKIPPED_LOCKED

    def test_a_locked_run_writes_nothing_at_all(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        with state.collector_lock(tmp_path / ".collect.lock"):
            capture(tmp_path, HEALTHY, HEALTHY_NOW)
        assert not (tmp_path / "raw").exists()
        assert not (tmp_path / "state.json").exists()

    def test_a_locked_run_does_not_touch_health(self, tmp_path):
        """It is not the run that should record an outcome for this tick."""
        tmp_path.mkdir(exist_ok=True)
        with state.collector_lock(tmp_path / ".collect.lock"):
            capture(tmp_path, HEALTHY, HEALTHY_NOW)
        assert health.read_health(tmp_path / health.HEALTH_FILENAME) == health.Health()


class TestErrors:
    def test_a_bad_payload_raises(self, tmp_path):
        with pytest.raises(PayloadError):
            collect_once(tmp_path, payload={"nope": True}, now=HEALTHY_NOW)

    def test_a_bad_payload_writes_no_archive_entry(self, tmp_path):
        with pytest.raises(PayloadError):
            collect_once(tmp_path, payload={"nope": True}, now=HEALTHY_NOW)
        assert not (tmp_path / "raw").exists()

    def test_a_bad_payload_is_recorded_in_health(self, tmp_path):
        with pytest.raises(PayloadError):
            collect_once(tmp_path, payload={"nope": True}, now=HEALTHY_NOW)
        record = health.read_health(tmp_path / health.HEALTH_FILENAME)
        assert record.consecutive_failures == 1
        assert record.last_verdict == "PayloadError"
        assert record.last_error

    def test_the_lock_is_released_after_a_bad_payload(self, tmp_path):
        with pytest.raises(PayloadError):
            collect_once(tmp_path, payload={"nope": True}, now=HEALTHY_NOW)
        assert not (tmp_path / ".collect.lock").exists()


class TestHealthRecording:
    def test_a_write_is_recorded(self, tmp_path):
        capture(tmp_path, HEALTHY, HEALTHY_NOW)
        record = health.read_health(tmp_path / health.HEALTH_FILENAME)
        assert record.total_writes == 1
        assert record.last_write == HEALTHY_NOW.isoformat(timespec="seconds")
        assert record.consecutive_failures == 0

    def test_a_skip_is_recorded_as_a_skip_not_a_failure(self, tmp_path):
        capture(tmp_path, CLOSE, CLOSE_NOW)
        capture(tmp_path, "2026-08-11T20-47-05Z", utc("2026-08-11T20:47:05"))
        record = health.read_health(tmp_path / health.HEALTH_FILENAME)
        assert record.total_skips == 1
        assert record.total_failures == 0
        assert record.consecutive_failures == 0

    def test_a_successful_skip_clears_a_failure_streak(self, tmp_path):
        """A skipped capture still proves fetch and parse work."""
        with pytest.raises(PayloadError):
            collect_once(tmp_path, payload={"nope": True}, now=HEALTHY_NOW)
        capture(tmp_path, "2026-08-09T22-38-43Z", utc("2026-08-09T22:38:43"))
        assert (
            health.read_health(tmp_path / health.HEALTH_FILENAME).consecutive_failures
            == 0
        )


class TestDefaults:
    def test_now_defaults_to_the_real_clock(self, tmp_path):
        """Every captured payload is long stale against today, so the default
        clock produces a skip -- which is exactly what proves it was used."""
        result = collect_once(tmp_path, payload=load_header(HEALTHY))
        assert result.decision.verdict is Verdict.STALE_FEED
        assert result.decision.age > timedelta(days=1)

    def test_relative_data_dir_is_resolved(self, tmp_path, monkeypatch):
        """A scheduled job does not inherit a working directory."""
        monkeypatch.chdir(tmp_path)
        result = collect_once("data", payload=load_header(HEALTHY), now=HEALTHY_NOW)
        assert result.raw_path.is_absolute()
        assert result.raw_path.exists()

    def test_ticker_defaults_to_spx(self, tmp_path):
        assert "_SPX_" in capture(tmp_path, HEALTHY, HEALTHY_NOW).raw_path.name

    def test_ticker_is_honoured(self, tmp_path):
        result = capture(tmp_path, HEALTHY, HEALTHY_NOW, ticker="SPY")
        assert result.raw_path.name.startswith("SPY_")
        assert result.snapshot.ticker == "SPY"


class TestFullSessionReplay:
    """Replay the captured timeline through the real collector, in order.

    End-to-end proof that the component would have produced the right archive
    from the payloads we actually received.
    """

    TIMELINE = [
        ("2026-08-09T22-38-43Z", "2026-08-09T22:38:43"),
        ("2026-08-10T23-57-37Z", "2026-08-10T23:57:37"),
        ("2026-08-11T00-07-40Z", "2026-08-11T00:07:40"),
        ("2026-08-11T16-11-33Z", "2026-08-11T16:11:33"),
        ("2026-08-11T20-37-03Z", "2026-08-11T20:37:03"),
        ("2026-08-11T20-47-05Z", "2026-08-11T20:47:05"),
        ("2026-08-11T20-57-15Z", "2026-08-11T20:57:15"),
        ("2026-08-11T21-07-17Z", "2026-08-11T21:07:17"),
        ("2026-08-11T23-57-57Z", "2026-08-11T23:57:57"),
    ]

    def replay(self, tmp_path):
        return [capture(tmp_path, name, utc(now)) for name, now in self.TIMELINE]

    def test_exactly_two_captures_are_archived(self, tmp_path):
        results = self.replay(tmp_path)
        written = [r.raw_path.name for r in results if r.written]
        assert written == [
            "_SPX_2026-08-11T16-11-33Z.json.gz",
            "_SPX_2026-08-11T20-37-03Z.json.gz",
        ]

    def test_the_settlement_capture_survives(self, tmp_path):
        """The one the inherited 22-minute threshold discarded."""
        self.replay(tmp_path)
        assert (tmp_path / "raw" / "_SPX_2026-08-11T20-37-03Z.json.gz").exists()

    def test_health_totals_add_up(self, tmp_path):
        self.replay(tmp_path)
        record = health.read_health(tmp_path / health.HEALTH_FILENAME)
        assert record.total_writes == 2
        assert record.total_skips == 7
        assert record.total_failures == 0

    def test_replaying_the_whole_timeline_twice_is_idempotent(self, tmp_path):
        self.replay(tmp_path)
        first = sorted(p.name for p in (tmp_path / "raw").iterdir())
        self.replay(tmp_path)
        assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == first

    def test_compression_ratio_is_worth_it(self, tmp_path):
        """Header fixtures are tiny; use the full chain for a real measurement."""
        from ..conftest import FIXTURES

        with gzip.open(FIXTURES / "chain_full_close.json.gz", "rt") as f:
            body = json.load(f)
        result = collect_once(tmp_path, payload=body, now=datetime.now(UTC), force=True)
        raw_bytes = len(json.dumps(body).encode())
        assert result.raw_path.stat().st_size < raw_bytes / 5
