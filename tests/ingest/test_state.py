"""Unit tests for `spxrnd.ingest.state`.

Durability under interruption. Every test here is about what a reader sees when
the writer was killed mid-flight.
"""

from __future__ import annotations

import json
import os
import time
from datetime import timedelta

import pytest

from spxrnd.ingest.state import (
    DEFAULT_LOCK_TIMEOUT,
    CollectorState,
    collector_lock,
    read_state,
    write_atomic,
    write_state,
)

SAMPLE = CollectorState(
    index_last_trade="2026-08-11T16:14:59",
    seqno=16938026517,
    capture_utc="2026-08-11T20-37-03Z",
    spot=7728.2002,
)


class TestWriteAtomic:
    def test_writes_the_bytes(self, tmp_path):
        target = tmp_path / "f.bin"
        write_atomic(target, b"hello")
        assert target.read_bytes() == b"hello"

    def test_overwrites_an_existing_file(self, tmp_path):
        target = tmp_path / "f.bin"
        target.write_bytes(b"old")
        write_atomic(target, b"new")
        assert target.read_bytes() == b"new"

    def test_creates_missing_parent_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.bin"
        write_atomic(target, b"x")
        assert target.read_bytes() == b"x"

    def test_leaves_no_temp_file_behind(self, tmp_path):
        write_atomic(tmp_path / "f.bin", b"x")
        assert [p.name for p in tmp_path.iterdir()] == ["f.bin"]

    def test_temp_file_is_removed_when_the_write_fails(self, tmp_path, monkeypatch):
        """A failed write must not leave debris that a later run trips over."""
        target = tmp_path / "f.bin"

        def boom(fd):
            raise OSError("disk full")

        monkeypatch.setattr(os, "fsync", boom)
        with pytest.raises(OSError, match="disk full"):
            write_atomic(target, b"x")
        assert list(tmp_path.iterdir()) == []

    def test_original_survives_a_failed_overwrite(self, tmp_path, monkeypatch):
        """The point of the temp-and-rename dance.

        A failure partway through must leave the previous version intact, not a
        truncated file that parses as valid but wrong.
        """
        target = tmp_path / "f.bin"
        write_atomic(target, b"original")
        monkeypatch.setattr(
            os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("nope"))
        )
        with pytest.raises(OSError):
            write_atomic(target, b"replacement")
        assert target.read_bytes() == b"original"

    def test_empty_payload_is_allowed(self, tmp_path):
        write_atomic(tmp_path / "f.bin", b"")
        assert (tmp_path / "f.bin").read_bytes() == b""

    def test_temp_name_is_process_scoped(self, tmp_path, monkeypatch):
        """Two processes writing the same path must not share a temp file."""
        seen = []
        real_replace = os.replace
        monkeypatch.setattr(
            os,
            "replace",
            lambda src, dst: (seen.append(str(src)), real_replace(src, dst))[1],
        )
        write_atomic(tmp_path / "f.bin", b"x")
        assert str(os.getpid()) in seen[0]


class TestReadWriteState:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "state.json"
        write_state(path, SAMPLE)
        assert read_state(path) == SAMPLE

    def test_missing_file_is_an_empty_state(self, tmp_path):
        state = read_state(tmp_path / "nope.json")
        assert state == CollectorState()
        assert state.is_empty

    def test_written_file_is_readable_json(self, tmp_path):
        """Operators read this file by hand when diagnosing a stuck collector."""
        path = tmp_path / "state.json"
        write_state(path, SAMPLE)
        assert json.loads(path.read_text())["seqno"] == 16938026517

    @pytest.mark.parametrize(
        "junk", ["", "{", "not json", "[]", '"a string"', "null", "123"]
    )
    def test_corrupt_state_reads_as_empty_rather_than_raising(self, tmp_path, junk):
        """Refusing to run costs a permanent gap; re-capturing costs a duplicate
        row the gate then drops. The asymmetry decides this."""
        path = tmp_path / "state.json"
        path.write_text(junk)
        assert read_state(path) == CollectorState()

    def test_unknown_keys_from_a_future_version_are_ignored(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps({"index_last_trade": "2026-08-11T16:14:59", "something_new": 42})
        )
        assert read_state(path).index_last_trade == "2026-08-11T16:14:59"

    @pytest.mark.parametrize(
        ("field", "junk"),
        [
            ("index_last_trade", 12345),
            ("seqno", "not an int"),
            ("capture_utc", []),
            ("spot", "7728.2"),
        ],
    )
    def test_wrongly_typed_fields_fall_back_to_none(self, tmp_path, field, junk):
        path = tmp_path / "state.json"
        path.write_text(json.dumps({field: junk}))
        assert getattr(read_state(path), field) is None

    def test_a_directory_where_the_state_file_should_be(self, tmp_path):
        """Not contrived: `mkdir -p` on the wrong path produces exactly this."""
        path = tmp_path / "state.json"
        path.mkdir()
        assert read_state(path) == CollectorState()

    def test_is_empty_reflects_only_the_print(self, tmp_path):
        assert CollectorState().is_empty
        assert not CollectorState(index_last_trade="x").is_empty
        assert CollectorState(seqno=1).is_empty


class TestCollectorLock:
    def test_acquires_when_free(self, tmp_path):
        with collector_lock(tmp_path / "l") as got:
            assert got is True

    def test_lock_file_exists_while_held(self, tmp_path):
        path = tmp_path / "l"
        with collector_lock(path):
            assert path.exists()

    def test_lock_file_is_removed_on_exit(self, tmp_path):
        path = tmp_path / "l"
        with collector_lock(path):
            pass
        assert not path.exists()

    def test_lock_file_is_removed_even_on_exception(self, tmp_path):
        path = tmp_path / "l"
        with pytest.raises(RuntimeError), collector_lock(path):
            raise RuntimeError("boom")
        assert not path.exists()

    def test_second_holder_is_refused(self, tmp_path):
        path = tmp_path / "l"
        with collector_lock(path), collector_lock(path) as second:
            assert second is False

    def test_refusal_is_not_an_error(self, tmp_path):
        """Overlapping scheduler runs are expected. Yielding False rather than
        raising keeps them out of the failure logs."""
        path = tmp_path / "l"
        with collector_lock(path), collector_lock(path) as second:
            assert second is False  # no exception raised

    def test_lock_records_the_holding_pid(self, tmp_path):
        """A stale lock found in three months should identify its owner."""
        path = tmp_path / "l"
        with collector_lock(path):
            assert path.read_text().strip() == str(os.getpid())

    def test_abandoned_lock_is_reclaimed(self, tmp_path):
        path = tmp_path / "l"
        path.write_text("99999\n")
        ancient = os.stat(path).st_mtime - DEFAULT_LOCK_TIMEOUT.total_seconds() - 60
        os.utime(path, (ancient, ancient))
        with collector_lock(path) as got:
            assert got is True

    def test_fresh_lock_is_not_reclaimed(self, tmp_path):
        path = tmp_path / "l"
        path.write_text("99999\n")
        with collector_lock(path) as got:
            assert got is False

    def test_reclaim_threshold_is_configurable(self, tmp_path):
        path = tmp_path / "l"
        path.write_text("99999\n")
        old = os.stat(path).st_mtime - 10
        os.utime(path, (old, old))
        with collector_lock(path, stale_after=timedelta(seconds=5)) as got:
            assert got is True

    def test_creates_missing_parent_directories(self, tmp_path):
        with collector_lock(tmp_path / "a" / "b" / "l") as got:
            assert got is True

    def test_a_refused_lock_does_not_delete_the_holders_file(self, tmp_path):
        """The bug this guards: releasing a lock we never acquired."""
        path = tmp_path / "l"
        with collector_lock(path):
            with collector_lock(path) as second:
                assert second is False
            assert path.exists(), "the inner refusal must not remove the outer lock"


class TestLockRaces:
    """The narrow windows between checking the lock and acting on it.

    Two collectors on a ten-minute schedule will not hit these often, but "not
    often" over years of unattended running is "eventually", and the failure
    mode is a permanently stuck collector.
    """

    def test_lock_released_between_our_open_and_our_stat(self, tmp_path, monkeypatch):
        """Held when we tried to create it, gone when we tried to age it.

        The holder finished in the microseconds between. Their lock is free,
        so we should take it rather than reporting a spurious refusal.
        """
        path = tmp_path / "l"
        path.write_text("99999\n")
        real_open = os.open
        calls = {"n": 0}

        def flaky(p, flags, mode=0o777):
            calls["n"] += 1
            if calls["n"] == 1:
                os.unlink(p)  # the holder releases, right here
                raise FileExistsError
            return real_open(p, flags, mode)

        monkeypatch.setattr(os, "open", flaky)
        with collector_lock(path) as got:
            assert got is True
        assert calls["n"] == 2, "must retry exactly once, not spin"

    def test_lock_retaken_between_our_open_and_our_stat(self, tmp_path, monkeypatch):
        """Same window, but a third collector got there first. Not ours."""
        from pathlib import Path

        path = tmp_path / "l"
        path.write_text("99999\n")
        real_stat = Path.stat

        def vanished(self, *a, **k):
            # Scoped to the lock file: `collector_lock` also stats directories
            # on its way in, and failing those tests nothing.
            if self == path:
                raise FileNotFoundError
            return real_stat(self, *a, **k)

        monkeypatch.setattr(Path, "stat", vanished)
        with collector_lock(path) as got:
            assert got is False
        # os.path.exists, not path.exists() -- the latter routes through the
        # very Path.stat this test has replaced.
        assert os.path.exists(path), "a lock we never held must not be removed"

    def test_abandoned_lock_reclaimed_by_someone_else_first(
        self, tmp_path, monkeypatch
    ):
        """We judged it abandoned and removed it; another collector recreated it
        before we could. Theirs, and we must not steal it back."""
        path = tmp_path / "l"
        path.write_text("99999\n")
        ancient = time.time() - DEFAULT_LOCK_TIMEOUT.total_seconds() - 60
        os.utime(path, (ancient, ancient))

        def always_taken(*a, **k):
            raise FileExistsError

        monkeypatch.setattr(os, "open", always_taken)
        with collector_lock(path) as got:
            assert got is False
