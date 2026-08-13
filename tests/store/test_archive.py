"""Unit tests for `spxrnd.store.archive`.

The archive cannot be rebuilt. Every test here is about not losing it.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime

import pytest

from spxrnd.store import archive

from ..conftest import load_header

PAYLOAD = load_header("2026-08-11T20-37-03Z")


@pytest.fixture
def raw_dir(tmp_path):
    """A miniature archive: two uncompressed captures and one gzipped."""
    d = tmp_path / "raw"
    d.mkdir()
    (d / "_SPX_2026-08-11T16-11-33Z.json").write_text(json.dumps(PAYLOAD))
    (d / "_SPX_2026-08-11T20-37-03Z.json").write_text(json.dumps(PAYLOAD))
    (d / "_SPX_2026-08-13T01-25-33Z.json.gz").write_bytes(
        gzip.compress(json.dumps(PAYLOAD).encode(), mtime=0)
    )
    return d


class TestParseName:
    def test_uncompressed(self, tmp_path):
        c = archive.parse_name(tmp_path / "_SPX_2026-08-11T20-37-03Z.json")
        assert c.ticker == "_SPX"
        assert c.captured_at == datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC)
        assert c.compressed is False

    def test_compressed(self, tmp_path):
        c = archive.parse_name(tmp_path / "_SPX_2026-08-11T20-37-03Z.json.gz")
        assert c.compressed is True
        assert c.captured_at == datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC)

    def test_stem_is_shared_with_the_curated_layer(self, tmp_path):
        """The identity that makes divergence a set difference."""
        a = archive.parse_name(tmp_path / "_SPX_2026-08-11T20-37-03Z.json")
        b = archive.parse_name(tmp_path / "_SPX_2026-08-11T20-37-03Z.json.gz")
        assert a.stem == b.stem == "_SPX_2026-08-11T20-37-03Z"

    def test_other_tickers(self, tmp_path):
        assert (
            archive.parse_name(tmp_path / "SPY_2026-08-11T20-37-03Z.json").ticker
            == "SPY"
        )

    @pytest.mark.parametrize(
        "name",
        [
            ".DS_Store",
            "meta.csv",
            "_SPX_2026-08-11T20-37-03Z.json.tmp",
            "_SPX_2026-08-11.json",
            "_SPX_2026-08-11T20:37:03Z.json",
            "notes.txt",
        ],
    )
    def test_non_captures_return_none(self, tmp_path, name):
        """`data/raw` accumulates other things. Ignore them, do not crash."""
        assert archive.parse_name(tmp_path / name) is None


class TestScan:
    def test_finds_every_capture(self, raw_dir):
        assert len(archive.scan(raw_dir)) == 3

    def test_sorted_oldest_first(self, raw_dir):
        stamps = [c.captured_at for c in archive.scan(raw_dir)]
        assert stamps == sorted(stamps)

    def test_mixed_compression_is_transparent(self, raw_dir):
        assert [c.compressed for c in archive.scan(raw_dir)] == [False, False, True]

    def test_ignores_non_captures(self, raw_dir):
        (raw_dir / ".DS_Store").write_text("junk")
        (raw_dir / "_SPX_2026-08-11T20-37-03Z.json.tmp").write_text("partial")
        assert len(archive.scan(raw_dir)) == 3

    def test_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert archive.scan(tmp_path / "nope") == []


class TestLoad:
    def test_reads_uncompressed(self, raw_dir):
        assert archive.load(raw_dir / "_SPX_2026-08-11T16-11-33Z.json") == PAYLOAD

    def test_reads_compressed(self, raw_dir):
        assert archive.load(raw_dir / "_SPX_2026-08-13T01-25-33Z.json.gz") == PAYLOAD

    def test_caller_need_not_know_which(self, raw_dir):
        assert all(archive.load(c.path) == PAYLOAD for c in archive.scan(raw_dir))


class TestCompress:
    def test_replaces_the_original(self, raw_dir):
        capture = archive.scan(raw_dir)[0]
        original = capture.path
        archive.compress(capture)
        assert not original.exists()
        assert original.with_name(original.name + ".gz").exists()

    def test_content_round_trips_exactly(self, raw_dir):
        capture = archive.scan(raw_dir)[0]
        before = capture.path.read_bytes()
        archive.compress(capture)
        gz = capture.path.with_name(capture.path.name + ".gz")
        with gzip.open(gz, "rb") as f:
            assert f.read() == before

    def test_reports_the_sizes(self, raw_dir):
        result = archive.compress(archive.scan(raw_dir)[0])
        assert result.after < result.before
        assert result.ratio > 1

    def test_already_compressed_is_skipped(self, raw_dir):
        compressed = [c for c in archive.scan(raw_dir) if c.compressed][0]
        result = archive.compress(compressed)
        assert result.skipped
        assert compressed.path.exists()

    def test_output_is_reproducible(self, tmp_path):
        """mtime=0, so recompressing the same bytes gives the same bytes."""
        outs = []
        for i in (1, 2):
            d = tmp_path / f"r{i}"
            d.mkdir()
            (d / "_SPX_2026-08-11T20-37-03Z.json").write_text(json.dumps(PAYLOAD))
            archive.compress(archive.scan(d)[0])
            outs.append((d / "_SPX_2026-08-11T20-37-03Z.json.gz").read_bytes())
        assert outs[0] == outs[1]

    def test_dry_run_changes_nothing(self, raw_dir):
        capture = archive.scan(raw_dir)[0]
        result = archive.compress(capture, dry_run=True)
        assert capture.path.exists(), "the original must survive a dry run"
        assert not capture.path.with_name(capture.path.name + ".gz").exists()
        assert result.after < result.before, "but it still reports the saving"


@pytest.fixture
def corrupting_compressor(monkeypatch):
    """Make compression produce bytes that will not read back equal.

    `real` is bound before patching: `archive.gzip` *is* the gzip module, so a
    lambda that called `gzip.compress` would call itself.
    """
    real = gzip.compress
    monkeypatch.setattr(
        archive.gzip, "compress", lambda *a, **k: real(b"wrong", mtime=0)
    )


class TestCompressSafety:
    """The verify-then-replace contract: no original is released until its
    replacement has been proved to reproduce it."""

    def test_original_survives_a_failed_round_trip(
        self, raw_dir, corrupting_compressor
    ):
        capture = archive.scan(raw_dir)[0]
        before = capture.path.read_bytes()
        with pytest.raises(OSError, match="round-trip mismatch"):
            archive.compress(capture)
        assert capture.path.exists()
        assert capture.path.read_bytes() == before

    def test_no_gz_is_left_after_a_failed_round_trip(
        self, raw_dir, corrupting_compressor
    ):
        capture = archive.scan(raw_dir)[0]
        with pytest.raises(OSError):
            archive.compress(capture)
        assert not capture.path.with_name(capture.path.name + ".gz").exists()

    def test_no_temp_file_is_left_behind(self, raw_dir, corrupting_compressor):
        capture = archive.scan(raw_dir)[0]
        with pytest.raises(OSError):
            archive.compress(capture)
        assert not any(p.name.endswith(".tmp") for p in raw_dir.iterdir())

    def test_the_error_says_the_original_is_safe(self, raw_dir, corrupting_compressor):
        """Whoever reads this at 2am needs to know nothing was destroyed."""
        capture = archive.scan(raw_dir)[0]
        with pytest.raises(OSError, match="has NOT been removed"):
            archive.compress(capture)

    def test_an_interrupted_run_is_resumable(self, raw_dir, monkeypatch):
        """Kill it partway: some captures converted, none lost, rerun finishes."""
        calls = {"n": 0}
        real = archive.compress

        def die_on_second(capture, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt
            return real(capture, **kw)

        monkeypatch.setattr(archive, "compress", die_on_second)
        with pytest.raises(KeyboardInterrupt):
            archive.compress_archive(raw_dir)

        monkeypatch.undo()
        assert len(archive.scan(raw_dir)) == 3, "no capture may be lost"
        archive.compress_archive(raw_dir)
        assert all(c.compressed for c in archive.scan(raw_dir))
        assert len(archive.scan(raw_dir)) == 3


class TestCompressArchive:
    def test_converts_everything_uncompressed(self, raw_dir):
        archive.compress_archive(raw_dir)
        assert all(c.compressed for c in archive.scan(raw_dir))

    def test_capture_count_is_unchanged(self, raw_dir):
        before = len(archive.scan(raw_dir))
        archive.compress_archive(raw_dir)
        assert len(archive.scan(raw_dir)) == before

    def test_all_payloads_still_load(self, raw_dir):
        archive.compress_archive(raw_dir)
        assert all(archive.load(c.path) == PAYLOAD for c in archive.scan(raw_dir))

    def test_running_twice_is_a_no_op(self, raw_dir):
        archive.compress_archive(raw_dir)
        second = archive.compress_archive(raw_dir)
        assert all(r.skipped for r in second)


class TestManifest:
    def test_covers_every_capture(self, raw_dir, tmp_path):
        n = archive.write_manifest(raw_dir, tmp_path / "m.sha256")
        assert n == 3
        assert len(archive.read_manifest(tmp_path / "m.sha256")) == 3

    def test_format_is_shasum_compatible(self, raw_dir, tmp_path):
        """Checkable by someone with no Python environment and no context."""
        path = tmp_path / "m.sha256"
        archive.write_manifest(raw_dir, path)
        body = [
            ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")
        ]
        for line in body:
            digest, sep, name = line.partition("  ")
            assert sep and len(digest) == 64 and name

    def test_verify_passes_on_an_intact_archive(self, raw_dir, tmp_path):
        archive.write_manifest(raw_dir, tmp_path / "m.sha256")
        report = archive.verify(raw_dir, tmp_path / "m.sha256")
        assert report.healthy
        assert len(report.ok) == 3

    def test_verify_detects_corruption(self, raw_dir, tmp_path):
        """The case the manifest exists for: silent bit rot."""
        archive.write_manifest(raw_dir, tmp_path / "m.sha256")
        victim = archive.scan(raw_dir)[0].path
        victim.write_text(victim.read_text().replace("7728", "9999"))
        report = archive.verify(raw_dir, tmp_path / "m.sha256")
        assert report.corrupted == [victim.name]
        assert not report.healthy

    def test_verify_detects_a_missing_capture(self, raw_dir, tmp_path):
        archive.write_manifest(raw_dir, tmp_path / "m.sha256")
        victim = archive.scan(raw_dir)[0].path
        victim.unlink()
        report = archive.verify(raw_dir, tmp_path / "m.sha256")
        assert report.missing == [victim.name]
        assert not report.healthy

    def test_new_captures_are_unlisted_not_a_failure(self, raw_dir, tmp_path):
        """The archive grows between manifest writes. Expected, not alarming."""
        archive.write_manifest(raw_dir, tmp_path / "m.sha256")
        (raw_dir / "_SPX_2026-08-14T20-37-03Z.json").write_text(json.dumps(PAYLOAD))
        report = archive.verify(raw_dir, tmp_path / "m.sha256")
        assert report.unlisted == ["_SPX_2026-08-14T20-37-03Z.json"]
        assert report.healthy, "an added capture is not a corruption"

    def test_compression_changes_the_digest_but_not_the_content(
        self, raw_dir, tmp_path
    ):
        """Why the pre-compression manifest is kept as chain of custody.

        The container's hash necessarily changes; what must not change is what
        comes back out of it.
        """
        archive.write_manifest(raw_dir, tmp_path / "before.sha256")
        before = archive.read_manifest(tmp_path / "before.sha256")

        capture = [c for c in archive.scan(raw_dir) if not c.compressed][0]
        original_bytes = capture.path.read_bytes()
        original_name = capture.path.name
        archive.compress(capture)

        archive.write_manifest(raw_dir, tmp_path / "after.sha256")
        after = archive.read_manifest(tmp_path / "after.sha256")
        assert original_name not in after, "the uncompressed entry is gone"

        gz = capture.path.with_name(original_name + ".gz")
        with gzip.open(gz, "rb") as f:
            restored = f.read()
        assert restored == original_bytes
        import hashlib

        assert hashlib.sha256(restored).hexdigest() == before[original_name]


class TestIterPayloads:
    def test_streams_every_capture_with_its_payload(self, raw_dir):
        seen = list(archive.iter_payloads(raw_dir))
        assert len(seen) == 3
        assert all(body == PAYLOAD for _c, body in seen)

    def test_ordered_oldest_first(self, raw_dir):
        stamps = [c.captured_at for c, _ in archive.iter_payloads(raw_dir)]
        assert stamps == sorted(stamps)
