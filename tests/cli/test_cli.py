"""Unit tests for `spxrnd.cli.main` and the launchd job.

The most important test in this file is `TestCollectStaysStdlibOnly`. Everything
else checks that a subcommand does what it says; that one checks that the
scheduled capture cannot be broken by a dependency upgrade.
"""

from __future__ import annotations

import builtins
import json
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from spxrnd.cli import main as cli

from ..conftest import FIXTURES, load_header

REPO = Path(__file__).resolve().parents[2]
HEAVY = ("numpy", "scipy", "pandas", "pyarrow", "duckdb")


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


def run(*argv) -> int:
    return cli.main(list(argv))


class TestCollectStaysStdlibOnly:
    """The constraint that keeps unattended collection unbreakable.

    `collect` runs every ten minutes forever, and a missed capture cannot be
    recovered -- a delayed-quote endpoint serves the present. So the one
    subcommand that runs unattended must import nothing that a dependency
    upgrade could break.
    """

    def test_collect_runs_with_the_scientific_stack_blocked(self, data_dir, capsys):
        """Not a spiritual claim: numpy, scipy, pandas, pyarrow and duckdb are
        blocked at the import hook while the capture runs."""
        real_import = builtins.__import__
        blocked = []

        def guard(name, *args, **kwargs):
            root = name.split(".")[0]
            if root in HEAVY:
                blocked.append(name)
                raise ImportError(f"{root} is blocked for this test")
            return real_import(name, *args, **kwargs)

        # Deliberately NOT clearing sys.modules. `import numpy` calls
        # __import__ before consulting the cache, so the guard catches it
        # either way -- and evicting a half-initialised pandas from the cache
        # corrupts every test that runs afterwards, which it duly did.
        payload = FIXTURES / "headers" / "_SPX_2026-08-11T16-11-33Z.json"
        builtins.__import__ = guard
        try:
            code = run("--dir", str(data_dir), "collect", "--from-file", str(payload))
        finally:
            builtins.__import__ = real_import

        assert code == cli.OK
        assert blocked == [], f"collect reached for {blocked}"
        assert "stale_feed" in capsys.readouterr().out

    def test_the_cli_module_imports_nothing_heavy_at_module_level(self):
        """Deferred imports live inside the subcommand functions. If one moved
        to the top of the file this fails, and the test above would still pass
        only by luck of module caching."""
        import ast

        source = (REPO / "src" / "spxrnd" / "cli" / "main.py").read_text()
        tree = ast.parse(source)
        top_level = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.append(node.module.split(".")[0])
        assert not (set(top_level) & set(HEAVY)), top_level


class TestCollect:
    def test_replays_a_payload_without_the_network(self, data_dir, capsys):
        payload = FIXTURES / "headers" / "_SPX_2026-08-11T16-11-33Z.json"
        assert run("--dir", str(data_dir), "collect", "--from-file", str(payload)) == 0
        assert "no network" in capsys.readouterr().out

    def test_force_writes_a_stale_capture(self, data_dir, capsys):
        payload = FIXTURES / "headers" / "_SPX_2026-08-11T16-11-33Z.json"
        code = run(
            "--dir", str(data_dir), "collect", "--from-file", str(payload), "--force"
        )
        assert code == 0
        assert "wrote" in capsys.readouterr().out
        assert list((data_dir / "raw").iterdir())

    def test_reads_a_gzipped_payload(self, data_dir, tmp_path, capsys):

        src = FIXTURES / "chain_full_close.json.gz"
        target = tmp_path / "capture.json.gz"
        target.write_bytes(src.read_bytes())
        code = run(
            "--dir", str(data_dir), "collect", "--from-file", str(target), "--force"
        )
        assert code == 0
        assert "wrote" in capsys.readouterr().out

    def test_a_skip_exits_zero(self, data_dir):
        """A holiday, a weekend and an off-hours tick are successes that write
        nothing. Exiting non-zero would fill the scheduler's log with failures
        that are not failures."""
        payload = FIXTURES / "headers" / "_SPX_2026-08-09T22-38-43Z.json"
        assert run("--dir", str(data_dir), "collect", "--from-file", str(payload)) == 0

    def test_a_bad_payload_exits_one(self, data_dir, tmp_path, capsys):
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps({"nope": True}))
        code = run("--dir", str(data_dir), "collect", "--from-file", str(broken))
        assert code == cli.FAILED
        assert "FAILED" in capsys.readouterr().out

    def test_output_is_timestamped_for_the_scheduler_log(self, data_dir, capsys):
        payload = FIXTURES / "headers" / "_SPX_2026-08-11T16-11-33Z.json"
        run("--dir", str(data_dir), "collect", "--from-file", str(payload))
        first = capsys.readouterr().out.splitlines()[0]
        assert first.startswith("20") and first[10] == "T" and "Z " in first


class TestStatus:
    def test_reports_nothing_recorded_yet(self, data_dir, capsys):
        assert run("--dir", str(data_dir), "status") == cli.ATTENTION
        assert "no collection recorded" in capsys.readouterr().out

    def test_reports_a_healthy_collector(self, data_dir, capsys):
        from datetime import UTC, datetime

        from spxrnd.ingest import health

        health.record(
            data_dir / health.HEALTH_FILENAME,
            now=datetime.now(UTC),
            verdict="accept",
            written=True,
        )
        assert run("--dir", str(data_dir), "status") == cli.OK
        assert "healthy" in capsys.readouterr().out

    def test_reports_a_failure_streak(self, data_dir, capsys):
        from datetime import UTC, datetime

        from spxrnd.ingest import health

        for _ in range(3):
            health.record(
                data_dir / health.HEALTH_FILENAME,
                now=datetime.now(UTC),
                verdict="FetchError",
                error="connection reset",
            )
        assert run("--dir", str(data_dir), "status") == cli.ATTENTION
        out = capsys.readouterr().out
        assert "FAILING: 3 consecutive" in out
        assert "connection reset" in out

    def test_distinguishes_a_dead_scheduler_from_a_dead_feed(self, data_dir, capsys):
        """They need different fixes, so they are reported separately."""
        from datetime import UTC, datetime, timedelta

        from spxrnd.ingest import health

        old = datetime.now(UTC) - timedelta(hours=5)
        health.record(
            data_dir / health.HEALTH_FILENAME, now=old, verdict="accept", written=True
        )
        assert run("--dir", str(data_dir), "status") == cli.ATTENTION
        out = capsys.readouterr().out
        assert "SCHEDULER" in out
        assert "launchctl" in out


class TestArchiveCommands:
    def test_captures_reports_the_real_archive(self, capsys):
        assert run("--dir", str(REPO / "data"), "captures") == cli.OK
        out = capsys.readouterr().out
        assert "captures" in out
        assert "curated layer aligned" in out

    def test_captures_on_an_empty_archive(self, data_dir, capsys):
        assert run("--dir", str(data_dir), "captures") == cli.ATTENTION
        assert "no captures" in capsys.readouterr().out

    def test_captures_can_list_every_file(self, capsys):
        run("--dir", str(REPO / "data"), "captures", "--list")
        assert "_SPX_2026-08-11T20-37-03Z" in capsys.readouterr().out

    def test_verify_passes_on_the_real_archive(self, capsys):
        code = run(
            "--dir",
            str(REPO / "data"),
            "verify",
            "--manifest",
            str(REPO / "manifests" / "raw_archive.sha256"),
        )
        assert code == cli.OK
        assert "corrupted 0" in capsys.readouterr().out

    def test_verify_flags_a_missing_capture(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T16-11-33Z.json").write_text(
            json.dumps(load_header("2026-08-11T16-11-33Z"))
        )
        manifest = tmp_path / "m.sha256"
        from spxrnd.store import archive

        archive.write_manifest(raw, manifest)
        next(raw.iterdir()).unlink()
        code = run("--dir", str(tmp_path), "verify", "--manifest", str(manifest))
        assert code == cli.ATTENTION
        assert "MISSING" in capsys.readouterr().out


class TestPipelineCommands:
    def test_backfill_and_curate_round_trip(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T20-37-03Z.json.gz").write_bytes(
            (FIXTURES / "chain_full_close.json.gz").read_bytes()
        )
        assert run("--dir", str(tmp_path), "backfill") == cli.OK
        out = capsys.readouterr().out
        assert "30,692 rows" in out
        assert "aligned" in out

        assert run("--dir", str(tmp_path), "curate") == cli.OK
        out = capsys.readouterr().out
        assert "quotes in" in out
        assert "parity R2" in out

    def test_estimate_reports_a_term_structure(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T20-37-03Z.json.gz").write_bytes(
            (FIXTURES / "chain_full_close.json.gz").read_bytes()
        )
        run("--dir", str(tmp_path), "backfill")
        capsys.readouterr()
        code = run("--dir", str(tmp_path), "estimate", "--term-structure")
        assert code == cli.OK
        out = capsys.readouterr().out
        assert "trustworthy" in out
        assert "kurt" in out

    def test_estimate_writes_a_csv(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T20-37-03Z.json.gz").write_bytes(
            (FIXTURES / "chain_full_close.json.gz").read_bytes()
        )
        run("--dir", str(tmp_path), "backfill")
        target = tmp_path / "terms.csv"
        run("--dir", str(tmp_path), "estimate", "--csv", str(target))
        assert target.exists()
        assert "trustworthy" in target.read_text().splitlines()[0]

    def test_curate_without_a_backfill_says_so(self, data_dir):
        with pytest.raises(FileNotFoundError, match="backfill"):
            run("--dir", str(data_dir), "curate")

    def test_an_unknown_capture_is_reported_with_the_newest(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T20-37-03Z.json.gz").write_bytes(
            (FIXTURES / "chain_full_close.json.gz").read_bytes()
        )
        run("--dir", str(tmp_path), "backfill")
        with pytest.raises(SystemExit, match="newest is"):
            run("--dir", str(tmp_path), "curate", "--capture", "1999-01-01")


class TestParser:
    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    def test_every_subcommand_is_wired_to_a_function(self):
        parser = cli.build_parser()
        for command in (
            "collect",
            "status",
            "verify",
            "captures",
            "backfill",
            "curate",
            "estimate",
        ):
            args = parser.parse_args([command])
            assert callable(args.func), command

    def test_the_module_runs_as_a_script(self):
        """launchd invokes `python -m spxrnd.cli.main`, so that path must work
        independently of the console-script entry point."""
        result = subprocess.run(
            [sys.executable, "-m", "spxrnd.cli.main", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        assert result.returncode == 0
        assert "risk-neutral densities" in result.stdout


class TestLaunchdJob:
    @pytest.fixture(scope="class")
    @classmethod
    def plist(cls):
        return plistlib.loads(
            (REPO / "deploy" / "com.spxrnd.collect.plist").read_bytes()
        )

    def test_the_checked_in_plist_matches_the_generator(self, plist):
        """So an edit to the schedule cannot land only in the generated file."""
        sys.path.insert(0, str(REPO / "tools"))
        import make_launchd_plist

        assert plist == make_launchd_plist.build(REPO)

    def test_it_is_a_valid_plist(self, plist):
        assert plist["Label"] == "com.spxrnd.collect"

    def test_paths_are_absolute(self, plist):
        """launchd inherits no working directory."""
        assert Path(plist["ProgramArguments"][0]).is_absolute()
        assert Path(plist["WorkingDirectory"]).is_absolute()
        assert "--dir" in plist["ProgramArguments"]
        assert Path(
            plist["ProgramArguments"][plist["ProgramArguments"].index("--dir") + 1]
        ).is_absolute()

    def test_it_runs_the_venv_python_not_the_system_one(self, plist):
        assert ".venv" in plist["ProgramArguments"][0]

    def test_it_invokes_collect_and_nothing_else(self, plist):
        """The scheduled job must never run backfill or estimate: those import
        the scientific stack and take minutes."""
        assert plist["ProgramArguments"][-1] == "collect"

    def test_it_does_not_fire_on_load(self, plist):
        """Loading the job should not trigger a capture at an arbitrary moment
        that the freshness gate would reject anyway."""
        assert plist["RunAtLoad"] is False

    def test_the_schedule_is_every_ten_minutes(self, plist):
        entries = plist["StartCalendarInterval"]
        monday = sorted(
            e["Hour"] * 60 + e["Minute"] for e in entries if e["Weekday"] == 1
        )
        assert set(b - a for a, b in zip(monday, monday[1:], strict=False)) == {10}

    def test_the_schedule_covers_the_session(self, plist):
        entries = [e for e in plist["StartCalendarInterval"] if e["Weekday"] == 1]
        minutes = [e["Hour"] * 60 + e["Minute"] for e in entries]
        assert min(minutes) == 9 * 60 + 35
        assert max(minutes) == 16 * 60 + 25

    def test_it_runs_on_weekdays_only(self, plist):
        """The freshness gate would reject weekend runs correctly, but they
        would still cost ~84 pointless requests against a free endpoint."""
        assert {e["Weekday"] for e in plist["StartCalendarInterval"]} == {1, 2, 3, 4, 5}

    def test_the_capture_count_matches_the_agreed_cadence(self, plist):
        entries = plist["StartCalendarInterval"]
        assert len(entries) == 210
        assert len(entries) // 5 == 42

    def test_logs_are_captured(self, plist):
        assert plist["StandardOutPath"].endswith(".log")
        assert plist["StandardErrorPath"].endswith(".log")


class TestRemainingCliPaths:
    """The reporting branches that only fire when something is wrong -- which
    is exactly when they need to have worked."""

    def test_status_reports_a_stale_feed_separately_from_a_dead_scheduler(
        self, data_dir, capsys, monkeypatch
    ):
        """last_attempt fresh but last_ok old: the job is running and the feed
        is not answering. A different fix from a dead scheduler."""
        from datetime import UTC, datetime, timedelta

        from spxrnd.ingest import health

        path = data_dir / health.HEALTH_FILENAME
        health.record(
            path,
            now=datetime.now(UTC) - timedelta(hours=5),
            verdict="accept",
            written=True,
        )
        current = health.read_health(path)
        from dataclasses import replace

        stamped = replace(
            current, last_attempt=datetime.now(UTC).isoformat(timespec="seconds")
        )
        import json
        from dataclasses import asdict

        path.write_text(json.dumps(asdict(stamped)))

        assert run("--dir", str(data_dir), "status") == cli.ATTENTION
        assert "FEED:" in capsys.readouterr().out

    def test_verify_can_rewrite_the_manifest(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T16-11-33Z.json").write_text(
            json.dumps(load_header("2026-08-11T16-11-33Z"))
        )
        manifest = tmp_path / "m.sha256"
        assert (
            run(
                "--dir", str(tmp_path), "verify", "--manifest", str(manifest), "--write"
            )
            == cli.OK
        )
        assert "manifest created" in capsys.readouterr().out
        assert manifest.exists()

        # And again, now that one exists: verifies first, then rewrites.
        assert (
            run(
                "--dir", str(tmp_path), "verify", "--manifest", str(manifest), "--write"
            )
            == cli.OK
        )
        assert "manifest rewritten" in capsys.readouterr().out

    def test_verify_without_a_manifest_says_how_to_make_one(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        assert (
            run("--dir", str(tmp_path), "verify", "--manifest", str(tmp_path / "none"))
            == cli.ATTENTION
        )
        assert "verify --write" in capsys.readouterr().out

    def test_dir_may_follow_the_subcommand(self, capsys):
        """`spxrnd captures --dir data` is how a person types it."""
        assert run("captures", "--dir", str(REPO / "data")) == cli.OK
        assert "captures" in capsys.readouterr().out

    def test_verify_reports_captures_newer_than_the_manifest(self, tmp_path, capsys):
        """Expected as the archive grows, so it is a note rather than a failure
        of the checksum itself."""
        from spxrnd.store import archive

        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T16-11-33Z.json").write_text(
            json.dumps(load_header("2026-08-11T16-11-33Z"))
        )
        manifest = tmp_path / "m.sha256"
        archive.write_manifest(raw, manifest)
        (raw / "_SPX_2026-08-14T16-11-33Z.json").write_text(
            json.dumps(load_header("2026-08-11T16-11-33Z"))
        )
        run("--dir", str(tmp_path), "verify", "--manifest", str(manifest))
        assert "newer than the manifest" in capsys.readouterr().out

    def test_captures_flags_a_missing_curated_layer(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T16-11-33Z.json").write_text(
            json.dumps(load_header("2026-08-11T16-11-33Z"))
        )
        assert run("--dir", str(tmp_path), "captures") == cli.ATTENTION
        assert "run `spxrnd backfill`" in capsys.readouterr().out

    def test_captures_flags_an_orphaned_parquet(self, tmp_path, capsys):
        """The raw archive should only ever grow, so a curated file with no
        capture behind it is the alarming direction."""
        from spxrnd.store import catalog

        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T20-37-03Z.json.gz").write_bytes(
            (FIXTURES / "chain_full_close.json.gz").read_bytes()
        )
        # Two captures, so removing one leaves an archive to report on -- an
        # empty archive short-circuits to "no captures" before divergence is
        # ever checked.
        (raw / "_SPX_2026-08-12T20-37-03Z.json.gz").write_bytes(
            (FIXTURES / "chain_full_close.json.gz").read_bytes()
        )
        catalog.backfill(raw, tmp_path / "curated")
        (raw / "_SPX_2026-08-12T20-37-03Z.json.gz").unlink()
        assert run("--dir", str(tmp_path), "captures") == cli.ATTENTION
        assert "only ever grow" in capsys.readouterr().out

    def test_backfill_reports_a_capture_it_could_not_convert(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T16-11-33Z.json").write_text('{"garbage": true}')
        assert run("--dir", str(tmp_path), "backfill") == cli.ATTENTION
        assert "FAILED" in capsys.readouterr().out

    def test_backfill_can_resume(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T20-37-03Z.json.gz").write_bytes(
            (FIXTURES / "chain_full_close.json.gz").read_bytes()
        )
        run("--dir", str(tmp_path), "backfill")
        capsys.readouterr()
        run("--dir", str(tmp_path), "backfill", "--resume")
        assert "0 written, 1 skipped" in capsys.readouterr().out

    def test_curate_accepts_an_explicit_capture(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T20-37-03Z.json.gz").write_bytes(
            (FIXTURES / "chain_full_close.json.gz").read_bytes()
        )
        run("--dir", str(tmp_path), "backfill")
        capsys.readouterr()
        assert run("--dir", str(tmp_path), "curate", "--capture", "20-37-03") == cli.OK
        assert "quotes in" in capsys.readouterr().out

    def test_main_is_callable_as_a_module_entry_point(self):
        """`python -m spxrnd.cli.main` is what launchd runs."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "spxrnd.cli.main",
                "captures",
                "--dir",
                str(REPO / "data"),
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        assert result.returncode == 0
        assert "captures" in result.stdout


class TestFinalReportingPaths:
    def test_verify_names_a_corrupted_capture(self, tmp_path, capsys):
        """Silent bit rot is the case the manifest exists for, so the report
        must say which file."""
        from spxrnd.store import archive

        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        target = raw / "_SPX_2026-08-11T16-11-33Z.json"
        target.write_text(json.dumps(load_header("2026-08-11T16-11-33Z")))
        manifest = tmp_path / "m.sha256"
        archive.write_manifest(raw, manifest)
        target.write_text(target.read_text().replace("7741", "9999"))

        assert run("--dir", str(tmp_path), "verify", "--manifest", str(manifest)) == (
            cli.ATTENTION
        )
        out = capsys.readouterr().out
        assert "CORRUPT" in out
        assert "_SPX_2026-08-11T16-11-33Z.json" in out

    def test_backfill_warns_about_unparseable_symbols(self, tmp_path, capsys):
        """Dropping contracts silently is how a chain loses an expiry."""
        import gzip

        with gzip.open(FIXTURES / "chain_full_close.json.gz", "rt") as f:
            payload = json.load(f)
        payload["data"]["options"][0]["option"] = "NOT-AN-OSI-SYMBOL"
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "_SPX_2026-08-11T20-37-03Z.json.gz").write_bytes(
            gzip.compress(json.dumps(payload).encode(), mtime=0)
        )
        run("--dir", str(tmp_path), "backfill")
        assert "WARNING: 1 option symbols did not parse" in capsys.readouterr().out

    def test_estimate_without_a_curated_layer_says_to_backfill(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (tmp_path / "curated").mkdir()
        with pytest.raises(FileNotFoundError, match="backfill"):
            run("--dir", str(tmp_path), "estimate")


def test_a_curated_layer_with_no_captures_says_to_backfill(tmp_path):
    """A Parquet file that exists but holds no rows. Reachable if a capture
    with an empty chain was backfilled -- a real payload shape, tested in
    `store`."""
    import pyarrow.parquet as pq

    from spxrnd.store import schema

    curated = tmp_path / "curated"
    curated.mkdir(parents=True)
    pq.write_table(
        schema.pa.table({f.name: [] for f in schema.SCHEMA}, schema=schema.SCHEMA),
        curated / "_SPX_2026-08-11T16-11-33Z.parquet",
    )
    with pytest.raises(SystemExit, match="run `spxrnd backfill`"):
        run("--dir", str(tmp_path), "curate")
