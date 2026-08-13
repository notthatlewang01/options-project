"""The `spxrnd` command line.

Subcommands map one-to-one onto the components:

    collect     ingest      take exactly one snapshot
    status      ingest      is collection actually working?
    verify      store       check the archive against its checksum manifest
    captures    store       what do we have?
    backfill    store       rebuild the curated layer from the raw archive
    curate      curate      one capture's quality report and forward curve
    estimate    analytics   densities and moments for every expiry

Heavy imports are deliberately deferred
---------------------------------------
`collect` is the one subcommand that runs unattended, every ten minutes,
forever. It imports **only** `spxrnd.ingest`, which is stdlib-only -- so a numpy
or pyarrow upgrade cannot stop a capture, and a capture missed is a capture
lost. Every other subcommand imports its dependencies inside the function that
needs them.

This is not a micro-optimisation and it is not optional. There is a test
(`tests/cli/test_cli.py::TestCollectStaysStdlibOnly`) that runs `collect` with
numpy, scipy, pandas, pyarrow and duckdb blocked at the import hook, and it must
keep passing.

Exit codes
----------
    0   worked, including a deliberate skip -- a holiday, a weekend and an
        off-hours tick are all successes that write nothing
    1   failed
    2   worked, but something needs attention (a failed check, a stale feed)
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

OK = 0
FAILED = 1
ATTENTION = 2

DEFAULT_DATA_DIR = "data"


def _log(message: str) -> None:
    """Single-line stdout, timestamped. launchd's log picks this up verbatim."""
    print(f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {message}", flush=True)


# ---------------------------------------------------------------------------
# collect -- the only subcommand that must stay stdlib-only
# ---------------------------------------------------------------------------


def cmd_collect(args) -> int:
    import json

    from spxrnd.ingest import collector
    from spxrnd.ingest.errors import IngestError

    payload = None
    if args.from_file:
        path = Path(args.from_file)
        if path.suffix == ".gz":
            import gzip

            with gzip.open(path, "rt") as f:
                payload = json.load(f)
        else:
            payload = json.loads(path.read_text())
        _log(f"replaying {path} (no network)")

    try:
        result = collector.collect_once(
            Path(args.dir),
            ticker=args.ticker,
            force=args.force,
            retries=args.retries,
            payload=payload,
        )
    except IngestError as exc:
        _log(f"FAILED {type(exc).__name__}: {exc}")
        return FAILED

    _log(f"{result.decision.verdict}: {result.decision.detail}")
    if result.written:
        size = result.raw_path.stat().st_size
        _log(f"wrote {result.raw_path.name} ({size / 1e6:.2f} MB)")
    return OK


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def cmd_status(args) -> int:
    from datetime import timedelta

    from spxrnd.ingest import health

    data_dir = Path(args.dir)
    record = health.read_health(data_dir / health.HEALTH_FILENAME)
    now = datetime.now(UTC)

    if record.last_attempt is None:
        print("no collection recorded yet")
        return ATTENTION

    def age(field: str) -> str:
        delta = record.stale_by(now, field=field)
        if delta is None:
            return "never"
        hours = delta.total_seconds() / 3600
        return f"{hours:.1f}h ago" if hours >= 1 else f"{delta.seconds // 60}m ago"

    print(f"last attempt   {record.last_attempt}  ({age('last_attempt')})")
    print(f"last ok        {record.last_ok}  ({age('last_ok')})")
    print(f"last write     {record.last_write}  ({age('last_write')})")
    print(f"last verdict   {record.last_verdict}")
    print(
        f"totals         {record.total_writes} written, "
        f"{record.total_skips} skipped, {record.total_failures} failed"
    )

    if record.consecutive_failures:
        print(f"\nFAILING: {record.consecutive_failures} consecutive")
        print(f"  {record.last_error}")
        return ATTENTION

    # A stale `last_attempt` means the scheduler is broken; a stale `last_ok`
    # means the network or the feed is. They need different fixes, so they are
    # reported separately rather than as one "unhealthy".
    attempt_age = record.stale_by(now, field="last_attempt")
    if attempt_age and attempt_age > timedelta(hours=1):
        print(f"\nSCHEDULER: no attempt for {attempt_age.total_seconds() / 3600:.1f}h")
        print(
            "  the collector is not being run -- check `launchctl list | grep spxrnd`"
        )
        return ATTENTION

    ok_age = record.stale_by(now, field="last_ok")
    if ok_age and ok_age > timedelta(hours=1):
        print(f"\nFEED: last successful fetch {ok_age.total_seconds() / 3600:.1f}h ago")
        return ATTENTION

    print("\nhealthy")
    return OK


# ---------------------------------------------------------------------------
# verify / captures / backfill
# ---------------------------------------------------------------------------


def cmd_verify(args) -> int:
    from spxrnd.store import archive

    raw_dir = Path(args.dir) / "raw"
    manifest = Path(args.manifest)
    if not manifest.exists():
        # A first run, not a failure. Refusing here would mean the only way to
        # create a manifest was to already have one.
        if args.write:
            n = archive.write_manifest(raw_dir, manifest)
            print(f"manifest created over {n} captures")
            return OK
        print(f"no manifest at {manifest} -- run `spxrnd verify --write` to create it")
        return ATTENTION

    report = archive.verify(raw_dir, manifest)
    print(
        f"ok {len(report.ok)}   corrupted {len(report.corrupted)}   "
        f"missing {len(report.missing)}   unlisted {len(report.unlisted)}"
    )
    for name in report.corrupted:
        print(f"  CORRUPT  {name}")
    for name in report.missing:
        print(f"  MISSING  {name}")
    if report.unlisted:
        print(
            f"  {len(report.unlisted)} captures newer than the manifest "
            f"(run `spxrnd verify --write` to refresh)"
        )
    if args.write:
        n = archive.write_manifest(Path(args.dir) / "raw", Path(args.manifest))
        print(f"manifest rewritten over {n} captures")
    return OK if report.healthy else ATTENTION


def cmd_captures(args) -> int:
    from spxrnd.store import archive, catalog

    raw_dir = Path(args.dir) / "raw"
    found = archive.scan(raw_dir)
    if not found:
        print(f"no captures in {raw_dir}")
        return ATTENTION

    total = sum(c.path.stat().st_size for c in found)
    print(f"{len(found)} captures, {total / 1e6:.1f} MB")
    print(
        f"  {found[0].captured_at:%Y-%m-%d %H:%M}Z to "
        f"{found[-1].captured_at:%Y-%m-%d %H:%M}Z"
    )

    div = catalog.divergence(raw_dir, Path(args.dir) / "curated")
    if div.aligned:
        print("  curated layer aligned")
    else:
        if div.missing_curated:
            print(
                f"  {len(div.missing_curated)} archived but not curated "
                f"-- run `spxrnd backfill`"
            )
        if div.orphan_curated:
            print(
                f"  {len(div.orphan_curated)} curated with no raw capture "
                f"-- the raw archive should only ever grow"
            )
        return ATTENTION

    if args.list:
        for capture in found:
            print(f"    {capture.stem}  {capture.path.stat().st_size / 1e6:.2f} MB")
    return OK


def cmd_backfill(args) -> int:
    from spxrnd.store import catalog

    raw_dir, curated_dir = Path(args.dir) / "raw", Path(args.dir) / "curated"
    report = catalog.backfill(raw_dir, curated_dir, overwrite=not args.resume)
    print(
        f"{len(report.written)} written, {len(report.skipped)} skipped, "
        f"{len(report.failed)} failed"
    )
    print(f"{report.total_rows:,} rows, {report.total_bytes / 1e6:.1f} MB")
    if report.total_unparsed:
        print(f"WARNING: {report.total_unparsed} option symbols did not parse")
    for stem, error in report.failed:
        print(f"  FAILED {stem}: {error}")

    div = catalog.divergence(raw_dir, curated_dir)
    print("divergence:", "aligned" if div.aligned else div)
    return OK if (not report.failed and div.aligned) else ATTENTION


# ---------------------------------------------------------------------------
# curate / estimate
# ---------------------------------------------------------------------------


def _resolve_capture(con, wanted: str | None) -> datetime:
    """Pick a capture: the newest by default, or the one whose name matches."""
    rows = con.execute(
        "SELECT capture_utc FROM captures ORDER BY capture_utc"
    ).fetchall()
    stamps = [r[0] for r in rows]
    if not stamps:
        raise SystemExit("no captures in the curated layer -- run `spxrnd backfill`")
    if wanted is None:
        return stamps[-1]
    matches = [s for s in stamps if wanted in s.strftime("%Y-%m-%dT%H-%M-%SZ")]
    if not matches:
        raise SystemExit(
            f"no capture matching {wanted!r}; newest is {stamps[-1]:%Y-%m-%dT%H-%M-%SZ}"
        )
    return matches[-1]


def cmd_curate(args) -> int:
    from spxrnd.curate import chain
    from spxrnd.store import catalog

    con = catalog.connect(Path(args.dir) / "curated")
    when = _resolve_capture(con, args.capture)
    curated = chain.from_catalog(con, when)
    print(curated.summary())
    return OK


def cmd_estimate(args) -> int:
    from spxrnd.analytics import surface
    from spxrnd.curate import chain
    from spxrnd.store import catalog

    con = catalog.connect(Path(args.dir) / "curated")
    when = _resolve_capture(con, args.capture)
    curated = chain.from_catalog(con, when)
    print(f"capture {when:%Y-%m-%d %H:%M:%S}Z   spot {curated.spot}\n")

    estimates = surface.estimate_all(curated.quotes)
    print(surface.summary(estimates))

    frame = surface.to_frame(estimates)
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")

    if args.term_structure:
        usable = frame[frame["trustworthy"]]
        print(
            f"\n{'expiry':<12}{'root':<6}{'days':>7}{'n':>6}{'vol':>8}"
            f"{'skew':>8}{'kurt':>8}"
        )
        for row in usable.itertuples():
            print(
                f"{str(row.expiry)[:10]:<12}{row.root:<6}{row.days:>7.1f}"
                f"{row.n_quotes:>6}{row.annualised_vol:>8.2%}"
                f"{row.skewness:>8.2f}{row.kurtosis:>8.1f}"
            )

    return OK if any(e.trustworthy for e in estimates) else ATTENTION


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spxrnd",
        description="Collect SPX option chains and estimate risk-neutral densities.",
    )
    parser.add_argument(
        "--dir",
        default=DEFAULT_DATA_DIR,
        help="data directory (default: %(default)s)",
    )
    # Repeated on every subcommand so that `spxrnd collect --dir data` works as
    # well as `spxrnd --dir data collect`. The natural way to type it is after
    # the subcommand, and argparse rejects that unless the option exists there
    # too. SUPPRESS on the copies, so an unpassed subcommand-level --dir does
    # not overwrite a value given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", default=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser(
        "collect", parents=[common], help="take exactly one snapshot"
    )
    collect.add_argument("--ticker", default="_SPX")
    collect.add_argument("--retries", type=int, default=3)
    collect.add_argument(
        "--force",
        action="store_true",
        help="write even if the capture is stale or duplicated",
    )
    collect.add_argument(
        "--from-file", help="replay a saved payload instead of fetching"
    )
    collect.set_defaults(func=cmd_collect)

    status = sub.add_parser("status", parents=[common], help="is collection working?")
    status.set_defaults(func=cmd_status)

    verify = sub.add_parser(
        "verify", parents=[common], help="check the archive against its manifest"
    )
    verify.add_argument("--manifest", default="manifests/raw_archive.sha256")
    verify.add_argument(
        "--write", action="store_true", help="rewrite the manifest afterwards"
    )
    verify.set_defaults(func=cmd_verify)

    captures = sub.add_parser(
        "captures", parents=[common], help="what is in the archive"
    )
    captures.add_argument("--list", action="store_true", help="name every capture")
    captures.set_defaults(func=cmd_captures)

    backfill = sub.add_parser(
        "backfill", parents=[common], help="rebuild the curated layer"
    )
    backfill.add_argument(
        "--resume", action="store_true", help="skip captures already converted"
    )
    backfill.set_defaults(func=cmd_backfill)

    curate = sub.add_parser(
        "curate", parents=[common], help="quality report and forward curve"
    )
    curate.add_argument("--capture", help="capture timestamp (default: newest)")
    curate.set_defaults(func=cmd_curate)

    estimate = sub.add_parser(
        "estimate", parents=[common], help="densities and moments"
    )
    estimate.add_argument("--capture", help="capture timestamp (default: newest)")
    estimate.add_argument("--csv", help="write the term structure to this path")
    estimate.add_argument(
        "--term-structure", action="store_true", help="print the per-expiry table"
    )
    estimate.set_defaults(func=cmd_estimate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
