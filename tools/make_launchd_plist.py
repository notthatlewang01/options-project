#!/usr/bin/env python3
"""Generate the launchd job that collects on a schedule.

Generated rather than hand-written for two reasons: the schedule is 210
`StartCalendarInterval` entries, and the paths must be absolute for the machine
it will run on. `launchd` inherits no working directory and no shell profile.

Usage:
    python3 tools/make_launchd_plist.py            # write deploy/<label>.plist
    python3 tools/make_launchd_plist.py --print    # stdout, change nothing
"""

from __future__ import annotations

import argparse
import plistlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABEL = "com.spxrnd.collect"

FIRST = (9, 35)
LAST = (16, 25)
STEP_MINUTES = 10
WEEKDAYS = (1, 2, 3, 4, 5)
"""launchd weekdays: 1 = Monday through 5 = Friday.

Weekends are excluded rather than left to the freshness gate. The gate would
reject them correctly -- on a non-trading day the feed never freshens -- but it
would cost ~84 pointless requests every weekend against a free courtesy
endpoint, and politeness is cheap here.
"""


def schedule() -> list[dict[str, int]]:
    """Every capture time, as launchd calendar entries.

    **launchd schedules in the machine's local time, not UTC.** This host is on
    America/New_York, so these are Eastern and line up with the session. On a
    machine in another zone the schedule would slide -- and the freshness gate
    would then reject most of the runs, which is a safe failure but a silent
    one. See deploy/README.md.
    """
    start = FIRST[0] * 60 + FIRST[1]
    end = LAST[0] * 60 + LAST[1]
    return [
        {"Weekday": day, "Hour": minute // 60, "Minute": minute % 60}
        for day in WEEKDAYS
        for minute in range(start, end + 1, STEP_MINUTES)
    ]


def build(repo: Path = REPO) -> dict:
    logs = repo / "logs"
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(repo / ".venv" / "bin" / "python"),
            "-m",
            "spxrnd.cli.main",
            "--dir",
            str(repo / "data"),
            "collect",
        ],
        "WorkingDirectory": str(repo),
        "StartCalendarInterval": schedule(),
        # False on purpose. Loading the job should not fire a capture; the
        # schedule decides when to run, and a load-time run would be at an
        # arbitrary moment that the freshness gate then rejects anyway.
        "RunAtLoad": False,
        "StandardOutPath": str(logs / "collect.log"),
        "StandardErrorPath": str(logs / "collect.err.log"),
        # launchd gives an agent almost no PATH. The collector shells out to
        # nothing, but anything it might later call would not be found.
        "EnvironmentVariables": {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        # One capture is a single HTTP request and a ~2 MB write. A run still
        # alive after five minutes is wedged, not slow.
        "ExitTimeOut": 300,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", dest="to_stdout")
    ap.add_argument("--repo", default=str(REPO))
    args = ap.parse_args()

    plist = build(Path(args.repo).resolve())
    blob = plistlib.dumps(plist, sort_keys=False)

    if args.to_stdout:
        sys.stdout.write(blob.decode())
        return 0

    target = REPO / "deploy" / f"{LABEL}.plist"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    entries = len(plist["StartCalendarInterval"])
    per_day = entries // len(WEEKDAYS)
    print(f"wrote {target.relative_to(REPO)}")
    print(
        f"  {entries} calendar entries -- {per_day} per day x {len(WEEKDAYS)} weekdays"
    )
    print(
        f"  {FIRST[0]:02d}:{FIRST[1]:02d} to {LAST[0]:02d}:{LAST[1]:02d} "
        f"local time, every {STEP_MINUTES} minutes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
