"""Shared fixture loading.

Every fixture here is a real captured payload from `data/raw`, trimmed by
`tools/build_fixtures.py`. Nothing is invented. An invented fixture tests what
we believe the feed does; a captured one tests what it actually did, which is
the whole reason this corpus exists.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def load_header(name: str) -> dict[str, Any]:
    """Load a metadata-only capture (chain emptied) by bare capture name."""
    path = FIXTURES / "headers" / f"_SPX_{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no header fixture {name!r}; run `python3 tools/build_fixtures.py`"
        )
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def trimmed_chain() -> dict[str, Any]:
    """The Aug 11 close capture with 674 options preserving every pathology."""
    return json.loads((FIXTURES / "chain_trimmed.json").read_text())


@pytest.fixture(scope="session")
def full_chain() -> dict[str, Any]:
    """The complete 30,692-option Aug 11 close capture.

    Session-scoped: decompressing and parsing 13 MB per test would dominate the
    suite's runtime. Tests must not mutate it.
    """
    with gzip.open(FIXTURES / "chain_full_close.json.gz", "rt") as f:
        return json.load(f)


@pytest.fixture
def headers() -> dict[str, dict[str, Any]]:
    """All metadata-only captures, keyed by bare capture name.

    The timeline these span, in order:

        2026-08-09T22-38-43Z  weekend fetch serving Aug 7's close
        2026-08-10T23-57-37Z  after-close Aug 10
        2026-08-11T00-07-40Z  repeat of the above
        2026-08-11T16-11-33Z  healthy mid-session capture
        2026-08-11T20-37-03Z  first capture at the 16:14:59 close print
        2026-08-11T20-47-05Z  repeat, same seqno
        2026-08-11T20-57-15Z  repeat, seqno ADVANCED
        2026-08-11T21-07-17Z  repeat
        2026-08-11T23-57-57Z  repeat, 7h45m stale
    """
    out = {}
    for path in sorted((FIXTURES / "headers").glob("*.json")):
        name = path.stem.removeprefix("_SPX_")
        out[name] = json.loads(path.read_text())
    return out


def et(iso: str) -> datetime:
    """Build an aware US/Eastern datetime from a naive ISO string."""
    return datetime.fromisoformat(iso).replace(tzinfo=EASTERN)


def utc(iso: str) -> datetime:
    """Build an aware UTC datetime from a naive ISO string."""
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)
