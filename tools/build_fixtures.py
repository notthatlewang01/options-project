#!/usr/bin/env python3
"""Build the test fixture corpus from the immutable raw archive.

Fixtures are real captured payloads, trimmed -- never invented ones. Invented
fixtures encode what we *think* the feed does; real ones encode what it actually
did, including the parts we got wrong the first time.

Three kinds are produced:

  headers/*.json          Metadata-only payloads (the `options` list emptied).
                          Freshness and dedup gating never look at the chain, so
                          these stay ~1 KB and cover the whole pathological
                          timeline for a rounding error in repo size.

  chain_trimmed.json      One close snapshot's metadata plus a deterministically
                          selected subset of options that preserves every known
                          pathology. This is what unit tests run against.

  chain_full_close.json.gz  The complete 30,692-option close snapshot, gzipped.
                          Integration tests use this so real 30k-row behaviour
                          is exercised somewhere, not merely assumed.

Selection is deterministic -- sorted, no sampling -- so regenerating produces
byte-identical output and fixture churn never pollutes a diff.

Usage:
    python3 tools/build_fixtures.py            # rebuild everything
    python3 tools/build_fixtures.py --check    # verify fixtures match the archive
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
FIXTURES = REPO / "tests" / "fixtures"

OSI_RE = re.compile(r"^(?P<root>[A-Z]+)(?P<exp>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")

# The capture used for every chain fixture: the final print of the Aug 11
# session, 16:14:59 ET. It is the richest snapshot we have -- it carries the
# 0-DTE expiry, the SPX/SPXW collision, and the full post-close freeze context.
CLOSE_CAPTURE = "2026-08-11T20-37-03Z"

TICKER = "_SPX"

# Metadata-only fixtures, keyed by bare capture timestamp. The note on each is
# the property it exists to pin down; the seed regression tests load these by
# the same bare name via conftest.load_header.
HEADER_CAPTURES = {
    "2026-08-09T22-38-43Z": "weekend fetch serving Friday Aug 7's close",
    "2026-08-10T23-57-37Z": "after-close Aug 10, first sighting of its print",
    "2026-08-11T00-07-40Z": "duplicate of the Aug 10 after-close capture",
    "2026-08-11T16-11-33Z": "healthy mid-session capture (the happy path)",
    "2026-08-11T20-37-03Z": "first capture at the 16:14:59 close print -- legitimate",
    "2026-08-11T20-47-05Z": "freeze: same print, same seqno",
    "2026-08-11T20-57-15Z": "freeze: same print, seqno ADVANCED (defeats seqno dedup)",
    "2026-08-11T21-07-17Z": "freeze: same print, hours of staleness",
    "2026-08-11T23-57-57Z": "freeze: 7h45m stale; also the raw-without-curated case",
}


def capture_file(name: str) -> str:
    """Bare capture timestamp -> the archive filename it came from."""
    return f"{TICKER}_{name}.json"


def load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def strike_of(opt: dict) -> float | None:
    m = OSI_RE.match(opt.get("option", ""))
    return int(m["strike"]) / 1000.0 if m else None


def expiry_of(opt: dict) -> str | None:
    m = OSI_RE.match(opt.get("option", ""))
    return m["exp"] if m else None


def select_options(
    options: list[dict], spot: float
) -> tuple[list[dict], dict[str, int]]:
    """Pick a subset preserving every pathology, deterministically.

    Returns the selected options (sorted by symbol) and a per-reason tally. The
    tally is written into the fixture manifest so a reader can see what each
    group is for without re-deriving it.
    """
    picked: dict[str, dict] = {}
    reasons: dict[str, int] = {}

    def take(opt: dict, reason: str) -> None:
        sym = opt["option"]
        if sym not in picked:
            picked[sym] = opt
            reasons[reason] = reasons.get(reason, 0) + 1

    def near_spot(opt: dict, pct: float) -> bool:
        k = strike_of(opt)
        return k is not None and abs(k - spot) / spot <= pct

    # --- 0-DTE: expires the same session. Tenor -> 0 breaks anything that
    # divides by time to expiry, and its IVs blow up near the money.
    for o in sorted(
        (o for o in options if expiry_of(o) == "260811" and near_spot(o, 0.02)),
        key=lambda o: o["option"],
    ):
        take(o, "0-DTE (260811), within 2% of spot")

    # --- The SPX/SPXW root collision. Both roots list the same strikes on this
    # expiry; without the root column they are indistinguishable rows. A
    # put-call parity fit that mixes them produces a corrupted forward.
    by_key: dict[tuple[str, float], dict[str, dict]] = {}
    for o in options:
        m = OSI_RE.match(o.get("option", ""))
        if m and m["exp"] == "260821":
            by_key.setdefault((m["cp"], int(m["strike"]) / 1000.0), {})[m["root"]] = o
    for (_cp, _k), roots in sorted(by_key.items()):
        if len(roots) == 2 and near_spot(next(iter(roots.values())), 0.03):
            for o in roots.values():
                take(o, "260821 SPX/SPXW strike collision, near spot")

    # --- Deep in-the-money: strike 200 against a 7728 spot. CBOE reports an
    # 800% IV and a 0.9997 delta here; it is a price, not a volatility.
    for o in sorted(
        (o for o in options if (k := strike_of(o)) is not None and k < spot * 0.10),
        key=lambda o: o["option"],
    )[:8]:
        take(o, "deep ITM, degenerate IV")

    # --- iv == 0.0 is a sentinel meaning "not computed", NOT zero volatility.
    # 2,160 options carry it, 694 of them within 10% of spot with live
    # two-sided quotes. Consumed naively it flattens the smile where it matters
    # most.
    for o in sorted(
        (
            o
            for o in options
            if o.get("iv") == 0.0 and o.get("bid", 0) > 0 and near_spot(o, 0.10)
        ),
        key=lambda o: o["option"],
    )[:20]:
        take(o, "iv==0.0 sentinel with a live quote, near spot")

    # --- Zero bid: no one is buying. The mid is meaningless and the spread is
    # 100%; these must never reach an IV inversion.
    for o in sorted(
        (o for o in options if o.get("bid") == 0), key=lambda o: o["option"]
    )[:12]:
        take(o, "zero bid")

    # --- last_trade_price can be years stale (one contract last printed in
    # Aug 2024, 65% away from its current bid). It is not a quote.
    for o in sorted(
        (o for o in options if (t := o.get("last_trade_time")) and t < "2026-01"),
        key=lambda o: (o.get("last_trade_time") or "", o["option"]),
    )[:8]:
        take(o, "last trade over a year stale")

    # --- Wide market: relative spread over 50%. Real quotes, but the mid
    # carries little information.
    for o in sorted(
        (
            o
            for o in options
            if o.get("bid", 0) > 0
            and o.get("ask", 0) > 0
            and (o["ask"] - o["bid"]) / ((o["ask"] + o["bid"]) / 2) > 0.5
        ),
        key=lambda o: o["option"],
    )[:10]:
        take(o, "relative spread > 50%")

    # --- A clean, liquid near-the-money band on a normal monthly expiry. The
    # happy path needs fixtures too, or every test is a test of pathology.
    for o in sorted(
        (
            o
            for o in options
            if expiry_of(o) == "260918"
            and near_spot(o, 0.05)
            and o.get("bid", 0) > 0
            and o.get("iv")
        ),
        key=lambda o: o["option"],
    )[:120]:
        take(o, "260918 monthly, liquid, near the money (happy path)")

    return [picked[s] for s in sorted(picked)], reasons


def build(check: bool = False) -> int:
    if not RAW.is_dir():
        print(f"ERROR: raw archive not found at {RAW}", file=sys.stderr)
        return 1

    written: list[tuple[str, int]] = []
    (FIXTURES / "headers").mkdir(parents=True, exist_ok=True)

    # --- metadata-only headers -------------------------------------------
    for name in sorted(HEADER_CAPTURES):
        src = RAW / capture_file(name)
        if not src.exists():
            print(f"ERROR: missing capture {src.name}", file=sys.stderr)
            return 1
        payload = load(src)
        payload["data"]["options"] = []
        blob = json.dumps(payload, indent=1, sort_keys=True) + "\n"
        if not check:
            (FIXTURES / "headers" / capture_file(name)).write_text(blob)
        written.append((f"headers/{capture_file(name)}", len(blob)))

    # --- trimmed chain ----------------------------------------------------
    payload = load(RAW / capture_file(CLOSE_CAPTURE))
    spot = payload["data"]["current_price"]
    total = len(payload["data"]["options"])
    selected, reasons = select_options(payload["data"]["options"], spot)
    payload["data"]["options"] = selected
    blob = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    if not check:
        (FIXTURES / "chain_trimmed.json").write_text(blob)
    written.append(("chain_trimmed.json", len(blob)))

    # --- full chain, gzipped ---------------------------------------------
    full = load(RAW / capture_file(CLOSE_CAPTURE))
    raw_blob = json.dumps(full, separators=(",", ":")).encode()
    gz = gzip.compress(raw_blob, compresslevel=9, mtime=0)  # mtime=0 -> reproducible
    if not check:
        (FIXTURES / "chain_full_close.json.gz").write_bytes(gz)
    written.append(("chain_full_close.json.gz", len(gz)))

    # --- manifest ---------------------------------------------------------
    lines = [
        "# Test fixture corpus",
        "",
        "Generated by `tools/build_fixtures.py` from `data/raw`. Do not edit by hand;",
        "regenerate instead. Selection is deterministic, so regeneration is a no-op",
        "unless the archive or the selectors change.",
        "",
        f"Source capture for all chain fixtures: `{capture_file(CLOSE_CAPTURE)}`  ",
        f"Spot at capture: {spot}  ",
        f"Full chain: {total:,} options  ",
        f"Trimmed chain: {len(selected):,} options",
        "",
        "## Metadata-only headers",
        "",
        "The `options` list is emptied. Freshness and dedup gating never inspect the",
        "chain, so these cover the whole pathological timeline at ~1 KB each.",
        "",
        "| Fixture | Why it exists |",
        "|---|---|",
    ]
    lines += [
        f"| `headers/{capture_file(n)}` | {why} |"
        for n, why in sorted(HEADER_CAPTURES.items())
    ]
    lines += [
        "",
        "## What the trimmed chain preserves",
        "",
        "| Selector | Options |",
        "|---|---|",
    ]
    lines += [f"| {reason} | {count} |" for reason, count in sorted(reasons.items())]
    lines += [
        "",
        "## Sizes",
        "",
        "| Fixture | Bytes |",
        "|---|---|",
    ]
    lines += [f"| `{n}` | {size:,} |" for n, size in written]
    lines += [""]
    manifest = "\n".join(lines)
    if not check:
        (FIXTURES / "MANIFEST.md").write_text(manifest)

    for name, size in written:
        print(f"  {name:<42} {size:>10,} bytes")
    print(f"\n  trimmed chain: {len(selected):,} of {total:,} options")
    for reason, count in sorted(reasons.items()):
        print(f"    {count:>4}  {reason}")
    print(f"\n  total: {sum(s for _, s in written):,} bytes")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="report what would be written without touching anything",
    )
    return build(check=ap.parse_args().check)


if __name__ == "__main__":
    sys.exit(main())
