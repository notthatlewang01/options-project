# `spxrnd.ingest`

Fetch one option-chain payload from CBOE, decide whether it carries new
information, and write it to the immutable archive. Nothing else.

**Status: complete.** 319 tests, 100% statement coverage, plus a `--live` smoke
test verified against the real endpoint.

## Why this component is stdlib-only

The collector runs unattended every ten minutes. A numpy or pyarrow upgrade must
never be able to stop a capture, because a missed capture cannot be recovered —
a delayed-quote endpoint serves the present, and 14:30 on a given Tuesday
happens once. `pyproject.toml` declares no runtime dependencies, and
`tests/test_ingest_is_stdlib_only.py` AST-parses this package to keep that
declaration honest.

The same constraint keeps the transport isolated in `endpoint.py`, so every
other module here is a pure function of a `dict`. That is what lets the whole
component be tested against real captured payloads with no network and no
mocking framework.

## Public API

### `osi` — contract symbols

```python
parse(symbol: str) -> OptionSymbol          # raises OsiParseError
try_parse(symbol: str) -> OptionSymbol | None
```

`OptionSymbol` is a frozen dataclass: `root`, `expiry` (a real `date`), `right`
(`"call"`/`"put"`), `strike` (float), `raw`.

`root` is not bookkeeping. SPX (AM-settled monthlies) and SPXW (PM-settled
weeklies) list the same strikes on the same expiries — 986 collisions on
2026-08-21 alone — at different quotes. Dropping the root makes those rows
indistinguishable and corrupts any put–call parity fit that mixes them.

### `payload` — validation

```python
parse(payload: dict, *, ticker: str) -> Snapshot     # raises PayloadError
parse_eastern(timestamp: str) -> datetime            # raises PayloadError
```

`Snapshot` carries `feed_timestamp`, `index_last_trade` (tz-aware),
`seqno`, `ticker`, `spot`, `options`, and `raw`.

**The payload has two timestamps and they are not interchangeable.**
`payload["timestamp"]` is when the CDN served the body — it advances forever,
including at 2am on a holiday. `data["last_trade_time"]` is when the cash index
last printed, and it is the only field that says whether there is new
information. The original collector read `payload["data"]["timestamp"]`, a key
that does not exist, so its `cboe_timestamp` column was empty in every row it
ever wrote.

### `freshness` — should this be kept?

```python
evaluate(snapshot, previous, *, now, max_stale=DEFAULT_MAX_STALE, force=False) -> Decision
```

Two gates, both keyed on the index print:

- **Duplicate** — the print has not moved since the last accepted capture.
- **Stale** — the print is older than `max_stale`. This also handles holidays
  for free: on a non-trading day the feed simply never freshens, so no exchange
  calendar is needed.

`now` is injected, never read from the clock inside the function. A gate that
decides what enters a permanent archive must be deterministically testable.

`DEFAULT_MAX_STALE` is **27 minutes**, derived as `FEED_DELAY + CADENCE + 2`.
Any genuinely new print is at most delay + one cadence old when first seen. The
inherited default was 22 minutes, which discards the 16:14:59 settlement capture
by four seconds — see below.

`seqno` is recorded but **never** used for deduplication. Across the post-close
freeze it advanced from 16938026517 to 16947306680 while the print stayed frozen
at 16:14:59; a seqno-keyed gate admits that capture as new data.

### `state` — durability

```python
read_state(path) -> CollectorState
write_state(path, state) -> None
write_atomic(path, data: bytes) -> None
collector_lock(path, *, stale_after=1h) -> ContextManager[bool]
```

Atomic writes are not decoration: a torn state file makes the *next* run
misjudge freshness, and a torn archive entry is a corrupt capture that cannot be
re-fetched. The lock yields `False` rather than raising when another collector
holds it — overlapping scheduler runs are expected, and exiting 0 keeps them out
of the failure logs. Locks older than an hour are reclaimed; a collector that
has held one that long is dead, not slow.

A *corrupt* state file is treated as empty rather than fatal. Re-capturing one
snapshot costs a duplicate row that the gate drops; refusing to run costs a
permanent hole in a series that cannot be backfilled.

### `endpoint` — transport

```python
fetch(ticker, *, retries=3, timeout=30, fetcher=None, sleep=None) -> dict
```

Requests gzip (≈1.7 MB instead of ≈13 MB) and backs off exponentially. This is a
free courtesy endpoint. `fetcher` and `sleep` are injectable so retry behaviour
is testable without sockets or waiting.

### `health` — is collection actually working?

```python
read_health(path) -> Health
record(path, *, now, verdict, written=False, error=None) -> Health
```

A scheduled collector fails silently by default: the job stops, the log scrolls,
and you find out weeks later when an analysis has a hole in it — by which point
the captures are permanently gone. `data/health.json` is the standing answer,
and nothing in this module can raise: telemetry is worth strictly less than the
data it reports on.

Three separate clocks, because they fail differently and only one of them means
something is broken:

| Field | Meaning | If it's old… |
|---|---|---|
| `last_attempt` | the collector process ran | **the scheduler** is broken |
| `last_ok` | fetch and parse both succeeded | **the network or feed** is broken |
| `last_write` | a capture reached the archive | nothing — this is old every weekend |

**A skip is not a failure.** Weekends, holidays and off-hours ticks all end in
one, and `consecutive_failures` is cleared by any successful fetch — including a
skipped one, since a skip still proves the pipeline works end to end.

### `collector` — orchestration

```python
collect_once(data_dir, *, ticker, now, max_stale, retries, force, payload) -> CaptureResult
archive_path(data_dir, ticker, captured_at) -> Path
```

One snapshot per invocation; scheduling is the operating system's job. The
original slept between captures inside the process, so a crash cost the rest of
the session instead of one sample.

**Ordering matters:** state is recorded only *after* the archive write succeeds.
A crash between the two re-captures a snapshot we already hold — harmless, the
gate drops it — rather than advancing past one that was never persisted, which
would leave a permanent hole.

A skip is a success. Holidays, weekends, and off-hours ticks cost one HTTP
request and write nothing; reporting that as failure trains the reader to ignore
the logs.

## Failure modes

| Exception | Cause | Operator response |
|---|---|---|
| `OsiParseError` | Symbol does not match OSI layout | Feed changed shape; investigate before trusting the capture |
| `PayloadError` | Missing field, or unparseable `last_trade_time` | Never defaulted — a snapshot that cannot be gated must not be archived |
| `FetchError` | All HTTP attempts failed | Carries the last underlying error; a dead network and a 500 need different responses |
| `LockError` | Lock could not be acquired or released | Check for an abandoned lock file |

## Example

```python
from pathlib import Path
from spxrnd.ingest.collector import collect_once

result = collect_once(Path("data"), ticker="_SPX")
print(result.decision.verdict, "-", result.decision.detail)
if result.written:
    print("archived to", result.raw_path)
```

Run against the live feed outside market hours, this prints:

```
stale_feed - feed is stale: index last printed 2026-08-12T16:14:59-04:00 ET,
310.5 min ago, over the 27 min limit -- market closed or holiday
```

…and writes nothing but a health record. That is the intended outcome, not a
failure.

Replaying a captured payload through the full pipeline, no network:

```python
import json

result = collect_once(
    Path("/tmp/scratch"), payload=json.load(open(capture)), force=True
)
```

## Files written under `data/`

| Path | Written when | Notes |
|---|---|---|
| `raw/<ticker>_<UTC>.json.gz` | a capture is accepted | Immutable. ~1.8 MB, ~7× smaller than the JSON. |
| `state.json` | **after** the archive write succeeds | The ordering decides what a crash costs. |
| `health.json` | every invocation that acquires the lock | Never blocks a capture. |
| `.collect.lock` | for the duration of a run | Removed on exit, including on exception. |

## The staleness boundary, measured

Ages of every capture in the archive, relative to its own index print:

| Group | Age (min) | n |
|---|---|---|
| In-session | 15.12 – 15.92 | 22 |
| **The 16:14:59 close** | **22.07** | 1 |
| First post-close repeat | 32.10 | 1 |
| Off-hours / weekend | 222 – 3024 | 6 |

The close capture is structurally older than in-session ones — the index stops
printing at 16:14:59 while the collector keeps its cadence. It is not stale, and
it is the settlement snapshot. The usable threshold window is
**(22.07, 32.10) minutes**; 27 sits near the middle with ~5 minutes of margin on
each side.
