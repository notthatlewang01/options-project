# spxrnd

Collect synchronized SPX option-chain snapshots from CBOE's public delayed-quote
endpoint, curate them into a validated time series, and estimate risk-neutral
densities from them.

The endpoint returns the **entire chain and the underlying index level in one
payload under one timestamp**. That single property is what makes the advertised
15-minute delay irrelevant: every snapshot is internally consistent, and internal
consistency — not freshness — is what density estimation requires.

**782 tests, 100% statement coverage**, plus an opt-in smoke test against the
live endpoint.

## Quickstart

```bash
make install          # venv + package with all extras
make test             # 782 tests, offline, deterministic
spxrnd captures       # what is in the archive
spxrnd estimate --term-structure
```

To collect on a schedule:

```bash
make install-agent    # launchd, every 10 min, 09:35-16:25 ET, weekdays
make status           # is it working?
```

## Architecture

Components form a strict one-way chain. Nothing flows backwards.

```
ingest ──▶ store ──▶ curate ──▶ analytics ──▶ cli
```

| Component | Responsibility | Dependencies | Tests |
|---|---|---|---|
| [`ingest`](src/spxrnd/ingest/README.md) | fetch, validate, freshness-gate, atomic write | **stdlib only** | 319 |
| [`store`](src/spxrnd/store/README.md) | immutable archive, Parquet, DuckDB catalog | pyarrow, duckdb | 86 |
| [`curate`](src/spxrnd/curate/README.md) | quality filters, put–call parity forward curve | numpy, pandas | 105 |
| [`analytics`](src/spxrnd/analytics/README.md) | Black-76, IV, arbitrage, SVI, density, moments | numpy, scipy | 344 |
| [`cli`](src/spxrnd/cli/README.md) | `collect` / `status` / `backfill` / `estimate` … | all of the above | 47 |

Each component directory has its own README with the API, the failure modes, and
the measurements behind its design choices.

**`ingest` is stdlib-only** so the collector deploys anywhere and cannot be
broken by a dependency upgrade — a missed capture is unrecoverable. Two tests
enforce it: one AST-parses the package and rejects any non-stdlib import,
another runs `collect` with numpy, scipy, pandas, pyarrow and duckdb blocked at
the import hook.

## Data

`data/` is gitignored. `manifests/` is not — integrity is verifiable from a
fresh clone.

```
data/
├── raw/                    31 captures, 55 MB gzipped — the source of truth
├── curated/                952,564 rows, 56 MB Parquet — derived, rebuildable
└── archive/csv_original/   29 retired CSVs (see below)
```

**The raw archive is the only source of truth.** Everything else is reproducible
by `spxrnd backfill`. Never edit `data/raw/`.

| | |
|---|---|
| Captures | 31, spanning Aug 9 – Aug 13 2026 |
| Raw archive | **55 MB** gzipped (was 392 MB, 7.1×) |
| Curated | 952,564 rows, zero unparseable symbols |
| Manifest | 31/31 verified |

One full session (**Aug 11**, 23 captures at 10-minute cadence, 12:11–16:37 ET)
plus two settlement snapshots: Aug 11 (spot 7728.20, 30,692 contracts) and
Aug 12 (spot 7748.50, 30,600).

### Why the original CSVs were retired

They are in `data/archive/csv_original/` and nothing reads them. All three
defects are reproducible from the raw JSON, so they were redundant as well as
wrong: `cboe_timestamp` empty in **every row ever written**, no `root` column,
and `bid_size`/`ask_size`/`theo` dropped.

## What the pipeline produces

From the Aug 11 close, 30,692 quotes:

| Stage | Result |
|---|---|
| Curated | 13,414 OTM quotes kept (43.7%), attrition reported per rule |
| Forward curve | 59 of 60 expiries, parity **R² min 0.999997** |
| Implied vol | **13,414 / 13,414 inverted**, median 0.000284 from CBOE's |
| Arbitrage | 13,298 butterflies checked, **1 executable breach** (~9¢) |
| Densities | **38 of 58** expiries trustworthy by six independent checks |
| Term structure | vol 11.65% (6d) → 29.03% (5.4y), skew −1.15 → −4.83 → −2.25 |

`E[S_T]` computed from each density reproduces the put–call-parity forward to
**8×10⁻¹⁰**, and Breeden–Litzenberger agrees with BKM to **7–8 significant
figures** where quotes are dense. Those are independent computations; agreement
is the evidence.

## Testing

Every public function is tested on its nominal path, its boundary, and its
failure mode. Fixtures are **real captured payloads**, trimmed — including the
pathological ones. Every bug found becomes a permanent regression test.

```bash
make test        # 782 tests, offline, ~60s
make test-live   # one opt-in test against the real endpoint
make cov         # coverage report
```

The CBOE endpoint is a free courtesy service. The default suite never touches
it.

### The seed regression corpus

Twelve defects, each found in real collected data, each pinned by a test built
from the capture that exhibits it — see
[`tests/test_seed_regressions.py`](tests/test_seed_regressions.py). Every one
produced output that looked entirely reasonable:

1. `cboe_timestamp` read from a key that does not exist — blank in every row ever written
2. The OSI root parsed then discarded — 986 SPX/SPXW collisions on one expiry
3. Post-close freeze — the CDN serves a frozen body under a fresh timestamp
4. `seqno` advances while the index print does not, defeating seqno dedup
5. Weekend fetches serving Friday's close
6. A capture that reached the archive and never reached the curated layer
7. 1,118 zero-bid contracts
8. `last_trade_price` up to two years stale; one quoted contract in six sits >25% from its own mid
9. Degenerate IVs — 800% deep-ITM, and again at 0-DTE as vega vanishes
10. 562 same-day expiries, tenor zero
11. `iv == 0.0` is a "not computed" sentinel, not a volatility — 2,160 contracts
12. `prev_day_close` rolls forward overnight, so a daily return computed from it is exactly zero

Several more were found and fixed during construction and are documented in the
component READMEs — a 22-minute staleness limit that discarded the settlement
capture by four seconds, an SVI fit that reached `b = 76.6` and returned
`E[S_T] = −1.4×10⁻⁶`, a density grid narrower than its own data, and a calendar
arbitrage check that compared SPX against SPXW of the same expiry.

## Scheduling

```bash
make install-agent
make status
```

launchd, 09:35–16:25 local time, every 10 minutes, weekdays. **launchd schedules
in local time, not UTC** — see [`deploy/README.md`](deploy/README.md) for what
that means if the machine moves, and for the known cost of scheduling on a
laptop that sleeps.
