# spxrnd

Collect synchronized SPX option-chain snapshots from CBOE's public delayed-quote
endpoint, curate them into a validated time series, and estimate risk-neutral
densities from them.

The endpoint returns the **entire chain and the underlying index level in one
payload under one timestamp**. That single property is what makes the advertised
15-minute delay irrelevant: every snapshot is internally consistent, and internal
consistency — not freshness — is what density estimation requires.

## Status

Under construction, built in gated stages. See
`.claude/plans/` for the build plan.

| Stage | Component | State |
|---|---|---|
| 0 | Relocation, repo skeleton | done |
| 1 | Fixture corpus, seed regression tests | done |
| 2 | `ingest` — fetch, validate, gate, write | done |
| 3 | `store` — archive, Parquet, DuckDB catalog | done |
| 4 | `curate` — quality filters, forward curve | done |
| 5 | `analytics` — Black-76, IV, arbitrage checks | done |
| 6 | RND — smile fit, Breeden–Litzenberger, BKM | done |
| 7 | CLI, launchd scheduling, docs | — |

## Quickstart

```bash
make install     # venv + package with all extras
make test        # full offline suite, no network
make help        # everything else
```

## Architecture

Components form a strict one-way chain. Nothing flows backwards.

```
ingest ──▶ store ──▶ curate ──▶ analytics ──▶ cli
```

| Component | Responsibility | Dependencies |
|---|---|---|
| `ingest` | HTTP fetch, payload validation, freshness/dedup gating, atomic write | **stdlib only** |
| `store` | Immutable raw archive, canonical schema, Parquet, DuckDB catalog | pyarrow, duckdb |
| `curate` | Quote-quality filters, put–call parity forward curve, moneyness | numpy, pandas |
| `analytics` | Black-76, IV inversion, arbitrage checks, smile fit, density, moments | numpy, scipy |
| `cli` | `collect` / `backfill` / `curate` / `estimate` | all of the above |

`ingest` is stdlib-only so the collector deploys anywhere and cannot break from a
dependency upgrade. That constraint is enforced by a test that AST-parses the
package and rejects any non-stdlib import — not by convention.

Every component directory carries its own `README.md` documenting its public
API, inputs and outputs, failure modes, and a runnable example.

## Data layout

`data/` is gitignored — it holds 500 MB+ and grows with every capture.

```
data/
├── raw/                    immutable JSON archive; the source of truth
├── curated/                Parquet, derived, safe to delete and rebuild
└── archive/csv_original/   the 29 defective CSVs, retired (see below)
```

**The raw archive is the only source of truth.** Everything else is derived and
reproducible by `make backfill`. Never edit `data/raw/`.

### What is currently collected

| | |
|---|---|
| Captures | 31 |
| Raw archive | **55 MB** gzipped (was 392 MB, 7.1× reduction) |
| Curated | 55.9 MB Parquet, **952,564 rows** |
| Unparseable symbols | 0 |

- One full session: **Aug 11 2026, 23 captures, 16:11Z–20:37Z** (12:11–16:37 ET)
  at ~10-minute cadence, index 7741 → 7728. Gaps at 16:32→16:50 and 17:10→17:56.
- Two settlement snapshots: **Aug 11** (spot 7728.20, 30,692 contracts) and
  **Aug 12** (spot 7748.50, 30,600).
- Five post-close repeats and two off-hours strays, all retained — the Aug 9
  capture is a *weekend* fetch serving Aug 7's close.
- The Aug 11 close holds **30,692 options** across 55 expiries (same-day out to
  Dec 2031), split SPX 10,156 / SPXW 20,536.

### Why the original CSVs were retired

They are in `data/archive/csv_original/` and nothing reads them. All three of
their defects are reproducible from the raw JSON, so they are redundant as well
as wrong:

1. `cboe_timestamp` is empty in **every row ever written** — it was read from
   `payload["data"]["timestamp"]`, but that key lives at the top level.
2. No `root` column. SPX and SPXW share strikes on five expiries (260821,
   260918, 261016, 261120, 261218 — 1,164 vs 992 contracts on 260821 alone).
   Different contracts, different quotes, indistinguishable rows.
3. `bid_size`, `ask_size`, and `theo` were dropped.

Keeping them where a future script could find them is a correctness hazard.

## Testing

Every public function is tested on its nominal path, its boundary, and its
failure mode. Fixtures are **real captured payloads**, trimmed — including the
pathological ones. Every bug found becomes a permanent regression test.

```bash
make test        # offline, deterministic
make test-live   # the single opt-in test against the real endpoint
make cov         # coverage report
```

The CBOE endpoint is a free courtesy service. The default suite never touches
it.
