# `spxrnd.cli`

The command line. Seven subcommands, one per thing you actually want to do.

**Status: complete.** 47 tests, 100% statement coverage.

```
spxrnd collect     take exactly one snapshot
spxrnd status      is collection working?
spxrnd verify      check the archive against its checksum manifest
spxrnd captures    what is in the archive
spxrnd backfill    rebuild the curated layer from the raw archive
spxrnd curate      one capture's quality report and forward curve
spxrnd estimate    densities and moments for every expiry
```

`--dir` works before or after the subcommand — `spxrnd --dir data collect` and
`spxrnd collect --dir data` are the same thing. argparse does not do this by
default, and the second form is how a person types it.

## `collect` must stay stdlib-only

It is the one subcommand that runs unattended, every ten minutes, forever, and
a missed capture cannot be recovered — a delayed-quote endpoint serves the
present. So it imports **only** `spxrnd.ingest`, which has no dependencies. A
numpy or pyarrow upgrade cannot stop a capture.

Every other subcommand imports what it needs *inside the function*, not at
module scope. Two tests keep that honest:

- one runs `collect` with numpy, scipy, pandas, pyarrow and duckdb **blocked at
  the import hook** and asserts nothing reached for them
- one AST-parses this module and fails if any of them appears at module level,
  because the first test would otherwise pass by luck of module caching

## Exit codes

| Code | Meaning |
|---|---|
| `0` | worked — **including a deliberate skip** |
| `1` | failed |
| `2` | worked, but something needs attention |

A holiday, a weekend and an off-hours tick all end in a skip that writes
nothing, and all exit `0`. Reporting those as failures would fill the
scheduler's log with non-failures and train whoever reads it to ignore them.

## `status` names which thing is broken

Three clocks, because they fail differently and need different fixes:

| Output | Meaning | Where to look |
|---|---|---|
| `SCHEDULER: no attempt for Nh` | the job is not being run | `launchctl list \| grep spxrnd` |
| `FEED: last successful fetch Nh ago` | job runs, endpoint silent | network, then CBOE |
| `FAILING: N consecutive` | fetch or parse erroring | the message names the cause |
| `healthy` | — | — |

```
$ spxrnd status
last attempt   2026-08-13T05:26:26+00:00  (0m ago)
last ok        2026-08-13T05:26:26+00:00  (0m ago)
last write     None  (never)
last verdict   stale_feed
totals         0 written, 1 skipped, 0 failed

healthy
```

## Examples

```bash
spxrnd captures                    # 31 captures, 57.8 MB, curated layer aligned
spxrnd verify                      # ok 31   corrupted 0   missing 0
spxrnd backfill                    # 952,564 rows, 55.9 MB
spxrnd curate                      # attrition table + forward curve
spxrnd estimate --term-structure   # per-expiry vol, skew, kurtosis
spxrnd estimate --csv terms.csv    # the same, as a file
```

`curate` and `estimate` default to the **newest** capture; `--capture 20-37-03`
selects one by any distinctive part of its timestamp. An unknown value reports
the newest available rather than an empty result.

## Replaying a capture, offline

```bash
spxrnd collect --from-file data/raw/_SPX_2026-08-11T20-37-03Z.json.gz --force
```

Drives the entire ingest path — validation, gating, atomic write, health
record — with no network. `.gz` and plain `.json` both work.
