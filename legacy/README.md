# legacy/

The three original scripts, kept verbatim for reference. **Nothing in the
package imports from here.** They are history, not dependencies.

| File | Date | What it was |
|---|---|---|
| `collect_cboe_snapshots.py` | Aug 9 | The original sleep-loop collector. Every byte of data in `data/raw/` came from this script. |
| `collect_snapshot.py` | Aug 11 | A one-shot rewrite with freshness gating, dedup, gzip, atomic writes, locking, and a meta manifest. Never executed. |
| `rebuild_csv.py` | Aug 11 | Repairs the defective CSVs from retained raw JSON. Never executed. |

## Why they are kept

`collect_snapshot.py` diagnosed four real defects in the original, and the
reasoning in its docstring is the specification the `ingest` component is built
against:

1. `cboe_timestamp` was read from `payload["data"]["timestamp"]`, but that key
   is at the **top level** of the payload — so the column was empty in every row
   ever written.
2. The OSI root was parsed and then discarded. SPX (AM-settled) and SPXW
   (PM-settled weeklies) share strikes on third-Friday expiries; without the
   root those rows are indistinguishable.
3. After the cash index stops printing at 16:15 ET, `last_trade_time` freezes
   while the CDN keeps serving the same body under a fresh top-level timestamp.
   The original wrote several identical post-close snapshots as a result.
4. `seqno` is not a safe dedup key: it advanced once (16947306680) while the
   index print stayed frozen at 16:14:59.

Those four, plus six more found while surveying the collected data, are the seed
regression tests in `tests/`. See the top-level `README.md`.

## Running them

Don't. They write to relative paths and have no tests. Use the `spxrnd` CLI.
