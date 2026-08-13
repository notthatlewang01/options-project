# deploy/

The launchd job that collects on a schedule.

## Install

```bash
make install-agent
```

Or by hand:

```bash
cp deploy/com.spxrnd.collect.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.spxrnd.collect.plist
```

Verify it registered:

```bash
launchctl list | grep spxrnd
```

Then, after the next scheduled tick:

```bash
spxrnd status
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.spxrnd.collect.plist
rm ~/Library/LaunchAgents/com.spxrnd.collect.plist
```

Nothing in `data/` is touched. The archive is the point; the scheduler is
replaceable.

## The schedule

**09:35 to 16:25, every 10 minutes, Monday to Friday** — 42 captures a day, 210
calendar entries.

The plist is **generated**, not hand-written:

```bash
python3 tools/make_launchd_plist.py
```

A test asserts the checked-in file still matches the generator, so a schedule
change cannot land in only one of them.

### Why weekdays only

The freshness gate would reject weekend runs correctly — on a non-trading day
the feed never freshens, which is also how holidays are handled without an
exchange calendar. But it would cost ~84 pointless requests every weekend
against a free courtesy endpoint. Politeness is cheap here.

### Why 09:35 and not 09:30

The feed is 15 minutes delayed, so nothing new exists at the open. Starting at
09:35 means the first capture carries a real print rather than the previous
session's close.

### Why 16:25

The cash index stops printing at 16:14:59 ET and the feed serves that print
about 15 minutes later. A 16:25 tick is the one that captures **settlement** —
the most valuable snapshot of the day, and the one an inherited 22-minute
staleness limit would have discarded by four seconds. See
`src/spxrnd/ingest/README.md`.

## ⚠️ launchd schedules in local time

`StartCalendarInterval` is the machine's local clock, not UTC. This host is on
`America/New_York`, so the entries line up with the session.

**Move the machine to another timezone and the schedule slides with it.** The
freshness gate makes that a safe failure — off-session runs cost one HTTP
request and write nothing — but a *silent* one: you would see skips, not
errors. `spxrnd status` is what surfaces it, by reporting how long since the
last successful write.

If the machine changes timezone, regenerate and reload:

```bash
python3 tools/make_launchd_plist.py
launchctl unload ~/Library/LaunchAgents/com.spxrnd.collect.plist
cp deploy/com.spxrnd.collect.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.spxrnd.collect.plist
```

## What the job actually runs

```
/Users/1off_k/options/.venv/bin/python -m spxrnd.cli.main --dir <repo>/data collect
```

Four things about that line are deliberate:

- **Absolute paths throughout.** launchd inherits no working directory and no
  shell profile.
- **The venv's python**, not `/usr/bin/python3`, which has none of the package.
- **`collect` and nothing else.** `backfill` and `estimate` import the
  scientific stack and take tens of seconds; the scheduled job must stay a
  single HTTP request.
- **`python -m`**, not the console script, so it works whether or not the
  package is on `PATH`.

`RunAtLoad` is `false`: loading the job should not fire a capture at an
arbitrary moment that the gate would reject anyway.

## Logs

```
logs/collect.log       stdout, one timestamped line per run
logs/collect.err.log   stderr
```

Gitignored. A normal off-hours run looks like this — **a skip is a success**:

```
2026-08-13T05:26:27Z stale_feed: feed is stale: index last printed
2026-08-12T16:14:59-04:00 ET, 551.5 min ago, over the 27 min limit
```

## When something is wrong

`spxrnd status` distinguishes the two failures, because they need different
fixes:

| Symptom | Meaning | Fix |
|---|---|---|
| `SCHEDULER: no attempt for Nh` | the job is not being run | `launchctl list \| grep spxrnd` |
| `FEED: last successful fetch Nh ago` | job runs, endpoint does not answer | check the network, then CBOE |
| `FAILING: N consecutive` | fetch or parse is erroring | the message names the cause |

A sleeping laptop misses captures silently and unrecoverably — launchd does not
run a missed calendar entry on wake. That is the known cost of scheduling this
on a laptop rather than a server, and `spxrnd status` is how you notice.
