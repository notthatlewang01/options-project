# `spxrnd.curate`

Turn a stored capture into a chain an estimator can consume: a forward and
discount factor per expiry, a tenor and log-moneyness per quote, and a defensible
answer to which quotes are usable.

**Status: complete.** 97 tests.

## Order of operations

Fixed here rather than left to callers, because two of the steps look
interchangeable and are not.

```
1. mid prices          needed by the parity fit
2. forward per expiry  needed by moneyness -> needed by the ITM flag -> a filter
3. tenor + moneyness   joined onto every quote
4. quality flags       computed for every row
5. drop                and report what it cost
```

**The forward is fitted on ITM quotes too; the density is estimated only on OTM
ones.** Both are right, and the asymmetry is the subtle part. Put–call parity is
an identity holding across the whole strike range, best measured where both legs
are liquid. A density wants OTM options because their prices are mostly time
value. Filtering before fitting would fit parity on a set with one leg already
removed.

## `moneyness` — time and where a strike sits

```python
expiry_instant(expiry, root) -> datetime     # timezone-aware
tenor_years(expiry, root, *, as_of) -> float # ACT/365
log_moneyness(strike, forward) -> float
is_otm(right, strike, forward) -> bool
```

**Settlement time depends on the root.** SPX is AM-settled — SET is built from
the opening prints, so it is done at 09:30 ET. SPXW runs to 16:00 ET the same
day. On a third Friday both list the same date, and treating them alike
misprices the shorter one by 6.5 hours — 27% of a one-day option's remaining
life. This is the `root` collision from `ingest.osi` reappearing in the time
dimension.

Tenor **may be zero or negative**: the Aug 11 capture at 16:37 ET contains SPXW
contracts that settled at 16:00. That is information, not an error. `quality`
filters it explicitly.

Moneyness is measured against the **forward**, not spot. They differ by 24% at
the 5-year expiry; using spot would misclassify the entire long end.

## `quality` — which quotes are usable

Every rule is a named flag computed for every row; dropping is a separate step
that consumes them. So attrition is always reportable, and any rule can be
switched off to test how much the answer depends on it.

| Flag | Rule | Aug 11 close |
|---|---|---|
| `itm` | ITM against the forward | 15,065 (49.1%) |
| `zero_iv` | feed's `iv == 0.0` marker | 1,612 (5.3%) |
| `wide_spread` | relative spread > 50% | 1,548 (5.0%) |
| `zero_bid` | nobody buying | 837 (2.7%) |
| `zero_dte` | under one day to expiry | 562 (1.8%) |
| `crossed` | bid > ask | **0** |
| `zero_ask` | nobody offering | **0** |

Rules overlap — a zero-bid quote is usually wide-spread too — so the counts do
not sum to the number dropped.

### `iv == 0.0` is flagged, not dropped

This is a deliberate deviation from the filter list originally proposed, and the
data is the reason. `iv == 0.0` is the feed's "not computed" marker, not a
property of the quote; those contracts have live two-sided markets and perfectly
good prices. Measured on the Aug 11 close it tracks *moneyness*, not the smile:

| Bucket | quoted | `iv==0` | | mean \|delta\| |
|---|---|---|---|---|
| deep ITM calls | 5,837 | 1,156 | 19.8% | 0.995 |
| within 10% of spot | 16,421 | 694 | 4.2% | 0.988 |
| deep ITM puts | 1,159 | 31 | 2.7% | 0.916 |
| everything else | 6,157 | 9 | 0.1% | 0.182 |

**684 of those 694 "near-spot" contracts are ITM**, |delta| 0.665–1.0. So the
sentinel marks contracts whose price is nearly all intrinsic — exactly where
CBOE declines to invert a vol, and reasonably so. The OTM rule removes that
whole region for an independent and better reason: inverting a vol from an
all-intrinsic price amplifies quoting noise without limit.

Filtering on `iv == 0.0` directly would be treating a symptom. It stays a flag,
it stays in the attrition table, and `Filters(drop={Flag.ZERO_IV})` turns it on
for a sensitivity check.

`last_trade_price` is never consulted at all — one quoted contract in six sits
over 25% from its own live mid.

## `forward` — implied from the quotes, not from a rate feed

```python
fit_one(chain, *, expiry, root, tenor_years, ...) -> ForwardFit
fit_all(quotes, *, as_of, ...) -> list[ForwardFit]
```

For European options, `C(K) - P(K) = D · (F - K)`. Regressing `C - P` on `K`
gives `D = -slope` and `F = intercept / D`. Self-contained, internally
consistent with the quotes being differentiated, and what CBOE's own VIX
methodology does — no second data feed to collect, align and version.

**Never mixes roots.** Pairing an SPX call with an SPXW put at a shared strike
produces a `C - P` that is not a parity relation, and the forward is silently
wrong. A test constructs exactly that and shows the error.

**Thin or poorly-fitting expiries are excluded with a recorded reason**, never
extrapolated: `< 8 pairs`, `R² < 0.99`, non-positive or implausible discount,
implied rate outside ±25%, already expired. A fitted-but-wrong forward poisons
every density built on it while looking exactly like a good one.

### Measured on the Aug 11 close

59 usable forwards of 60 (expiry, root) pairs. The single exclusion is the
0-DTE SPXW expiry, which had settled 37 minutes before the capture.

| | |
|---|---|
| Parity R² | min 0.999997, median 1.000000 |
| Forward vs spot | +0.027% at 1 day → +23.9% at 5.4 years |
| Implied rate, beyond 2 weeks | 3.89% – 6.30%, converging to ~4.65% |
| Implied rate, inside 2 weeks | 3.96% – 9.96% |

The short-end rate scatter is **structural, not a fit failure**. `rate =
-ln(D)/T`, so as `T → 0` any error in `D` is amplified by `1/T` — 365× at one
day. The forwards themselves are fine, and the density needs the forward and
discount factor, not the rate.

The forward curve is **not exactly monotonic**, and asserting that it should be
would be demanding more precision than the quotes carry. Eight steps decrease;
five of them are the third-Friday SPX/SPXW pairs, which are different contracts
settling 6.5 hours apart rather than kinks in one curve. The remaining three are
under 0.002% of the level — a fraction of a tick on a 7731 index.

## `chain` — the assembled result

```python
build(quotes, *, as_of, spot, filters=None) -> CuratedChain
from_catalog(con, capture_utc, **kwargs) -> CuratedChain
```

`CuratedChain.summary()` prints the attrition table, the parity fit quality, the
implied rate range, and every excluded expiry with its reason. **Read it before
trusting `.quotes`** — a rule that ate an entire wing is invisible in the kept
frame alone.

```python
from spxrnd.store import catalog
from spxrnd.curate import chain

con = catalog.connect(Path("data/curated"))
c = chain.from_catalog(con, datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC))
print(c.summary())
c.for_expiry(date(2026, 9, 18), "SPXW")  # one clean smile, sorted by strike
```

Result on the Aug 11 close: **30,692 quotes in, 13,414 kept (43.7%)**, every
survivor OTM with a live two-sided market, and every expiry retaining strikes on
both sides of the forward.
