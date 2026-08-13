# `spxrnd.analytics`

Black-76 pricing, implied-volatility inversion, and the static no-arbitrage
conditions a chain must satisfy before a density can be read off it.

**Status: complete.** 344 tests, 100% statement coverage — `bs`, `arbitrage`,
`smile`, `density`, `moments`, `surface`.

## `bs` — Black-76

Black-76, not Black–Scholes, because the model is written on the **forward** —
which `curate.forward` measures directly from put–call parity. Black–Scholes
would need a spot, a rate and a dividend yield, and the dividend yield is
precisely the quantity nobody observes; it would have to be backed out of the
same parity relation and fed back in. Working on the forward skips that round
trip and its error.

```python
price(forward, strike, total_vol, *, discount, right) -> array
vega(forward, strike, total_vol, *, discount, tenor=None) -> array
price_bounds(forward, strike, *, discount, right) -> (lower, upper)
implied_total_vol(observed, forward, strike, *, discount, right) -> (v, ok)
implied_vol(observed, forward, strike, tenor, *, discount, right) -> (sigma, ok)
```

Everything is written in **total volatility** `v = σ√T` rather than `σ` and `T`
separately. It is what the formula actually depends on, it stays well-scaled as
`T → 0`, and it is the natural variable for the no-arbitrage conditions.
`implied_vol` divides by `√T` at the very end — which is where a short tenor
hurts, magnifying any error by 19× at one day.

**Inversion is Newton, then Brent.** The price is monotone in `v` with an
analytic derivative, so Newton converges in a handful of steps almost
everywhere. Almost: in the far wings vega underflows to zero and the step is
undefined — exactly where quotes are widest and an unnoticed failure most
likely. Brent on a bracket catches those.

**It never raises on a bad price.** A chain of 13,414 quotes contains prices no
volatility can produce; the result is `NaN` with `ok=False`. Aborting a run over
one contract would be the wrong trade.

### Verified against the real chain

| | |
|---|---|
| Round-trip precision | **2×10⁻¹⁶** (price a vol, invert, recover) |
| Model put–call parity | holds to 7×10⁻¹⁵ |
| Aug 11 close, OTM quotes inverted | **13,414 / 13,414 (100%)** |
| Median abs. diff vs CBOE's `iv` | **0.000284** vol points |
| Within 1% relative | 97.3% |

Correlation with CBOE by tenor bucket: 0.99988 (<1w), 0.99997 (1w–1m), 0.99999
(1–3m), 0.99999 (3m–1y).

### Where we disagree with CBOE — and why we're right

The `>1y` bucket correlates at only 0.667. The disagreement is concentrated in
**long-dated deep-OTM puts**, and it is CBOE's numbers that are broken. The
2030-12-20 put smile, `F = 9171.71`:

| Strike | ours | CBOE |
|---|---|---|
| 5100 | 0.2667 | 0.3472 |
| 5300 | 0.2623 | **0.7493** |
| 5500 | 0.2581 | **0.0000** |
| 5675 | 0.2546 | **0.0000** |
| 5800 | 0.2521 | 0.7330 |

Their values jump 0.35 → 0.75 between adjacent strikes with `0.0000` sentinels
interleaved. Ours is monotone across all 45 strikes; theirs is not. Measured on
the standard deviation of first differences, **ours is 6.6× smoother** (0.029 vs
0.192).

That is not a different-but-valid convention — a smile cannot be discontinuous
in strike and also correct. This is the concrete payoff of inverting our own
vols and treating the published column as a cross-check only.

## `arbitrage` — static conditions

These follow from the payoffs alone. They are not model assumptions, and a
breach means the quotes are inconsistent or we have processed them wrongly.

```python
check_vertical(strikes, calls, *, discount, bids=None, asks=None) -> [Violation]
check_butterfly(strikes, calls, *, forward, bids=None, asks=None) -> [Violation]
check_calendar(frame) -> [Violation]
check_chain(frame) -> ArbitrageReport
```

| Condition | Statement | Equivalent to |
|---|---|---|
| **Vertical** | `-D ≤ dC/dK ≤ 0` | implied CDF in [0, 1] |
| **Butterfly** | `C` convex in `K` | **implied density ≥ 0** |
| **Calendar** | total variance non-decreasing in `T` | no calendar spread arbitrage |

Butterfly convexity is not a diagnostic here — Breeden–Litzenberger reads the
density off `d²C/dK²`, so **it is the density's non-negativity**. A chain that
breaches it yields negative probability mass, and a smile fit allowed to
interpolate through such a region produces a density that integrates to 1 while
being negative in the middle, which looks entirely plausible in a plot.

### On mids vs executable

The distinction the whole report turns on. Stated on mid prices, these
conditions are breached constantly by nothing more than tick granularity. An
**executable** breach is one you could actually put on at the quoted prices —
wings at the ask, body at the bid, and still collect.

Aug 11 close, 13,414 curated OTM quotes:

| Condition | checked | on mids | **executable** |
|---|---|---|---|
| butterfly | 13,298 | 1,929 | **1** |
| vertical | 13,356 | 10 | **0** |
| calendar | 58 | 0 | **0** |

All ten vertical breaches are **exactly one tick** (SPX quotes in 0.05
increments). The single surviving butterfly is at-the-money on the 2026-08-21
SPXW expiry, worth about **nine cents** — 6.4×10⁻⁵ of the forward. Reporting the
mid count alone would claim 1,929 arbitrages in a chain that has at most one.

### Two implementation details that matter

**Uneven strike spacing.** The butterfly weights are the linear-interpolation
ones, `(k₃-k₂)/(k₃-k₁)` and `(k₂-k₁)/(k₃-k₁)`, not `(1, -2, 1)`. This chain mixes
5-point and 400-point gaps; treating them as even manufactures violations that
are not there.

**Calendar compares within a root only.** Sorting every `(expiry, root)` curve
into one sequence by tenor puts the SPX and SPXW legs of a third Friday
adjacent — two different contracts 6.5 hours apart — and comparing them is not a
calendar condition. It was, briefly, and it manufactured all four "violations"
the first run reported. There is now a test for it.

The checks run on OTM quotes converted to call equivalents by parity,
`C(K) = P(K) + D·(F − K)`, using the same forward and discount the chain was
curated with. Exact, and it avoids needing the ITM calls we deliberately
excluded.

## Example

```python
from spxrnd.analytics import bs, arbitrage

sigma, ok = bs.implied_vol(
    chain.quotes["mid"],
    chain.quotes["forward"],
    chain.quotes["strike"],
    chain.quotes["tenor_years"],
    discount=chain.quotes["discount"],
    right="call",
)
report = arbitrage.check_chain(chain.quotes.assign(total_variance=sigma**2 * t))
print(report.summary())
assert report.clean  # no executable breaches
```


---

## `smile` — SVI, constrained

Raw SVI in total variance, `w(k) = a + b(ρ(k−m) + √((k−m)² + σ²))`. A hyperbola
with two linear asymptotes, which is the right shape because Lee's moment
formula requires total variance to grow at most linearly in `|k|`.

The fit is penalised on the **Durrleman function**
`g(k) = (1 − kw'/2w)² − (w'/4)²(1/w + ¼) + w''/2`, which satisfies `q(K) ≥ 0 ⟺
g(k) ≥ 0`. So the arbitrage constraint is not bolted on afterwards — it is the
density's non-negativity written in the smile's own variables.

### Lee's bound is enforced, and here is why

`b(1+|ρ|) ≤ 2`. Left unconstrained on the 2027-04-16 expiry the solver found
**`b = 76.6`, `ρ = +0.996`** — a wing slope of 153, and a *positive* skew on an
equity index. It matched the quoted strikes to **0.002 vol points** and was
nonsense everywhere else: the density spread across strikes from 2.7 to
2.4×10⁷, went negative, and returned **`E[S_T] = −1.4×10⁻⁶` against a forward of
7923**.

Six expiries did this. The lesson generalises: in-sample residual says nothing
about extrapolation, and the wings are where a density integral spends its time.

## `density` — Breeden–Litzenberger

Closed form, `q(K) = g(k) / (K√(2πw)) · e^{−d₂²/2}`, rather than differencing a
numerically-priced call twice. The finite-difference route is kept as an
independent check on the algebra; the two agree to 10⁻⁵ across the core.

**The grid is adaptive, and that was a bug worth having.** SVI's wings make the
density decay like `e^{−|k|/2b}` — a rate set by the *wing slope*, not by ATM
volatility. On the 2026-09-18 expiry the ATM total vol is 0.0422, so a
"generous" eight standard deviations spanned ±0.34 in log-moneyness — **narrower
than the quoted strikes**, which reach −0.885. That truncated 0.28% of the mass
and pulled `E[S_T]` 0.18% below the forward. Expansion now stops when the edge
density falls below 10⁻¹³ of its peak.

## `moments` — BKM, the independent route

Carr–Madan spanning integrals of option prices: no density, no Durrleman
function, no density grid. `E[X]` comes from the log contract itself rather than
BKM's series approximation for `μ`, which is stated for small returns and put
the skewness 34% and kurtosis 64% off on a 38-day equity smile.

### The cross-check, on real data

| Expiry | quotes | variance | skewness | kurtosis |
|---|---|---|---|---|
| 10-day | 377 | 2.1×10⁻⁹ | 4.5×10⁻⁸ | −2.0×10⁻⁷ |
| 38-day | 386 | 1.0×10⁻⁸ | 5.5×10⁻⁸ | −1.4×10⁻⁷ |
| 1-year | 199 | −5.4×10⁻⁵ | 3.6×10⁻⁴ | −1.6×10⁻³ |
| 5.4-year | 18 | −2.0×10⁻³ | 7.8×10⁻³ | −2.2×10⁻² |

Seven to eight significant figures where quotes are dense, degrading exactly
where they thin out — and the degradation is itself the quality signal.

**Both routes must integrate over the same range.** Truncating BKM's while the
density kept its own left the variance 7% low and the kurtosis 64% off. Two
estimates over different ranges are not a cross-check of anything.

## `surface` — every expiry, with a verdict

```python
estimate_all(quotes) -> [ExpiryEstimate]
to_frame(estimates) -> DataFrame
summary(estimates) -> str
```

Six checks per expiry: fit RMSE, Durrleman minimum, density mass, **`E[S_T]` vs
the parity forward**, minimum density relative to peak, and BL-vs-BKM agreement.
Nothing is silently dropped; what varies is the verdict.

The `E[S_T]` check was computed and displayed for a while before being *gated*
on, which let six broken expiries through reporting `E[S_T]` near zero. It is
the strongest single check available — the density and the parity forward are
entirely independent computations.

### Aug 11 close, all expiries

| | |
|---|---|
| (expiry, root) pairs | 58 |
| **Trustworthy** | **38** |
| Not estimable | 1 |
| Fit RMSE | max 0.0039 vol pts |
| Density mass | worst \|1−m\| = 1.3×10⁻⁵ |
| **`E[S_T]` vs forward** | **worst 4.8×10⁻⁷** |
| Annualised vol | 11.65% (6d) → 29.03% (5.4y) |
| Skew | −1.15 → −4.83 → −2.25 |

The 20 flagged expiries are all long-dated, and they are flagged for marginal
Durrleman negativity (~10⁻⁶) or BL/BKM disagreement above 1% — both symptoms of
thin data past a year, both reported rather than hidden.
