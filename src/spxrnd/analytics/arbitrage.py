"""Static no-arbitrage conditions on a chain.

These are the conditions a set of option prices must satisfy for *any*
risk-neutral density to exist. They are not model assumptions and not
statistical tendencies -- they follow from the payoffs alone, and a violation
means either the quotes are inconsistent or we have processed them wrongly.

Which is exactly why they are worth checking before Stage 6 rather than after.
Breeden-Litzenberger reads the density off the second derivative of the call
price in strike:

    q(K) = (1/D) * d^2C/dK^2

so **butterfly convexity is not a diagnostic here, it is the density's
non-negativity**. A chain that violates convexity produces a "density" with
negative mass, and a smile fit that is allowed to interpolate through such a
region produces one that integrates to 1 while being negative somewhere in the
middle -- which looks entirely plausible in a plot.

Three conditions, in increasing strength:

**Vertical** -- `-D <= dC/dK <= 0`. A call spread costs between nothing and the
discounted strike difference. Equivalently the implied CDF stays in [0, 1].

**Butterfly** -- `C` is convex in `K`. Equivalently the implied density is
non-negative.

**Calendar** -- total implied variance is non-decreasing in tenor at fixed
log-moneyness. Equivalently a longer-dated option cannot be worth less than a
shorter one struck at the same relative level.

Everything is checked on **mid prices as we curated them**, so a violation is a
statement about our pipeline's output rather than about a raw feed we do not
control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Violation:
    """One breach of a static condition, with enough context to act on it."""

    kind: str
    expiry: object
    root: str
    strikes: tuple[float, ...]
    magnitude: float
    """How badly, in the natural units of the condition. Always positive."""

    relative: float
    """Magnitude scaled by something comparable -- the forward for price
    conditions, the variance level for calendar. Lets violations from different
    expiries be ranked against each other."""

    executable: bool = False
    """Whether the breach survives the quoted bid-ask.

    The distinction that makes this report usable. A condition stated on mids
    is violated constantly by nothing more than tick granularity: SPX quotes in
    0.05 increments, and every one of the vertical breaches in the Aug 11 close
    is *exactly* one tick. Those are not arbitrages -- you cannot trade a mid.

    An executable breach is one you could actually put on at the quoted prices:
    buy the wings at the ask, sell the body at the bid, and still collect. Only
    those mean the chain is genuinely inconsistent.
    """

    detail: str = ""


@dataclass
class ArbitrageReport:
    violations: list[Violation] = field(default_factory=list)
    n_checked: dict[str, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        """No *executable* breaches. Mid-level noise does not count."""
        return not self.executable

    @property
    def executable(self) -> list[Violation]:
        """Breaches that survive the quoted bid-ask. The ones that matter."""
        return [v for v in self.violations if v.executable]

    def of_kind(self, kind: str, *, executable_only: bool = False) -> list[Violation]:
        pool = self.executable if executable_only else self.violations
        return [v for v in pool if v.kind == kind]

    def worst(self, kind: str | None = None) -> Violation | None:
        pool = self.of_kind(kind) if kind else self.violations
        return max(pool, key=lambda v: v.relative, default=None)

    def summary(self) -> str:
        lines = [
            f"  {'condition':<10}{'checked':>10}{'on mids':>10}{'executable':>12}"
            f"{'worst rel':>12}"
        ]
        for kind, n in sorted(self.n_checked.items()):
            hits = self.of_kind(kind)
            real = [h for h in hits if h.executable]
            worst = f"{max((h.relative for h in real), default=0.0):.2e}"
            lines.append(
                f"  {kind:<10}{n:>10,}{len(hits):>10,}{len(real):>12,}{worst:>12}"
            )
        lines.append("")
        lines.append(
            "  'on mids' counts breaches of the stated condition; 'executable'"
        )
        lines.append("  counts those tradeable at the quoted bid-ask. Only the latter")
        lines.append("  means the chain is inconsistent.")
        return "\n".join(lines)


def check_vertical(
    strikes,
    calls,
    *,
    discount: float = 1.0,
    tol: float = 1e-9,
    expiry=None,
    root="",
    bids=None,
    asks=None,
) -> list[Violation]:
    """`-D <= dC/dK <= 0` across adjacent strikes.

    The call price must fall as the strike rises (you cannot pay more for the
    right to buy at a worse price) and cannot fall faster than the discounted
    strike difference (or the spread would cost more than it can ever pay).
    Together they are the implied CDF staying in [0, 1], stated in prices.

    With `bids`/`asks` given, each breach is additionally classified as
    executable: whether the spread could be sold at the quoted prices for more
    than its maximum payoff.
    """
    strikes = np.asarray(strikes, dtype=float)
    calls = np.asarray(calls, dtype=float)
    order = np.argsort(strikes)
    strikes, calls = strikes[order], calls[order]
    have_quotes = bids is not None and asks is not None
    if have_quotes:
        bids = np.asarray(bids, dtype=float)[order]
        asks = np.asarray(asks, dtype=float)[order]

    dk = np.diff(strikes)
    dc = np.diff(calls)
    slope = np.divide(dc, dk, out=np.zeros_like(dc), where=dk > 0)

    out = []
    for i in np.flatnonzero(slope > tol):
        # Tradeable only if buying the low strike and selling the high one is a
        # credit: a position with a payoff that is never negative.
        executable = bool(have_quotes and (bids[i + 1] - asks[i] > tol))
        out.append(
            Violation(
                "vertical",
                expiry,
                root,
                (float(strikes[i]), float(strikes[i + 1])),
                magnitude=float(slope[i]),
                relative=float(slope[i]),
                executable=executable,
                detail=f"call price rises with strike: dC/dK = {slope[i]:+.6f} > 0",
            )
        )
    for i in np.flatnonzero(slope < -discount - tol):
        # Sell the spread for more than its discounted maximum payoff.
        executable = bool(
            have_quotes and (bids[i] - asks[i + 1] - discount * dk[i] > tol)
        )
        out.append(
            Violation(
                "vertical",
                expiry,
                root,
                (float(strikes[i]), float(strikes[i + 1])),
                magnitude=float(-slope[i] - discount),
                relative=float(-slope[i] - discount),
                executable=executable,
                detail=(
                    f"call spread cheaper than its worst case: dC/dK = "
                    f"{slope[i]:.6f} < -D = {-discount:.6f}"
                ),
            )
        )
    return out


def check_butterfly(
    strikes,
    calls,
    *,
    tol: float = 1e-9,
    expiry=None,
    root="",
    forward: float = 1.0,
    bids=None,
    asks=None,
) -> list[Violation]:
    """Convexity of `C` in `K`, on every consecutive strike triple.

    A butterfly -- long the wings, short two bodies -- has a payoff that is
    never negative, so it cannot cost less than zero. With unevenly spaced
    strikes the weights are the linear-interpolation ones rather than
    (1, -2, 1); treating uneven spacing as even manufactures violations that
    are not there.

    **This is the density's non-negativity.** An executable breach means
    Stage 6 would produce negative probability mass over that strike interval.

    With `bids`/`asks` given, executability is the cost of actually putting the
    structure on -- wings at the ask, body at the bid. On mids alone this
    condition is breached constantly by tick granularity, which is noise rather
    than an inconsistency.
    """
    strikes = np.asarray(strikes, dtype=float)
    calls = np.asarray(calls, dtype=float)
    order = np.argsort(strikes)
    strikes, calls = strikes[order], calls[order]
    have_quotes = bids is not None and asks is not None
    if have_quotes:
        bids = np.asarray(bids, dtype=float)[order]
        asks = np.asarray(asks, dtype=float)[order]
    if len(strikes) < 3:
        return []

    k1, k2, k3 = strikes[:-2], strikes[1:-1], strikes[2:]
    c1, c2, c3 = calls[:-2], calls[1:-1], calls[2:]

    # Weights making this a butterfly with unit body: long (k3-k2)/(k3-k1) at
    # k1, long (k2-k1)/(k3-k1) at k3, short 1 at k2.
    span = k3 - k1
    w1 = np.divide(k3 - k2, span, out=np.zeros_like(span), where=span > 0)
    w3 = np.divide(k2 - k1, span, out=np.zeros_like(span), where=span > 0)
    value = w1 * c1 + w3 * c3 - c2

    cost = (w1 * asks[:-2] + w3 * asks[2:] - bids[1:-1]) if have_quotes else value

    out = []
    for i in np.flatnonzero(value < -tol):
        out.append(
            Violation(
                "butterfly",
                expiry,
                root,
                (float(k1[i]), float(k2[i]), float(k3[i])),
                magnitude=float(-value[i]),
                relative=float(-value[i] / forward) if forward else float(-value[i]),
                executable=bool(cost[i] < -tol),
                detail=(
                    f"butterfly value {value[i]:.6e} on mids"
                    + (f", {cost[i]:.6e} at the quoted bid-ask" if have_quotes else "")
                    + f" -- negative density between {k1[i]:.0f} and {k3[i]:.0f}"
                ),
            )
        )
    return out


def check_calendar(
    frame: pd.DataFrame, *, tol: float = 1e-6, n_grid: int = 41
) -> list[Violation]:
    """Total implied variance non-decreasing in tenor, at fixed log-moneyness.

    `w(k, T) = sigma(k, T)^2 * T` must not fall as `T` rises, or a calendar
    spread would have negative value.

    Comparison is at fixed **log-moneyness**, not fixed strike. At fixed strike
    the forward moves between expiries -- by 24% across this chain -- so the two
    options are not at comparable places on their own smiles, and a spurious
    violation is guaranteed. Each expiry is interpolated onto a shared
    log-moneyness grid over the range they actually have in common.

    Args:
        frame: needs `expiry`, `root`, `tenor_years`, `log_moneyness`,
            `total_variance`.
    """
    needed = {"tenor_years", "log_moneyness", "total_variance"}
    missing = needed - set(frame.columns)
    if missing:
        raise KeyError(f"check_calendar needs columns {sorted(missing)}")

    by_root: dict[str, list] = {}
    for (expiry, root), g in frame.groupby(["expiry", "root"], observed=True):
        g = g.dropna(subset=["log_moneyness", "total_variance"]).sort_values(
            "log_moneyness"
        )
        # Average duplicate log-moneyness points (a call and a put can land on
        # the same one) so the curve is a function.
        g = g.groupby("log_moneyness", as_index=False)["total_variance"].mean()
        if len(g) >= 4:
            tenor = float(
                frame.loc[
                    (frame["expiry"] == expiry) & (frame["root"] == root),
                    "tenor_years",
                ].iloc[0]
            )
            by_root.setdefault(root, []).append(
                (
                    tenor,
                    expiry,
                    root,
                    g["log_moneyness"].to_numpy(),
                    g["total_variance"].to_numpy(),
                )
            )

    # Within a root only. Sorting every (expiry, root) curve into one sequence
    # puts the SPX and SPXW legs of a third Friday next to each other -- two
    # different contracts 6.5 hours apart -- and comparing those is not a
    # calendar condition. It was, briefly, and it manufactured every violation
    # the first run reported.
    pairs = []
    for curves in by_root.values():
        curves.sort(key=lambda c: c[0])
        pairs += list(zip(curves, curves[1:], strict=False))

    out = []
    for (t_a, _exp_a, _root_a, k_a, w_a), (t_b, exp_b, root_b, k_b, w_b) in pairs:
        lo = max(k_a.min(), k_b.min())
        hi = min(k_a.max(), k_b.max())
        if not (hi > lo):
            continue
        grid = np.linspace(lo, hi, n_grid)
        wa = np.interp(grid, k_a, w_a)
        wb = np.interp(grid, k_b, w_b)
        drop = wa - wb  # positive where the longer expiry has LESS variance
        worst = int(np.argmax(drop))
        if drop[worst] > tol:
            out.append(
                Violation(
                    "calendar",
                    exp_b,
                    root_b,
                    (float(grid[worst]),),
                    magnitude=float(drop[worst]),
                    relative=float(drop[worst] / max(wa[worst], 1e-12)),
                    detail=(
                        f"total variance falls from {wa[worst]:.6f} at T={t_a:.4f} "
                        f"to {wb[worst]:.6f} at T={t_b:.4f} "
                        f"at log-moneyness {grid[worst]:+.4f}"
                    ),
                )
            )
    return out


def check_chain(frame: pd.DataFrame, *, tol: float = 1e-9) -> ArbitrageReport:
    """Run every static check over a curated chain.

    Vertical and butterfly need call prices across a full strike range, but a
    curated chain holds OTM options only -- puts below the forward, calls above.
    Put-call parity converts the puts to their call equivalents, which is exact
    and uses the same forward and discount the chain was curated with:

        C(K) = P(K) + D * (F - K)

    So the checks run on a synthetic all-call curve that is equivalent to the
    real quotes, without needing the ITM calls we deliberately excluded.
    """
    report = ArbitrageReport(n_checked={"vertical": 0, "butterfly": 0, "calendar": 0})

    for (expiry, root), g in frame.groupby(["expiry", "root"], observed=True):
        g = g.dropna(subset=["mid", "strike", "forward", "discount"])
        if len(g) < 3:
            continue
        forward = float(g["forward"].iloc[0])
        discount = float(g["discount"].iloc[0])

        strikes = g["strike"].to_numpy()
        is_put = g["option_right"].to_numpy() != "call"
        # Parity shift is deterministic, so bid and ask carry through it
        # unchanged in width: C = P + D*(F - K) applies to every quote level.
        shift = np.where(is_put, discount * (forward - strikes), 0.0)
        call_equiv = g["mid"].to_numpy() + shift
        call_bid = g["bid"].to_numpy() + shift
        call_ask = g["ask"].to_numpy() + shift

        report.n_checked["vertical"] += max(len(strikes) - 1, 0)
        report.n_checked["butterfly"] += max(len(strikes) - 2, 0)
        report.violations += check_vertical(
            strikes,
            call_equiv,
            discount=discount,
            tol=tol,
            expiry=expiry,
            root=root,
            bids=call_bid,
            asks=call_ask,
        )
        report.violations += check_butterfly(
            strikes,
            call_equiv,
            tol=tol,
            expiry=expiry,
            root=root,
            forward=forward,
            bids=call_bid,
            asks=call_ask,
        )

    if "total_variance" in frame.columns:
        cal = check_calendar(frame)
        report.n_checked["calendar"] = frame.groupby(
            ["expiry", "root"], observed=True
        ).ngroups
        report.violations += cal

    return report
