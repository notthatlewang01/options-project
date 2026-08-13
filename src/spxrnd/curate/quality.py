"""Deciding which quotes are usable, and saying so out loud.

Every rule here is a named flag computed for every row, and dropping is a
separate step that consumes those flags. Two reasons for the split:

**Attrition is reported, never silent.** A filter that quietly removes half an
expiry's wing produces a density that looks entirely reasonable and is wrong.
`attrition` returns a per-expiry, per-rule count so the cost of every rule is
visible before it reaches an estimator.

**Rules are individually toggleable.** A rule you cannot turn off is a rule you
cannot test the sensitivity of, and "how much does the answer depend on the
50% spread cut?" is a question worth being able to answer.

What is NOT filtered here
-------------------------
`iv == 0.0`. It is the feed's "not computed" marker and it is recorded as a
flag, but it is deliberately not a drop rule, because it is not a property of
the *quote* -- those contracts have live two-sided markets and perfectly good
prices. Measured on the Aug 11 close, it is overwhelmingly a marker of deep ITM:

    deep ITM calls        1,156 of 5,837 quoted   (19.8%),  mean |delta| 0.995
    within 10% of spot      694 of 16,421         ( 4.2%),  mean |delta| 0.988
                                                  -- but 684 of those 694 are ITM
    everything else           9 of 6,157          ( 0.1%)

So the sentinel tracks moneyness, and the OTM selection in
`moneyness.is_otm` removes that whole region for an independent and better
reason: an ITM option's price is nearly all intrinsic, and inverting a vol from
it amplifies noise. Filtering on `iv == 0.0` directly would be treating a
symptom.

`last_trade_price` is likewise never consulted. One quoted contract in six sits
over 25% from its own live mid -- it is not a quote, it is a historical fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from . import moneyness


class Flag(StrEnum):
    """Per-quote quality markers. Computed always, applied selectively."""

    ZERO_BID = "zero_bid"
    """Nobody is buying at any price. The mid is not a price, and the relative
    spread is 200% by construction."""

    ZERO_ASK = "zero_ask"
    """Nobody is offering. Not observed in any capture so far; guarded because
    "never seen" is not "cannot happen"."""

    CROSSED = "crossed"
    """bid > ask. A broken book. Zero occurrences across 952,564 rows, which is
    worth continuing to assert rather than assume."""

    WIDE_SPREAD = "wide_spread"
    """Relative spread over the tolerance. A real market, but the mid carries
    little information about where the contract would actually trade."""

    ZERO_IV = "zero_iv"
    """The feed's iv == 0.0 sentinel. Recorded, never a drop reason -- see the
    module docstring."""

    EXPIRED = "expired"
    """Tenor at or below zero: the capture was taken after this contract's
    settlement instant."""

    ZERO_DTE = "zero_dte"
    """Expires the same session. Vega vanishes as tenor goes to zero, so any
    pricing error explodes into implied vol; 562 such contracts sit in the
    Aug 11 close."""

    ITM = "itm"
    """In the money against the forward. Excluded from density estimation, and
    the rule that subsumes the iv sentinel."""


@dataclass(frozen=True, slots=True)
class Filters:
    """Which flags drop a row, and the thresholds that produce them.

    The defaults are deliberately strict -- the input to a density should be
    quotes you would be willing to trade on -- and every one of them is
    reported in the attrition table.
    """

    max_relative_spread: float = 0.5
    """(ask - bid) / mid above which a quote is `WIDE_SPREAD`."""

    min_tenor_years: float = 1.0 / 365.0
    """Contracts with less life than this are `ZERO_DTE`. One calendar day."""

    drop: frozenset[Flag] = frozenset(
        {
            Flag.ZERO_BID,
            Flag.ZERO_ASK,
            Flag.CROSSED,
            Flag.WIDE_SPREAD,
            Flag.EXPIRED,
            Flag.ZERO_DTE,
            Flag.ITM,
        }
    )
    """Flags that remove a row. Note `ZERO_IV` is absent by design."""

    def without(self, *flags: Flag) -> Filters:
        """A copy with those rules disabled, for sensitivity analysis."""
        return Filters(
            max_relative_spread=self.max_relative_spread,
            min_tenor_years=self.min_tenor_years,
            drop=frozenset(self.drop - set(flags)),
        )


STRICT = Filters()
"""The default rule set, as a module-level singleton.

`Filters` is frozen, so sharing one instance is safe -- and it gives callers a
name to refer to the defaults by rather than reconstructing them."""

PERMISSIVE = Filters(drop=frozenset())
"""Flag everything, drop nothing. For inspecting what the rules would do."""


def add_quality_columns(df: pd.DataFrame, *, filters: Filters = STRICT) -> pd.DataFrame:
    """Attach mid, spread, tenor and one boolean column per flag.

    Requires `forward` and `tenor_years` columns to already be present -- both
    are per-expiry quantities that this module does not compute. Run
    `curate.forward` first.

    Returns a copy; the input is not modified.
    """
    out = df.copy()

    bid, ask = out["bid"].to_numpy(), out["ask"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mid = (bid + ask) / 2.0
        rel = np.where(mid > 0, (ask - bid) / mid, np.inf)
    out["mid"] = mid
    out["spread"] = ask - bid
    out["relative_spread"] = rel

    out[Flag.ZERO_BID] = bid <= 0
    out[Flag.ZERO_ASK] = ask <= 0
    out[Flag.CROSSED] = bid > ask
    out[Flag.WIDE_SPREAD] = rel > filters.max_relative_spread
    out[Flag.ZERO_IV] = out["iv"].to_numpy() == 0.0
    out[Flag.EXPIRED] = out["tenor_years"].to_numpy() <= 0
    out[Flag.ZERO_DTE] = out["tenor_years"].to_numpy() < filters.min_tenor_years
    out[Flag.ITM] = ~np.fromiter(
        (
            moneyness.is_otm(r, k, f)
            for r, k, f in zip(
                out["option_right"], out["strike"], out["forward"], strict=True
            )
        ),
        dtype=bool,
        count=len(out),
    )
    return out


@dataclass
class Attrition:
    """What each rule cost, per expiry and in total."""

    by_rule: dict[str, int] = field(default_factory=dict)
    """Rows carrying each flag. Rules overlap, so these do not sum to the
    number dropped -- a zero-bid contract is usually wide-spread too."""

    by_expiry: pd.DataFrame = field(default_factory=pd.DataFrame)
    """One row per expiry: input count, kept count, and per-rule counts."""

    total_in: int = 0
    total_kept: int = 0

    @property
    def total_dropped(self) -> int:
        return self.total_in - self.total_kept

    @property
    def kept_fraction(self) -> float:
        return self.total_kept / self.total_in if self.total_in else 0.0

    def summary(self) -> str:
        lines = [
            f"{self.total_in:,} quotes in, {self.total_kept:,} kept "
            f"({self.kept_fraction:.1%}), {self.total_dropped:,} dropped",
            "",
            f"  {'rule':<16}{'flagged':>10}{'% of input':>12}",
        ]
        for rule, n in sorted(self.by_rule.items(), key=lambda kv: -kv[1]):
            pct = n / self.total_in if self.total_in else 0.0
            lines.append(f"  {rule:<16}{n:>10,}{pct:>11.1%}")
        return "\n".join(lines)


def apply(
    df: pd.DataFrame, *, filters: Filters = STRICT
) -> tuple[pd.DataFrame, Attrition]:
    """Drop flagged rows, returning what survived and what it cost.

    Args:
        df: quotes with quality columns already attached.
        filters: which flags drop a row.

    Returns:
        `(kept, attrition)`. Always inspect the attrition -- a rule that ate an
        entire wing is invisible in the kept frame alone.
    """
    flags = [f for f in Flag if f in df.columns]
    by_rule = {str(f): int(df[f].sum()) for f in flags}

    if filters.drop:
        drop_mask = np.zeros(len(df), dtype=bool)
        for flag in filters.drop:
            if flag in df.columns:
                drop_mask |= df[flag].to_numpy()
    else:
        drop_mask = np.zeros(len(df), dtype=bool)

    kept = df.loc[~drop_mask].copy()

    per_expiry = (
        df.assign(_dropped=drop_mask)
        .groupby(["expiry", "root"], observed=True)
        .agg(
            n_in=("option_symbol", "size"),
            n_kept=("_dropped", lambda s: int((~s).sum())),
            **{str(f): (f, "sum") for f in flags},
        )
        .reset_index()
    )

    return kept, Attrition(
        by_rule=by_rule,
        by_expiry=per_expiry,
        total_in=len(df),
        total_kept=len(kept),
    )
