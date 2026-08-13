"""Assembling one capture into a chain an estimator can consume.

The order of operations matters and is not obvious, so it is fixed here rather
than left to each caller:

    1. mid prices          -- needed by the parity fit
    2. forward per expiry  -- needed by moneyness, which is needed by the ITM
                              flag, which is a filter
    3. tenor and moneyness -- joined back onto every quote
    4. quality flags       -- computed for every row
    5. drop                -- and report what it cost

Steps 2 and 4 look like they could swap. They cannot: the ITM rule is defined
against the forward, so filtering before fitting would mean filtering on a
forward that does not exist yet, and fitting after filtering would mean fitting
parity on a set from which one side has already been removed. Parity needs both
legs, including the ITM ones.

That last point is worth stating plainly, because it is the subtle bit: **the
forward is fitted on ITM quotes too, and the density is estimated only on OTM
ones.** Both are correct. Parity is an identity that holds across the whole
strike range and is best measured where both legs are liquid; the density wants
OTM options because their prices are mostly time value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from . import forward as forward_mod
from . import quality


@dataclass
class CuratedChain:
    """One capture, curated. Everything an estimator needs and nothing implicit."""

    capture_utc: datetime
    spot: float

    quotes: pd.DataFrame
    """Surviving quotes, with mid, tenor_years, forward, discount, rate,
    log_moneyness and every quality flag attached."""

    forwards: pd.DataFrame
    """One row per (expiry, root), including excluded ones with their reason."""

    attrition: quality.Attrition

    @property
    def usable_forwards(self) -> pd.DataFrame:
        return self.forwards.loc[self.forwards["usable"]]

    @property
    def expiries(self) -> list:
        return sorted(self.quotes["expiry"].unique())

    def for_expiry(self, expiry, root: str) -> pd.DataFrame:
        """One expiry's surviving quotes, sorted by strike."""
        mask = (self.quotes["expiry"] == expiry) & (self.quotes["root"] == root)
        return self.quotes.loc[mask].sort_values("strike")

    def summary(self) -> str:
        usable = self.usable_forwards
        lines = [
            f"capture {self.capture_utc:%Y-%m-%d %H:%M:%S}Z   spot {self.spot}",
            "",
            self.attrition.summary(),
            "",
            f"  forwards: {len(usable)} usable of {len(self.forwards)} "
            f"(expiry, root) pairs",
        ]
        if len(usable):
            lines.append(
                f"  parity R2: min {usable['r2'].min():.6f}  "
                f"median {usable['r2'].median():.6f}"
            )
            lines.append(
                f"  implied rate: {usable['rate'].min():.2%} to "
                f"{usable['rate'].max():.2%}"
            )
        excluded = self.forwards.loc[~self.forwards["usable"]]
        if len(excluded):
            lines.append(f"  excluded expiries: {len(excluded)}")
            for _, row in excluded.iterrows():
                lines.append(f"    {row['expiry']} {row['root']:<5} {row['excluded']}")
        return "\n".join(lines)


def build(
    quotes: pd.DataFrame,
    *,
    as_of: datetime,
    spot: float,
    filters: quality.Filters | None = None,
    min_pairs: int = forward_mod.MIN_PAIRS,
    min_r2: float = forward_mod.MIN_R2,
) -> CuratedChain:
    """Curate one capture's quotes into an estimable chain.

    Args:
        quotes: one capture, straight from the store. Must carry `expiry`,
            `root`, `option_right`, `strike`, `bid`, `ask`, `iv`.
        as_of: the capture instant, timezone-aware.
        spot: the underlying level from the same payload.
        filters: quality rules. Defaults to the strict set.

    Returns:
        A :class:`CuratedChain`. Read `.summary()` before trusting `.quotes` --
        the attrition table is where a rule that ate a whole wing shows up.
    """
    filters = filters or quality.STRICT
    df = quotes.copy()

    # 1. mid, needed by the parity fit
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # 2. forwards, fitted on ALL quotes including ITM -- parity needs both legs
    fits = forward_mod.fit_all(
        df.loc[df["bid"] > 0], as_of=as_of, min_pairs=min_pairs, min_r2=min_r2
    )
    forwards = forward_mod.to_frame(fits)

    # 3. join tenor and forward onto every quote
    df = df.merge(
        forwards[["expiry", "root", "tenor_years", "forward", "discount", "rate"]],
        on=["expiry", "root"],
        how="left",
    )

    # An expiry with no usable forward cannot be assessed for moneyness, so it
    # is dropped here rather than carried forward with a NaN that would silently
    # make every comparison against it False.
    unusable = df["forward"].isna()
    n_unusable = int(unusable.sum())
    df = df.loc[~unusable].copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_moneyness"] = np.log(df["strike"].to_numpy() / df["forward"].to_numpy())

    # 4. flags, then 5. drop
    df = quality.add_quality_columns(df, filters=filters)
    kept, attrition = quality.apply(df, filters=filters)

    attrition.total_in += n_unusable
    if n_unusable:
        attrition.by_rule["no_usable_forward"] = n_unusable

    return CuratedChain(
        capture_utc=as_of,
        spot=spot,
        quotes=kept,
        forwards=forwards,
        attrition=attrition,
    )


def from_catalog(con, capture_utc: datetime, **kwargs) -> CuratedChain:
    """Curate one capture read straight out of the DuckDB catalog.

    Convenience over `build`, so the common path is one call.
    """
    df = con.execute(
        """
        SELECT option_symbol, root, option_right, expiry, strike,
               bid, ask, iv, theo, delta, open_interest, volume,
               underlying_price
        FROM quotes WHERE capture_utc = ?
        """,
        [capture_utc],
    ).df()
    if df.empty:
        raise ValueError(f"no quotes for capture {capture_utc}")
    spot = float(df["underlying_price"].iloc[0])
    return build(
        df.drop(columns=["underlying_price"]), as_of=capture_utc, spot=spot, **kwargs
    )
