"""Black-76 pricing and implied-volatility inversion.

Black-76, not Black-Scholes: the model is written on the **forward**, which is
what `curate.forward` measures directly from put-call parity. Black-Scholes
would require a spot, a rate and a dividend yield, and the dividend yield is
exactly the quantity nobody observes -- it would have to be backed out of the
same parity relation anyway, then fed back in as an input. Working on the
forward skips that round trip and its error.

    C = D * [F N(d1) - K N(d2)]
    P = D * [K N(-d2) - F N(-d1)]

    d1 = (ln(F/K) + v^2/2) / v,   d2 = d1 - v,   v = sigma * sqrt(T)

`v` -- total volatility, `sigma * sqrt(T)` -- is carried around rather than
`sigma` and `T` separately wherever possible. It is the quantity the formula
actually depends on, it is what stays well-scaled as `T` goes to zero, and it
is what the no-arbitrage conditions are naturally written in.

Why not use CBOE's published `iv`
---------------------------------
Because 2,160 contracts per capture carry `iv == 0.0`, the feed's "not
computed" marker, and because a vol computed from a mid we chose, under a
forward we measured, is the only one whose assumptions we know. CBOE's column
is kept as an **independent cross-check** -- two routes to the same number,
which is what catches a mistake in either.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr  # standard normal CDF, vectorised

CALL = "call"
PUT = "put"

SQRT_2PI = float(np.sqrt(2.0 * np.pi))

MAX_VOL = 5.0
"""Upper bracket for the inversion: 500% annualised.

Chosen to be far outside anything economically meaningful while still finite.
Real SPX vols in this archive run 8%-80%; a quote implying more than 500% is
not a volatility, it is a broken price."""

MIN_VOL = 1e-8


def _as_arrays(*args):
    """Broadcast inputs to a common shape as float arrays."""
    return np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])


def normal_pdf(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / SQRT_2PI


def d1_d2(forward, strike, total_vol):
    """The two Black arguments, in terms of total volatility `v = sigma*sqrt(T)`.

    Returns `(d1, d2)`. Where `v <= 0` both are +/-inf by the sign of
    `ln(F/K)`, which is the correct limit: with no volatility the option is
    worth its forward intrinsic value, and `N(+-inf)` delivers exactly that.
    """
    forward, strike, total_vol = _as_arrays(forward, strike, total_vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_fk = np.log(forward / strike)
        d1 = np.where(
            total_vol > 0,
            (log_fk + 0.5 * total_vol**2) / np.where(total_vol > 0, total_vol, 1.0),
            np.where(log_fk > 0, np.inf, np.where(log_fk < 0, -np.inf, 0.0)),
        )
        d2 = np.where(total_vol > 0, d1 - total_vol, d1)
    return d1, d2


def price(forward, strike, total_vol, *, discount=1.0, right=CALL):
    """Undiscounted-forward Black-76 price, times the discount factor.

    Args:
        forward: forward price of the underlying at expiry.
        strike: strike.
        total_vol: `sigma * sqrt(T)`. Zero gives the intrinsic value.
        discount: discount factor to the settlement date.
        right: ``"call"`` or ``"put"``.

    Returns:
        Option price. Array-shaped to the broadcast of the inputs.
    """
    forward, strike, total_vol, discount = _as_arrays(
        forward, strike, total_vol, discount
    )
    d1, d2 = d1_d2(forward, strike, total_vol)
    if right == CALL:
        undiscounted = forward * ndtr(d1) - strike * ndtr(d2)
    elif right == PUT:
        undiscounted = strike * ndtr(-d2) - forward * ndtr(-d1)
    else:
        raise ValueError(f"right must be {CALL!r} or {PUT!r}, got {right!r}")
    return discount * undiscounted


def vega(forward, strike, total_vol, *, discount=1.0, tenor=None):
    """Sensitivity to volatility.

    With `tenor` given, this is dPrice/dSigma -- the conventional vega. Without
    it, dPrice/dv, the derivative with respect to *total* volatility, which is
    what the inversion iterates on and what stays finite as `T` goes to zero.

    Vega is the same for calls and puts: they differ by a forward contract,
    which has no volatility exposure. That identity is a test.
    """
    forward, strike, total_vol, discount = _as_arrays(
        forward, strike, total_vol, discount
    )
    d1, _ = d1_d2(forward, strike, total_vol)
    base = discount * forward * normal_pdf(d1)
    if tenor is None:
        return base
    return base * np.sqrt(np.asarray(tenor, dtype=float))


def price_bounds(forward, strike, *, discount=1.0, right=CALL):
    """The no-arbitrage range a price must sit in for an implied vol to exist.

    Call: `D*max(F-K, 0) <= C <= D*F`. Put: `D*max(K-F, 0) <= P <= D*K`.

    The lower bound is the zero-vol limit and the upper is the infinite-vol
    limit, so a price outside them is not a mispricing to be inverted -- it is
    a price no Black-76 volatility can produce. Returning the bounds rather
    than just a boolean lets the caller report *how far* outside, which is the
    difference between "one tick of quote noise" and "this chain is wrong".
    """
    forward, strike, discount = _as_arrays(forward, strike, discount)
    if right == CALL:
        return discount * np.maximum(forward - strike, 0.0), discount * forward
    if right == PUT:
        return discount * np.maximum(strike - forward, 0.0), discount * strike
    raise ValueError(f"right must be {CALL!r} or {PUT!r}, got {right!r}")


def implied_total_vol(
    observed,
    forward,
    strike,
    *,
    discount=1.0,
    right=CALL,
    tol: float = 1e-10,
    max_iter: int = 64,
):
    """Invert Black-76 for total volatility `v = sigma*sqrt(T)`.

    Newton first -- the price is monotone in `v` with an analytic derivative,
    so it converges in a handful of steps almost everywhere -- and Brent's
    method on a bracket for whatever Newton fails to resolve. Newton alone is
    not enough: in the far wings vega underflows and the step becomes
    numerically undefined, which is precisely where quotes are widest and an
    unnoticed failure most likely.

    Returns:
        `(total_vol, ok)`. `total_vol` is NaN where no solution exists; `ok` is
        a boolean array. **Never raises on a bad price** -- a chain of 13,414
        quotes will contain some that cannot be inverted, and aborting the run
        for one of them would be the wrong trade.
    """
    observed, forward, strike, discount = _as_arrays(
        observed, forward, strike, discount
    )
    shape = observed.shape
    observed, forward, strike, discount = (
        a.ravel() for a in (observed, forward, strike, discount)
    )

    lower, upper = price_bounds(forward, strike, discount=discount, right=right)
    # Strictly inside: at the boundary the vol is 0 or infinite, neither of
    # which is a number a smile fit can use.
    ok = (observed > lower + 1e-12) & (observed < upper - 1e-12) & (discount > 0)

    v = np.full(observed.shape, np.nan)
    if not ok.any():
        return v.reshape(shape), ok.reshape(shape)

    idx = np.flatnonzero(ok)
    f, k, d, target = forward[idx], strike[idx], discount[idx], observed[idx]

    # Brenner-Subrahmanyam ATM approximation as the starting point: for F ~ K
    # the price is about 0.4 * D * F * v, which is exact enough to put Newton
    # in the right basin even well away from the money.
    guess = np.clip(target / (0.4 * d * f), 1e-4, 2.0)

    for _ in range(max_iter):
        diff = price(f, k, guess, discount=d, right=right) - target
        dv = vega(f, k, guess, discount=d)
        # `np.divide(..., where=)` rather than `np.where(cond, a/b, 0)`: the
        # latter evaluates the division for every element before selecting, so
        # it still divides by the underflowed vegas it was meant to skip. In
        # the far wings that is most of the array.
        step = np.zeros_like(diff)
        np.divide(diff, dv, out=step, where=dv > 1e-300)
        guess_next = np.clip(guess - step, MIN_VOL, MAX_VOL)
        if np.all(np.abs(guess_next - guess) < tol):
            guess = guess_next
            break
        guess = guess_next

    residual = np.abs(price(f, k, guess, discount=d, right=right) - target)
    scale = np.maximum(np.abs(target), 1e-8)
    converged = residual / scale < 1e-8

    # Brent for the stragglers: bracketed, derivative-free, and guaranteed on a
    # monotone function.
    for j in np.flatnonzero(~converged):
        try:
            guess[j] = brentq(
                lambda v_, jj=j: (
                    price(f[jj], k[jj], v_, discount=d[jj], right=right) - target[jj]
                ),
                MIN_VOL,
                MAX_VOL,
                xtol=1e-12,
                maxiter=200,
            )
        except (ValueError, RuntimeError):
            ok[idx[j]] = False
            guess[j] = np.nan

    v[idx] = guess
    return v.reshape(shape), ok.reshape(shape)


def implied_vol(
    observed, forward, strike, tenor, *, discount=1.0, right=CALL, **kwargs
):
    """Annualised implied volatility, `sigma`.

    Thin wrapper over :func:`implied_total_vol` that divides by `sqrt(T)`. Note
    that this is where a short tenor hurts: dividing by `sqrt(T)` magnifies any
    error in `v` by `1/sqrt(T)`, which is 19x at one day. The total volatility
    is the better-conditioned quantity, and the smile fit in Stage 6 works in
    it for that reason.
    """
    tenor_arr = np.asarray(tenor, dtype=float)
    total, ok = implied_total_vol(
        observed, forward, strike, discount=discount, right=right, **kwargs
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(tenor_arr > 0, total / np.sqrt(tenor_arr), np.nan)
    return sigma, ok & (tenor_arr > 0)
