"""HTTP access to CBOE's delayed-quotes endpoint.

This is a free, public, courtesy endpoint. Two obligations follow: request gzip
so we transfer ~1.7 MB instead of ~13 MB, and back off rather than retry
tightly. Neither is negotiable for a job that runs every ten minutes forever.

The transport is isolated in this module so that everything else in `ingest` is
a pure function of a dict. That is what lets the entire pipeline be tested
against real captured payloads with no network and no mocking framework.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from .errors import FetchError

ENDPOINT = "https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"

USER_AGENT = "Mozilla/5.0 (academic research; RND estimation project)"

DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 30.0

Fetcher = Callable[[str, float], bytes]
"""Signature of the low-level transport: ``(url, timeout) -> body bytes``.

Injectable so tests can drive :func:`fetch` -- including its retry and backoff
behaviour -- without a socket or a mocking library.
"""


def fetch(
    ticker: str,
    *,
    retries: int = DEFAULT_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
    fetcher: Fetcher | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Fetch and decode one chain payload.

    Args:
        ticker: ``"_SPX"`` for the S&P 500 index, ``"SPY"`` for the ETF.
        retries: total attempts, not additional ones. Backoff is 2^n seconds.
        timeout: per-attempt socket timeout in seconds.
        fetcher: transport override. Defaults to the urllib implementation.
        sleep: sleep override, so retry tests do not actually wait.

    Returns:
        The decoded payload.

    Raises:
        FetchError: every attempt failed. Carries the last underlying error --
            a dead network and a 500 need different responses from an operator.
    """
    if retries < 1:
        raise ValueError(f"retries must be at least 1, got {retries}")

    transport = fetcher or _urllib_fetcher
    pause = sleep or time.sleep
    url = ENDPOINT.format(ticker=ticker)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            body = transport(url, timeout)
            return json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 -- see below
            # Deliberately broad. Transports raise from urllib, ssl, socket,
            # zlib and json; enumerating them means a new failure mode escapes
            # the retry loop and kills the capture outright. Whatever it was is
            # attached to the FetchError for the operator.
            last_error = exc
            if attempt < retries:
                pause(2**attempt)

    raise FetchError(
        f"all {retries} attempts failed for {url}: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _urllib_fetcher(url: str, timeout: float) -> bytes:
    """Default transport: urllib with gzip transfer encoding.

    Kept private and separate from :func:`fetch` so the retry policy can be
    tested independently of sockets.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        # urllib does not transparently decompress; without this the payload is
        # a gzip member and json.loads fails on byte one.
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return body
