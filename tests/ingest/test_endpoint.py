"""Unit tests for `spxrnd.ingest.endpoint`.

The transport is injectable, so retry and backoff are tested exactly -- number
of attempts, sleep durations, which error survives -- with no socket, no mock
library, and no waiting.
"""

from __future__ import annotations

import gzip
import json
import urllib.request

import pytest

from spxrnd.ingest.endpoint import (
    DEFAULT_RETRIES,
    ENDPOINT,
    USER_AGENT,
    _urllib_fetcher,
    fetch,
)
from spxrnd.ingest.errors import FetchError

BODY = {"timestamp": "2026-08-11 20:36:41", "data": {"current_price": 7728.2002}}


def ok_fetcher(url, timeout):
    return json.dumps(BODY).encode()


class Recorder:
    """A fetcher that fails a set number of times, recording every call."""

    def __init__(self, failures=0, error=None):
        self.failures = failures
        self.error = error or ConnectionResetError("connection reset")
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append((url, timeout))
        if len(self.calls) <= self.failures:
            raise self.error
        return json.dumps(BODY).encode()


@pytest.fixture
def naps():
    """Collects sleep durations instead of sleeping."""
    return []


class TestSuccess:
    def test_returns_the_decoded_payload(self):
        assert fetch("_SPX", fetcher=ok_fetcher) == BODY

    def test_builds_the_documented_url(self):
        rec = Recorder()
        fetch("_SPX", fetcher=rec)
        assert rec.calls[0][0] == ENDPOINT.format(ticker="_SPX")

    def test_ticker_is_substituted(self):
        rec = Recorder()
        fetch("SPY", fetcher=rec)
        assert "SPY.json" in rec.calls[0][0]

    def test_timeout_is_passed_through(self):
        rec = Recorder()
        fetch("_SPX", fetcher=rec, timeout=7.5)
        assert rec.calls[0][1] == 7.5

    def test_a_single_attempt_on_success(self, naps):
        rec = Recorder()
        fetch("_SPX", fetcher=rec, sleep=naps.append)
        assert len(rec.calls) == 1
        assert naps == []


class TestRetry:
    def test_recovers_after_transient_failures(self, naps):
        rec = Recorder(failures=2)
        assert fetch("_SPX", fetcher=rec, sleep=naps.append) == BODY
        assert len(rec.calls) == 3

    def test_backoff_is_exponential(self, naps):
        rec = Recorder(failures=2)
        fetch("_SPX", fetcher=rec, retries=3, sleep=naps.append)
        assert naps == [2, 4]

    def test_no_sleep_after_the_final_attempt(self, naps):
        """Sleeping before giving up delays the failure for no benefit."""
        rec = Recorder(failures=99)
        with pytest.raises(FetchError):
            fetch("_SPX", fetcher=rec, retries=3, sleep=naps.append)
        assert len(naps) == 2, "3 attempts means 2 gaps"

    def test_exhausts_exactly_the_requested_attempts(self, naps):
        rec = Recorder(failures=99)
        with pytest.raises(FetchError):
            fetch("_SPX", fetcher=rec, retries=5, sleep=naps.append)
        assert len(rec.calls) == 5

    def test_retries_is_a_total_not_an_extra(self, naps):
        """`retries=1` means one attempt, not one plus a retry."""
        rec = Recorder(failures=99)
        with pytest.raises(FetchError):
            fetch("_SPX", fetcher=rec, retries=1, sleep=naps.append)
        assert len(rec.calls) == 1
        assert naps == []

    def test_default_is_three_attempts(self, naps):
        rec = Recorder(failures=99)
        with pytest.raises(FetchError):
            fetch("_SPX", fetcher=rec, sleep=naps.append)
        assert len(rec.calls) == DEFAULT_RETRIES

    @pytest.mark.parametrize("bad", [0, -1])
    def test_nonsensical_retry_counts_are_rejected(self, bad):
        """Zero attempts would report "all attempts failed" having tried none."""
        with pytest.raises(ValueError, match="at least 1"):
            fetch("_SPX", retries=bad, fetcher=ok_fetcher)


class TestFailure:
    def test_raises_fetch_error_when_all_attempts_fail(self, naps):
        with pytest.raises(FetchError):
            fetch("_SPX", fetcher=Recorder(failures=99), sleep=naps.append)

    def test_error_names_the_url_and_the_cause(self, naps):
        """An operator needs to tell a dead network from a 500."""
        rec = Recorder(failures=99, error=TimeoutError("timed out"))
        with pytest.raises(FetchError) as exc:
            fetch("_SPX", fetcher=rec, sleep=naps.append)
        assert "_SPX.json" in str(exc.value)
        assert "TimeoutError" in str(exc.value)
        assert "timed out" in str(exc.value)

    def test_original_exception_is_chained(self, naps):
        original = TimeoutError("timed out")
        with pytest.raises(FetchError) as exc:
            fetch(
                "_SPX", fetcher=Recorder(failures=99, error=original), sleep=naps.append
            )
        assert exc.value.__cause__ is original

    def test_malformed_json_is_retried_then_reported(self, naps):
        """A truncated body is a transport failure, not a payload problem.

        It must not escape the retry loop -- a CDN hiccup mid-transfer is
        exactly the case retrying exists for.
        """
        calls = []

        def truncating(url, timeout):
            calls.append(url)
            return b'{"timestamp": "2026'

        with pytest.raises(FetchError):
            fetch("_SPX", fetcher=truncating, retries=3, sleep=naps.append)
        assert len(calls) == 3

    def test_undecodable_bytes_are_retried(self, naps):
        calls = []

        def garbage(url, timeout):
            calls.append(url)
            return b"\xff\xfe\x00"

        with pytest.raises(FetchError):
            fetch("_SPX", fetcher=garbage, retries=2, sleep=naps.append)
        assert len(calls) == 2

    def test_an_unexpected_exception_type_still_retries(self, naps):
        """The catch is deliberately broad. Transports raise from urllib, ssl,
        socket, zlib and json; enumerating them lets a new failure mode escape
        the loop and kill the capture outright."""
        rec = Recorder(failures=99, error=MemoryError("weird"))
        with pytest.raises(FetchError):
            fetch("_SPX", fetcher=rec, retries=2, sleep=naps.append)
        assert len(rec.calls) == 2


class TestCourtesy:
    def test_user_agent_identifies_the_project(self):
        """This is a free endpoint. An anonymous scraper is a rude one."""
        assert "research" in USER_AGENT.lower()

    def test_gzip_transfer_is_handled(self):
        """The uncompressed chain is ~13 MB and the gzipped one ~1.7 MB.

        Exercises the decompression path that `_urllib_fetcher` performs, at
        the boundary where a wrong answer produces a gzip member fed to
        json.loads.
        """
        blob = gzip.compress(json.dumps(BODY).encode())
        assert fetch("_SPX", fetcher=lambda u, t: gzip.decompress(blob)) == BODY


class FakeResponse:
    """Stands in for urlopen's context manager."""

    def __init__(self, body: bytes, headers: dict):
        self._body = body
        self.headers = headers

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestUrllibTransport:
    """The default transport, tested without a socket.

    Covered here rather than left to the live test: the gzip branch is the one
    place where being wrong yields a compressed member handed to json.loads,
    and that must not need a network to catch.
    """

    @pytest.fixture
    def sent(self, monkeypatch):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["timeout"] = timeout
            return FakeResponse(*captured["response"])

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return captured

    def test_decompresses_a_gzip_encoded_response(self, sent):
        sent["response"] = (
            gzip.compress(json.dumps(BODY).encode()),
            {"Content-Encoding": "gzip"},
        )
        assert _urllib_fetcher("http://x", 30) == json.dumps(BODY).encode()

    def test_passes_through_an_unencoded_response(self, sent):
        sent["response"] = (b"plain", {})
        assert _urllib_fetcher("http://x", 30) == b"plain"

    def test_requests_gzip(self, sent):
        """Not requesting it costs ~11 MB per capture on a courtesy endpoint."""
        sent["response"] = (b"{}", {})
        _urllib_fetcher("http://x", 30)
        assert sent["headers"]["Accept-encoding"] == "gzip"

    def test_identifies_itself(self, sent):
        sent["response"] = (b"{}", {})
        _urllib_fetcher("http://x", 30)
        assert sent["headers"]["User-agent"] == USER_AGENT

    def test_passes_the_timeout_through(self, sent):
        sent["response"] = (b"{}", {})
        _urllib_fetcher("http://x", 12.5)
        assert sent["timeout"] == 12.5

    def test_is_wired_up_as_the_default_transport(self, sent):
        """`fetch` with no fetcher must reach this function."""
        sent["response"] = (json.dumps(BODY).encode(), {})
        assert fetch("_SPX") == BODY
        assert sent["url"] == ENDPOINT.format(ticker="_SPX")
