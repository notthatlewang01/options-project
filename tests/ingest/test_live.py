"""The single opt-in test that touches the real CBOE endpoint.

Excluded from the default suite. Run it by hand with `make test-live` when you
want to know whether the feed still looks the way this package believes it does.

Why exactly one test, and why opt-in
------------------------------------
The endpoint is a free courtesy service; a suite that hits it on every commit is
rude and would make the suite slow, flaky and offline-hostile. But a purely
hermetic suite has a real blind spot: CBOE renames a field, every mocked test
still passes, and the next scheduled capture silently degrades. So there is one
test, it is deliberately hard to run by accident, and it asserts the *shape* of
the feed rather than any particular value.

Nothing here asserts on prices. The chain changes every ten minutes; a test that
pinned a number would fail for the wrong reason within the hour.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from spxrnd.ingest import endpoint, osi
from spxrnd.ingest.payload import parse

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_payload():
    """One real fetch, shared by every assertion below.

    Module-scoped on purpose: one request per run, not one per test.
    """
    return endpoint.fetch("_SPX")


class TestFeedShape:
    def test_the_payload_still_parses(self, live_payload):
        assert parse(live_payload, ticker="_SPX").spot > 0

    def test_the_feed_timestamp_is_still_top_level(self, live_payload):
        """The defect that emptied every row ever written. If CBOE ever moves
        this key, this is where we find out."""
        assert live_payload.get("timestamp")

    def test_data_still_has_no_timestamp_key(self, live_payload):
        """The key the original collector read. It must stay absent, or the
        question of which timestamp is authoritative reopens."""
        assert "timestamp" not in live_payload["data"]

    def test_every_documented_index_field_is_present(self, live_payload):
        expected = {
            "current_price",
            "last_trade_time",
            "seqno",
            "bid",
            "ask",
            "open",
            "high",
            "low",
            "prev_day_close",
            "iv30",
        }
        assert expected <= set(live_payload["data"])

    def test_every_documented_option_field_is_present(self, live_payload):
        expected = {
            "option",
            "bid",
            "bid_size",
            "ask",
            "ask_size",
            "iv",
            "theo",
            "open_interest",
            "volume",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
            "last_trade_price",
            "last_trade_time",
            "prev_day_close",
        }
        assert expected <= set(live_payload["data"]["options"][0])


class TestChainShape:
    def test_the_chain_is_still_large(self, live_payload):
        """~30k contracts. An order-of-magnitude drop means a truncated
        response, not a quiet market."""
        assert len(live_payload["data"]["options"]) > 10_000

    def test_every_symbol_still_parses_as_osi(self, live_payload):
        bad = [
            o["option"]
            for o in live_payload["data"]["options"]
            if osi.try_parse(o["option"]) is None
        ]
        assert bad == [], f"{len(bad)} unparseable symbols, e.g. {bad[:3]}"

    def test_both_roots_are_still_present(self, live_payload):
        roots = {osi.parse(o["option"]).root for o in live_payload["data"]["options"]}
        assert {"SPX", "SPXW"} <= roots

    def test_the_root_collision_still_happens(self, live_payload):
        """If this ever stops being true the root column stops being load-
        bearing -- worth knowing either way."""
        seen: dict[tuple, set[str]] = {}
        for o in live_payload["data"]["options"]:
            sym = osi.parse(o["option"])
            seen.setdefault((sym.expiry, sym.right, sym.strike), set()).add(sym.root)
        assert any(len(roots) > 1 for roots in seen.values())


class TestDelayContract:
    def test_the_index_print_is_recent_but_not_current(self, live_payload):
        """The advertised ~15-minute delay, checked rather than assumed.

        Skips outside market hours: after the close the print freezes, which is
        correct behaviour and not a failure.
        """
        snap = parse(live_payload, ticker="_SPX")
        age = datetime.now(UTC) - snap.index_last_trade
        if age > timedelta(minutes=30):
            pytest.skip(f"market closed; index print is {age} old")
        assert age > timedelta(minutes=5), (
            "the feed is fresher than its advertised delay -- verify the delay "
            "assumption in freshness.FEED_DELAY still holds"
        )
        assert age < timedelta(minutes=30)


class TestKnownPathologiesPersist:
    def test_the_zero_iv_sentinel_is_still_being_emitted(self, live_payload):
        """2,160 contracts carried iv == 0.0 in the archived snapshot.

        If this ever stops, the curation layer's handling of it becomes dead
        code -- and if it grows, the smile fit has a bigger hole to work around.
        """
        sentinel = [o for o in live_payload["data"]["options"] if o["iv"] == 0.0]
        assert sentinel, "iv==0.0 sentinel has disappeared; revisit curate/quality"

    def test_zero_bid_contracts_still_exist(self, live_payload):
        assert [o for o in live_payload["data"]["options"] if o["bid"] == 0]
