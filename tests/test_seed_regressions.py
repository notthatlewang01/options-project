"""The seed regression corpus.

Eleven defects, each found in real collected data, each pinned by a test built
from the capture that exhibits it. Five were diagnosed in the original
collector's rewrite; six more surfaced while surveying the archive.

These are written before the implementation exists and they define its contract.
They fail with NotImplementedError until Stage 2 fills the stubs in -- that is
the intended state, not a broken suite.

**These tests are permanent.** They are not scaffolding to be deleted once the
code works. Each one encodes a way this pipeline has already been observed to
produce silently wrong data, and silence is the operative word: every one of
these defects produced output that looked entirely reasonable.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from spxrnd.ingest import collector, freshness, osi, payload
from spxrnd.ingest.state import CollectorState

from .conftest import load_header, utc

# ---------------------------------------------------------------------------
# 1. cboe_timestamp was empty in every row ever written
# ---------------------------------------------------------------------------
# The original read payload["data"]["timestamp"]. That key does not exist -- the
# feed timestamp is at the TOP level -- so .get() returned "" for every row of
# every snapshot ever collected. Nothing failed. The column was simply blank,
# and stayed blank across 29 files and 890,000 rows.


class TestSeed01FeedTimestampIsTopLevel:
    def test_feed_timestamp_is_populated(self, trimmed_chain):
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        assert snap.feed_timestamp, "feed timestamp must never be empty"

    def test_feed_timestamp_comes_from_the_top_level_key(self, trimmed_chain):
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        assert snap.feed_timestamp == trimmed_chain["timestamp"]

    def test_data_has_no_timestamp_key_at_all(self, trimmed_chain):
        """The bug was reading a key that does not exist. Pin that it doesn't.

        If CBOE ever adds `data.timestamp`, this fails and forces a decision
        about which one is authoritative, rather than letting a plausible-looking
        wrong value slip in.
        """
        assert "timestamp" not in trimmed_chain["data"]

    def test_feed_timestamp_differs_from_the_index_print(self, trimmed_chain):
        """They are different clocks and must not be conflated.

        The feed timestamp advances forever, including at 2am on a holiday. Only
        the index print says whether the body holds new information.
        """
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        assert snap.feed_timestamp != snap.index_last_trade.isoformat()


# ---------------------------------------------------------------------------
# 2. The OSI root was parsed and then thrown away
# ---------------------------------------------------------------------------
# SPX is AM-settled monthlies, SPXW is PM-settled weeklies. On third-Friday
# expiries BOTH roots list the same strikes: 986 (right, strike) pairs collide
# on 2026-08-21 alone. Different contracts, different quotes. Dropping the root
# makes those rows indistinguishable -- and a put-call parity regression that
# mixes them returns a corrupted forward, which poisons every density built on
# top of it.


class TestSeed02RootIsPreserved:
    def test_root_is_parsed(self):
        assert osi.parse("SPXW260821C01400000").root == "SPXW"
        assert osi.parse("SPX260821C01400000").root == "SPX"

    def test_roots_differ_where_everything_else_matches(self):
        a = osi.parse("SPX260821C01400000")
        b = osi.parse("SPXW260821C01400000")
        assert (a.expiry, a.right, a.strike) == (b.expiry, b.right, b.strike)
        assert a != b, "same expiry/right/strike must still be distinct contracts"
        assert len({a, b}) == 2, "and must not collapse in a set or dict key"

    def test_collision_is_real_in_the_captured_chain(self, trimmed_chain):
        """Not hypothetical. Count the collisions in real captured data."""
        seen: dict[tuple, set[str]] = {}
        for opt in trimmed_chain["data"]["options"]:
            sym = osi.try_parse(opt["option"])
            if sym and sym.expiry.isoformat() == "2026-08-21":
                seen.setdefault((sym.right, sym.strike), set()).add(sym.root)
        collisions = [k for k, roots in seen.items() if len(roots) > 1]
        assert collisions, "fixture must contain SPX/SPXW collisions"

    def test_colliding_contracts_carry_different_quotes(self, trimmed_chain):
        """The two roots are not merely distinct labels -- they price apart.

        If they always quoted identically the root would be bookkeeping. They
        do not, so it is correctness.
        """
        by_key: dict[tuple, dict[str, dict]] = {}
        for opt in trimmed_chain["data"]["options"]:
            sym = osi.try_parse(opt["option"])
            if sym and sym.expiry.isoformat() == "2026-08-21":
                by_key.setdefault((sym.right, sym.strike), {})[sym.root] = opt
        differing = [
            k
            for k, roots in by_key.items()
            if len(roots) == 2
            and (roots["SPX"]["bid"], roots["SPX"]["ask"])
            != (roots["SPXW"]["bid"], roots["SPXW"]["ask"])
        ]
        assert differing, "colliding roots must be shown to quote differently"


# ---------------------------------------------------------------------------
# 3. The post-close freeze
# ---------------------------------------------------------------------------
# The cash index stops printing at 16:15 ET. The CDN keeps serving the identical
# body under a fresh top-level timestamp. The original collector wrote every one
# of them: four duplicate captures on Aug 11 plus one on Aug 10, all frozen at
# the same 16:14:59 print and the same 7728.2002 spot.


class TestSeed03PostCloseFreezeIsRejected:
    ACCEPTED_STATE = CollectorState(
        index_last_trade="2026-08-11T16:14:59",
        seqno=16938026517,
        capture_utc="2026-08-11T20-37-03Z",
        spot=7728.2002,
    )

    @pytest.mark.parametrize(
        ("capture", "now"),
        [
            ("2026-08-11T20-47-05Z", "2026-08-11T20:47:05"),
            ("2026-08-11T20-57-15Z", "2026-08-11T20:57:15"),
            ("2026-08-11T21-07-17Z", "2026-08-11T21:07:17"),
            ("2026-08-11T23-57-57Z", "2026-08-11T23:57:57"),
        ],
    )
    def test_each_frozen_repeat_is_rejected(self, capture, now):
        snap = payload.parse(load_header(capture), ticker="_SPX")
        decision = freshness.evaluate(snap, self.ACCEPTED_STATE, now=utc(now))
        assert not decision.accepted
        assert decision.verdict is freshness.Verdict.DUPLICATE_PRINT

    def test_the_close_capture_itself_is_accepted(self):
        """The 16:14:59 print is the settlement snapshot. Keep it.

        This is the boundary the inherited 22-minute threshold got wrong -- it
        discarded this capture by four seconds. See freshness.DEFAULT_MAX_STALE.
        """
        snap = payload.parse(load_header("2026-08-11T20-37-03Z"), ticker="_SPX")
        decision = freshness.evaluate(
            snap,
            CollectorState(index_last_trade="2026-08-11T16:11:36"),
            now=utc("2026-08-11T20:37:03"),
        )
        assert decision.accepted, "the close capture must survive the staleness gate"

    def test_close_is_accepted_even_with_no_prior_state(self):
        """A lost state file must not cost us the close print."""
        snap = payload.parse(load_header("2026-08-11T20-37-03Z"), ticker="_SPX")
        decision = freshness.evaluate(
            snap, CollectorState(), now=utc("2026-08-11T20:37:03")
        )
        assert decision.accepted

    def test_first_post_close_repeat_is_rejected_even_with_no_prior_state(self):
        """With state lost, the staleness gate is the only line of defence."""
        snap = payload.parse(load_header("2026-08-11T20-47-05Z"), ticker="_SPX")
        decision = freshness.evaluate(
            snap, CollectorState(), now=utc("2026-08-11T20:47:05")
        )
        assert not decision.accepted
        assert decision.verdict is freshness.Verdict.STALE_FEED


# ---------------------------------------------------------------------------
# 4. seqno is not a safe deduplication key
# ---------------------------------------------------------------------------
# It is the obvious choice and it is wrong. Across the post-close freeze the
# seqno advanced from 16938026517 to 16947306680 while the index print stayed
# frozen at 16:14:59. A seqno-keyed gate admits the 20:57 capture as new data.
# It is not new data -- the chain is unchanged.


class TestSeed04SeqnoIsNotADedupKey:
    def test_seqno_advanced_while_the_index_print_did_not(self):
        """Pin the raw observation the rule is derived from."""
        first = payload.parse(load_header("2026-08-11T20-37-03Z"), ticker="_SPX")
        later = payload.parse(load_header("2026-08-11T20-57-15Z"), ticker="_SPX")
        assert later.seqno != first.seqno, "seqno moved"
        assert later.index_last_trade == first.index_last_trade, "print did not"

    def test_advanced_seqno_does_not_admit_a_frozen_capture(self):
        """The actual regression: this capture must still be rejected."""
        snap = payload.parse(load_header("2026-08-11T20-57-15Z"), ticker="_SPX")
        state = CollectorState(
            index_last_trade="2026-08-11T16:14:59",
            seqno=16938026517,  # deliberately stale relative to the payload
        )
        decision = freshness.evaluate(snap, state, now=utc("2026-08-11T20:57:15"))
        assert not decision.accepted
        assert decision.verdict is freshness.Verdict.DUPLICATE_PRINT

    def test_matching_seqno_does_not_by_itself_reject_a_moved_print(self):
        """The converse. Only the print decides; seqno never overrides it."""
        snap = payload.parse(load_header("2026-08-11T16-11-33Z"), ticker="_SPX")
        state = CollectorState(
            index_last_trade="2026-08-11T11:46:00",
            seqno=snap.seqno,  # identical seqno, earlier print
        )
        decision = freshness.evaluate(snap, state, now=utc("2026-08-11T16:11:33"))
        assert decision.accepted


# ---------------------------------------------------------------------------
# 5. Weekend and holiday fetches
# ---------------------------------------------------------------------------
# The Aug 9 capture is a Sunday fetch serving Friday Aug 7's close -- a payload
# 50 hours old, served without complaint. No exchange calendar is needed to
# reject it: on a non-trading day the feed simply never freshens, so the
# staleness gate covers holidays for free.


class TestSeed05StaleOffHoursFetchIsRejected:
    @pytest.mark.parametrize(
        ("capture", "now", "expected_age_min"),
        [
            ("2026-08-09T22-38-43Z", "2026-08-09T22:38:43", 3023.73),
            ("2026-08-10T23-57-37Z", "2026-08-10T23:57:37", 222.63),
            ("2026-08-11T23-57-57Z", "2026-08-11T23:57:57", 222.97),
        ],
    )
    def test_rejected_as_stale(self, capture, now, expected_age_min):
        snap = payload.parse(load_header(capture), ticker="_SPX")
        decision = freshness.evaluate(snap, CollectorState(), now=utc(now))
        assert not decision.accepted
        assert decision.verdict is freshness.Verdict.STALE_FEED
        assert decision.age is not None
        assert decision.age.total_seconds() / 60 == pytest.approx(
            expected_age_min, abs=0.02
        )

    def test_no_exchange_calendar_is_consulted(self):
        """Holiday handling is a consequence of the staleness gate, not a
        feature with a dependency. Rejection must need nothing but the payload,
        the previous state, and the clock."""
        snap = payload.parse(load_header("2026-08-09T22-38-43Z"), ticker="_SPX")
        decision = freshness.evaluate(
            snap, CollectorState(), now=utc("2026-08-09T22:38:43")
        )
        assert decision.verdict is freshness.Verdict.STALE_FEED


# ---------------------------------------------------------------------------
# 6. A capture that reached the archive but never reached the curated layer
# ---------------------------------------------------------------------------
# _SPX_2026-08-11T23-57-57Z has raw JSON and no CSV: the original collector wrote
# the raw payload, then failed or was killed before writing the flattened rows.
# Nothing detected it. The archive must be re-derivable, and re-deriving must be
# idempotent, so a partial run is repaired by simply running again.
#
# The catalog half of this -- reporting raw/curated divergence -- lands in
# Stage 3. What is pinned here is the precondition: archive paths are a pure
# function of their inputs, so a re-run overwrites rather than duplicates.


class TestSeed06ArchiveWritesAreIdempotent:
    def test_archive_path_is_deterministic(self, tmp_path):
        when = utc("2026-08-11T20:37:03")
        first = collector.archive_path(tmp_path, "_SPX", when)
        second = collector.archive_path(tmp_path, "_SPX", when)
        assert first == second

    def test_archive_path_encodes_the_capture_time(self, tmp_path):
        path = collector.archive_path(tmp_path, "_SPX", utc("2026-08-11T20:37:03"))
        assert "2026-08-11T20-37-03Z" in path.name
        assert ":" not in path.name, "colons are hostile in filenames"

    def test_distinct_captures_get_distinct_paths(self, tmp_path):
        a = collector.archive_path(tmp_path, "_SPX", utc("2026-08-11T20:37:03"))
        b = collector.archive_path(tmp_path, "_SPX", utc("2026-08-11T20:47:05"))
        assert a != b

    def test_distinct_tickers_do_not_collide(self, tmp_path):
        when = utc("2026-08-11T20:37:03")
        assert collector.archive_path(tmp_path, "_SPX", when) != collector.archive_path(
            tmp_path, "SPY", when
        )


# ---------------------------------------------------------------------------
# 7. Zero-bid contracts
# ---------------------------------------------------------------------------
# 1,118 of the 30,692 options in the close snapshot have bid == 0: nobody is
# buying at any price. The mid is meaningless, the relative spread is 200%, and
# an IV inverted from that mid is noise wearing a number's clothing.
#
# Parsing must NOT drop them -- open interest on a zero-bid contract is real
# information. The obligation is that they are identifiable, so the curation
# layer can exclude them deliberately rather than by accident.


class TestSeed07ZeroBidIsPreservedAndIdentifiable:
    def test_zero_bid_contracts_survive_parsing(self, trimmed_chain):
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        zero_bid = [o for o in snap.options if o["bid"] == 0]
        assert zero_bid, "zero-bid contracts must not be silently dropped"

    def test_zero_bid_contracts_still_have_a_live_ask(self, trimmed_chain):
        """bid == 0 with ask > 0 is a one-sided market, not a missing quote.

        Distinguishing the two matters: one is information, the other is a hole.
        """
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        zero_bid = [o for o in snap.options if o["bid"] == 0]
        assert all(o["ask"] > 0 for o in zero_bid)

    def test_no_contract_is_quoted_zero_on_both_sides(self, full_chain):
        """Measured across the full 30,692-option chain: there are none.

        If this ever fails the feed has changed shape and the "one-sided market"
        reading above no longer holds.
        """
        both_zero = [
            o for o in full_chain["data"]["options"] if o["bid"] == 0 and o["ask"] == 0
        ]
        assert both_zero == []


# ---------------------------------------------------------------------------
# 8. last_trade_price is not a quote
# ---------------------------------------------------------------------------
# Two contracts in the close snapshot last printed in 2024 -- SPX271217C01800000
# on 2024-08-09 at 3588.02, against a mid of 5949.90 today, 39.7% away.
#
# The individual cases are the vivid part; the population is the important part.
# Of 22,716 quoted contracts that have ever traded, 3,751 -- one in six -- sit
# more than 25% from their own live mid. This is not a handful of dead strikes,
# it is the normal condition of an option chain, and any pipeline that reaches
# for last_trade_price when bid/ask look inconvenient is wrong at that scale.


def _mid_deviation(opt: dict) -> float:
    mid = (opt["bid"] + opt["ask"]) / 2
    return abs(opt["last_trade_price"] - mid) / mid


class TestSeed08LastTradePriceIsNotAQuote:
    def test_last_trade_times_can_predate_the_capture_by_years(self, trimmed_chain):
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        ancient = [
            o
            for o in snap.options
            if o.get("last_trade_time") and o["last_trade_time"] < "2025-01-01"
        ]
        assert ancient, "fixture must retain contracts with years-stale prints"

    def test_years_stale_trades_diverge_materially_from_the_live_quote(
        self, trimmed_chain
    ):
        """The specific case. Observed deviations are 38.5% and 39.7%."""
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        deviations = [
            _mid_deviation(o)
            for o in snap.options
            if o.get("last_trade_time")
            and o["last_trade_time"] < "2025-01-01"
            and o["bid"] > 0
            and o["last_trade_price"]
        ]
        assert deviations, "fixture must retain a priceable years-stale contract"
        assert max(deviations) > 0.25, (
            "expected a years-stale last trade at least 25% from the live mid; "
            f"worst observed was {max(deviations):.1%}"
        )

    def test_stale_last_trades_are_pervasive_not_exceptional(self, full_chain):
        """The population claim, measured across the full 30,692-option chain.

        This is the one that justifies the rule. If it were two dead contracts
        you could special-case them; at one in six you cannot, and the only
        correct policy is to never read last_trade_price as a price.
        """
        quoted = [
            o
            for o in full_chain["data"]["options"]
            if o.get("last_trade_price") and o["bid"] > 0
        ]
        far = [o for o in quoted if _mid_deviation(o) > 0.25]
        assert len(quoted) > 20_000, f"unexpected chain shape: {len(quoted)} quoted"
        assert len(far) / len(quoted) > 0.10, (
            "expected >10% of quoted contracts to sit over 25% from their own "
            f"mid; observed {len(far)}/{len(quoted)} = {len(far) / len(quoted):.1%}"
        )


# ---------------------------------------------------------------------------
# 9. Degenerate implied volatilities
# ---------------------------------------------------------------------------
# CBOE publishes an 800% IV on a strike-200 call against a 7728 spot, with delta
# 0.9997. That contract is worth its intrinsic value; there is no volatility
# information in it, and the number is an artefact of inverting a price that is
# all intrinsic. 0-DTE contracts blow up the same way for a different reason:
# tenor near zero makes vega vanish, so any pricing error explodes into IV.


class TestSeed09DegenerateImpliedVolsAreIdentifiable:
    def test_absurd_ivs_exist_in_real_captured_data(self, trimmed_chain):
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        absurd = [o for o in snap.options if o.get("iv") and o["iv"] > 5.0]
        assert absurd, "fixture must retain contracts with IV > 500%"

    def test_deep_itm_calls_carry_degenerate_ivs(self, trimmed_chain):
        """Strike 200 against a 7728 spot: pure intrinsic, no vol content."""
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        spot = snap.spot
        deep = [
            o
            for o in snap.options
            if (sym := osi.try_parse(o["option"]))
            and sym.is_call
            and sym.strike < spot * 0.10
        ]
        assert deep, "fixture must retain deep-ITM contracts"
        assert any(o["iv"] > 5.0 for o in deep)

    def test_zero_dte_contracts_carry_degenerate_ivs(self, trimmed_chain):
        """A different mechanism, the same symptom: vega -> 0 as tenor -> 0."""
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        zero_dte = [
            o
            for o in snap.options
            if (sym := osi.try_parse(o["option"]))
            and sym.expiry == snap.index_last_trade.date()
        ]
        assert zero_dte, "fixture must retain 0-DTE contracts"
        assert any(o["iv"] > 5.0 for o in zero_dte)


# ---------------------------------------------------------------------------
# 10. Same-day expiry
# ---------------------------------------------------------------------------
# The Aug 11 close snapshot contains 562 contracts expiring 2026-08-11. Tenor is
# zero, or negative if measured against a 16:15 close from a 16:14:59 print.
# Anything dividing by time to expiry -- IV inversion, forward implication, the
# Breeden-Litzenberger scaling -- divides by zero.


class TestSeed10SameDayExpiryIsHandled:
    def test_zero_dte_contracts_exist_in_the_capture(self, trimmed_chain):
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        same_day = [
            o
            for o in snap.options
            if (sym := osi.try_parse(o["option"]))
            and sym.expiry == snap.index_last_trade.date()
        ]
        assert same_day, "fixture must retain same-day expiries"

    def test_zero_dte_is_parsed_not_rejected(self):
        """Parsing must not be where 0-DTE is filtered.

        Exclusion is a modelling decision that belongs to the curation layer,
        where it is visible and reversible -- not a silent side effect of a
        symbol parser.
        """
        sym = osi.parse("SPXW260811C07730000")
        assert sym.expiry.isoformat() == "2026-08-11"

    def test_expiry_date_is_a_real_date_object(self):
        """Not a string. A string expiry is how tenor arithmetic gets skipped."""
        from datetime import date

        assert isinstance(osi.parse("SPXW260811C07730000").expiry, date)


# ---------------------------------------------------------------------------
# 11. iv == 0.0 is a sentinel, not a volatility
# ---------------------------------------------------------------------------
# 2,160 contracts in the close snapshot report iv == 0.0. Of those, 1,890 carry
# live two-sided quotes, and 694 sit within 10% of spot -- precisely the region
# that determines the shape of the density.
#
# Zero volatility is not a thing. This is CBOE's "not computed" marker. Consumed
# as a number it flattens the smile exactly where the smile matters, and the
# resulting density looks plausible. This is the strongest argument for
# inverting IV ourselves in Stage 5 and treating the published iv as a
# cross-check only, never an input.


class TestSeed11ZeroIvIsASentinel:
    def test_zero_iv_contracts_exist_with_live_quotes(self, trimmed_chain):
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        sentinel = [o for o in snap.options if o["iv"] == 0.0 and o["bid"] > 0]
        assert sentinel, "fixture must retain iv==0 contracts that are really quoted"

    def test_zero_iv_reaches_near_the_money(self, trimmed_chain):
        """The damaging part. Far-wing sentinels would be ignorable."""
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        near = [
            o
            for o in snap.options
            if o["iv"] == 0.0
            and o["bid"] > 0
            and (sym := osi.try_parse(o["option"]))
            and abs(sym.strike - snap.spot) / snap.spot <= 0.10
        ]
        assert near, "iv==0 must be shown to occur within 10% of spot"

    def test_zero_iv_contracts_have_a_nonzero_theoretical_value(self, trimmed_chain):
        """Proof it is a sentinel and not a real computation.

        CBOE prices these contracts -- theo is populated and delta is
        non-degenerate. Only the IV field is blank. A genuine zero-vol
        contract would be worth exactly its intrinsic value.
        """
        snap = payload.parse(trimmed_chain, ticker="_SPX")
        sentinel = [o for o in snap.options if o["iv"] == 0.0 and o["bid"] > 0]
        assert all(o["theo"] > 0 for o in sentinel)

    def test_zero_iv_scale_across_the_full_chain(self, full_chain):
        """Measured, so a change in the feed's behaviour is visible.

        Loose bounds on purpose: this must fail when the feed's shape changes
        materially, not when a few contracts move.
        """
        opts = full_chain["data"]["options"]
        sentinel = [o for o in opts if o["iv"] == 0.0]
        quoted = [o for o in sentinel if o["bid"] > 0]
        assert 1500 < len(sentinel) < 3000, f"iv==0 count moved: {len(sentinel)}"
        assert len(quoted) > 1000, f"quoted iv==0 count moved: {len(quoted)}"


# ---------------------------------------------------------------------------
# Boundary: the staleness threshold itself
# ---------------------------------------------------------------------------
# Derived from measured data rather than chosen. Pinned here because the
# inherited value was wrong by four seconds in the direction that loses the most
# valuable capture of the day, and nothing would have reported it.


class TestStalenessThresholdBoundary:
    def test_threshold_admits_the_observed_close_capture(self):
        assert timedelta(minutes=22.07) < freshness.DEFAULT_MAX_STALE

    def test_threshold_rejects_the_first_observed_post_close_repeat(self):
        assert timedelta(minutes=32.10) > freshness.DEFAULT_MAX_STALE

    def test_threshold_is_derived_from_delay_plus_cadence(self):
        """Not a magic number: feed delay + one cadence + jitter headroom."""
        assert freshness.DEFAULT_MAX_STALE >= (freshness.FEED_DELAY + freshness.CADENCE)


# ---------------------------------------------------------------------------
# 12. prev_day_close rolls forward overnight
# ---------------------------------------------------------------------------
# It is not reliably "the previous day's close". Some hours after the session
# ends the feed advances it to the CURRENT day's close, so on an evening
# snapshot prev_day_close == current_price and any daily return computed from
# it is exactly zero.
#
# Observed in the archive:
#
#     Aug 11 19:57 ET  (3h42m after close)  prev=7753.11  current=7728.20  ok
#     Aug 12 21:25 ET  (5h10m after close)  prev=7748.50  current=7748.50  rolled
#
# So the rollover happens somewhere in the evening, and whether a given capture
# has crossed it is not knowable from the payload alone. The only safe reading
# is that prev_day_close is unusable for return calculations; the previous
# close must come from the previous capture in our own archive.


class TestSeed12PrevDayCloseRollsForward:
    def test_an_evening_capture_shows_prev_equal_to_current(self):
        raw = load_header("2026-08-13T01-25-33Z")["data"]
        assert raw["prev_day_close"] == raw["current_price"] == 7748.5

    def test_an_earlier_evening_capture_has_not_yet_rolled(self):
        """The same session, four hours earlier, still reports Aug 10's close.

        The pair is the whole point: identical field, same phase of the day,
        opposite meanings.
        """
        raw = load_header("2026-08-11T23-57-57Z")["data"]
        assert raw["prev_day_close"] == 7753.1099
        assert raw["current_price"] == 7728.2002
        assert raw["prev_day_close"] != raw["current_price"]

    def test_intraday_captures_report_a_genuine_previous_close(self):
        """During the session the field means what it says."""
        raw = load_header("2026-08-11T16-11-33Z")["data"]
        assert raw["prev_day_close"] == 7753.1099
        assert raw["current_price"] == 7741.3999

    def test_a_daily_return_from_prev_day_close_would_be_zero(self):
        """State the damage as the calculation someone would actually write."""
        raw = load_header("2026-08-13T01-25-33Z")["data"]
        daily_return = raw["current_price"] / raw["prev_day_close"] - 1
        assert daily_return == 0.0, (
            "if this ever stops being 0, the feed changed and the rule that "
            "prev_day_close is unusable for returns should be revisited"
        )
