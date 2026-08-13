"""Unit tests for `spxrnd.ingest.freshness`.

The gate that decides what enters a permanent archive. Every branch is covered
on both sides of its boundary, because a wrong answer here is either a lost
capture that cannot be recovered or a duplicate that quietly corrupts a series.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from spxrnd.ingest.freshness import (
    CADENCE,
    DEFAULT_MAX_STALE,
    FEED_DELAY,
    Decision,
    Verdict,
    evaluate,
)
from spxrnd.ingest.payload import parse
from spxrnd.ingest.state import CollectorState

from ..conftest import load_header, utc

# Every capture below is real. `now` is each capture's own wall-clock time.
HEALTHY = "2026-08-11T16-11-33Z"
HEALTHY_NOW = "2026-08-11T16:11:33"
HEALTHY_PRINT = "2026-08-11T11:56:26"

CLOSE = "2026-08-11T20-37-03Z"
CLOSE_NOW = "2026-08-11T20:37:03"
CLOSE_PRINT = "2026-08-11T16:14:59"


def snap(capture):
    return parse(load_header(capture), ticker="_SPX")


class TestAcceptance:
    def test_first_ever_run_accepts_a_fresh_capture(self):
        d = evaluate(snap(HEALTHY), CollectorState(), now=utc(HEALTHY_NOW))
        assert d.verdict is Verdict.ACCEPT
        assert d.accepted

    def test_moved_print_is_accepted(self):
        d = evaluate(
            snap(HEALTHY),
            CollectorState(index_last_trade="2026-08-11T11:46:26"),
            now=utc(HEALTHY_NOW),
        )
        assert d.verdict is Verdict.ACCEPT

    def test_detail_carries_the_numbers_an_operator_needs(self):
        d = evaluate(snap(HEALTHY), CollectorState(), now=utc(HEALTHY_NOW))
        assert HEALTHY_PRINT in d.detail
        assert "7741.3999" in d.detail

    def test_age_is_reported(self):
        d = evaluate(snap(HEALTHY), CollectorState(), now=utc(HEALTHY_NOW))
        assert d.age.total_seconds() / 60 == pytest.approx(15.12, abs=0.02)


class TestDuplicateGate:
    def test_unmoved_print_is_a_duplicate(self):
        d = evaluate(
            snap(CLOSE),
            CollectorState(index_last_trade=CLOSE_PRINT),
            now=utc(CLOSE_NOW),
        )
        assert d.verdict is Verdict.DUPLICATE_PRINT
        assert not d.accepted

    def test_duplicate_wins_over_stale_when_both_apply(self):
        """A repeat we have seen before is better diagnosed as a duplicate.

        The 23:57 capture is both frozen and 3h43m old. "No new data" is the
        more precise statement and the one that recurs; "stale" is the backstop
        for when state is missing.
        """
        d = evaluate(
            snap("2026-08-11T23-57-57Z"),
            CollectorState(index_last_trade=CLOSE_PRINT),
            now=utc("2026-08-11T23:57:57"),
        )
        assert d.verdict is Verdict.DUPLICATE_PRINT

    def test_duplicate_detail_calls_out_a_moved_seqno(self):
        """When seqno moved but the print did not, say so explicitly.

        This is the trap that would have been fallen into; the log line should
        name it rather than leave the reader to rediscover it.
        """
        d = evaluate(
            snap("2026-08-11T20-57-15Z"),
            CollectorState(index_last_trade=CLOSE_PRINT, seqno=16938026517),
            now=utc("2026-08-11T20:57:15"),
        )
        assert d.verdict is Verdict.DUPLICATE_PRINT
        assert "not a dedup key" in d.detail
        assert "16938026517" in d.detail

    def test_duplicate_detail_stays_quiet_when_seqno_also_matched(self):
        d = evaluate(
            snap("2026-08-11T20-47-05Z"),
            CollectorState(index_last_trade=CLOSE_PRINT, seqno=16938026517),
            now=utc("2026-08-11T20:47:05"),
        )
        assert "not a dedup key" not in d.detail

    def test_empty_state_cannot_trigger_the_duplicate_gate(self):
        """A first run must never be diagnosed as a repeat of nothing."""
        d = evaluate(snap(HEALTHY), CollectorState(), now=utc(HEALTHY_NOW))
        assert d.verdict is not Verdict.DUPLICATE_PRINT


class TestStalenessGate:
    def test_exactly_at_the_limit_is_accepted(self):
        """Boundary: `age > max_stale` rejects, so equality is accepted."""
        s = snap(CLOSE)
        at_limit = s.index_last_trade + DEFAULT_MAX_STALE
        assert evaluate(s, CollectorState(), now=at_limit).verdict is Verdict.ACCEPT

    def test_one_second_past_the_limit_is_rejected(self):
        s = snap(CLOSE)
        past = s.index_last_trade + DEFAULT_MAX_STALE + timedelta(seconds=1)
        assert evaluate(s, CollectorState(), now=past).verdict is Verdict.STALE_FEED

    def test_custom_limit_is_honoured(self):
        d = evaluate(
            snap(HEALTHY),
            CollectorState(),
            now=utc(HEALTHY_NOW),
            max_stale=timedelta(minutes=1),
        )
        assert d.verdict is Verdict.STALE_FEED

    def test_stale_detail_reports_both_the_age_and_the_limit(self):
        d = evaluate(
            snap("2026-08-09T22-38-43Z"),
            CollectorState(),
            now=utc("2026-08-09T22:38:43"),
        )
        assert "3023.7 min" in d.detail
        assert "27 min limit" in d.detail

    def test_a_future_print_is_not_stale(self):
        """Clock skew between us and the feed must not discard a capture.

        A negative age is under any positive limit, so it passes. Whether a
        future print is *plausible* is a different question, and not one the
        staleness gate should answer by throwing data away.
        """
        s = snap(HEALTHY)
        d = evaluate(s, CollectorState(), now=s.index_last_trade - timedelta(minutes=5))
        assert d.verdict is Verdict.ACCEPT
        assert d.age < timedelta(0)


class TestForce:
    def test_force_overrides_the_duplicate_gate(self):
        d = evaluate(
            snap(CLOSE),
            CollectorState(index_last_trade=CLOSE_PRINT),
            now=utc(CLOSE_NOW),
            force=True,
        )
        assert d.verdict is Verdict.FORCED
        assert d.accepted

    def test_force_overrides_the_staleness_gate(self):
        d = evaluate(
            snap("2026-08-09T22-38-43Z"),
            CollectorState(),
            now=utc("2026-08-09T22:38:43"),
            force=True,
        )
        assert d.verdict is Verdict.FORCED
        assert d.accepted

    def test_forced_is_distinguishable_from_a_clean_accept(self):
        """So a forced capture is identifiable after the fact."""
        assert Verdict.FORCED != Verdict.ACCEPT


class TestClockDiscipline:
    def test_naive_now_is_rejected(self):
        """A naive `now` would silently compare against an Eastern print and be
        wrong by the UTC offset -- four hours in summer, five in winter."""
        from datetime import datetime

        with pytest.raises(ValueError, match="timezone-aware"):
            evaluate(snap(HEALTHY), CollectorState(), now=datetime(2026, 8, 11, 16, 11))

    def test_evaluate_never_reads_the_wall_clock(self, monkeypatch):
        """Pin the injection. If `evaluate` ever calls datetime.now(), the gate
        stops being deterministically testable."""
        import spxrnd.ingest.freshness as mod

        class Exploding:
            def now(self, *a, **k):
                raise AssertionError("evaluate must not read the clock")

        monkeypatch.setattr(mod, "datetime", Exploding(), raising=False)
        assert evaluate(snap(HEALTHY), CollectorState(), now=utc(HEALTHY_NOW)).accepted


class TestThresholdDerivation:
    def test_default_is_delay_plus_cadence_plus_headroom(self):
        assert FEED_DELAY + CADENCE + timedelta(minutes=2) == DEFAULT_MAX_STALE
        assert timedelta(minutes=27) == DEFAULT_MAX_STALE

    def test_margin_on_both_sides_of_the_observed_window(self):
        """Observed: close capture 22.07 min, first post-close repeat 32.10."""
        assert DEFAULT_MAX_STALE - timedelta(minutes=22.07) >= timedelta(minutes=4)
        assert timedelta(minutes=32.10) - DEFAULT_MAX_STALE >= timedelta(minutes=4)


class TestDecisionValue:
    def test_accepted_is_true_only_for_accept_and_forced(self):
        for verdict in Verdict:
            d = Decision(verdict=verdict, detail="")
            assert d.accepted == (verdict in (Verdict.ACCEPT, Verdict.FORCED))

    def test_verdicts_are_stable_strings(self):
        """Logged and asserted on; renaming one is a breaking change."""
        assert Verdict.ACCEPT == "accept"
        assert Verdict.DUPLICATE_PRINT == "duplicate_print"
        assert Verdict.STALE_FEED == "stale_feed"
        assert Verdict.FORCED == "forced"
        assert Verdict.SKIPPED_LOCKED == "skipped_locked"


class TestWholeArchiveReplay:
    """Replay the whole captured timeline through the gate in order.

    The single most valuable test in this module: it asserts the gate would
    have made the right call on every capture we actually took, which is the
    claim the component exists to support.
    """

    TIMELINE = [
        ("2026-08-09T22-38-43Z", "2026-08-09T22:38:43", Verdict.STALE_FEED),
        ("2026-08-10T23-57-37Z", "2026-08-10T23:57:37", Verdict.STALE_FEED),
        ("2026-08-11T00-07-40Z", "2026-08-11T00:07:40", Verdict.STALE_FEED),
        ("2026-08-11T16-11-33Z", "2026-08-11T16:11:33", Verdict.ACCEPT),
        ("2026-08-11T20-37-03Z", "2026-08-11T20:37:03", Verdict.ACCEPT),
        ("2026-08-11T20-47-05Z", "2026-08-11T20:47:05", Verdict.DUPLICATE_PRINT),
        ("2026-08-11T20-57-15Z", "2026-08-11T20:57:15", Verdict.DUPLICATE_PRINT),
        ("2026-08-11T21-07-17Z", "2026-08-11T21:07:17", Verdict.DUPLICATE_PRINT),
        ("2026-08-11T23-57-57Z", "2026-08-11T23:57:57", Verdict.DUPLICATE_PRINT),
    ]

    def test_every_capture_gets_the_right_verdict_in_sequence(self):
        state = CollectorState()
        got = []
        for capture, now, _expected in self.TIMELINE:
            s = snap(capture)
            d = evaluate(s, state, now=utc(now))
            got.append((capture, d.verdict))
            if d.accepted:
                state = CollectorState(
                    index_last_trade=s.index_last_trade.replace(tzinfo=None).isoformat(
                        timespec="seconds"
                    ),
                    seqno=s.seqno,
                    spot=s.spot,
                )
        assert got == [(c, v) for c, _n, v in self.TIMELINE]

    def test_exactly_two_captures_survive_the_gate(self):
        """Of the nine header fixtures, only the healthy mid-session capture and
        the settlement capture carry new information."""
        state = CollectorState()
        accepted = []
        for capture, now, _ in self.TIMELINE:
            s = snap(capture)
            d = evaluate(s, state, now=utc(now))
            if d.accepted:
                accepted.append(capture)
                state = CollectorState(
                    index_last_trade=s.index_last_trade.replace(tzinfo=None).isoformat(
                        timespec="seconds"
                    ),
                    seqno=s.seqno,
                )
        assert accepted == ["2026-08-11T16-11-33Z", "2026-08-11T20-37-03Z"]
