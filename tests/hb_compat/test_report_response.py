import asyncio
from email.utils import formatdate

import pytest

import rate_limit
from rate_limit.hb_compat.async_throttler import AsyncThrottler
from rate_limit.hb_compat.data_types import RateLimit
from rate_limit.testing import ManualClock

WALL_NOW = 1_700_000_000.0


def build(clock: ManualClock) -> AsyncThrottler:
    return AsyncThrottler(
        rate_limits=[
            RateLimit("a", limit=100, time_interval=60.0),
            RateLimit("b", limit=100, time_interval=60.0),
        ],
        retry_interval=0.005,
        clock=clock,
        wall_clock=lambda: WALL_NOW,
    )


def test_2xx_returns_not_limited_and_changes_nothing():
    throttler = build(ManualClock())

    verdict = throttler.report_response("a", 200, {})

    assert verdict == rate_limit.RateLimitVerdict(False, 200, None, 0.0, False)
    assert throttler.paused_until("a") is None


@pytest.mark.parametrize("status", [99, 302, 400, 401, 404, 500, 503])
def test_other_statuses_change_nothing_and_never_raise(status):
    throttler = build(ManualClock())

    verdict = throttler.report_response("a", status, {"Retry-After": "30"})

    assert verdict.rate_limited is False
    assert verdict.status == status
    assert throttler.paused_until("a") is None


def test_429_with_retry_after_seconds_pauses_the_limit_id():
    clock = ManualClock(10.0)
    throttler = build(clock)

    verdict = throttler.report_response("a", 429, {"Retry-After": "7"})

    assert verdict == rate_limit.RateLimitVerdict(True, 429, 7.0, 7.0, False)
    assert throttler.paused_until("a") == 17.0
    assert throttler.paused_until("b") is None


def test_429_retry_after_header_lookup_is_case_insensitive():
    throttler = build(ManualClock())

    assert throttler.report_response("a", 429, {"retry-after": "3"}).retry_after == 3.0


def test_429_with_http_date_retry_after_uses_the_injected_wall_clock():
    clock = ManualClock(0.0)
    throttler = build(clock)
    header = formatdate(WALL_NOW + 30, usegmt=True)

    verdict = throttler.report_response("a", 429, {"Retry-After": header})

    assert verdict.retry_after == pytest.approx(30.0, abs=1.0)
    assert throttler.paused_until("a") == pytest.approx(30.0, abs=1.0)


def test_429_without_retry_after_backs_off_exponentially_up_to_the_cap():
    clock = ManualClock()
    throttler = build(clock)
    delays = []

    for _ in range(9):
        delays.append(throttler.report_response("a", 429, None).delay)
        clock.advance(delays[-1])  # let the pause lapse so the next 429 is a fresh signal

    assert delays == [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 300.0, 300.0]


def test_signal_during_an_active_pause_does_not_escalate_the_streak():
    throttler = build(ManualClock())

    first = throttler.report_response("a", 429, None)
    burst = throttler.report_response("a", 429, None)

    assert (first.delay, burst.delay) == (5.0, 0.0)
    assert burst.rate_limited is True


def test_2xx_resets_the_backoff_streak():
    clock = ManualClock()
    throttler = build(clock)
    for expected in (5.0, 10.0):
        verdict = throttler.report_response("a", 429, None)
        assert verdict.delay == expected
        clock.advance(verdict.delay)

    throttler.report_response("a", 200, None)

    assert throttler.report_response("a", 429, None).delay == 5.0


def test_backoff_streak_is_per_limit_id():
    clock = ManualClock()
    throttler = build(clock)
    clock.advance(throttler.report_response("a", 429, None).delay)
    assert throttler.report_response("a", 429, None).delay == 10.0

    assert throttler.report_response("b", 429, None).delay == 5.0


def test_418_pauses_everything_for_the_default_ban_without_retry_after():
    clock = ManualClock(1.0)
    throttler = build(clock)

    verdict = throttler.report_response("a", 418, None)

    assert verdict == rate_limit.RateLimitVerdict(True, 418, None, 120.0, True)
    for limit_id in ("a", "b", "never_registered"):
        assert throttler.paused_until(limit_id) == 121.0


def test_418_with_retry_after_uses_it_for_pause_all():
    throttler = build(ManualClock())

    verdict = throttler.report_response("a", 418, {"Retry-After": "9"})

    assert verdict.paused_all is True
    assert throttler.paused_until("b") == 9.0


async def test_report_response_pause_is_the_same_state_that_blocks_acquirers():
    clock = ManualClock()
    throttler = build(clock)

    throttler.report_response("a", 429, {"Retry-After": "5"})

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(throttler.execute_task("a").acquire(), 0.05)
    clock.advance(5.0)
    await asyncio.wait_for(throttler.execute_task("a").acquire(), 0.5)


def test_explicit_pause_counts_as_in_burst_for_backoff():
    throttler = build(ManualClock())
    throttler.pause("a", 30.0)

    verdict = throttler.report_response("a", 429, None)

    assert verdict.delay == 0.0  # joins the active pause; single source of truth is pause()
    assert throttler.paused_until("a") == 30.0
