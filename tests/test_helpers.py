import asyncio
import logging

import pytest

from rate_limit import (
    BackoffPolicy,
    RateLimit,
    RateLimiter,
    RateLimitExceededError,
    note_rate_limited,
    note_request_succeeded,
    run_paced,
    run_paced_limit,
)
from rate_limit.testing import ManualClock


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(1000.0)


def bound(clock: ManualClock, limit: int = 100):  # noqa: ANN201
    return RateLimiter([RateLimit("a", limit, 1)], clock=clock, sleep=clock.sleep).bind("a")


async def ok() -> str:
    return "ok"


def failing(error: Exception):  # noqa: ANN201
    async def call() -> str:
        await asyncio.sleep(0)
        raise error

    return call


class TestRunPaced:
    async def test_success_resets_streak(self, clock: ManualClock) -> None:
        limiter = bound(clock)
        limiter.backoff.streak = 3
        assert await run_paced(limiter, ok) == "ok"
        assert limiter.backoff.streak == 0

    async def test_rate_limit_pauses_logs_once_and_reraises(
        self, clock: ManualClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        limiter = bound(clock)
        error = RateLimitExceededError(status=429, retry_after=7.0)
        with caplog.at_level(logging.WARNING), pytest.raises(RateLimitExceededError):
            await run_paced(limiter, failing(error))
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "429" in warnings[0].getMessage()
        assert "7.0" in warnings[0].getMessage()
        await limiter.acquire()
        assert clock.sleeps == [pytest.approx(7.0)]

    async def test_other_errors_do_not_touch_backoff(self, clock: ManualClock) -> None:
        limiter = bound(clock)
        limiter.backoff.streak = 2
        with pytest.raises(RuntimeError):
            await run_paced(limiter, failing(RuntimeError("boom")))
        assert limiter.backoff.streak == 2
        assert not limiter.is_paused()

    async def test_streak_grows_then_resets(self, clock: ManualClock) -> None:
        limiter = bound(clock)
        for expected in (5.0, 10.0):
            with pytest.raises(RateLimitExceededError):
                await run_paced(limiter, failing(RateLimitExceededError(status=429)))
            before = clock()
            await limiter.acquire()
            assert clock() - before == pytest.approx(expected, abs=0.01)
        await run_paced(limiter, ok)
        assert limiter.backoff.streak == 0

    async def test_concurrent_burst_is_one_pause(self, clock: ManualClock) -> None:
        limiter = bound(clock)
        error = RateLimitExceededError(status=429)
        results = await asyncio.gather(
            *(run_paced(limiter, failing(error)) for _ in range(5)), return_exceptions=True
        )
        assert all(isinstance(r, RateLimitExceededError) for r in results)
        assert limiter.backoff.streak == 1

    async def test_retry_after_and_418_extend_an_active_pause(self, clock: ManualClock) -> None:
        limiter = bound(clock)
        limiter.pause(5)
        with pytest.raises(RateLimitExceededError):
            await run_paced(limiter, failing(RateLimitExceededError(status=429, retry_after=20.0)))
        with pytest.raises(RateLimitExceededError):
            await run_paced(limiter, failing(RateLimitExceededError(status=418)))
        before = clock()
        await limiter.acquire()
        assert clock() - before == pytest.approx(120.0, abs=0.01)


class TestBoundLimiterDelegation:
    def test_418_on_bound_limiter_pauses_all_limits(self, clock: ManualClock) -> None:
        rl = RateLimiter(
            [RateLimit("a", 10, 1), RateLimit("b", 10, 1)], clock=clock, sleep=clock.sleep
        )
        note_rate_limited(rl.bind("a"), RateLimitExceededError(status=418))
        assert rl.is_paused("b")


class TestUnpausableLimiter:
    class AcquireOnly:
        async def acquire(self) -> None:
            return None

    async def test_warns_once_and_reraises(self, caplog: pytest.LogCaptureFixture) -> None:
        limiter = self.AcquireOnly()
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                with pytest.raises(RateLimitExceededError):
                    await run_paced(limiter, failing(RateLimitExceededError(status=429)))
        assert len([r for r in caplog.records if "cannot pause" in r.getMessage()]) == 1

    def test_non_weakrefable_limiter_does_not_crash(self) -> None:
        class Slotted:
            __slots__ = ()

            async def acquire(self) -> None:
                return None

        limiter = Slotted()
        note_rate_limited(limiter, RateLimitExceededError(status=429))
        note_request_succeeded(limiter)

    def test_fallback_policy_is_remembered_and_reset(self) -> None:
        class Pausable:
            def __init__(self) -> None:
                self.paused: list[float] = []

            async def acquire(self) -> None:
                return None

            def pause(self, seconds: float) -> None:
                self.paused.append(seconds)

        limiter = Pausable()
        note_rate_limited(limiter, RateLimitExceededError(status=429))
        note_rate_limited(limiter, RateLimitExceededError(status=429))
        assert limiter.paused == [5.0, 10.0]
        note_request_succeeded(limiter)
        note_rate_limited(limiter, RateLimitExceededError(status=None))
        assert limiter.paused[-1] == 5.0

    def test_limiter_label_falls_back_to_type_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            note_rate_limited(self.AcquireOnly(), RateLimitExceededError(status=418))  # type: ignore[arg-type]
        assert "AcquireOnly" in caplog.text


class TestRunPacedLimit:
    async def test_success_and_weights(self, clock: ManualClock) -> None:
        rl = RateLimiter([RateLimit("a", 10, 1)], clock=clock, sleep=clock.sleep)
        rl.backoff("a").streak = 4
        assert await run_paced_limit(rl, "a", ok, weights={"a": 10}) == "ok"
        assert rl.backoff("a").streak == 0
        assert rl.blocked_for("a") == pytest.approx(1.0)

    async def test_418_pauses_all_limits(self, clock: ManualClock) -> None:
        rl = RateLimiter(
            [RateLimit("a", 10, 1), RateLimit("b", 10, 1)],
            clock=clock,
            sleep=clock.sleep,
            backoff_factory=BackoffPolicy,
        )
        with pytest.raises(RateLimitExceededError):
            await run_paced_limit(rl, "a", failing(RateLimitExceededError(status=418)))
        assert rl.is_paused("b")

    async def test_other_errors_propagate(self, clock: ManualClock) -> None:
        rl = RateLimiter([RateLimit("a", 10, 1)], clock=clock, sleep=clock.sleep)
        with pytest.raises(RuntimeError):
            await run_paced_limit(rl, "a", failing(RuntimeError("x")))
        assert not rl.is_paused("a")
