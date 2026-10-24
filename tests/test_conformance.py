from collections.abc import Sequence
from decimal import Decimal
from types import SimpleNamespace

import pytest

from rate_limit import (
    BackoffPolicy,
    PausableRateLimiterProtocol,
    RateLimit,
    RateLimiter,
    RestRateLimiterProtocol,
    ThrottlerProtocol,
)
from rate_limit.testing import (
    ManualClock,
    assert_conforms_backoff,
    assert_conforms_pausable,
    assert_conforms_rest_limiter,
    assert_conforms_throttler,
)
from rate_limit.types import Clock, Sleep


def make_rest(limit: int, interval: float):  # noqa: ANN201
    def make(clock: Clock, sleep: Sleep) -> RestRateLimiterProtocol:
        return RateLimiter([RateLimit("x", limit, interval)], clock=clock, sleep=sleep).bind("x")

    return make


def make_throttler(limits: Sequence[RateLimit], clock: Clock, sleep: Sleep) -> ThrottlerProtocol:
    return RateLimiter(limits, clock=clock, sleep=sleep)


@pytest.mark.parametrize(("limit", "interval"), [(1, 1.0), (3, 10.0), (5, 0.25)])
def test_rest_limiter_conformance(limit: int, interval: float) -> None:
    assert_conforms_rest_limiter(make_rest(limit, interval), limit=limit, interval=interval)


def test_pausable_conformance() -> None:
    def make(clock: Clock, sleep: Sleep) -> PausableRateLimiterProtocol:
        return RateLimiter([RateLimit("x", 100, 1)], clock=clock, sleep=sleep).bind("x")

    assert_conforms_pausable(make)


def test_throttler_conformance() -> None:
    assert_conforms_throttler(make_throttler, weighted=True, registry=True)


def test_throttler_conformance_basic_only() -> None:
    assert_conforms_throttler(make_throttler)


def test_backoff_conformance_with_custom_policy() -> None:
    assert_conforms_backoff(lambda: BackoffPolicy(1.0, 8.0, 30.0))


def test_conformance_detects_a_broken_limiter() -> None:
    class NoLimit:
        async def acquire(self) -> None:
            return None

    with pytest.raises(AssertionError):
        assert_conforms_rest_limiter(lambda _c, _s: NoLimit(), limit=1, interval=1.0)


def test_conformance_detects_future_dated_reservations() -> None:
    class Reserving:
        """Records the charge before sleeping (the future-dated reservation defect)."""

        def __init__(self, clock: Clock, sleep: Sleep, limit: int, interval: float) -> None:
            self.clock, self.sleep, self.limit, self.interval = clock, sleep, limit, interval
            self.stamps: list[float] = []

        async def acquire(self) -> None:
            now = self.clock()
            self.stamps = [s for s in self.stamps if s + self.interval >= now]
            if len(self.stamps) >= self.limit:
                at = self.stamps[0] + self.interval
                self.stamps.append(at)
                await self.sleep(at - now)
                return
            self.stamps.append(now)

    with pytest.raises(AssertionError):
        assert_conforms_rest_limiter(lambda c, s: Reserving(c, s, 1, 10.0), limit=1, interval=10.0)


def test_conformance_detects_unscaled_pause() -> None:
    class Ignores:
        async def acquire(self) -> None:
            return None

        def pause(self, seconds: float) -> None:
            return None

    with pytest.raises(AssertionError):
        assert_conforms_pausable(lambda _c, _s: Ignores())


def test_conformance_detects_swallowed_body_exception() -> None:
    class Swallow:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *exc: object) -> bool:
            return True

    class Throttler:
        def execute_task(self, limit_id: str) -> Swallow:
            return Swallow()

    with pytest.raises(AssertionError):
        assert_conforms_throttler(lambda _l, _c, _s: Throttler())


def test_plugin_fixtures(manual_clock: ManualClock, conformance_suites: SimpleNamespace) -> None:
    assert manual_clock() == 0.0
    assert conformance_suites.backoff is assert_conforms_backoff
    assert conformance_suites.rest_limiter is assert_conforms_rest_limiter
    assert conformance_suites.pausable is assert_conforms_pausable
    assert conformance_suites.throttler is assert_conforms_throttler


def test_manual_clock() -> None:
    clock = ManualClock(5.0)
    assert clock() == 5.0
    clock.advance(2)
    assert clock() == 7.0


async def test_manual_clock_sleep_records_and_advances() -> None:
    clock = ManualClock()
    seen: list[float] = []
    clock.on_sleep = seen.append
    await clock.sleep(2.5)
    assert (clock(), clock.sleeps, seen) == (2.5, [2.5], [2.5])


def test_default_pct_decimal_is_exact() -> None:
    assert RateLimiter().limits_pct == Decimal(1)
