from collections.abc import Iterator

import pytest

from rate_limit import (
    BackoffPolicy,
    LinkedLimitWeightPair,
    RateLimit,
    RateLimitPoolConflictError,
    get_pool_limiter,
    reset_pool_limiters,
)
from rate_limit.testing import ManualClock


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    reset_pool_limiters()
    yield
    reset_pool_limiters()


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(1000.0)


async def test_same_key_shares_one_limiter(clock: ManualClock) -> None:
    first = get_pool_limiter("p", RateLimit("ignored", 1, 10), clock=clock, sleep=clock.sleep)
    second = get_pool_limiter("p", RateLimit("other-id", 1, 10), clock=clock, sleep=clock.sleep)
    assert first.limiter is second.limiter
    assert first.limit_id == "p"
    await first.acquire()
    await second.acquire()
    assert clock.sleeps == [pytest.approx(10.0)]


async def test_pause_on_one_sharer_blocks_the_other(clock: ManualClock) -> None:
    a = get_pool_limiter("p", RateLimit("x", 100, 1), clock=clock, sleep=clock.sleep)
    b = get_pool_limiter("p", RateLimit("x", 100, 1))
    a.pause(5)
    await b.acquire()
    assert clock.sleeps == [pytest.approx(5.0)]


def test_first_caller_wins_clock_and_backoff(clock: ManualClock) -> None:
    other = ManualClock(5.0)
    first = get_pool_limiter(
        "p",
        RateLimit("x", 1, 1),
        clock=clock,
        sleep=clock.sleep,
        backoff_factory=lambda: BackoffPolicy.fixed(7),
    )
    get_pool_limiter("p", RateLimit("x", 1, 1), clock=other, sleep=other.sleep)
    first.pause(1)
    assert first.limiter.paused_until("p") == pytest.approx(1001.0)
    assert first.backoff.next_delay(429, None) == 7


def test_pools_are_isolated_by_key(clock: ManualClock) -> None:
    a = get_pool_limiter("a", RateLimit("x", 1, 1), clock=clock, sleep=clock.sleep)
    b = get_pool_limiter("b", RateLimit("x", 1, 1), clock=clock, sleep=clock.sleep)
    a.limiter.pause_all(10)
    assert a.is_paused()
    assert not b.is_paused()


@pytest.mark.parametrize(
    "other",
    [
        RateLimit("x", 2, 10),
        RateLimit("x", 1, 11),
        RateLimit("x", 1, 10, weight=2),
        RateLimit("x", 1, 10, linked_limits=[LinkedLimitWeightPair("y")]),
    ],
)
def test_raw_mismatch_raises_and_names_owner(other: RateLimit) -> None:
    get_pool_limiter("p", RateLimit("x", 1, 10), owner="first")
    with pytest.raises(RateLimitPoolConflictError, match="first") as excinfo:
        get_pool_limiter("p", other, owner="second")
    assert excinfo.value.pool == "p"
    assert "second" in str(excinfo.value)


def test_linked_limits_rejected_at_creation() -> None:
    linked = RateLimit("x", 1, 10, linked_limits=[LinkedLimitWeightPair("y")])
    with pytest.raises(ValueError, match="single-limit"):
        get_pool_limiter("p", linked)


def test_identical_raw_readd_is_noop() -> None:
    first = get_pool_limiter("p", RateLimit("x", 1, 10), owner="first")
    again = get_pool_limiter("p", RateLimit("y", 1, 10), owner="second")
    assert first.limiter is again.limiter


def test_reset_drops_limiters() -> None:
    first = get_pool_limiter("p", RateLimit("x", 1, 10))
    reset_pool_limiters()
    assert get_pool_limiter("p", RateLimit("x", 2, 10)).limiter is not first.limiter
