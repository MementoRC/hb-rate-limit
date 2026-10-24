from collections.abc import Iterator
from datetime import UTC, datetime
from email.utils import format_datetime

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


def test_first_caller_wins_wall_clock(clock: ManualClock) -> None:
    wall = datetime(2030, 1, 1, tzinfo=UTC).timestamp()
    header = format_datetime(datetime.fromtimestamp(wall + 30, UTC), usegmt=True)
    first = get_pool_limiter(
        "p", RateLimit("x", 1, 1), clock=clock, sleep=clock.sleep, wall_clock=lambda: wall
    )
    get_pool_limiter("p", RateLimit("x", 1, 1), wall_clock=lambda: wall + 1000)
    verdict = first.limiter.report_response("p", 429, {"Retry-After": header})
    assert verdict.retry_after == pytest.approx(30.0)


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


def test_kind_mismatch_is_a_pool_conflict() -> None:
    get_pool_limiter("p", RateLimit("x", 1, 10), owner="first")
    with pytest.raises(RateLimitPoolConflictError):
        get_pool_limiter("p", RateLimit("x", 1, 10, kind="token_bucket"), owner="second")


async def test_bucket_pool_is_created_and_shared(clock: ManualClock) -> None:
    limit = RateLimit("x", 2, 2, kind="token_bucket")
    first = get_pool_limiter("p", limit, clock=clock, sleep=clock.sleep)
    second = get_pool_limiter("p", limit)
    assert first.limiter is second.limiter
    assert first.limiter.get_limit("p") == RateLimit("p", 2, 2, kind="token_bucket")
    await first.acquire()
    await second.acquire()
    assert first.blocked_for() == pytest.approx(1.0)


async def test_margin_is_passed_to_the_limiter(clock: ManualClock) -> None:
    pool = get_pool_limiter(
        "p", RateLimit("x", 1, 10), clock=clock, sleep=clock.sleep, safety_margin_pct=0.05
    )
    await pool.acquire()
    assert pool.blocked_for() == pytest.approx(10.5)


async def test_margin_defaults_to_zero(clock: ManualClock) -> None:
    pool = get_pool_limiter("p", RateLimit("x", 1, 10), clock=clock, sleep=clock.sleep)
    await pool.acquire()
    assert pool.blocked_for() == pytest.approx(10.0)


def test_margin_mismatch_is_a_pool_conflict() -> None:
    get_pool_limiter("p", RateLimit("x", 1, 10), safety_margin_pct=0.05, owner="first")
    with pytest.raises(RateLimitPoolConflictError, match="first") as excinfo:
        get_pool_limiter("p", RateLimit("x", 1, 10), owner="second")
    assert "safety_margin_pct 0.05" in str(excinfo.value)
    assert get_pool_limiter("p", RateLimit("x", 1, 10), safety_margin_pct=0.05)


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_invalid_margin_rejected(margin: float) -> None:
    with pytest.raises(ValueError, match="safety_margin_pct"):
        get_pool_limiter("p", RateLimit("x", 1, 10), safety_margin_pct=margin)


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
