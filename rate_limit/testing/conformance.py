"""Conformance suites that consumers run against their own limiter / throttler.

Every suite takes a factory and drives it with a :class:`ManualClock`, so runs are
deterministic with no real sleeps. The suites are synchronous (they use ``asyncio.run``):
call them from plain test functions, not from inside a running event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import math
from typing import TYPE_CHECKING, Any

from rate_limit.backoff import BackoffPolicy
from rate_limit.errors import RateLimitPoolConflictError, RateLimitWeightError
from rate_limit.protocols import (
    PausableRateLimiterProtocol,
    PoolPauseProtocol,
    RestRateLimiterProtocol,
    ThrottlerProtocol,
    WeightedThrottlerProtocol,
)
from rate_limit.testing.clock import ManualClock
from rate_limit.types import Clock, LinkedLimitWeightPair, RateLimit, Sleep

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

_START = 1000.0
_EPS = 1e-9


class _BoomError(Exception):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=_EPS)


def _run(coro: Awaitable[None]) -> None:
    async def runner() -> None:
        await coro

    asyncio.run(runner())


# ---- REST limiter -------------------------------------------------------------------


def assert_conforms_rest_limiter(
    make: Callable[[Clock, Sleep], RestRateLimiterProtocol], *, limit: int, interval: float
) -> None:
    """Window accounting: ``limit`` per ``interval`` seconds, inclusive boundary."""
    _run(_window(make, limit, interval))
    _run(_boundary(make, limit, interval))
    _run(_concurrent(make, limit, interval))
    _run(_cancelled_acquire_records_nothing(make, limit, interval))


async def _fill(limiter: RestRateLimiterProtocol, count: int) -> None:
    for _ in range(count):
        await limiter.acquire()


async def _window(
    make: Callable[[Clock, Sleep], RestRateLimiterProtocol], limit: int, interval: float
) -> None:
    clock = ManualClock(_START)
    limiter = make(clock, clock.sleep)
    await _fill(limiter, limit)
    _require(clock.sleeps == [], "free capacity must not sleep")
    await limiter.acquire()
    _require(len(clock.sleeps) == 1, f"call limit+1 must sleep once, got {clock.sleeps}")
    _require(_close(clock.sleeps[0], interval), f"must wait exactly {interval}: {clock.sleeps}")


async def _boundary(
    make: Callable[[Clock, Sleep], RestRateLimiterProtocol], limit: int, interval: float
) -> None:
    clock = ManualClock(_START)
    limiter = make(clock, clock.sleep)
    await _fill(limiter, limit)
    clock.advance(interval)
    await limiter.acquire()
    _require(len(clock.sleeps) > 0, "an entry at age == window must still count (inclusive)")

    clock = ManualClock(_START)
    limiter = make(clock, clock.sleep)
    await _fill(limiter, limit)
    clock.advance(interval * 1.001)
    await limiter.acquire()
    _require(clock.sleeps == [], "an entry older than the window must not count")


async def _concurrent(
    make: Callable[[Clock, Sleep], RestRateLimiterProtocol], limit: int, interval: float
) -> None:
    clock = ManualClock(_START)
    limiter = make(clock, clock.sleep)
    total = limit * 3
    stamps: list[float] = []

    async def one() -> None:
        await limiter.acquire()
        stamps.append(clock())

    await asyncio.gather(*(one() for _ in range(total)))
    stamps.sort()
    for i in range(total - limit):
        _require(
            stamps[i + limit] - stamps[i] >= interval - _EPS,
            f"more than {limit} acquisitions inside one window: {stamps}",
        )
    bound = math.ceil(total / limit) * interval
    _require(clock() - _START <= bound + _EPS, f"waiters took {clock() - _START} > {bound}")


async def _cancelled_acquire_records_nothing(
    make: Callable[[Clock, Sleep], RestRateLimiterProtocol], limit: int, interval: float
) -> None:
    clock = ManualClock(_START)
    started = asyncio.Event()

    async def blocked_sleep(seconds: float) -> None:
        started.set()
        await asyncio.Event().wait()

    limiter = make(clock, blocked_sleep)
    await _fill(limiter, limit)
    task = asyncio.create_task(limiter.acquire())
    await started.wait()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    clock.advance(interval * 1.001)
    try:
        await asyncio.wait_for(_fill(limiter, limit), timeout=1.0)
    except TimeoutError:
        raise AssertionError("a cancelled acquire must not record a charge") from None


# ---- pausable -----------------------------------------------------------------------


def assert_conforms_pausable(make: Callable[[Clock, Sleep], PausableRateLimiterProtocol]) -> None:
    """Pause semantics shared by every pausable limiter."""
    _run(_pause_blocks(make))
    _run(_pause_never_shortens(make))
    _run(_pause_extended_while_waiting(make))


async def _pause_blocks(make: Callable[[Clock, Sleep], PausableRateLimiterProtocol]) -> None:
    clock = ManualClock(_START)
    limiter = make(clock, clock.sleep)
    _require(isinstance(limiter, PausableRateLimiterProtocol), "must be pausable")
    limiter.pause(5.0)
    if isinstance(limiter, PoolPauseProtocol):
        _require(limiter.is_paused(), "is_paused must be True right after pause()")
        _require(_close(limiter.blocked_for(), 5.0), "blocked_for must report the remainder")
    await limiter.acquire()
    _require(len(clock.sleeps) == 1 and _close(clock.sleeps[0], 5.0), f"got {clock.sleeps}")
    if isinstance(limiter, PoolPauseProtocol):
        _require(not limiter.is_paused(), "pause must have elapsed")

    clock = ManualClock(_START)
    limiter = make(clock, clock.sleep)
    limiter.pause(2.0)
    clock.advance(3.0)
    await limiter.acquire()
    _require(clock.sleeps == [], "an elapsed pause must not sleep")


async def _pause_never_shortens(
    make: Callable[[Clock, Sleep], PausableRateLimiterProtocol],
) -> None:
    clock = ManualClock(_START)
    limiter = make(clock, clock.sleep)
    limiter.pause(10.0)
    limiter.pause(1.0)
    await limiter.acquire()
    _require(_close(sum(clock.sleeps), 10.0), f"a shorter pause must not win: {clock.sleeps}")


async def _pause_extended_while_waiting(
    make: Callable[[Clock, Sleep], PausableRateLimiterProtocol],
) -> None:
    clock = ManualClock(_START)
    holder: list[PausableRateLimiterProtocol] = []
    extended = False

    async def sleep(seconds: float) -> None:
        nonlocal extended
        await clock.sleep(seconds)
        if not extended:
            extended = True
            holder[0].pause(3.0)  # another consumer got a 429 during our wait

    limiter = make(clock, sleep)
    holder.append(limiter)
    limiter.pause(2.0)
    await limiter.acquire()
    _require(
        _close(sum(clock.sleeps), 5.0), f"a pause set meanwhile must be honoured: {clock.sleeps}"
    )


# ---- throttler ----------------------------------------------------------------------


def assert_conforms_throttler(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], ThrottlerProtocol],
    *,
    weighted: bool = False,
    registry: bool = False,
) -> None:
    """``execute_task`` contract, linked pools, and optionally weights and the registry."""
    _run(_throttler_basics(make))
    _run(_throttler_linked(make))
    if weighted:
        _run(_throttler_weights(make))
    if registry:
        _run(_throttler_registry(make))


async def _enter(throttler: ThrottlerProtocol, limit_id: str, **kw: Any) -> None:
    async with throttler.execute_task(limit_id=limit_id, **kw):
        pass


def _waited(clock: ManualClock) -> bool:
    return clock() - _START >= 1.0 - _EPS


async def _throttler_basics(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], ThrottlerProtocol],
) -> None:
    clock = ManualClock(_START)
    throttler = make([RateLimit("a", 2, 1.0)], clock, clock.sleep)
    _require(isinstance(throttler, ThrottlerProtocol), "must satisfy ThrottlerProtocol")
    _require(
        not inspect.iscoroutinefunction(throttler.execute_task), "execute_task must be non-async"
    )
    context = throttler.execute_task(limit_id="a")
    _require(
        hasattr(context, "__aenter__") and hasattr(context, "__aexit__"),
        "execute_task must return an async context manager",
    )
    async with context:
        pass
    await _enter(throttler, "a")
    _require(clock() == _START, "capacity free: must not wait")
    await _enter(throttler, "a")
    _require(_waited(clock), "entering past capacity must block until it frees")

    try:
        async with throttler.execute_task("a"):
            raise _BoomError
    except _BoomError:
        pass
    else:
        raise AssertionError("__aexit__ must not swallow the body's exception")


async def _throttler_linked(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], ThrottlerProtocol],
) -> None:
    clock = ManualClock(_START)
    link = [LinkedLimitWeightPair("shared", 1)]
    throttler = make(
        [
            RateLimit("a", 5, 1.0, linked_limits=link),
            RateLimit("b", 5, 1.0, linked_limits=link),
            RateLimit("shared", 1, 1.0),
        ],
        clock,
        clock.sleep,
    )
    await _enter(throttler, "a")
    _require(clock() == _START, "first call must pass")
    await _enter(throttler, "b")
    _require(_waited(clock), "a linked pool of 1 must gate two different primaries")


def _weighted_limits() -> list[RateLimit]:
    return [
        RateLimit(
            "a",
            100,
            1.0,
            linked_limits=[LinkedLimitWeightPair("pool", 1), LinkedLimitWeightPair("raw", 1)],
        ),
        RateLimit("pool", 10, 1.0),
        RateLimit("raw", 100, 1.0),
    ]


async def _expect_value_error(throttler: ThrottlerProtocol, **kw: Any) -> None:
    try:
        await _enter(throttler, "a", **kw)
    except ValueError as exc:
        _require(not isinstance(exc, RateLimitWeightError), f"wrong error type for {kw}")
    else:
        raise AssertionError(f"expected ValueError for {kw}")


async def _throttler_weights(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], ThrottlerProtocol],
) -> None:
    def fresh() -> tuple[ManualClock, ThrottlerProtocol]:
        clock = ManualClock(_START)
        return clock, make(_weighted_limits(), clock, clock.sleep)

    clock, throttler = fresh()  # weights=None charges each declared weight
    for _ in range(10):
        await _enter(throttler, "a")
    _require(clock() == _START, "10 x declared weight 1 fits the pool of 10")
    await _enter(throttler, "a")
    _require(_waited(clock), "the 11th declared-weight call must wait")

    clock, throttler = fresh()  # a linked override replaces that id's charge
    await _enter(throttler, "a", weights={"pool": 10})
    _require(clock() == _START, "pool override fits")
    await _enter(throttler, "a")
    _require(_waited(clock), "the pool is exhausted by the override")

    clock, throttler = fresh()  # a primary override
    await _enter(throttler, "a", weights={"a": 100})
    await _enter(throttler, "a")
    _require(_waited(clock), "the primary is exhausted by the override")

    clock, throttler = fresh()  # unlisted ids keep their declared weight
    await _enter(throttler, "a", weights={"pool": 5})
    await _enter(throttler, "a", weights={"pool": 5})
    _require(clock() == _START, "5 + 5 fits the pool")
    await _enter(throttler, "a")
    _require(_waited(clock), "the declared pool charge of 1 no longer fits")

    _, throttler = fresh()
    await _expect_value_error(throttler, weights={"nope": 1})
    await _expect_value_error(throttler, weights={"pool": 0})
    try:
        await _enter(throttler, "a", weights={"pool": 11})
    except RateLimitWeightError:
        pass
    else:
        raise AssertionError("a weight above the limit must raise RateLimitWeightError")


async def _throttler_registry(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], ThrottlerProtocol],
) -> None:
    clock = ManualClock(_START)
    throttler = make([RateLimit("a", 5, 1.0)], clock, clock.sleep)
    _require(isinstance(throttler, WeightedThrottlerProtocol), "must satisfy the registry API")
    assert isinstance(throttler, WeightedThrottlerProtocol)
    _require(throttler.has_limit("a") and not throttler.has_limit("zzz"), "has_limit")
    stored = throttler.get_limit("a")
    _require(stored is not None and stored.limit == 5, "get_limit must return the stored limit")
    _require(throttler.get_limit("zzz") is None, "get_limit of an unknown id is None")
    throttler.add_limits([RateLimit("a", 5, 1.0)], owner="x")  # identical: no-op
    throttler.add_limits([RateLimit("c", 3, 1.0)])
    _require(throttler.has_limit("c"), "a new limit must register")
    try:
        throttler.add_limits([RateLimit("a", 6, 1.0)], owner="y")
    except RateLimitPoolConflictError:
        pass
    else:
        raise AssertionError("a raw mismatch must raise RateLimitPoolConflictError")
    throttler.pause("c", 2.0)
    await _enter(throttler, "c")
    _require(clock() - _START >= 2.0 - _EPS, "a paused id must wait out the pause")


# ---- backoff ------------------------------------------------------------------------


def assert_conforms_backoff(policy_factory: Callable[[], BackoffPolicy]) -> None:
    """``BackoffPolicy.next_delay`` ordering and clamps."""
    policy = policy_factory()
    _require(policy.next_delay(429, 1.5) == 1.5, "Retry-After wins")
    _require(policy.next_delay(418, 0.0) == 0.0, "an explicit Retry-After of 0 wins over a ban")

    policy = policy_factory()
    _require(policy.next_delay(418, None) == policy.ban_seconds, "418 gives ban_seconds")

    policy = policy_factory()
    delays = [policy.next_delay(429, None) for _ in range(12)]
    _require(delays[0] == min(policy.cap_seconds, policy.base_seconds), "first 429 gives base")
    _require(
        all(b >= a for a, b in zip(delays, delays[1:], strict=False)), "delays must not shrink"
    )
    _require(delays[-1] == policy.cap_seconds, "429 doubles up to the cap")
    policy.reset()
    _require(policy.streak == 0, "reset clears the streak")

    policy = policy_factory()
    policy.streak = 5000
    _require(policy.next_delay(429, None) == policy.cap_seconds, "the exponent clamp holds at 5000")

    policy = policy_factory()
    _require(policy.next_delay(429, None, in_burst=True) == 0.0, "in_burst joins the pause")
    _require(policy.streak == 0, "in_burst must leave the streak unchanged")
    _require(policy.next_delay(429, 2.0, in_burst=True) == 2.0, "in_burst + Retry-After counts")
    policy = policy_factory()
    _require(
        policy.next_delay(418, None, in_burst=True) == policy.ban_seconds, "in_burst + 418 counts"
    )

    fixed = BackoffPolicy.fixed(63.0)
    _require(
        all(fixed.next_delay(s, None) == 63.0 for s in (429, 429, 418)), "fixed(s) is constant"
    )
