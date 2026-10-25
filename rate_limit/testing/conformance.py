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
from typing import TYPE_CHECKING, Any, Protocol

from rate_limit.backoff import BackoffPolicy
from rate_limit.errors import RateLimitPoolConflictError, RateLimitWeightError
from rate_limit.protocols import (
    PausableRateLimiterProtocol,
    PoolPauseProtocol,
    RestRateLimiterProtocol,
    ThrottlerProtocol,
    WeightedThrottlerProtocol,
)
from rate_limit.testing.clock import BlockingSleep, ManualClock
from rate_limit.types import Clock, LinkedLimitWeightPair, RateLimit, Sleep

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence

    from rate_limit.backoff import RateLimitVerdict
    from rate_limit.types import PoolKind, RateLimitLike

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


# ---- opt-in suites (v0.2 features; not part of the required protocols) --------------
# Run these only if your limiter claims the feature.


# ---- token bucket -------------------------------------------------------------------


def assert_conforms_token_bucket(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], ThrottlerProtocol],
) -> None:
    """``RateLimit(kind="token_bucket")``: full burst, then ``cost / refill`` per deficit.

    The factory receives limits whose ``kind`` is ``"token_bucket"`` and must build a limiter
    with no safety margin (refill is exactly ``limit / time_interval``).
    """
    _run(_bucket_burst_and_deficit(make))
    _run(_bucket_fractional_refill(make))
    _run(_bucket_capacity_cap(make))


_BUCKET_CAPACITY = 4
_BUCKET_REFILL = 2.0  # tokens per second: 4 per 2.0s


def _bucket(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], ThrottlerProtocol],
) -> tuple[ManualClock, ThrottlerProtocol]:
    clock = ManualClock(_START)
    limit = RateLimit("a", _BUCKET_CAPACITY, _BUCKET_CAPACITY / _BUCKET_REFILL, kind="token_bucket")
    return clock, make([limit], clock, clock.sleep)


async def _drain(throttler: ThrottlerProtocol, count: int = _BUCKET_CAPACITY) -> None:
    for _ in range(count):
        await _enter(throttler, "a")


def _single_sleep(clock: ManualClock, expected: float, what: str) -> None:
    _require(len(clock.sleeps) == 1, f"{what}: expected one sleep, got {clock.sleeps}")
    _require(_close(clock.sleeps[0], expected), f"{what}: must wait {expected}: {clock.sleeps}")


async def _bucket_burst_and_deficit(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], ThrottlerProtocol],
) -> None:
    clock, throttler = _bucket(make)
    await _drain(throttler)
    _require(clock.sleeps == [], "a full bucket admits a burst of `limit` without waiting")
    await _enter(throttler, "a")
    _single_sleep(clock, 1 / _BUCKET_REFILL, "the call after the burst")

    clock, throttler = _bucket(make)
    await _drain(throttler)
    await _enter(throttler, "a", weights={"a": 3})
    _single_sleep(clock, 3 / _BUCKET_REFILL, "a weight-3 call on an empty bucket")


async def _bucket_fractional_refill(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], ThrottlerProtocol],
) -> None:
    clock, throttler = _bucket(make)
    await _drain(throttler)
    clock.advance(0.25)  # half a token back: the deficit is the other half
    await _enter(throttler, "a")
    _single_sleep(clock, 0.25, "a half-refilled bucket")


async def _bucket_capacity_cap(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], ThrottlerProtocol],
) -> None:
    clock, throttler = _bucket(make)
    await _enter(throttler, "a")
    clock.advance(1000.0)
    await _drain(throttler)
    _require(clock.sleeps == [], "an idle bucket refills to capacity")
    await _enter(throttler, "a")
    _single_sleep(clock, 1 / _BUCKET_REFILL, "idle time must not bank more than capacity")


# ---- unpause ------------------------------------------------------------------------


class UnpausableLimiter(Protocol):
    """Opt-in surface for :func:`assert_conforms_unpausable` (``RateLimiter`` satisfies it)."""

    async def acquire(self, limit_id: str) -> None: ...

    def pause(self, limit_id: str, seconds: float) -> None: ...

    def pause_all(self, seconds: float) -> None: ...

    def unpause(self, limit_id: str) -> None: ...

    def unpause_all(self) -> None: ...

    def is_paused(self, limit_id: str) -> bool: ...


def assert_conforms_unpausable(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], UnpausableLimiter],
) -> None:
    """``unpause`` / ``unpause_all``: sleeping acquirers wake without the clock advancing.

    The factory's ``sleep`` is a :class:`BlockingSleep`, so a woken acquirer is the only way
    out of a pause. The limiter must wake acquirers that are sleeping out a pause.
    """
    _run(_unpause_wakes_sleeper(make))
    _run(_unpause_keeps_pause_all(make))
    _run(_unpause_all_lifts_everything(make))
    _run(_unpause_keeps_spent_capacity(make))


async def _yield_until(condition: Callable[[], bool], what: str) -> None:
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"timed out waiting for {what}")


async def _finishes(task: asyncio.Task[None], what: str) -> None:
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except TimeoutError:
        raise AssertionError(f"{what}: the acquirer must wake and finish") from None


async def _abandon(task: asyncio.Task[None]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _unpause_wakes_sleeper(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], UnpausableLimiter],
) -> None:
    clock, sleeper = ManualClock(_START), BlockingSleep()
    limiter = make([RateLimit("a", 5, 1.0)], clock, sleeper)
    limiter.pause("a", 7200.0)
    task = asyncio.create_task(limiter.acquire("a"))
    try:
        await asyncio.wait_for(sleeper.started.wait(), timeout=1.0)
        _require(not task.done(), "an acquirer must wait out a pause")
        limiter.unpause("a")
        await _finishes(task, "unpause")
    finally:
        await _abandon(task)
    _require(clock() == _START, "unpause must not need the clock to advance")
    _require(_close(sleeper.durations[0], 7200.0), f"slept the pause: {sleeper.durations}")
    _require(not limiter.is_paused("a"), "unpause must lift the pause")


async def _unpause_keeps_pause_all(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], UnpausableLimiter],
) -> None:
    clock, sleeper = ManualClock(_START), BlockingSleep()
    limiter = make([RateLimit("a", 5, 1.0)], clock, sleeper)
    limiter.pause("a", 7200.0)
    limiter.pause_all(100.0)
    task = asyncio.create_task(limiter.acquire("a"))
    try:
        await asyncio.wait_for(sleeper.started.wait(), timeout=1.0)
        limiter.unpause("a")
        await _yield_until(lambda: len(sleeper.durations) >= 2, "a second sleep")
        _require(not task.done(), "unpause(id) must leave pause_all in force")
        _require(limiter.is_paused("a"), "pause_all must still block the id")
        _require(_close(sleeper.durations[1], 100.0), f"got {sleeper.durations}")
        limiter.unpause_all()
        await _finishes(task, "unpause_all")
    finally:
        await _abandon(task)


async def _unpause_all_lifts_everything(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], UnpausableLimiter],
) -> None:
    clock, sleeper = ManualClock(_START), BlockingSleep()
    limiter = make([RateLimit("a", 5, 1.0), RateLimit("b", 5, 1.0)], clock, sleeper)
    limiter.pause_all(7200.0)
    limiter.pause("b", 9000.0)
    task = asyncio.create_task(limiter.acquire("b"))
    try:
        await asyncio.wait_for(sleeper.started.wait(), timeout=1.0)
        limiter.unpause_all()
        await _finishes(task, "unpause_all")
    finally:
        await _abandon(task)
    _require(not limiter.is_paused("a") and not limiter.is_paused("b"), "both pauses lifted")
    _require(clock() == _START, "unpause_all must not need the clock to advance")


async def _unpause_keeps_spent_capacity(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], UnpausableLimiter],
) -> None:
    clock, sleeper = ManualClock(_START), BlockingSleep()
    limiter = make([RateLimit("a", 1, 10.0)], clock, sleeper)
    await limiter.acquire("a")  # spends the only slot
    limiter.pause("a", 7200.0)
    task = asyncio.create_task(limiter.acquire("a"))
    try:
        await asyncio.wait_for(sleeper.started.wait(), timeout=1.0)
        limiter.unpause("a")
        await _yield_until(lambda: len(sleeper.durations) >= 2, "the window sleep")
        _require(not task.done(), "unpause must not refund capacity already spent")
        _require(_close(sleeper.durations[1], 10.0), f"must wait the window: {sleeper.durations}")
    finally:
        await _abandon(task)


class BoundUnpausableLimiter(Protocol):
    """Opt-in surface for :func:`assert_conforms_bound_unpausable` (``BoundLimiter`` fits)."""

    async def acquire(self) -> None: ...

    def pause(self, seconds: float) -> None: ...

    def unpause(self) -> None: ...

    def unpause_all(self) -> None: ...

    def is_paused(self) -> bool: ...


class _UnboundUnpausable:
    """Id-taking view of a bound limiter, so the unpause scenarios run unchanged."""

    def __init__(self, bound: BoundUnpausableLimiter) -> None:
        self._bound = bound

    async def acquire(self, limit_id: str) -> None:
        await self._bound.acquire()

    def pause(self, limit_id: str, seconds: float) -> None:
        self._bound.pause(seconds)

    def pause_all(self, seconds: float) -> None:
        raise NotImplementedError("a bound limiter has no pause_all")

    def unpause(self, limit_id: str) -> None:
        self._bound.unpause()

    def unpause_all(self) -> None:
        self._bound.unpause_all()

    def is_paused(self, limit_id: str) -> bool:
        return self._bound.is_paused()


def assert_conforms_bound_unpausable(
    make: Callable[[RateLimit, Clock, Sleep], BoundUnpausableLimiter],
) -> None:
    """:func:`assert_conforms_unpausable` for a limiter bound to one limit id.

    ``make(limit, clock, sleep)`` returns the bound limiter for ``limit``. Only the scenarios
    expressible without ``pause_all`` or a second limit id are ported: an unpause wakes a
    sleeping acquirer, and an unpause keeps spent capacity. A bound-only scenario covers
    ``unpause_all``: it lifts a pause set through the bound object and wakes its sleeper.
    """

    def unbound(limits: Sequence[RateLimit], clock: Clock, sleep: Sleep) -> UnpausableLimiter:
        return _UnboundUnpausable(make(limits[0], clock, sleep))

    _run(_unpause_wakes_sleeper(unbound))
    _run(_unpause_keeps_spent_capacity(unbound))
    _run(_bound_unpause_all_wakes_sleeper(make))


async def _bound_unpause_all_wakes_sleeper(
    make: Callable[[RateLimit, Clock, Sleep], BoundUnpausableLimiter],
) -> None:
    clock, sleeper = ManualClock(_START), BlockingSleep()
    bound = make(RateLimit("a", 5, 1.0), clock, sleeper)
    bound.pause(7200.0)
    task = asyncio.create_task(bound.acquire())
    try:
        await asyncio.wait_for(sleeper.started.wait(), timeout=1.0)
        _require(not task.done(), "an acquirer must wait out a pause")
        bound.unpause_all()
        await _finishes(task, "unpause_all")
    finally:
        await _abandon(task)
    _require(clock() == _START, "unpause_all must not need the clock to advance")
    _require(_close(sleeper.durations[0], 7200.0), f"slept the pause: {sleeper.durations}")
    _require(not bound.is_paused(), "unpause_all must lift the pause")


# ---- Retry-After cap ----------------------------------------------------------------


class RetryAfterCapLimiter(Protocol):
    """Opt-in surface for :func:`assert_conforms_retry_after_cap` (``RateLimiter`` satisfies it)."""

    def report_throttled(
        self, limit_id: str, retry_after: float | None = None, *, status: int | None = None
    ) -> RateLimitVerdict: ...

    def paused_until(self, limit_id: str) -> float: ...


def assert_conforms_retry_after_cap(
    make: Callable[[float, Clock, Callable[[], BackoffPolicy]], RetryAfterCapLimiter],
) -> None:
    """``max_retry_after_seconds``: a server ``Retry-After`` is clamped, derived delays are not.

    ``make(cap, clock, backoff_factory)`` builds a limiter whose cap is ``cap`` seconds.
    """
    _cap_clamps(make)
    _cap_honours_lower_values(make)
    _cap_spares_backoff_delays(make)


_CAP = 60.0


def _cap_clamps(
    make: Callable[[float, Clock, Callable[[], BackoffPolicy]], RetryAfterCapLimiter],
) -> None:
    clock = ManualClock(_START)
    limiter = make(_CAP, clock, BackoffPolicy)
    verdict = limiter.report_throttled("a", 7200.0)
    _require(_close(verdict.delay, _CAP), f"a Retry-After above the cap pauses the cap: {verdict}")
    _require(
        verdict.retry_after is not None and _close(verdict.retry_after, _CAP),
        f"the verdict must report the capped Retry-After: {verdict}",
    )
    _require(_close(limiter.paused_until("a") - clock(), _CAP), "the pause itself is the cap")


def _cap_honours_lower_values(
    make: Callable[[float, Clock, Callable[[], BackoffPolicy]], RetryAfterCapLimiter],
) -> None:
    for retry_after in (_CAP / 2, _CAP):
        clock = ManualClock(_START)
        limiter = make(_CAP, clock, BackoffPolicy)
        verdict = limiter.report_throttled("a", retry_after)
        _require(_close(verdict.delay, retry_after), f"{retry_after} is within the cap: {verdict}")
        _require(verdict.retry_after == retry_after, f"retry_after must be as sent: {verdict}")


def _cap_spares_backoff_delays(
    make: Callable[[float, Clock, Callable[[], BackoffPolicy]], RetryAfterCapLimiter],
) -> None:
    clock = ManualClock(_START)
    limiter = make(1.0, clock, lambda: BackoffPolicy.fixed(30.0))
    verdict = limiter.report_throttled("a")
    _require(_close(verdict.delay, 30.0), f"a backoff-derived delay is never capped: {verdict}")
    _require(verdict.retry_after is None, f"no Retry-After was sent: {verdict}")


class BoundRetryAfterCapLimiter(Protocol):
    """Opt-in surface for :func:`assert_conforms_bound_retry_after_cap`."""

    def report_throttled(
        self, retry_after: float | None = None, *, status: int | None = None
    ) -> RateLimitVerdict: ...

    def paused_until(self) -> float: ...


class _UnboundRetryAfterCap:
    """Id-taking view of a bound limiter, so the cap scenarios run unchanged."""

    def __init__(self, bound: BoundRetryAfterCapLimiter) -> None:
        self._bound = bound

    def report_throttled(
        self, limit_id: str, retry_after: float | None = None, *, status: int | None = None
    ) -> RateLimitVerdict:
        return self._bound.report_throttled(retry_after, status=status)

    def paused_until(self, limit_id: str) -> float:
        return self._bound.paused_until()


def assert_conforms_bound_retry_after_cap(
    make: Callable[[float, Clock, Callable[[], BackoffPolicy]], BoundRetryAfterCapLimiter],
) -> None:
    """:func:`assert_conforms_retry_after_cap` for a limiter bound to one limit id.

    ``make(cap, clock, backoff_factory)`` returns the bound limiter; every scenario is ported.
    """

    def unbound(
        cap: float, clock: Clock, backoff: Callable[[], BackoffPolicy]
    ) -> RetryAfterCapLimiter:
        return _UnboundRetryAfterCap(make(cap, clock, backoff))

    _cap_clamps(unbound)
    _cap_honours_lower_values(unbound)
    _cap_spares_backoff_delays(unbound)


# ---- kind change --------------------------------------------------------------------


class KindChangeLimiter(Protocol):
    """Opt-in surface for :func:`assert_conforms_kind_change` (``RateLimiter`` satisfies it)."""

    async def acquire(self, limit_id: str) -> None: ...

    def set_limits(self, limits: Iterable[RateLimitLike]) -> None: ...


def assert_conforms_kind_change(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], KindChangeLimiter],
) -> None:
    """``set_limits`` keeps recorded usage across a change of ``RateLimit.kind``.

    The factory must build a limiter with no safety margin. Window to bucket becomes a deficit
    equal to the window's usage; bucket to window becomes one charge of ``ceil(used)`` stamped
    at the switch. Retained ids of the same kind keep usage; a removed id starts fresh.
    """
    _run(_kind_window_to_bucket(make))
    _run(_kind_bucket_to_window(make))
    _run(_kind_same_kind_retains(make))
    _run(_kind_removed_id_starts_fresh(make))


_KIND_LIMIT = 4
_KIND_INTERVAL = 2.0  # a bucket of 4 per 2.0s refills 2 tokens per second


def _kind_limit(kind: PoolKind, limit_id: str = "a") -> RateLimit:
    return RateLimit(limit_id, _KIND_LIMIT, _KIND_INTERVAL, kind=kind)


async def _spend(limiter: KindChangeLimiter, count: int) -> None:
    for _ in range(count):
        await limiter.acquire("a")


async def _kind_window_to_bucket(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], KindChangeLimiter],
) -> None:
    clock = ManualClock(_START)
    limiter = make([_kind_limit("window")], clock, clock.sleep)
    await _spend(limiter, _KIND_LIMIT)
    limiter.set_limits([_kind_limit("token_bucket")])
    await limiter.acquire("a")
    _single_sleep(clock, 0.5, "a bucket seeded with the window's usage refills one token")

    clock = ManualClock(_START)
    limiter = make([_kind_limit("window")], clock, clock.sleep)
    await _spend(limiter, 3)
    limiter.set_limits([_kind_limit("token_bucket")])
    await limiter.acquire("a")
    _require(clock.sleeps == [], "the one token left after spending 3 of 4 must be free")
    await limiter.acquire("a")
    _single_sleep(clock, 0.5, "the call past the carried deficit")


async def _kind_bucket_to_window(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], KindChangeLimiter],
) -> None:
    clock = ManualClock(_START)
    limiter = make([_kind_limit("token_bucket")], clock, clock.sleep)
    await _spend(limiter, 3)
    clock.advance(0.25)  # used is now 2.5 tokens: the window is charged ceil(2.5) == 3
    limiter.set_limits([_kind_limit("window")])
    await limiter.acquire("a")
    _require(clock.sleeps == [], "limit - ceil(used) == 1 call must still be free")
    await limiter.acquire("a")
    _single_sleep(clock, _KIND_INTERVAL, "the carried charge lasts a full window from the switch")


async def _kind_same_kind_retains(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], KindChangeLimiter],
) -> None:
    cases: list[tuple[PoolKind, float]] = [("window", _KIND_INTERVAL), ("token_bucket", 0.5)]
    for kind, expected in cases:
        clock = ManualClock(_START)
        limiter = make([_kind_limit(kind)], clock, clock.sleep)
        await _spend(limiter, _KIND_LIMIT)
        limiter.set_limits([_kind_limit(kind)])
        await limiter.acquire("a")
        _single_sleep(clock, expected, f"a retained {kind} id keeps its usage")


async def _kind_removed_id_starts_fresh(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], KindChangeLimiter],
) -> None:
    clock = ManualClock(_START)
    limiter = make([_kind_limit("window")], clock, clock.sleep)
    await _spend(limiter, _KIND_LIMIT)
    limiter.set_limits([_kind_limit("window", "other")])
    limiter.set_limits([_kind_limit("window"), _kind_limit("window", "other")])
    await _spend(limiter, _KIND_LIMIT)
    _require(clock.sleeps == [], "an id removed then re-added must start with a fresh pool")


# ---- refunds ------------------------------------------------------------------------


class RefundableCharge(Protocol):
    """What :meth:`RefundableLimiter.reserve` returns (``Charge`` satisfies it)."""

    def refund(self) -> None: ...


class RefundableLimiter(Protocol):
    """Opt-in surface for :func:`assert_conforms_refundable` (``RateLimiter`` satisfies it)."""

    async def reserve(
        self, limit_id: str, weights: Mapping[str, int] | None = None
    ) -> RefundableCharge: ...

    def try_acquire(self, limit_id: str, weights: Mapping[str, int] | None = None) -> float: ...


def assert_conforms_refundable(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], RefundableLimiter],
) -> None:
    """``reserve()`` returns a charge whose ``refund()`` hands its capacity straight back.

    The factory must build a limiter with no safety margin; ``try_acquire`` returns 0.0 and
    records when a call fits, else the wait. A refund frees capacity without the clock
    advancing (window and token bucket), is idempotent, frees only its own charge, gives nothing
    back once the charge has expired (a window aged out, a bucket refilled ``cost`` since), and
    reaches every linked pool.
    """
    _run(_refund_restores_capacity(make))
    _run(_refund_is_idempotent(make))
    _run(_refund_leaves_other_charges(make))
    _run(_refund_after_expiry_is_a_noop(make))
    _run(_refund_covers_linked_pools(make))


_REFUND_INTERVAL = 10.0
_REFUND_KINDS: tuple[PoolKind, ...] = ("window", "token_bucket")


def _refund_limit(kind: PoolKind = "window") -> RateLimit:
    return RateLimit("a", 2, _REFUND_INTERVAL, kind=kind)


def _full(limiter: RefundableLimiter, what: str) -> None:
    _require(limiter.try_acquire("a") > 0, f"{what}: the pool must be full")


async def _refund_restores_capacity(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], RefundableLimiter],
) -> None:
    for kind in _REFUND_KINDS:
        clock = ManualClock(_START)
        limiter = make([_refund_limit(kind)], clock, clock.sleep)
        charge = await limiter.reserve("a")
        await limiter.reserve("a")
        _full(limiter, f"{kind} after two charges")
        charge.refund()
        _require(limiter.try_acquire("a") == 0.0, f"a {kind} refund must free its charge at once")
        _full(limiter, f"a {kind} refund frees its own charge, no more")
        _require(clock.sleeps == [], f"{kind}: nothing here may sleep: {clock.sleeps}")


async def _refund_is_idempotent(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], RefundableLimiter],
) -> None:
    for kind in _REFUND_KINDS:
        clock = ManualClock(_START)
        limiter = make([_refund_limit(kind)], clock, clock.sleep)
        charge = await limiter.reserve("a")
        await limiter.reserve("a")
        charge.refund()
        charge.refund()
        _require(limiter.try_acquire("a") == 0.0, f"a {kind} refund must free its charge")
        _full(limiter, f"a second {kind} refund must not credit again")


async def _refund_leaves_other_charges(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], RefundableLimiter],
) -> None:
    clock = ManualClock(_START)
    limiter = make([_refund_limit()], clock, clock.sleep)
    first = await limiter.reserve("a")
    clock.advance(1.0)
    await limiter.reserve("a")
    first.refund()
    _require(limiter.try_acquire("a") == 0.0, "refunding the first charge frees one slot")
    wait = limiter.try_acquire("a")
    _require(_close(wait, _REFUND_INTERVAL), f"the second charge must still count: wait {wait}")


async def _refund_after_expiry_is_a_noop(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], RefundableLimiter],
) -> None:
    for kind in _REFUND_KINDS:  # older than the window, and than cost / refill for a bucket
        clock = ManualClock(_START)
        limiter = make([_refund_limit(kind)], clock, clock.sleep)
        stale = await limiter.reserve("a")
        clock.advance(_REFUND_INTERVAL * 1.001)
        await limiter.reserve("a")
        await limiter.reserve("a")
        stale.refund()
        _full(limiter, f"refunding an expired {kind} charge must not add headroom")


async def _refund_covers_linked_pools(
    make: Callable[[Sequence[RateLimit], Clock, Sleep], RefundableLimiter],
) -> None:
    clock = ManualClock(_START)
    link = [LinkedLimitWeightPair("shared", 1)]
    limiter = make(
        [
            RateLimit("a", 1, _REFUND_INTERVAL, linked_limits=link),
            RateLimit("b", 1, _REFUND_INTERVAL, linked_limits=link),
            RateLimit("shared", 1, _REFUND_INTERVAL),
        ],
        clock,
        clock.sleep,
    )
    charge = await limiter.reserve("a")
    _require(limiter.try_acquire("b") > 0, "the linked pool must be charged")
    charge.refund()
    _require(
        limiter.try_acquire("a") == 0.0, "the refund must reach the primary and the linked pool"
    )
