import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest

from rate_limit import RateLimitWeightError
from rate_limit.backoff import BackoffPolicy
from rate_limit.hb_compat.async_throttler import AsyncThrottler
from rate_limit.hb_compat.data_types import LinkedLimitWeightPair, RateLimit
from rate_limit.testing import ManualClock
from rate_limit.testing.conformance import assert_conforms_pausable

POLL = 0.005
SETTLE = 0.1
PAST_WINDOW = 10.0


async def tick(_seconds: float) -> None:
    """Poll quickly in real time without moving the injected clock."""
    await asyncio.sleep(0.001)


def build(
    clock: ManualClock, limits: list[RateLimit], sleep: Callable[[float], Any] = tick, **kwargs
) -> AsyncThrottler:
    return AsyncThrottler(
        rate_limits=limits, retry_interval=POLL, clock=clock, sleep=sleep, **kwargs
    )


async def passes(
    throttler: AsyncThrottler,
    limit_id: str,
    weights: dict[str, int] | None = None,
    timeout: float = SETTLE,
) -> bool:
    try:
        await asyncio.wait_for(throttler.execute_task(limit_id, weights=weights).acquire(), timeout)
    except TimeoutError:
        return False
    return True


async def spawn_acquirers(
    throttler: AsyncThrottler, limit_id: str, count: int
) -> list[asyncio.Task[None]]:
    tasks = [asyncio.create_task(throttler.execute_task(limit_id).acquire()) for _ in range(count)]
    await asyncio.sleep(SETTLE)
    return tasks


async def cancel_all(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def completed(tasks: list[asyncio.Task[None]]) -> int:
    return sum(1 for task in tasks if task.done() and not task.cancelled())


def single(limit: int, interval: float = 5.0) -> list[RateLimit]:
    return [RateLimit("solo", limit=limit, time_interval=interval)]


def pooled(pool_limit: int) -> list[RateLimit]:
    return [
        RateLimit("pool", limit=pool_limit, time_interval=5.0),
        RateLimit(
            "ep", limit=1000, time_interval=5.0, linked_limits=[LinkedLimitWeightPair("pool")]
        ),
    ]


async def test_flush_only_elapsed_tasks_are_flushed():
    clock = ManualClock()
    throttler = build(clock, single(2))

    assert await passes(throttler, "solo")
    clock.advance(3.0)
    assert await passes(throttler, "solo")
    assert not await passes(throttler, "solo")

    clock.advance(3.0)

    assert await passes(throttler, "solo")
    assert not await passes(throttler, "solo")


async def test_within_capacity_singular_non_weighted_task_returns_false():
    clock = ManualClock()
    throttler = build(clock, single(1))

    assert await passes(throttler, "solo")
    assert not await passes(throttler, "solo")

    clock.advance(PAST_WINDOW)

    assert await passes(throttler, "solo")


async def test_within_capacity_singular_non_weighted_task_returns_true():
    clock = ManualClock()
    throttler = build(clock, single(1), sleep=clock.sleep)

    await asyncio.wait_for(throttler.execute_task("solo").acquire(), SETTLE)

    assert clock.sleeps == []


async def test_within_capacity_pool_non_weighted_task_returns_false():
    clock = ManualClock()
    throttler = build(clock, pooled(1))

    assert await passes(throttler, "ep")
    assert not await passes(throttler, "ep")

    clock.advance(PAST_WINDOW)

    assert await passes(throttler, "ep")


async def test_within_capacity_pool_non_weighted_task_returns_true():
    clock = ManualClock()
    throttler = build(clock, pooled(1), sleep=clock.sleep)

    await asyncio.wait_for(throttler.execute_task("ep").acquire(), SETTLE)

    assert clock.sleeps == []


async def test_within_capacity_pool_weighted_tasks():
    limits = [
        RateLimit("pool", limit=10, time_interval=5.0),
        RateLimit("task1", 1000, 5.0, linked_limits=[LinkedLimitWeightPair("pool", 5)]),
        RateLimit("task2", 1000, 5.0, linked_limits=[LinkedLimitWeightPair("pool", 1)]),
    ]
    throttler = build(ManualClock(), limits)

    assert await passes(throttler, "task1")
    assert await passes(throttler, "task2")

    assert not await passes(throttler, "task1")
    assert await passes(throttler, "task2")


async def test_acquire_appends_to_task_logs():
    throttler = build(ManualClock(), single(1))

    assert await passes(throttler, "solo")

    assert not await passes(throttler, "solo")


async def test_concurrent_acquirers_cannot_overbook_a_limit():
    clock = ManualClock()
    throttler = build(clock, single(3))

    tasks = await spawn_acquirers(throttler, "solo", 10)

    try:
        assert completed(tasks) == 3
        assert sum(1 for task in tasks if not task.done()) == 7
    finally:
        await cancel_all(tasks)


async def test_within_capacity_returns_true_for_throttler_without_configured_limits():
    clock = ManualClock()
    throttler = build(clock, [], sleep=clock.sleep)

    async with throttler.execute_task("x"):
        pass

    assert clock.sleeps == []


@pytest.mark.parametrize(
    ("age", "admitted"), [(0.19, False), (0.2, False), (0.21, True)], ids=["inside", "edge", "past"]
)
async def test_within_capacity_for_limits_with_milliseconds_interval(age, admitted):
    clock = ManualClock()
    throttler = build(clock, single(2, interval=0.2), safety_margin_pct=0.0)
    assert await passes(throttler, "solo")
    assert await passes(throttler, "solo")

    clock.advance(age)

    assert await passes(throttler, "solo") is admitted


async def test_unknown_limit_id_warns_once_and_runs_unthrottled(caplog):
    throttler = build(ManualClock(), [])

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            assert await passes(throttler, "ghost")

    assert sum("ghost" in record.getMessage() for record in caplog.records) == 1


def weighted_limits(**overrides: int) -> list[RateLimit]:
    caps = {"pool": 100, "other": 100, "ep": 100} | overrides
    return [
        RateLimit("pool", limit=caps["pool"], time_interval=5.0),
        RateLimit("other", limit=caps["other"], time_interval=5.0),
        RateLimit(
            "ep",
            limit=caps["ep"],
            time_interval=5.0,
            weight=2,
            linked_limits=[LinkedLimitWeightPair("pool", 1), LinkedLimitWeightPair("other", 1)],
        ),
    ]


@pytest.mark.parametrize(
    ("tight_id", "cap", "admitted"),
    [("ep", 5, 2), ("pool", 2, 2), ("other", 3, 3)],
)
async def test_weights_none_keeps_declared_weights(tight_id, cap, admitted):
    throttler = build(ManualClock(), weighted_limits(**{tight_id: cap}))

    for _ in range(admitted):
        assert await passes(throttler, "ep")

    assert not await passes(throttler, "ep")


async def test_weights_primary_only_override():
    throttler = build(ManualClock(), weighted_limits(ep=10))

    assert await passes(throttler, "ep", weights={"ep": 7})
    assert not await passes(throttler, "ep", weights={"ep": 4})
    assert await passes(throttler, "ep", weights={"ep": 3})


async def test_weights_linked_only_override():
    throttler = build(ManualClock(), weighted_limits(pool=10))

    assert await passes(throttler, "ep", weights={"pool": 6})
    assert not await passes(throttler, "ep", weights={"pool": 5})
    assert await passes(throttler, "ep", weights={"pool": 4})


async def test_weights_capacity_check_and_log_agree():
    throttler = build(ManualClock(), weighted_limits(pool=10))

    assert await passes(throttler, "ep", weights={"pool": 6})
    assert await passes(throttler, "ep", weights={"pool": 4})

    assert not await passes(throttler, "ep")


async def test_weights_primary_override_is_used_by_capacity_check():
    throttler = build(ManualClock(), weighted_limits(ep=100))

    assert await passes(throttler, "ep", weights={"ep": 100})

    with pytest.raises(RateLimitWeightError):
        await passes(throttler, "ep", weights={"ep": 101})
    assert not await passes(throttler, "ep", weights={"ep": 1})


async def test_unknown_id_honours_per_id_pause_and_other_unknown_ids_do_not():
    clock = ManualClock()
    throttler = build(clock, [])

    throttler.pause("ghost", 5.0)

    assert not await passes(throttler, "ghost")
    assert await passes(throttler, "other_ghost")
    clock.advance(5.0)
    assert await passes(throttler, "ghost")


async def test_paused_acquirer_does_not_hold_the_lock_while_waiting():
    limits = [
        RateLimit("pool", limit=100, time_interval=60.0),
        RateLimit("paused", limit=100, time_interval=60.0),
    ]
    throttler = build(ManualClock(), limits)
    throttler.pause("paused", 5.0)

    tasks = await spawn_acquirers(throttler, "paused", 1)
    try:
        assert completed(tasks) == 0
        assert await passes(throttler, "pool")
    finally:
        await cancel_all(tasks)


async def test_concurrent_acquirers_behind_a_pause_cannot_overbook():
    clock = ManualClock()
    throttler = build(clock, [RateLimit("capped", limit=3, time_interval=60.0)])
    throttler.pause("capped", 5.0)

    tasks = await spawn_acquirers(throttler, "capped", 10)
    try:
        assert completed(tasks) == 0
        clock.advance(5.0)
        await asyncio.sleep(SETTLE)
        assert completed(tasks) == 3
    finally:
        await cancel_all(tasks)


class ThrottlerAsPausable:
    def __init__(self, throttler: AsyncThrottler) -> None:
        self._throttler = throttler

    def pause(self, seconds: float) -> None:
        self._throttler.pause("solo", seconds)

    async def acquire(self) -> None:
        await self._throttler.execute_task("solo").acquire()


def test_pausable_conformance_kit_passes():
    def make(clock, sleep):
        return ThrottlerAsPausable(
            AsyncThrottler(
                rate_limits=single(1000, interval=60.0),
                retry_interval=POLL,
                clock=clock,
                sleep=sleep,
            )
        )

    assert assert_conforms_pausable(make) == "pausable"


def test_backoff_conformance_kit_passes_for_the_throttlers_policy():
    clock = ManualClock()
    throttler = build(clock, single(1000, interval=60.0))
    policy = BackoffPolicy()
    expected = [policy.next_delay(429, None) for _ in range(12)]

    observed = []
    for _ in range(12):
        observed.append(throttler.report_response("solo", 429, None).delay)
        clock.advance(observed[-1])

    assert observed == expected
    assert observed[-1] == policy.cap_seconds
    assert throttler.report_response("solo", 429, {"Retry-After": "1.5"}).delay == 1.5
    assert throttler.report_response("solo", 418, None).delay == policy.ban_seconds


async def test_inclusive_window_boundary_mirrors_the_conformance_kit():
    at_edge = ManualClock(1000.0)
    on_time = ManualClock(1000.0)
    edge = build(at_edge, single(2, interval=5.0), sleep=at_edge.sleep, safety_margin_pct=0.0)
    past = build(on_time, single(2, interval=5.0), sleep=on_time.sleep, safety_margin_pct=0.0)
    for throttler in (edge, past):
        for _ in range(2):
            await throttler.execute_task("solo").acquire()

    at_edge.advance(5.0)
    on_time.advance(5.0 * 1.001)
    await asyncio.wait_for(edge.execute_task("solo").acquire(), SETTLE)
    await asyncio.wait_for(past.execute_task("solo").acquire(), SETTLE)

    assert at_edge.sleeps
    assert on_time.sleeps == []
