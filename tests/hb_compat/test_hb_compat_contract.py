import asyncio
import copy
import dataclasses
import importlib
import logging
from decimal import Decimal

import pytest

from rate_limit import RateLimitWeightError
from rate_limit.hb_compat.async_throttler import AsyncThrottler
from rate_limit.hb_compat.data_types import LinkedLimitWeightPair, RateLimit, TaskLog
from rate_limit.testing import ManualClock

POLL = 0.005
SETTLE = 0.1
MODULES = (
    "rate_limit.hb_compat.data_types",
    "rate_limit.hb_compat.async_throttler_base",
    "rate_limit.hb_compat.async_request_context_base",
    "rate_limit.hb_compat.async_throttler",
)
PUBLIC_NAMES = {
    "MAX_CAPACITY_REACHED_WARNING_INTERVAL",
    "AsyncRequestContextBase",
    "AsyncRequestContext",
    "AsyncThrottler",
    "AsyncThrottlerBase",
    "DEFAULT_PATH",
    "DEFAULT_WEIGHT",
    "LinkedLimitWeightPair",
    "Limit",
    "RateLimit",
    "RequestPath",
    "RequestWeight",
    "Seconds",
    "TaskLog",
}


async def tick(_seconds: float) -> None:
    await asyncio.sleep(0.001)


def build(limits: list[RateLimit], clock: ManualClock | None = None, **kwargs) -> AsyncThrottler:
    return AsyncThrottler(
        rate_limits=limits, retry_interval=POLL, clock=clock or ManualClock(), sleep=tick, **kwargs
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


async def assert_admits_exactly(throttler: AsyncThrottler, limit_id: str, count: int) -> None:
    for _ in range(count):
        assert await passes(throttler, limit_id)
    assert not await passes(throttler, limit_id)


def warnings_in(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


def rl(limit_id: str, limit: float = 10, interval: float = 60.0, **kwargs) -> RateLimit:
    return RateLimit(limit_id, limit=limit, time_interval=interval, **kwargs)


def test_hb_compat_modules_import_and_define_the_public_names():
    modules = [importlib.import_module(name) for name in MODULES]
    defined = {name for module in modules for name in vars(module)}

    assert defined >= PUBLIC_NAMES


def test_package_reexports_the_public_names():
    package = importlib.import_module("rate_limit.hb_compat")

    assert all(hasattr(package, name) for name in PUBLIC_NAMES)


def test_data_type_constants_match_hummingbot():
    from rate_limit.hb_compat.data_types import DEFAULT_PATH, DEFAULT_WEIGHT

    assert DEFAULT_WEIGHT == 1
    assert DEFAULT_PATH == ""


def test_task_log_is_a_dataclass_with_timestamp_rate_limit_and_weight():
    assert dataclasses.is_dataclass(TaskLog)
    assert [field.name for field in dataclasses.fields(TaskLog)] == [
        "timestamp",
        "rate_limit",
        "weight",
    ]


@pytest.mark.parametrize(
    ("limit", "share", "stored"),
    [
        (10.9, Decimal(100), 10),
        (0, None, 1),
        (3, Decimal(10), 1),
        (10, Decimal(150), 15),
        (3.8, Decimal(80), 3),
    ],
)
async def test_legacy_scaling_is_floored_with_a_minimum_of_one(limit, share, stored):
    throttler = AsyncThrottler(
        rate_limits=[rl("x", limit)],
        retry_interval=POLL,
        limits_share_percentage=share,
        clock=ManualClock(),
        sleep=tick,
    )

    assert throttler.get_limit("x").limit == stored
    await assert_admits_exactly(throttler, "x", stored)


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_invalid_time_interval_is_rejected_by_the_constructor(interval):
    with pytest.raises(ValueError, match="bad_id"):
        AsyncThrottler(rate_limits=[rl("bad_id", interval=interval)])


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_invalid_time_interval_is_rejected_by_add_rate_limits(interval):
    throttler = AsyncThrottler(rate_limits=[])

    with pytest.raises(ValueError, match="bad_id"):
        throttler.add_rate_limits([rl("bad_id", interval=interval)])


async def test_weight_that_can_never_fit_raises_with_id_weight_and_limit():
    throttler = build([rl("api", 5)])

    with pytest.raises(RateLimitWeightError) as raised:
        async with throttler.execute_task("api", weights={"api": 6}):
            pass

    message = str(raised.value)
    assert "api" in message
    assert "6" in message
    assert "5" in message


async def test_full_limit_with_a_fitting_weight_blocks_instead_of_raising():
    throttler = build([rl("api", 5)])

    assert await passes(throttler, "api", weights={"api": 5})

    assert not await passes(throttler, "api", weights={"api": 1})


def duplicate_link_limits() -> list[RateLimit]:
    return [
        rl("p", 4),
        rl(
            "ep",
            1000,
            linked_limits=[LinkedLimitWeightPair("p", 1), LinkedLimitWeightPair("p", 1)],
        ),
    ]


async def test_duplicate_linked_ids_are_summed():
    throttler = build(duplicate_link_limits())

    await assert_admits_exactly(throttler, "ep", 2)


async def test_weights_override_replaces_the_summed_total_for_that_id():
    throttler = build(duplicate_link_limits())

    for _ in range(4):
        assert await passes(throttler, "ep", weights={"p": 1})

    assert not await passes(throttler, "ep", weights={"p": 1})


def test_readd_with_fewer_duplicate_links_is_a_mismatch_that_warns_once(caplog):
    throttler = AsyncThrottler(rate_limits=duplicate_link_limits())

    with caplog.at_level(logging.WARNING):
        throttler.add_rate_limits([rl("ep", 1000, linked_limits=[LinkedLimitWeightPair("p", 1)])])

    assert len(warnings_in(caplog)) == 1
    assert len(throttler.get_limit("ep").linked_limits) == 2


def test_readd_with_the_same_links_in_another_order_is_silent(caplog):
    links = [
        LinkedLimitWeightPair("p", 1),
        LinkedLimitWeightPair("q", 2),
        LinkedLimitWeightPair("p", 1),
    ]
    throttler = AsyncThrottler(rate_limits=[rl("ep", linked_limits=links)])

    with caplog.at_level(logging.WARNING):
        throttler.add_rate_limits([rl("ep", linked_limits=[links[1], links[0], links[2]])])

    assert warnings_in(caplog) == []


def test_candles_readd_of_request_weight_keeps_the_connector_limit_and_warns_once(caplog):
    throttler = AsyncThrottler(rate_limits=[RateLimit("REQUEST_WEIGHT", 2400, 60)])

    with caplog.at_level(logging.WARNING):
        throttler.add_rate_limits([RateLimit("REQUEST_WEIGHT", 1200, 60)])
        first = len(warnings_in(caplog))
        throttler.add_rate_limits([RateLimit("REQUEST_WEIGHT", 1200, 60)])

    assert throttler.get_limit("REQUEST_WEIGHT").limit == 2400
    assert first == 1
    assert len(warnings_in(caplog)) == 1


async def test_unknown_id_passes_repeatedly_without_blocking_and_warns_once(caplog):
    clock = ManualClock()
    throttler = AsyncThrottler(rate_limits=[], clock=clock, sleep=clock.sleep)

    with caplog.at_level(logging.WARNING):
        for _ in range(20):
            await asyncio.wait_for(throttler.execute_task("ghost").acquire(), SETTLE)

    assert clock.sleeps == []
    assert sum("ghost" in record.getMessage() for record in warnings_in(caplog)) == 1


async def test_unknown_id_is_warned_about_once_per_id(caplog):
    throttler = build([])

    with caplog.at_level(logging.WARNING):
        for limit_id in ("one", "two", "one", "two"):
            assert await passes(throttler, limit_id)

    assert len(warnings_in(caplog)) == 2


def test_paused_until_is_none_before_any_pause():
    assert build([rl("x")]).paused_until("x") is None


def test_pause_of_zero_seconds_reports_the_current_time_not_none():
    throttler = build([rl("x")], ManualClock(0.0))

    throttler.pause("x", 0)

    assert throttler.paused_until("x") == 0.0


def test_pool_pause_is_reported_for_a_linked_endpoint():
    throttler = build([rl("pool"), rl("ep", linked_limits=[LinkedLimitWeightPair("pool")])])

    throttler.pause("pool", 5)

    assert throttler.paused_until("ep") == 5.0


def test_pause_all_is_reported_for_an_unregistered_id():
    throttler = build([])

    throttler.pause_all(3)

    assert throttler.paused_until("nobody") == 3.0


@pytest.mark.parametrize("scope", ["id", "all"])
async def test_unpause_wakes_a_sleeping_acquirer_without_advancing_the_clock(scope):
    clock = ManualClock()
    throttler = build([rl("x")], clock)
    pause, unpause = (
        (lambda: throttler.pause("x", 100), lambda: throttler.unpause("x"))
        if scope == "id"
        else (lambda: throttler.pause_all(100), throttler.unpause_all)
    )
    pause()
    waiter = asyncio.create_task(throttler.execute_task("x").acquire())
    await asyncio.sleep(SETTLE)
    assert not waiter.done()

    unpause()

    await asyncio.wait_for(waiter, SETTLE)
    assert clock() == 0.0


async def test_bitrue_style_replacement_enforces_new_limit_and_frees_removed_id():
    throttler = build([rl("old", 1), rl("kept", 1)])
    limits = copy.deepcopy(throttler._rate_limits)
    limits = [limit for limit in limits if limit.limit_id != "old"]
    limits.append(rl("fresh", 1))

    throttler.set_rate_limits(limits)

    await assert_admits_exactly(throttler, "fresh", 1)
    for _ in range(5):
        assert await passes(throttler, "old")


async def test_coin_gecko_style_assignment_rebuilds_the_map_and_enforcement():
    throttler = build([rl("old", 1)])

    throttler._rate_limits = [rl("new", 2)]

    assert list(throttler._id_to_limit_map) == ["new"]
    await assert_admits_exactly(throttler, "new", 2)


async def test_okx_style_add_enforces_new_id_and_leaves_existing_usage_untouched():
    throttler = build([rl("used", 1), rl("idle", 1)])
    assert await passes(throttler, "used")

    throttler.add_rate_limits([rl("added", 1)])

    await assert_admits_exactly(throttler, "added", 1)
    assert not await passes(throttler, "used")
    assert await passes(throttler, "idle")


def test_get_related_limits_drops_unknown_links():
    throttler = build(
        [
            rl("known"),
            rl(
                "ep",
                linked_limits=[LinkedLimitWeightPair("known"), LinkedLimitWeightPair("ghost")],
            ),
        ]
    )

    primary, related = throttler.get_related_limits("ep")

    assert primary.limit_id == "ep"
    assert [limit.limit_id for limit, _weight in related] == ["known"]


async def test_window_is_driven_by_the_injected_clock():
    clock = ManualClock()
    throttler = build([rl("x", 1, interval=1.0)], clock, safety_margin_pct=0.05)
    assert await passes(throttler, "x")

    clock.advance(0.5)
    assert not await passes(throttler, "x")

    clock.advance(0.56)
    assert await passes(throttler, "x")


ARC_MODULE = "rate_limit.hb_compat.async_request_context_base"


def test_arc_logger_module_global_exists():
    module = importlib.import_module(ARC_MODULE)

    assert hasattr(module, "arc_logger")


def test_request_context_logger_is_lazily_created_and_cached(monkeypatch):
    module = importlib.import_module(ARC_MODULE)
    monkeypatch.setattr(module, "arc_logger", None)

    first = module.AsyncRequestContextBase.logger()
    second = module.AsyncRequestContextBase.logger()

    assert isinstance(first, logging.Logger)
    assert first.name == ARC_MODULE
    assert second is first
    assert module.arc_logger is first


def test_rate_limit_repr_matches_the_legacy_text():
    limit = RateLimit("a", 10, 1.0, 2, [LinkedLimitWeightPair("p", 3)])

    assert repr(limit) == (
        "limit_id: a, limit: 10, time interval: 1.0, weight: 2, "
        "linked_limits: [LinkedLimitWeightPair(limit_id='p', weight=3)]"
    )
