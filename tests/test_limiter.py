import asyncio
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from email.utils import format_datetime

import pytest

from rate_limit import (
    BackoffPolicy,
    BoundLimiter,
    Charge,
    LinkedLimitWeightPair,
    PausableRateLimiterProtocol,
    PoolPauseProtocol,
    RateLimit,
    RateLimiter,
    RateLimitExceededError,
    RateLimitPoolConflictError,
    RateLimitWeightError,
    RestRateLimiterProtocol,
    SlidingWindowPool,
    ThrottlerProtocol,
    UnknownRateLimitError,
    WeightedThrottlerProtocol,
)
from rate_limit.testing import BlockingSleep, ManualClock


def make(clock: ManualClock, *limits: RateLimit, **kw: object) -> RateLimiter:
    return RateLimiter(limits, clock=clock, sleep=clock.sleep, **kw)  # type: ignore[arg-type]


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(1000.0)


class TestRegistry:
    def test_add_get_and_snapshot(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 1))
        assert rl.has_limit("a")
        assert not rl.has_limit("b")
        assert rl.get_limit("b") is None
        assert rl.get_raw_limit("b") is None
        snapshot = rl.limits()
        assert snapshot["a"].limit == 10
        with pytest.raises(TypeError):
            snapshot["x"] = RateLimit("x", 1, 1)  # type: ignore[index]
        rl.add_limit(RateLimit("b", 2, 1))
        assert "b" not in snapshot

    def test_scaled_vs_raw(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 1), limits_pct=Decimal("0.5"))
        assert rl.limits_pct == Decimal("0.5")
        assert rl.get_limit("a") == RateLimit("a", 5, 1)
        assert rl.get_raw_limit("a") == RateLimit("a", 10, 1)

    def test_scaling_does_not_touch_weights(self, clock: ManualClock) -> None:
        link = LinkedLimitWeightPair("p", 3)
        rl = make(
            clock, RateLimit("a", 10, 1, weight=4, linked_limits=[link]), limits_pct=Decimal("0.5")
        )
        stored = rl.get_limit("a")
        assert stored is not None
        assert (stored.weight, stored.linked_limits) == (4, (link,))

    @pytest.mark.parametrize("pct", [Decimal(0), Decimal(-1), Decimal("1.5"), Decimal("NaN")])
    def test_invalid_pct(self, pct: Decimal) -> None:
        with pytest.raises(ValueError):
            RateLimiter(limits_pct=pct)

    def test_invalid_pct_type(self) -> None:
        with pytest.raises(ValueError):
            RateLimiter(limits_pct=0.5)  # type: ignore[arg-type]

    def test_invalid_margin_and_mode(self) -> None:
        with pytest.raises(ValueError):
            RateLimiter(safety_margin_pct=-0.1)
        with pytest.raises(ValueError):
            RateLimiter(unknown="nope")  # type: ignore[arg-type]

    def test_identical_raw_readd_is_noop_even_scaled(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 1), limits_pct=Decimal("0.5"))
        stored = rl.add_limit(RateLimit("a", 10, 1))
        assert stored.limit == 5

    @pytest.mark.parametrize(
        "other",
        [
            RateLimit("a", 11, 1),
            RateLimit("a", 10, 2),
            RateLimit("a", 10, 1, weight=2),
            RateLimit("a", 10, 1, linked_limits=[LinkedLimitWeightPair("z")]),
        ],
    )
    def test_raw_mismatch_conflicts_and_names_owner(
        self, clock: ManualClock, other: RateLimit
    ) -> None:
        rl = make(clock)
        rl.add_limit(RateLimit("a", 10, 1), owner="first")
        with pytest.raises(RateLimitPoolConflictError, match="first") as excinfo:
            rl.add_limit(other, owner="second")
        assert "second" in str(excinfo.value)
        assert excinfo.value.pool == "a"
        assert excinfo.value.existing == RateLimit("a", 10, 1)
        assert excinfo.value.requested == other
        assert rl.get_owner("a") == "first"

    def test_conflict_compares_raw_not_scaled(self, clock: ManualClock) -> None:
        rl = make(clock, limits_pct=Decimal("0.5"))
        rl.add_limit(RateLimit("a", 10, 1))
        with pytest.raises(RateLimitPoolConflictError):
            rl.add_limit(RateLimit("a", 5, 1))  # equals the scaled value, differs raw

    def test_batch_is_atomic(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 1))
        with pytest.raises(RateLimitPoolConflictError):
            rl.add_limits([RateLimit("new", 1, 1), RateLimit("a", 99, 1)])
        assert not rl.has_limit("new")

    def test_conflict_inside_one_batch(self, clock: ManualClock) -> None:
        rl = make(clock)
        with pytest.raises(RateLimitPoolConflictError):
            rl.add_limits([RateLimit("a", 1, 1), RateLimit("a", 2, 1)], owner="o")
        assert not rl.has_limit("a")
        rl.add_limits([RateLimit("a", 1, 1), RateLimit("a", 1, 1)])
        assert rl.has_limit("a")

    def test_same_batch_conflict_names_the_batch_not_a_phantom_owner(
        self, clock: ManualClock
    ) -> None:
        rl = make(clock)
        with pytest.raises(RateLimitPoolConflictError) as excinfo:
            rl.add_limits([RateLimit("a", 1, 1), RateLimit("a", 2, 1)], owner="o")
        assert "earlier in the same batch" in str(excinfo.value)

    async def test_set_limits_keeps_history_and_starts_new_ids_empty(
        self, clock: ManualClock
    ) -> None:
        rl = make(clock, RateLimit("keep", 1, 10), RateLimit("change", 1, 10))
        await rl.acquire("keep")
        await rl.acquire("change")
        rl.set_limits(
            [RateLimit("keep", 1, 10), RateLimit("change", 2, 10), RateLimit("new", 1, 1)]
        )
        assert not rl.has_limit("zzz")
        assert rl.try_acquire("keep") > 0  # history kept
        assert rl.try_acquire("change") == 0.0  # 1 used of 2: still fits
        assert rl.try_acquire("new") == 0.0
        rl.set_limits([RateLimit("only", 1, 1)])
        assert not rl.has_limit("keep")


class TestSetLimitsHistory:
    async def test_limit_decrease_keeps_used_history(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 10))
        await rl.acquire("a", {"a": 8})
        rl.set_limits([RateLimit("a", 5, 10)])
        assert rl.try_acquire("a") == pytest.approx(10.0)  # not a fresh window

    async def test_interval_change_evicts_per_new_window(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        await rl.acquire("a")
        clock.advance(15)
        rl.set_limits([RateLimit("a", 1, 20)])
        assert rl.try_acquire("a") == pytest.approx(5.0)  # entry still inside the 20s window
        clock.advance(6)
        assert rl.try_acquire("a") == 0.0
        rl.set_limits([RateLimit("a", 1, 5)])
        clock.advance(6)
        assert rl.try_acquire("a") == 0.0  # shrunk window drops the aged entry

    async def test_weight_and_linked_change_keep_history(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 2, 10))
        await rl.acquire("a")
        rl.set_limits([RateLimit("a", 2, 10, weight=2)])
        assert rl.try_acquire("a") > 0
        linked = RateLimit("a", 1, 10, linked_limits=[LinkedLimitWeightPair("b")])
        rl.set_limits([linked, RateLimit("b", 5, 10)])
        assert rl.try_acquire("a") > 0

    async def test_removed_then_readded_id_starts_empty(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        await rl.acquire("a")
        rl.set_limits([RateLimit("b", 1, 10)])
        rl.set_limits([RateLimit("a", 1, 10)])
        assert rl.try_acquire("a") == 0.0


def bucket(limit_id: str, limit: int, interval: float) -> RateLimit:
    return RateLimit(limit_id, limit, interval, kind="token_bucket")


class TestTokenBucket:
    async def test_burst_then_paced(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 5, 5))  # capacity 5, refill 1/s
        for _ in range(5):
            await rl.acquire("a")
        assert clock.sleeps == []
        await rl.acquire("a")
        await rl.acquire("a")
        assert clock.sleeps == [pytest.approx(1.0), pytest.approx(1.0)]
        assert clock.now == pytest.approx(1002.0)

    async def test_idle_time_refills_fractionally(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 4, 4))
        for _ in range(4):
            await rl.acquire("a")
        clock.advance(2.5)
        assert rl.try_acquire("a") == 0.0
        assert rl.try_acquire("a") == 0.0
        assert rl.try_acquire("a") == pytest.approx(0.5)

    async def test_weights_are_charged(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 10, 10))
        await rl.acquire("a", {"a": 10})
        assert rl.try_acquire("a", {"a": 4}) == pytest.approx(4.0)
        with pytest.raises(RateLimitWeightError):
            await rl.acquire("a", {"a": 11})

    async def test_limits_pct_scales_capacity_and_refill(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 10, 10), limits_pct=Decimal("0.5"))
        assert rl.get_limit("a") == bucket("a", 5, 10)
        for _ in range(5):
            assert rl.try_acquire("a") == 0.0
        assert rl.try_acquire("a") == pytest.approx(2.0)  # refill 5 / 10s = 0.5/s

    async def test_margin_slows_refill(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 4, 4), safety_margin_pct=0.25)
        for _ in range(4):
            await rl.acquire("a")
        assert rl.try_acquire("a") == pytest.approx(1.25)

    async def test_linked_buckets_and_windows_mix(self, clock: ManualClock) -> None:
        link = LinkedLimitWeightPair("w")
        rl = make(
            clock,
            RateLimit("a", 3, 3, linked_limits=[link], kind="token_bucket"),
            RateLimit("w", 2, 10),
        )
        await rl.acquire("a")
        await rl.acquire("a")
        assert rl.try_acquire("a") == pytest.approx(10.0)  # window is the tighter pool

    async def test_deadline_wait_passes_without_a_second_sleep(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 3, 7))  # awkward refill: 3 / 7 per second
        for _ in range(3):
            await rl.acquire("a")
        for _ in range(20):
            await rl.acquire("a")
        assert len(clock.sleeps) == 20


class TestSetLimitsKinds:
    async def test_bucket_keeps_deficit_across_resize(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 5, 5))
        await rl.acquire("a", {"a": 5})
        rl.set_limits([bucket("a", 10, 10)])
        assert rl.try_acquire("a", {"a": 5}) == 0.0  # 10 - 5 used
        assert rl.try_acquire("a") == pytest.approx(1.0)

    async def test_window_to_bucket_converts_used_to_deficit(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 3, 10))
        for _ in range(3):
            await rl.acquire("a")
        rl.set_limits([bucket("a", 3, 3)])
        assert rl.try_acquire("a") == pytest.approx(1.0)

    async def test_bucket_to_window_seeds_rounded_up_charge(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 5, 5))
        await rl.acquire("a", {"a": 5})
        clock.advance(2.5)
        rl.set_limits([RateLimit("a", 5, 10)])  # 2.5 used -> a 3-unit charge stamped now
        assert rl.try_acquire("a") == 0.0
        assert rl.try_acquire("a") == 0.0
        assert rl.try_acquire("a") == pytest.approx(10.0)

    async def test_window_to_window_still_rebuilds(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        await rl.acquire("a")
        rl.set_limits([RateLimit("a", 1, 10)])
        assert rl.try_acquire("a") > 0


class TestAcquireWindow:
    async def test_window_waits_exactly_interval(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 2, 10))
        await rl.acquire("a")
        await rl.acquire("a")
        assert clock.sleeps == []
        await rl.acquire("a")
        assert clock.sleeps == [pytest.approx(10.0)]

    async def test_inclusive_boundary(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        await rl.acquire("a")
        clock.advance(10)
        assert rl.try_acquire("a") > 0  # age == window still counts
        clock.advance(0.001)
        assert rl.try_acquire("a") == 0.0

    async def test_safety_margin_extends_window(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10), safety_margin_pct=0.5)
        await rl.acquire("a")
        await rl.acquire("a")
        assert clock.sleeps == [pytest.approx(15.0)]

    async def test_weights_default_to_declared(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 1, weight=4))
        await rl.acquire("a")
        await rl.acquire("a")
        assert clock.sleeps == []
        await rl.acquire("a")
        assert len(clock.sleeps) == 1

    async def test_weighted_wait_uses_deque_walk(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 10))
        await rl.acquire("a", {"a": 6})
        clock.advance(4)
        await rl.acquire("a", {"a": 4})
        clock.advance(1)
        assert rl.blocked_for("a", {"a": 5}) == pytest.approx(5.0)  # first (6) expires at +10
        assert rl.blocked_for("a", {"a": 1}) == pytest.approx(5.0)
        assert rl.blocked_for("a", {"a": 10}) == pytest.approx(9.0)

    async def test_no_polling_single_sleep(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 100))
        await rl.acquire("a")
        await rl.acquire("a")
        assert len(clock.sleeps) == 1

    async def test_no_overshoot_with_concurrent_waiters(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 3, 10))
        stamps: list[float] = []

        async def one() -> None:
            await rl.acquire("a")
            stamps.append(clock())

        await asyncio.gather(*(one() for _ in range(10)))
        stamps.sort()
        assert all(stamps[i + 3] - stamps[i] >= 10 for i in range(len(stamps) - 3))
        assert clock() - 1000 <= 4 * 10 + 1e-6  # ceil(10/3) windows

    async def test_almost_reached_warning_throttled(
        self, clock: ManualClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        rl = make(clock, RateLimit("a", 20, 100))
        with caplog.at_level(logging.WARNING, logger="rate_limit.limiter"):
            for _ in range(19):
                await rl.acquire("a")
            assert len([r for r in caplog.records if "almost reached" in r.message]) == 1
            clock.advance(31)
            await rl.acquire("a")
            assert len([r for r in caplog.records if "almost reached" in r.message]) == 2


class TestLinkedAndWeights:
    @pytest.fixture
    def rl(self, clock: ManualClock) -> RateLimiter:
        return make(
            clock,
            RateLimit(
                "a",
                100,
                1,
                linked_limits=[LinkedLimitWeightPair("pool", 1), LinkedLimitWeightPair("raw", 2)],
            ),
            RateLimit("pool", 10, 1),
            RateLimit("raw", 100, 1),
            RateLimit("b", 100, 1, linked_limits=[LinkedLimitWeightPair("pool", 1)]),
        )

    async def test_linked_pool_gates_two_primaries(self, clock: ManualClock) -> None:
        rl = make(
            clock,
            RateLimit("a", 5, 1, linked_limits=[LinkedLimitWeightPair("s")]),
            RateLimit("b", 5, 1, linked_limits=[LinkedLimitWeightPair("s")]),
            RateLimit("s", 1, 1),
        )
        await rl.acquire("a")
        await rl.acquire("b")
        assert clock.sleeps == [pytest.approx(1.0)]

    async def test_link_order_does_not_matter_at_registration(self, clock: ManualClock) -> None:
        rl = RateLimiter(
            [RateLimit("a", 5, 1, linked_limits=[LinkedLimitWeightPair("later")])],
            clock=clock,
            sleep=clock.sleep,
        )
        with pytest.raises(UnknownRateLimitError):
            await rl.acquire("a")
        rl.add_limit(RateLimit("later", 1, 1))
        await rl.acquire("a")

    async def test_declared_weights_charged(self, rl: RateLimiter, clock: ManualClock) -> None:
        for _ in range(10):
            await rl.acquire("a")
        assert clock.sleeps == []
        await rl.acquire("a")
        assert len(clock.sleeps) == 1

    async def test_linked_override_leaves_others_declared(
        self, rl: RateLimiter, clock: ManualClock
    ) -> None:
        await rl.acquire("a", {"pool": 9})
        await rl.acquire("a")
        assert clock.sleeps == []
        assert rl.try_acquire("b") > 0  # pool is full: 9 + 1

    async def test_primary_override(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 1))
        await rl.acquire("a", {"a": 10})
        assert rl.try_acquire("a") > 0

    async def test_unknown_key_raises_before_waiting(
        self, rl: RateLimiter, clock: ManualClock
    ) -> None:
        with pytest.raises(ValueError, match="nope"):
            await rl.acquire("a", {"nope": 1})
        assert clock.sleeps == []

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
    async def test_bad_weight_raises(self, rl: RateLimiter, bad: object) -> None:
        with pytest.raises(ValueError):
            await rl.acquire("a", {"pool": bad})  # type: ignore[dict-item]

    async def test_weight_above_limit_raises_immediately(
        self, rl: RateLimiter, clock: ManualClock
    ) -> None:
        with pytest.raises(RateLimitWeightError):
            await rl.acquire("a", {"pool": 11})
        assert clock.sleeps == []
        assert isinstance(RateLimitWeightError("x"), ValueError)

    async def test_weight_above_scaled_limit_raises(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 1), limits_pct=Decimal("0.5"))
        with pytest.raises(RateLimitWeightError):
            await rl.acquire("a", {"a": 6})

    def test_execute_task_validates_eagerly(self, rl: RateLimiter) -> None:
        with pytest.raises(ValueError):
            rl.execute_task("a", weights={"nope": 1})

    def test_duplicate_links_sum(self, clock: ManualClock) -> None:
        rl = make(
            clock,
            RateLimit(
                "a", 5, 1, linked_limits=[LinkedLimitWeightPair("s"), LinkedLimitWeightPair("s")]
            ),
            RateLimit("s", 2, 1),
        )
        assert rl.try_acquire("a") == 0.0
        assert rl.try_acquire("a") > 0


class TestUnknown:
    async def test_raise_mode(self, clock: ManualClock) -> None:
        rl = make(clock)
        with pytest.raises(UnknownRateLimitError):
            await rl.acquire("nope")
        assert isinstance(UnknownRateLimitError("x"), KeyError)

    async def test_allow_mode_warns_once_and_runs_unthrottled(
        self, clock: ManualClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        rl = make(clock, unknown="allow")
        with caplog.at_level(logging.WARNING, logger="rate_limit.limiter"):
            for _ in range(3):
                await rl.acquire("nope")
                async with rl.execute_task(limit_id="nope"):
                    pass
        assert len([r for r in caplog.records if "nope" in r.message]) == 1
        assert clock.sleeps == []
        assert rl.try_acquire("nope") == 0.0
        assert rl.blocked_for("nope") == 0.0

    @pytest.fixture(params=["pause_all", "pause_id"])
    def paused_unknown(self, clock: ManualClock, request: pytest.FixtureRequest) -> RateLimiter:
        rl = make(clock, unknown="allow")
        if request.param == "pause_all":
            rl.pause_all(7)
        else:
            rl.pause("nope", 7)
        return rl

    async def test_allow_mode_acquire_waits_out_pause(
        self, paused_unknown: RateLimiter, clock: ManualClock
    ) -> None:
        await paused_unknown.acquire("nope")
        assert clock.sleeps == [pytest.approx(7.0)]

    def test_allow_mode_try_acquire_reports_pause(self, paused_unknown: RateLimiter) -> None:
        assert paused_unknown.try_acquire("nope") == pytest.approx(7.0)

    def test_allow_mode_blocked_for_reports_pause(self, paused_unknown: RateLimiter) -> None:
        assert paused_unknown.blocked_for("nope") == pytest.approx(7.0)

    @pytest.mark.parametrize("weights", [{"ghost": 1}, {"registered": 1}, {"nope": 0}])
    async def test_allow_mode_still_validates_weights(
        self, clock: ManualClock, weights: dict[str, int]
    ) -> None:
        rl = make(clock, RateLimit("registered", 5, 1), unknown="allow")
        with pytest.raises(ValueError):
            await rl.acquire("nope", weights)
        with pytest.raises(ValueError):
            rl.execute_task("nope", weights=weights)

    async def test_allow_mode_accepts_valid_weight_on_the_unknown_primary(
        self, clock: ManualClock
    ) -> None:
        rl = make(clock, unknown="allow")
        await rl.acquire("nope", {"nope": 3})
        assert clock.sleeps == []

    async def test_unknown_warning_is_emitted_without_the_lock(self, clock: ManualClock) -> None:
        rl = make(clock, unknown="allow")
        held: list[bool] = []

        class Probe(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                held.append(rl._lock.locked())

        handler = Probe(logging.WARNING)
        target = logging.getLogger("rate_limit.limiter")
        target.addHandler(handler)
        try:
            await rl.acquire("nope")
            rl.execute_task("nope2")
        finally:
            target.removeHandler(handler)
        assert held == [False, False]

    async def test_linked_unknown_follows_mode(
        self, clock: ManualClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        a = RateLimit("a", 5, 1, linked_limits=[LinkedLimitWeightPair("ghost")])
        rl = make(clock, a, unknown="allow")
        with caplog.at_level(logging.WARNING, logger="rate_limit.limiter"):
            await rl.acquire("a")
            await rl.acquire("a")
        assert len([r for r in caplog.records if "ghost" in r.message]) == 1
        with pytest.raises(UnknownRateLimitError):
            await make(clock, a).acquire("a")


class TestPause:
    async def test_pause_blocks_and_never_shortens(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 100, 1))
        rl.pause("a", 10)
        rl.pause("a", 1)
        assert rl.is_paused("a")
        assert rl.paused_until("a") == pytest.approx(1010.0)
        await rl.acquire("a")
        assert clock.sleeps == [pytest.approx(10.0)]
        assert not rl.is_paused("a")

    def test_paused_until_zero_if_never_paused(self, clock: ManualClock) -> None:
        assert make(clock, RateLimit("a", 1, 1)).paused_until("a") == 0.0

    async def test_pause_on_linked_id_blocks_dependents_only(self, clock: ManualClock) -> None:
        rl = make(
            clock,
            RateLimit("feed", 100, 1, linked_limits=[LinkedLimitWeightPair("host")]),
            RateLimit("host", 100, 1),
            RateLimit("other", 100, 1),
        )
        rl.pause("host", 5)
        assert rl.is_paused("feed")
        assert rl.is_paused("host")
        assert not rl.is_paused("other")
        rl.pause("feed", 5)
        assert rl.blocked_for("other") == 0.0
        await rl.acquire("feed")
        assert clock.sleeps == [pytest.approx(5.0)]

    async def test_pause_on_dependent_does_not_block_host(self, clock: ManualClock) -> None:
        rl = make(
            clock,
            RateLimit("feed", 100, 1, linked_limits=[LinkedLimitWeightPair("host")]),
            RateLimit("host", 100, 1),
        )
        rl.pause("feed", 5)
        assert not rl.is_paused("host")

    async def test_pause_on_unknown_id_applies_after_add(self, clock: ManualClock) -> None:
        rl = make(clock)
        rl.pause("late", 4)
        rl.add_limit(RateLimit("late", 5, 1))
        await rl.acquire("late")
        assert clock.sleeps == [pytest.approx(4.0)]

    async def test_pause_all(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1), RateLimit("b", 5, 1))
        rl.pause_all(7)
        rl.pause_all(1)
        assert rl.is_paused("a") and rl.is_paused("b") and rl.is_paused("anything")
        await rl.acquire("b")
        assert clock.sleeps == [pytest.approx(7.0)]

    def test_pause_rejects_non_finite(self, clock: ManualClock) -> None:
        rl = make(clock)
        with pytest.raises(ValueError):
            rl.pause("a", float("inf"))
        with pytest.raises(ValueError):
            rl.pause_all(float("nan"))
        rl.pause("a", -5)  # clamps to 0
        assert not rl.is_paused("a")

    async def test_pause_arriving_during_window_sleep_is_honoured(self, clock: ManualClock) -> None:
        paused = False

        async def sleep(seconds: float) -> None:
            nonlocal paused
            await clock.sleep(seconds)
            if not paused:
                paused = True
                rl.pause("a", 4)

        rl = RateLimiter([RateLimit("a", 1, 1)], clock=clock, sleep=sleep)
        await rl.acquire("a")
        await rl.acquire("a")
        assert clock.sleeps == [pytest.approx(1.0), pytest.approx(4.0)]

    async def test_pause_from_another_thread_and_loop(self) -> None:
        rl = RateLimiter([RateLimit("a", 100, 1)])
        rl.pause("a", 0.15)
        started = asyncio.get_running_loop().time()
        await asyncio.to_thread(lambda: asyncio.run(rl.acquire("a")))
        assert asyncio.get_running_loop().time() - started >= 0.12


class TestRetryAfterCap:
    def test_default_honours_server_value(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1))
        assert rl.report_throttled("a", 7200).delay == 7200

    def test_cap_clamps_and_logs_both_values(
        self, clock: ManualClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        rl = make(clock, RateLimit("a", 5, 1), max_retry_after_seconds=60)
        with caplog.at_level(logging.WARNING, logger="rate_limit.limiter"):
            verdict = rl.report_throttled("a", 7200)
        assert (verdict.delay, verdict.retry_after) == (60, 60)
        assert rl.paused_until("a") == pytest.approx(clock.now + 60)
        capped = [r.getMessage() for r in caplog.records if "7200" in r.getMessage()]
        assert len(capped) == 1
        assert "60" in capped[0]
        assert not any("honoured" in r.getMessage() for r in caplog.records)

    def test_values_at_or_below_cap_are_untouched(
        self, clock: ManualClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        rl = make(clock, RateLimit("a", 5, 1), max_retry_after_seconds=60)
        with caplog.at_level(logging.WARNING, logger="rate_limit.limiter"):
            assert rl.report_throttled("a", 60).delay == 60
        assert all("exceeds" not in r.getMessage() for r in caplog.records)

    def test_cap_does_not_alter_backoff_derived_delays(self, clock: ManualClock) -> None:
        rl = make(
            clock,
            RateLimit("a", 5, 1),
            max_retry_after_seconds=1,
            backoff_factory=lambda: BackoffPolicy.fixed(30),
        )
        verdict = rl.report_throttled("a")
        assert (verdict.delay, verdict.retry_after) == (30, None)

    def test_cap_applies_to_418(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1), max_retry_after_seconds=5)
        verdict = rl.report_throttled("a", 900, status=418)
        assert (verdict.delay, verdict.paused_all) == (5, True)

    @pytest.mark.parametrize("cap", [0, -1, float("nan"), float("inf")])
    def test_cap_validation(self, cap: float) -> None:
        with pytest.raises(ValueError, match="max_retry_after_seconds"):
            RateLimiter(max_retry_after_seconds=cap)


class TestUnpause:
    @staticmethod
    def blocked(clock: ManualClock, sleeper: BlockingSleep, *limits: RateLimit) -> RateLimiter:
        return RateLimiter(limits, clock=clock, sleep=sleeper)

    async def test_unpause_wakes_a_sleeping_acquirer_without_advancing_time(
        self, clock: ManualClock
    ) -> None:
        sleeper = BlockingSleep()
        rl = self.blocked(clock, sleeper, RateLimit("a", 5, 1))
        rl.pause("a", 7200)
        task = asyncio.create_task(rl.acquire("a"))
        await sleeper.started.wait()
        assert not task.done()
        rl.unpause("a")
        await asyncio.wait_for(task, 1)
        assert clock.now == 1000.0
        assert sleeper.durations == [pytest.approx(7200.0)]
        assert not rl.is_paused("a")

    async def test_unpause_all_lifts_global_and_per_id_pauses(self, clock: ManualClock) -> None:
        sleeper = BlockingSleep()
        rl = self.blocked(clock, sleeper, RateLimit("a", 5, 1), RateLimit("b", 5, 1))
        rl.pause_all(7200)
        rl.pause("b", 9000)
        task = asyncio.create_task(rl.acquire("b"))
        await sleeper.started.wait()
        rl.unpause_all()
        await asyncio.wait_for(task, 1)
        assert not rl.is_paused("a") and not rl.is_paused("b")
        assert clock.now == 1000.0

    async def test_unpause_wakes_every_sleeper(self, clock: ManualClock) -> None:
        sleeper = BlockingSleep()
        rl = self.blocked(clock, sleeper, RateLimit("a", 5, 1))
        rl.pause("a", 7200)
        tasks = [asyncio.create_task(rl.acquire("a")) for _ in range(3)]
        while len(sleeper.durations) < 3:
            await asyncio.sleep(0)
        rl.unpause("a")
        await asyncio.wait_for(asyncio.gather(*tasks), 1)

    async def test_still_paused_acquirer_goes_back_to_sleep(self, clock: ManualClock) -> None:
        sleeper = BlockingSleep()
        rl = self.blocked(clock, sleeper, RateLimit("a", 5, 1))
        rl.pause("a", 7200)
        rl.pause_all(100)
        task = asyncio.create_task(rl.acquire("a"))
        await sleeper.started.wait()
        rl.unpause("a")  # pause_all remains
        while len(sleeper.durations) < 2:
            await asyncio.sleep(0)
        assert sleeper.durations[1] == pytest.approx(100.0)
        assert not task.done()
        rl.unpause_all()
        await asyncio.wait_for(task, 1)

    async def test_unpause_does_not_free_pool_capacity(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        await rl.acquire("a")
        rl.pause("a", 3)
        rl.unpause("a")
        assert rl.try_acquire("a") == pytest.approx(10.0)
        rl.pause_all(3)
        rl.unpause_all()
        assert rl.try_acquire("a") == pytest.approx(10.0)

    def test_unpause_unknown_id_is_a_noop(self, clock: ManualClock) -> None:
        rl = make(clock)
        rl.unpause("nope")
        rl.pause("late", 4)
        rl.unpause("late")
        assert not rl.is_paused("late")

    async def test_unpause_leaves_backoff_streak_alone(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1))
        rl.report_throttled("a")
        rl.unpause("a")
        assert rl.backoff("a").streak == 1
        rl.report_throttled("a")  # no longer a burst: the streak escalates
        assert rl.backoff("a").streak == 2

    async def test_cancelled_paused_acquire_leaks_nothing(self, clock: ManualClock) -> None:
        sleeper = BlockingSleep()
        rl = self.blocked(clock, sleeper, RateLimit("a", 5, 1))
        rl.pause("a", 7200)
        before = asyncio.all_tasks()
        task = asyncio.create_task(rl.acquire("a"))
        await sleeper.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert asyncio.all_tasks() == before
        rl.unpause("a")  # waker already deregistered: nothing to wake

    async def test_sleep_failure_propagates(self, clock: ManualClock) -> None:
        async def broken(seconds: float) -> None:
            raise RuntimeError("sleep failed")

        rl = RateLimiter([RateLimit("a", 5, 1)], clock=clock, sleep=broken)
        rl.pause("a", 5)
        with pytest.raises(RuntimeError, match="sleep failed"):
            await rl.acquire("a")

    async def test_unpause_from_another_thread(self, clock: ManualClock) -> None:
        sleeper = BlockingSleep()
        rl = self.blocked(clock, sleeper, RateLimit("a", 5, 1))
        rl.pause("a", 7200)
        task = asyncio.create_task(rl.acquire("a"))
        await sleeper.started.wait()
        await asyncio.to_thread(rl.unpause, "a")
        await asyncio.wait_for(task, 1)

    async def test_plain_pacing_sleep_is_not_interruptible_by_unpause(
        self, clock: ManualClock
    ) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        await rl.acquire("a")
        rl.unpause_all()
        await rl.acquire("a")
        assert clock.sleeps == [pytest.approx(10.0)]

    async def test_bound_limiter_forwards(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1), RateLimit("b", 5, 1))
        bound = rl.bind("a")
        bound.pause(50)
        rl.pause("b", 50)
        bound.unpause()
        assert not bound.is_paused()
        assert rl.is_paused("b")
        bound.unpause_all()
        assert not rl.is_paused("b")


class TestNonBlocking:
    async def test_try_acquire_reserves_only_when_zero(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        assert rl.try_acquire("a") == 0.0
        assert rl.try_acquire("a") == pytest.approx(10.0)
        assert rl.try_acquire("a") == pytest.approx(10.0)  # nothing was recorded
        clock.advance(10.001)
        assert rl.try_acquire("a") == 0.0

    async def test_try_acquire_reports_pause_without_recording(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        rl.pause("a", 3)
        assert rl.try_acquire("a") == pytest.approx(3.0)
        clock.advance(3)
        assert rl.try_acquire("a") == 0.0

    async def test_blocked_for_pause_window_and_max(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        assert rl.blocked_for("a") == 0.0
        await rl.acquire("a")
        assert rl.blocked_for("a") == pytest.approx(10.0)
        rl.pause("a", 20)
        assert rl.blocked_for("a") == pytest.approx(20.0)
        assert rl.blocked_for("a") == pytest.approx(20.0)  # still records nothing
        clock.advance(20)
        assert rl.blocked_for("a") == 0.0

    def test_blocked_for_validates(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 1))
        with pytest.raises(ValueError):
            rl.blocked_for("a", {"x": 1})
        with pytest.raises(ValueError):
            rl.try_acquire("a", {"x": 1})


class TestMustInvariants:
    async def test_cancelled_acquire_charges_nothing(self, clock: ManualClock) -> None:
        started = asyncio.Event()

        async def blocked_sleep(seconds: float) -> None:
            started.set()
            await asyncio.Event().wait()

        rl = RateLimiter([RateLimit("a", 1, 10)], clock=clock, sleep=blocked_sleep)
        await rl.acquire("a")
        task = asyncio.create_task(rl.acquire("a"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert rl._pools["a"].used == 1  # nothing recorded before the sleep
        clock.advance(10.001)
        assert rl.try_acquire("a") == 0.0  # only the first charge existed, and it expired
        assert rl.try_acquire("a") == pytest.approx(10.0)  # exactly one charge since

    async def test_cancel_while_waiting_out_pause_charges_nothing(self, clock: ManualClock) -> None:
        started = asyncio.Event()

        async def blocked_sleep(seconds: float) -> None:
            started.set()
            await asyncio.Event().wait()

        rl = RateLimiter([RateLimit("a", 1, 10)], clock=clock, sleep=blocked_sleep)
        rl.pause("a", 5)
        task = asyncio.create_task(rl.acquire("a"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        clock.advance(5)
        assert rl.try_acquire("a") == 0.0

    async def test_no_lock_held_across_a_sleep(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        rl.pause("a", 2)
        seen: list[bool] = []
        clock.on_sleep = lambda _seconds: seen.append(rl._lock.locked())
        await rl.acquire("a")  # pause sleep
        await rl.acquire("a")  # window sleep
        assert len(seen) >= 2
        assert seen == [False] * len(seen)

    async def test_a_sleeper_does_not_block_other_waiters(self, clock: ManualClock) -> None:
        """While one acquire sleeps, a different limit stays immediately usable."""
        gate = asyncio.Event()
        started = asyncio.Event()

        async def sleep(seconds: float) -> None:
            started.set()
            await gate.wait()

        rl = RateLimiter(
            [RateLimit("slow", 1, 10), RateLimit("fast", 5, 10)], clock=clock, sleep=sleep
        )
        await rl.acquire("slow")
        sleeper = asyncio.create_task(rl.acquire("slow"))
        await started.wait()
        await asyncio.wait_for(rl.acquire("fast"), timeout=1)
        assert rl.try_acquire("fast") == 0.0
        sleeper.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sleeper

    def test_two_loops_on_two_threads_share_one_limit(self) -> None:
        interval = 0.05
        reserved: list[float] = []

        class Recording(SlidingWindowPool):
            def reserve(self, cost: int, now: float) -> float:
                wait = super().reserve(cost, now)
                if wait == 0.0:
                    reserved.append(now)
                return wait

        rl = RateLimiter([RateLimit("a", 2, interval)])
        rl._pools["a"] = Recording(2, interval)
        errors: list[BaseException] = []

        def worker() -> None:
            async def go() -> None:
                for _ in range(3):
                    await rl.acquire("a")

            try:
                asyncio.run(go())
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == []
        assert len(reserved) == 6
        reserved.sort()
        assert all(reserved[i + 2] - reserved[i] >= interval for i in range(4))


class TestRefund:
    async def test_reserve_returns_a_read_only_receipt(self, clock: ManualClock) -> None:
        rl = make(
            clock,
            RateLimit("a", 2, 10, linked_limits=[LinkedLimitWeightPair("s")]),
            RateLimit("s", 5, 10),
        )
        charge = await rl.reserve("a", {"s": 3})
        assert isinstance(charge, Charge)
        assert (charge.limit_id, dict(charge.costs), charge.stamp) == ("a", {"a": 1, "s": 3}, 1000)
        assert not charge.refunded
        with pytest.raises(TypeError):
            charge.costs["a"] = 2  # type: ignore[index]
        charge.refund()
        assert charge.refunded
        assert (rl._pools["a"].used, rl._pools["s"].used) == (0, 0)

    async def test_reserve_waits_like_acquire(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        await rl.reserve("a")
        rl.pause("a", 15)
        charge = await rl.reserve("a")
        assert clock.sleeps == [pytest.approx(15.0)]
        assert charge.stamp == pytest.approx(1015.0)
        await rl.reserve("a")
        assert clock.sleeps == [pytest.approx(15.0), pytest.approx(10.0)]

    async def test_acquire_is_unchanged(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 2, 10))
        assert await rl.acquire("a", {"a": 2}) is None
        assert rl.try_acquire("a") == pytest.approx(10.0)

    async def test_double_refund_credits_once(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 4, 4))
        charge = await rl.reserve("a", {"a": 2})
        await rl.reserve("a", {"a": 2})
        charge.refund()
        charge.refund()
        assert rl._pools["a"].used == 2

    async def test_concurrent_refunds_credit_once(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 4, 4))
        charge = await rl.reserve("a", {"a": 2})
        await rl.reserve("a", {"a": 2})
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: charge.refund(), range(8)))
        assert rl._pools["a"].used == 2

    async def test_refund_survives_a_same_kind_set_limits(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        charge = await rl.reserve("a")
        rl.set_limits([RateLimit("a", 2, 10)])
        charge.refund()
        assert rl.try_acquire("a") == 0.0
        assert rl.try_acquire("a") == 0.0
        assert rl.try_acquire("a") == pytest.approx(10.0)

    async def test_bucket_refund_survives_a_same_kind_set_limits(self, clock: ManualClock) -> None:
        rl = make(clock, bucket("a", 4, 4))
        charge = await rl.reserve("a", {"a": 4})
        rl.set_limits([bucket("a", 4, 8)])
        charge.refund()
        assert rl.try_acquire("a", {"a": 4}) == 0.0

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            (RateLimit("a", 2, 10), bucket("a", 2, 10)),
            (bucket("a", 2, 10), RateLimit("a", 2, 10)),  # carry-over equals the charge
        ],
    )
    async def test_kind_change_makes_refund_a_noop(
        self, clock: ManualClock, before: RateLimit, after: RateLimit
    ) -> None:
        rl = make(clock, before)
        charge = await rl.reserve("a")
        rl.set_limits([after])
        charge.refund()
        assert charge.refunded
        assert rl._pools["a"].used == pytest.approx(1.0)

    async def test_removed_limit_is_skipped(self, clock: ManualClock) -> None:
        rl = make(
            clock,
            RateLimit("a", 1, 10, linked_limits=[LinkedLimitWeightPair("s")]),
            RateLimit("s", 1, 10),
        )
        charge = await rl.reserve("a")
        rl.set_limits([RateLimit("a", 1, 10)])  # "s" is gone
        charge.refund()
        assert rl.try_acquire("a") == 0.0

        rl = make(clock, RateLimit("a", 1, 10))
        charge = await rl.reserve("a")
        rl.set_limits([])
        charge.refund()  # nothing left to refund into
        assert charge.refunded

    async def test_unknown_allow_mode_gives_an_empty_charge(self, clock: ManualClock) -> None:
        rl = make(clock, unknown="allow")
        charge = await rl.reserve("nope")
        assert dict(charge.costs) == {}
        charge.refund()
        assert charge.refunded

    async def test_refund_does_not_wake_a_waiter(self, clock: ManualClock) -> None:
        sleeper = BlockingSleep()
        rl = RateLimiter([RateLimit("a", 1, 10)], clock=clock, sleep=sleeper)
        charge = await rl.reserve("a")
        task = asyncio.create_task(rl.acquire("a"))
        await sleeper.started.wait()
        charge.refund()
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done()  # it sees the capacity only when its sleep ends
        assert rl.try_acquire("a") == 0.0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_bound_reserve_uses_bound_weights(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 1))
        bound = rl.bind("a", {"a": 4})
        charge = await bound.reserve()
        assert dict(charge.costs) == {"a": 4}
        assert rl._pools["a"].used == 4
        charge.refund()
        assert rl._pools["a"].used == 0


class TestRefundOnError:
    async def test_exception_refunds_and_propagates(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        with pytest.raises(RuntimeError, match="sign"):
            async with rl.execute_task("a", refund_on_error=True):
                raise RuntimeError("sign failed")
        assert rl.try_acquire("a") == 0.0

    async def test_success_keeps_the_charge(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        async with rl.execute_task("a", refund_on_error=True):
            pass
        assert rl.try_acquire("a") == pytest.approx(10.0)

    async def test_off_by_default(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        with pytest.raises(RuntimeError):
            async with rl.execute_task("a"):
                raise RuntimeError("sign failed")
        assert rl.try_acquire("a") == pytest.approx(10.0)

    async def test_cancellation_keeps_the_charge(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        entered = asyncio.Event()

        async def body() -> None:
            async with rl.execute_task("a", refund_on_error=True):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(body())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert rl.try_acquire("a") == pytest.approx(10.0)  # the request may have gone out


class TestBackoffHooks:
    def test_report_throttled_429_escalates_and_pauses_only_that_id(
        self, clock: ManualClock
    ) -> None:
        rl = make(clock, RateLimit("a", 5, 1), RateLimit("b", 5, 1))
        verdict = rl.report_throttled("a")
        assert (verdict.rate_limited, verdict.status, verdict.delay) == (True, 429, 5.0)
        assert not verdict.paused_all
        assert rl.is_paused("a") and not rl.is_paused("b")
        clock.advance(5)
        assert rl.report_throttled("a").delay == 10.0
        assert rl.backoff("a").streak == 2

    def test_concurrent_reports_escalate_once(self, clock: ManualClock) -> None:
        class Slow(BackoffPolicy):
            def next_delay(
                self, status: int, retry_after: float | None, in_burst: bool = False
            ) -> float:
                time.sleep(0.02)  # widen the check-then-act window
                return super().next_delay(status, retry_after, in_burst)

        rl = make(clock, RateLimit("a", 5, 1), backoff_factory=Slow)
        with ThreadPoolExecutor(max_workers=8) as pool:
            verdicts = list(pool.map(lambda _: rl.report_throttled("a"), range(8)))
        assert sorted(v.delay for v in verdicts) == [0.0] * 7 + [5.0]
        assert rl.backoff("a").streak == 1
        assert rl.paused_until("a") == pytest.approx(1005.0)

    def test_report_throttled_418_pauses_all(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1), RateLimit("b", 5, 1))
        verdict = rl.report_throttled("a", status=418)
        assert (verdict.delay, verdict.paused_all) == (120.0, True)
        assert rl.is_paused("b")

    def test_report_throttled_retry_after_wins_and_long_is_flagged(
        self, clock: ManualClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        rl = make(clock, RateLimit("a", 5, 1))
        with caplog.at_level(logging.WARNING, logger="rate_limit.limiter"):
            verdict = rl.report_throttled("a", 7200.0)
        assert verdict.delay == 7200.0
        assert "unusually long" in caplog.text

    def test_in_burst_joins_pause_without_escalating(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1))
        rl.report_throttled("a")
        verdict = rl.report_throttled("a")
        assert (verdict.rate_limited, verdict.delay, verdict.paused_all) == (True, 0.0, False)
        assert rl.backoff("a").streak == 1

    def test_report_rate_limited_uses_exception_fields(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1))
        verdict = rl.report_rate_limited("a", RateLimitExceededError(status=429, retry_after=3.0))
        assert (verdict.delay, verdict.retry_after) == (3.0, 3.0)
        assert rl.paused_until("a") == pytest.approx(1003.0)
        assert rl.report_rate_limited("zzz", RateLimitExceededError()).status == 429

    def test_report_response_branches(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1))
        ok = rl.report_response("a", 200)
        assert (ok.rate_limited, ok.delay, ok.paused_all) == (False, 0.0, False)
        other = rl.report_response("a", 503, {"Retry-After": "9"})
        assert other.rate_limited is False
        assert not rl.is_paused("a")
        limited = rl.report_response("a", 429, {"retry-after": "4"})
        assert (limited.rate_limited, limited.status, limited.retry_after, limited.delay) == (
            True,
            429,
            4.0,
            4.0,
        )
        banned = rl.report_response("a", 418)
        assert banned.paused_all and banned.delay == 120.0

    def test_success_resets_streak(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1))
        rl.report_throttled("a")
        clock.advance(5)
        rl.report_response("a", 204)
        assert rl.backoff("a").streak == 0
        rl.report_success("never-seen")  # no state, no error

    def test_negative_retry_after_clamps_to_zero(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1))
        verdict = rl.report_throttled("a", -5.0)
        assert (verdict.delay, verdict.retry_after) == (0.0, 0.0)
        assert not rl.is_paused("a")

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_retry_after_is_rejected_before_the_streak(
        self, clock: ManualClock, bad: float
    ) -> None:
        rl = make(clock, RateLimit("a", 5, 1))
        with pytest.raises(ValueError, match="retry_after"):
            rl.report_throttled("a", bad)
        assert rl.backoff("a").streak == 0
        assert not rl.is_paused("a")
        assert not rl._report_lock.locked()

    def test_http_date_uses_wall_clock_not_the_limiter_clock(self, clock: ManualClock) -> None:
        wall = datetime(2030, 1, 1, tzinfo=UTC).timestamp()
        rl = RateLimiter(
            [RateLimit("a", 5, 1)], clock=clock, sleep=clock.sleep, wall_clock=lambda: wall
        )
        header = format_datetime(datetime.fromtimestamp(wall + 30, UTC), usegmt=True)
        verdict = rl.report_response("a", 429, {"Retry-After": header})
        assert verdict.retry_after == pytest.approx(30.0)
        assert verdict.delay == pytest.approx(30.0)

    def test_fixed_backoff_via_factory(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1), backoff_factory=lambda: BackoffPolicy.fixed(63))
        assert rl.report_response("a", 429).delay == 63.0


class TestProtocolsAndBinding:
    async def test_bound_limiter(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        bound = rl.bind("a")
        assert isinstance(bound, BoundLimiter)
        for proto in (RestRateLimiterProtocol, PausableRateLimiterProtocol, PoolPauseProtocol):
            assert isinstance(bound, proto)
        assert bound.limit_id == "a"
        assert bound.limiter is rl
        assert bound.backoff is rl.backoff("a")
        await bound.acquire()
        assert bound.blocked_for() == pytest.approx(10.0)
        bound.pause(30)
        assert bound.is_paused()
        await bound.acquire()
        assert clock() == pytest.approx(1030.0)

    def test_bound_paused_until(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1), RateLimit("b", 5, 1))
        bound = rl.bind("a")
        assert bound.paused_until() == 0.0
        bound.pause(50)
        assert bound.paused_until() == pytest.approx(1050.0)
        assert rl.paused_until("b") == 0.0

    def test_bound_report_throttled(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 5, 1), RateLimit("b", 5, 1))
        bound = rl.bind("a")
        verdict = bound.report_throttled(7.0)
        assert verdict.retry_after == 7.0
        assert verdict.delay == pytest.approx(7.0)
        assert bound.paused_until() == pytest.approx(1007.0)
        assert not rl.is_paused("b")
        assert bound.report_throttled(status=503).status == 503

    async def test_bound_weights(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("a", 10, 1))
        bound = rl.bind("a", {"a": 10})
        await bound.acquire()
        assert bound.blocked_for() == pytest.approx(1.0)

    def test_throttler_protocols(self, clock: ManualClock) -> None:
        rl = make(clock)
        assert isinstance(rl, ThrottlerProtocol)
        assert isinstance(rl, WeightedThrottlerProtocol)

    async def test_execute_task_is_plain_method_returning_async_cm(
        self, clock: ManualClock
    ) -> None:
        rl = make(clock, RateLimit("a", 1, 10))
        cm = rl.execute_task(limit_id="a")
        assert not asyncio.iscoroutine(cm)
        async with cm:
            pass
        with pytest.raises(RuntimeError, match="body"):
            async with rl.execute_task("a"):
                raise RuntimeError("body")
