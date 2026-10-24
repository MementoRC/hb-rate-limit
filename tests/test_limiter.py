import asyncio
import logging
import threading
from decimal import Decimal

import pytest

from rate_limit import (
    BackoffPolicy,
    BoundLimiter,
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
from rate_limit.testing import ManualClock


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

    async def test_set_limits_keeps_history_only_for_unchanged(self, clock: ManualClock) -> None:
        rl = make(clock, RateLimit("keep", 1, 10), RateLimit("change", 1, 10))
        await rl.acquire("keep")
        await rl.acquire("change")
        rl.set_limits(
            [RateLimit("keep", 1, 10), RateLimit("change", 2, 10), RateLimit("new", 1, 1)]
        )
        assert not rl.has_limit("zzz")
        assert rl.try_acquire("keep") > 0  # history kept
        assert rl.try_acquire("change") == 0.0  # fresh pool
        assert rl.try_acquire("new") == 0.0
        rl.set_limits([RateLimit("only", 1, 1)])
        assert not rl.has_limit("keep")


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
