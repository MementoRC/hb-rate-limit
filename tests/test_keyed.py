import asyncio

import pytest

from rate_limit import KeyedCooldown, RateLimitWeightError
from rate_limit.testing import BlockingSleep, ManualClock


def make(
    clock: ManualClock, limit: int, interval: float, safety_margin_pct: float = 0.0
) -> KeyedCooldown:
    return KeyedCooldown(
        limit, interval, safety_margin_pct=safety_margin_pct, clock=clock, sleep=clock.sleep
    )


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(1000.0)


class TestCooldown:
    async def test_identical_request_within_15s_waits(self, clock: ManualClock) -> None:
        kc = make(clock, 1, 15.0)
        await kc.acquire("req")
        clock.advance(10)
        assert kc.blocked_for("req") == pytest.approx(5.0)
        clock.advance(5)  # exactly 15s: the boundary is inclusive, so still blocked
        assert 0.0 < kc.try_acquire("req") < 1e-9
        clock.advance(0.001)
        assert kc.try_acquire("req") == 0.0
        assert kc.try_acquire("req") == pytest.approx(15.0)

    async def test_acquire_sleeps_once_through_the_boundary(self, clock: ManualClock) -> None:
        kc = make(clock, 1, 15.0)
        await kc.acquire("req")
        await kc.acquire("req")
        assert clock.sleeps == [pytest.approx(15.0)]
        assert clock() == pytest.approx(1015.0)

    def test_six_per_two_seconds_per_tuple_key(self, clock: ManualClock) -> None:
        kc = make(clock, 6, 2.0)
        key = (265598, "SMART", "233")
        assert [kc.try_acquire(key) for _ in range(6)] == [0.0] * 6
        assert kc.try_acquire(key) == pytest.approx(2.0)
        assert kc.try_acquire((265598, "ISLAND", "233")) == 0.0
        clock.advance(2.001)
        assert kc.try_acquire(key) == 0.0

    def test_keys_are_independent(self, clock: ManualClock) -> None:
        kc = make(clock, 1, 10)
        assert kc.try_acquire("a") == 0.0
        assert kc.try_acquire("b") == 0.0
        assert kc.try_acquire("a") == pytest.approx(10.0)
        kc.reset("a")
        assert kc.try_acquire("a") == 0.0
        assert kc.blocked_for("b") == pytest.approx(10.0)

    def test_weights(self, clock: ManualClock) -> None:
        kc = make(clock, 5, 10)
        assert kc.try_acquire("k", 3) == 0.0
        clock.advance(1)
        assert kc.try_acquire("k", 3) == pytest.approx(9.0)  # nothing recorded
        assert kc.try_acquire("k", 2) == 0.0
        assert kc.blocked_for("k", 1) == pytest.approx(9.0)
        assert kc.blocked_for("other", 5) == 0.0

    def test_weight_above_limit_raises(self, clock: ManualClock) -> None:
        kc = make(clock, 5, 10)
        with pytest.raises(RateLimitWeightError, match="weight 6 exceeds limit 5 of 'k'"):
            kc.try_acquire("k", 6)
        with pytest.raises(RateLimitWeightError):
            kc.blocked_for("k", 6)
        assert len(kc) == 0

    @pytest.mark.parametrize("weight", [0, -1, True, 1.5])
    def test_invalid_weight(self, clock: ManualClock, weight: object) -> None:
        kc = make(clock, 5, 10)
        with pytest.raises(ValueError, match="weight"):
            kc.try_acquire("k", weight)  # type: ignore[arg-type]

    async def test_acquire_rejects_weight_above_limit(self, clock: ManualClock) -> None:
        kc = make(clock, 1, 10)
        with pytest.raises(RateLimitWeightError):
            await kc.acquire("k", 2)

    def test_safety_margin_widens_the_window(self, clock: ManualClock) -> None:
        kc = make(clock, 1, 10, safety_margin_pct=0.5)
        assert kc.try_acquire("k") == 0.0
        clock.advance(14)
        assert kc.blocked_for("k") == pytest.approx(1.0)
        clock.advance(1)  # age 15 == 10 * 1.5: still counts
        assert kc.blocked_for("k") > 0.0
        clock.advance(0.001)
        assert kc.try_acquire("k") == 0.0

    def test_properties(self) -> None:
        kc = KeyedCooldown(6, 2)
        assert (kc.limit, kc.time_interval) == (6, 2.0)
        assert isinstance(kc.time_interval, float)


class TestValidation:
    @pytest.mark.parametrize("limit", [0, -1, True, 1.5])
    def test_invalid_limit(self, limit: object) -> None:
        with pytest.raises(ValueError, match="limit"):
            KeyedCooldown(limit, 1)  # type: ignore[arg-type]

    @pytest.mark.parametrize("interval", [0, -1.0, float("inf"), float("nan"), True])
    def test_invalid_interval(self, interval: object) -> None:
        with pytest.raises(ValueError, match="time_interval"):
            KeyedCooldown(1, interval)  # type: ignore[arg-type]

    @pytest.mark.parametrize("margin", [-0.1, float("inf"), float("nan")])
    def test_invalid_margin(self, margin: float) -> None:
        with pytest.raises(ValueError, match="safety_margin_pct"):
            KeyedCooldown(1, 1, safety_margin_pct=margin)


class TestMemory:
    def test_idle_keys_are_dropped(self, clock: ManualClock) -> None:
        kc = make(clock, 1, 15.0)
        for i in range(1000):
            assert kc.try_acquire(i) == 0.0
        assert len(kc) == 1000
        clock.advance(15.001)
        assert kc.try_acquire("fresh") == 0.0
        assert len(kc) == 1

    def test_boundary_key_is_kept(self, clock: ManualClock) -> None:
        kc = make(clock, 1, 10)
        kc.try_acquire("old")
        clock.advance(5)
        kc.try_acquire("new")
        clock.advance(5)  # "old" exactly at its boundary: still counts
        assert len(kc) == 2
        clock.advance(0.001)
        assert len(kc) == 1
        assert kc.blocked_for("new") == pytest.approx(4.999)

    def test_recharge_moves_a_key_to_the_back(self, clock: ManualClock) -> None:
        kc = make(clock, 2, 10)
        kc.try_acquire("a")
        kc.try_acquire("b")
        clock.advance(5)
        kc.try_acquire("a")  # "a" now outlives "b"
        clock.advance(5.001)
        assert len(kc) == 1
        assert kc.blocked_for("a") == 0.0  # one charge left of two
        assert kc.try_acquire("a") == 0.0
        assert kc.try_acquire("a") == pytest.approx(4.999)

    def test_blocked_for_creates_no_key(self, clock: ManualClock) -> None:
        kc = make(clock, 1, 10)
        assert kc.blocked_for("ghost") == 0.0
        assert len(kc) == 0
        kc.try_acquire("k")
        assert kc.try_acquire("k") > 0.0
        assert len(kc) == 1

    def test_reset(self, clock: ManualClock) -> None:
        kc = make(clock, 1, 10)
        for key in ("a", "b", "c"):
            kc.try_acquire(key)
        kc.reset("b")
        kc.reset("missing")
        assert len(kc) == 2
        assert kc.try_acquire("b") == 0.0
        kc.reset()
        assert len(kc) == 0
        assert kc.try_acquire("a") == 0.0


class TestConcurrency:
    async def test_cancelled_acquire_records_nothing(self, clock: ManualClock) -> None:
        sleeper = BlockingSleep()
        kc = KeyedCooldown(1, 10, clock=clock, sleep=sleeper)
        await kc.acquire("k")
        task = asyncio.create_task(kc.acquire("k"))
        await sleeper.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        clock.advance(10.001)
        assert kc.try_acquire("k") == 0.0  # only the first charge existed, and it expired
        assert kc.try_acquire("k") == pytest.approx(10.0)

    async def test_concurrent_acquirers_on_one_key(self, clock: ManualClock) -> None:
        kc = make(clock, 3, 10)
        stamps: list[float] = []

        async def one() -> None:
            await kc.acquire(("con", "SMART"))
            stamps.append(clock())

        await asyncio.gather(*(one() for _ in range(10)))
        stamps.sort()
        assert all(stamps[i + 3] - stamps[i] >= 10 for i in range(len(stamps) - 3))
        assert clock() - 1000 <= 4 * 10 + 1e-6  # ceil(10/3) windows

    async def test_a_sleeper_does_not_block_other_keys(self, clock: ManualClock) -> None:
        sleeper = BlockingSleep()
        kc = KeyedCooldown(1, 10, clock=clock, sleep=sleeper)
        await kc.acquire("slow")
        task = asyncio.create_task(kc.acquire("slow"))
        await sleeper.started.wait()
        await asyncio.wait_for(kc.acquire("fast"), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_no_lock_held_across_a_sleep(self, clock: ManualClock) -> None:
        kc = make(clock, 1, 10)
        seen: list[bool] = []
        clock.on_sleep = lambda _seconds: seen.append(kc._lock.locked())
        await kc.acquire("k")
        await kc.acquire("k")
        assert seen == [False]
