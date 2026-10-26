import logging

import pytest

from rate_limit import (
    LinkedLimitWeightPair,
    RateLimit,
    RateLimiter,
    RateLimitWeightError,
    UnknownRateLimitError,
)
from rate_limit.testing import ManualClock


def make(clock: ManualClock, *limits: RateLimit, **kw: object) -> RateLimiter:
    return RateLimiter(limits, clock=clock, sleep=clock.sleep, **kw)  # type: ignore[arg-type]


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(1000.0)


@pytest.fixture
def rl(clock: ManualClock) -> RateLimiter:
    return make(
        clock,
        RateLimit("ep", 5, 10, linked_limits=[LinkedLimitWeightPair("pool", 2)]),
        RateLimit("pool", 10, 10),
        RateLimit("x", 3, 10),
        RateLimit("y", 4, 10),
    )


def used(limiter: RateLimiter, pool_id: str) -> float:
    return limiter._pools[pool_id].used


class TestZeroWeightLinkedPool:
    async def test_zero_skips_linked_pool(self, rl: RateLimiter) -> None:
        charge = await rl.reserve("ep", pools={"pool": 0})
        assert dict(charge.costs) == {"ep": 1}
        assert used(rl, "ep") == 1
        assert used(rl, "pool") == 0

    async def test_nonzero_replaces_linked_total(self, rl: RateLimiter) -> None:
        charge = await rl.reserve("ep", pools={"pool": 5})
        assert dict(charge.costs) == {"ep": 1, "pool": 5}

    async def test_primary_cannot_be_zeroed(self, rl: RateLimiter) -> None:
        with pytest.raises(ValueError):
            await rl.acquire("ep", {"ep": 0})
        with pytest.raises(ValueError, match="weights"):
            await rl.acquire("ep", pools={"ep": 1})

    async def test_parity_without_pools(self, rl: RateLimiter) -> None:
        charge = await rl.reserve("ep")
        assert dict(charge.costs) == {"ep": 1, "pool": 2}
        charge = await rl.reserve("ep", {"pool": 3})
        assert dict(charge.costs) == {"ep": 1, "pool": 3}


class TestAdHocPools:
    async def test_charges_unlinked_pools_and_registers_nothing(self, rl: RateLimiter) -> None:
        before = dict(rl.limits())
        charge = await rl.reserve("ep", pools={"x": 1, "y": 2})
        assert dict(charge.costs) == {"ep": 1, "pool": 2, "x": 1, "y": 2}
        assert (used(rl, "x"), used(rl, "y")) == (1, 2)
        assert dict(rl.limits()) == before

    async def test_many_combinations_keep_registry_constant(self, clock: ManualClock) -> None:
        rl = make(
            clock, RateLimit("ep", 1000, 10), RateLimit("x", 1000, 10), RateLimit("y", 1000, 10)
        )
        for i in range(1, 20):
            await rl.acquire("ep", pools={"x": i % 3, "y": i % 5})
            rl.try_acquire("ep", pools={"y": i})
            assert rl.blocked_for("ep", pools={"x": i}) == 0.0
        assert len(rl.limits()) == 3

    def test_all_or_nothing_try(self, rl: RateLimiter) -> None:
        assert rl.try_acquire("ep", pools={"x": 3}) == 0.0
        assert rl.try_acquire("ep", pools={"y": 1, "x": 1}) > 0
        assert used(rl, "ep") == 1
        assert used(rl, "y") == 0
        assert used(rl, "pool") == 2
        assert used(rl, "x") == 3

    async def test_all_or_nothing_waits(self, rl: RateLimiter, clock: ManualClock) -> None:
        await rl.acquire("ep", pools={"x": 3})
        await rl.acquire("ep", pools={"x": 1, "y": 1})
        assert clock.sleeps == [pytest.approx(10.0)]
        assert used(rl, "y") == 1

    async def test_capacity_exceeded(self, rl: RateLimiter, clock: ManualClock) -> None:
        with pytest.raises(RateLimitWeightError):
            await rl.acquire("ep", pools={"x": 4})
        assert clock.sleeps == []
        assert used(rl, "ep") == 0

    async def test_pause_on_ad_hoc_pool_blocks(self, rl: RateLimiter) -> None:
        rl.pause("x", 50)
        assert rl.try_acquire("ep") == 0.0
        assert rl.try_acquire("ep", pools={"x": 1}) == pytest.approx(50.0)
        assert rl.blocked_for("ep", pools={"x": 1}) == pytest.approx(50.0)
        assert rl.try_acquire("ep", pools={"x": 0}) == 0.0  # not charged: pause ignored

    async def test_zero_weight_declared_linked_pause_still_applies(self, rl: RateLimiter) -> None:
        rl.pause("pool", 50)
        assert rl.try_acquire("ep", pools={"pool": 0}) == pytest.approx(50.0)

    async def test_refund_returns_ad_hoc_pools(self, rl: RateLimiter) -> None:
        charge = await rl.reserve("ep", pools={"x": 3, "y": 0})
        assert "y" not in charge.costs
        assert rl.try_acquire("ep", pools={"x": 1}) > 0
        charge.refund()
        assert used(rl, "x") == 0
        assert rl.try_acquire("ep", pools={"x": 3}) == 0.0


class TestPoolsValidation:
    @pytest.mark.parametrize("bad", [-1, True, 1.5, "2", None])
    async def test_bad_values(self, rl: RateLimiter, bad: object) -> None:
        with pytest.raises(ValueError, match="pools"):
            await rl.acquire("ep", pools={"x": bad})  # type: ignore[dict-item]

    async def test_key_in_weights_and_pools(self, rl: RateLimiter) -> None:
        with pytest.raises(ValueError, match="ambiguous"):
            await rl.acquire("ep", {"pool": 1}, pools={"pool": 1})

    async def test_unknown_key_raises(self, rl: RateLimiter, clock: ManualClock) -> None:
        with pytest.raises(UnknownRateLimitError):
            await rl.acquire("ep", pools={"ghost": 1})
        assert used(rl, "ep") == 0

    async def test_unknown_key_allow_mode_warns_once_and_skips(
        self, clock: ManualClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        rl = make(clock, RateLimit("ep", 5, 10), unknown="allow")
        with caplog.at_level(logging.WARNING, logger="rate_limit.limiter"):
            charge = await rl.reserve("ep", pools={"ghost": 1})
            await rl.acquire("ep", pools={"ghost": 1})
        assert dict(charge.costs) == {"ep": 1}
        assert len([r for r in caplog.records if "ghost" in r.message]) == 1

    async def test_validated_even_when_primary_unknown(self, clock: ManualClock) -> None:
        rl = make(clock, unknown="allow")
        with pytest.raises(ValueError):
            await rl.acquire("nope", pools={"x": -1})
        await rl.acquire("nope", pools={"x": 1})

    def test_execute_task_validates_eagerly(self, rl: RateLimiter) -> None:
        with pytest.raises(ValueError):
            rl.execute_task("ep", pools={"x": -1})


class TestPoolsPlumbing:
    async def test_execute_task(self, rl: RateLimiter) -> None:
        async with rl.execute_task("ep", pools={"x": 2}):
            pass
        assert used(rl, "x") == 2

    async def test_execute_task_refund_on_error(self, rl: RateLimiter) -> None:
        with pytest.raises(RuntimeError):
            async with rl.execute_task("ep", pools={"x": 2}, refund_on_error=True):
                raise RuntimeError
        assert used(rl, "x") == 0

    async def test_bound_pools(self, rl: RateLimiter) -> None:
        bound = rl.bind("ep", pools={"y": 2})
        assert bound.blocked_for() == 0.0
        charge = await bound.reserve()
        await bound.acquire()
        assert dict(charge.costs) == {"ep": 1, "pool": 2, "y": 2}
        assert used(rl, "y") == 4
        assert bound.blocked_for() > 0

    async def test_bound_weights_and_pools(self, rl: RateLimiter) -> None:
        bound = rl.bind("ep", {"pool": 4}, pools={"y": 1})
        charge = await bound.reserve()
        assert dict(charge.costs) == {"ep": 1, "pool": 4, "y": 1}
