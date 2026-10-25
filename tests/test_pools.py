import math

import pytest

from rate_limit import SlidingWindowPool, TokenBucketPool
from rate_limit.pools import carry_over, new_pool


class TestValidation:
    @pytest.mark.parametrize(
        ("capacity", "refill"),
        [
            (0, 1),
            (-1, 1),
            (1, 0),
            (1, -1),
            (math.nan, 1),
            (1, math.nan),
            (math.inf, 1),
            (1, math.inf),
        ],
    )
    def test_capacity_and_refill_must_be_finite_and_positive(
        self, capacity: float, refill: float
    ) -> None:
        with pytest.raises(ValueError):
            TokenBucketPool(capacity, refill)

    def test_rebuilt_validates_too(self) -> None:
        with pytest.raises(ValueError):
            TokenBucketPool(1, 1).rebuilt(0, 1, 0.0)


class TestRefill:
    def test_starts_full(self) -> None:
        pool = TokenBucketPool(10, 2)
        assert (pool.capacity, pool.refill_per_second, pool.used) == (10, 2, 0)
        assert pool.wait_for(10, 0.0) == 0.0

    def test_reserve_drains_then_refuses_without_recording(self) -> None:
        pool = TokenBucketPool(10, 2)
        assert pool.reserve(10, 0.0) == 0.0
        assert pool.used == 10
        assert pool.reserve(1, 0.0) == pytest.approx(0.5)
        assert pool.used == 10

    def test_fractional_refill(self) -> None:
        pool = TokenBucketPool(10, 2)
        pool.evict(0.0)
        pool.reserve(10, 0.0)
        pool.evict(0.25)
        assert pool.used == pytest.approx(9.5)
        assert pool.wait_for(1, 0.25) == pytest.approx(0.25)

    def test_refill_is_capped_at_capacity(self) -> None:
        pool = TokenBucketPool(10, 2)
        pool.evict(0.0)
        pool.reserve(4, 0.0)
        pool.evict(1000.0)
        assert pool.used == 0

    def test_first_evict_only_sets_reference_time(self) -> None:
        pool = TokenBucketPool(10, 2)
        pool.evict(500.0)
        assert pool.used == 0
        pool.reserve(10, 500.0)
        pool.evict(501.0)
        assert pool.used == pytest.approx(8.0)

    def test_time_going_backwards_is_ignored(self) -> None:
        pool = TokenBucketPool(10, 2)
        pool.evict(10.0)
        pool.reserve(10, 10.0)
        pool.evict(5.0)
        assert pool.used == 10
        assert pool.wait_for(1, 5.0) == pytest.approx(5.5)  # measured from the latest time seen
        pool.evict(11.0)
        assert pool.used == pytest.approx(8.0)

    def test_wait_for_is_exact_and_records_nothing(self) -> None:
        pool = TokenBucketPool(10, 2)
        pool.reserve(10, 0.0)
        assert pool.wait_for(1, 0.0) == pytest.approx(0.5)
        assert pool.wait_for(3, 0.0) == pytest.approx(1.5)
        assert pool.wait_for(10, 0.0) == pytest.approx(5.0)
        assert pool.used == 10

    @pytest.mark.parametrize("start", [0.0, 0.3, 123456.789, 1e9 + 0.3])
    def test_wait_never_undershoots(self, start: float) -> None:
        pool = TokenBucketPool(10, 3.3)
        pool.evict(start)
        pool.reserve(10, start)
        wait = pool.wait_for(7, start)
        assert wait > 0
        assert start + wait >= start + 7 / 3.3 - 1e-6
        pool.evict(start + wait)
        assert pool.wait_for(7, start + wait) == 0.0

    def test_wait_is_never_zero_when_refused(self) -> None:
        pool = TokenBucketPool(1, 1e12)
        pool.evict(1e9)
        pool.reserve(1, 1e9)
        wait = pool.wait_for(1, 1e9)
        assert wait > 0
        assert 1e9 + wait > 1e9


class TestLiveness:
    @pytest.mark.parametrize(
        ("capacity", "refill", "cost", "start"),
        [
            (10, 0.1, 1, 1e9 + 0.3),
            (3, 7 / 3, 2, 123456.789),
            (100, 33.3, 7, 5e8 + 0.1),
            (5, 1 / 3, 5, 0.1),
            (1000, 0.7, 999, 3e7 + 0.77),
        ],
    )
    def test_reaching_the_deadline_always_admits(
        self, capacity: int, refill: float, cost: int, start: float
    ) -> None:
        pool = TokenBucketPool(capacity, refill)
        pool.evict(start)
        pool.reserve(capacity, start)
        wait = pool.wait_for(cost, start)
        assert wait > 0
        deadline = start + wait
        pool.evict(deadline, expired_through=deadline)
        assert pool.wait_for(cost, deadline) == 0.0

    @pytest.mark.parametrize(("refill", "start"), [(0.1, 1e9 + 0.3), (7 / 3, 12345.6789)])
    def test_paced_loop_converges_in_one_wait_per_charge(self, refill: float, start: float) -> None:
        pool = TokenBucketPool(3, refill)
        now = start
        through = -math.inf
        waits = 0
        for _ in range(200):
            pool.evict(now, through)
            wait = pool.wait_for(2, now)
            if wait == 0.0:
                pool.reserve(2, now)
                through = -math.inf
                continue
            waits += 1
            through = now + wait
            now = through
            pool.evict(now, through)
            assert pool.wait_for(2, now) == 0.0
            pool.reserve(2, now)
            through = -math.inf
        assert waits >= 190


class TestRebuilt:
    def test_carries_the_deficit(self) -> None:
        pool = TokenBucketPool(10, 2)
        pool.evict(0.0)
        pool.reserve(7, 0.0)
        rebuilt = pool.rebuilt(20, 4, 0.0)
        assert (rebuilt.capacity, rebuilt.refill_per_second, rebuilt.used) == (20, 4, 7)
        assert rebuilt.wait_for(13, 0.0) == 0.0
        assert rebuilt.wait_for(14, 0.0) == pytest.approx(0.25)

    def test_refills_up_to_now_first(self) -> None:
        pool = TokenBucketPool(10, 2)
        pool.evict(0.0)
        pool.reserve(10, 0.0)
        assert pool.rebuilt(10, 2, 2.0).used == pytest.approx(6.0)

    def test_smaller_capacity_leaves_negative_tokens(self) -> None:
        pool = TokenBucketPool(10, 1)
        pool.evict(0.0)
        pool.reserve(8, 0.0)
        rebuilt = pool.rebuilt(5, 1, 0.0)
        assert rebuilt.used == 8
        assert rebuilt.wait_for(1, 0.0) == pytest.approx(4.0)


class TestCarryOver:
    def test_window_to_bucket_keeps_used(self) -> None:
        old = SlidingWindowPool(10, 10.0)
        old.reserve(4, 0.0)
        pool = carry_over(old, "token_bucket", 10, 10.0, 0.0, 0.0)
        assert isinstance(pool, TokenBucketPool)
        assert pool.used == 4

    def test_bucket_to_window_seeds_a_rounded_up_charge_at_now(self) -> None:
        old = TokenBucketPool(5, 1)
        old.evict(0.0)
        old.reserve(5, 0.0)
        pool = carry_over(old, "window", 5, 10.0, 0.0, 2.5)
        assert isinstance(pool, SlidingWindowPool)
        assert pool.used == 3
        assert pool.wait_for(3, 2.5) == pytest.approx(10.0)

    def test_idle_bucket_seeds_nothing(self) -> None:
        pool = carry_over(TokenBucketPool(5, 1), "window", 5, 10.0, 0.0, 0.0)
        assert pool.used == 0

    def test_same_kind_rebuilds(self) -> None:
        old = new_pool("window", 2, 10.0, 0.0)
        old.reserve(1, 0.0)
        pool = carry_over(old, "window", 4, 10.0, 0.0, 1.0)
        assert isinstance(pool, SlidingWindowPool)
        assert (pool.capacity, pool.used) == (4, 1)


class TestWindowRefund:
    def test_removes_the_exact_entry(self) -> None:
        pool = SlidingWindowPool(10, 10.0)
        pool.reserve(3, 0.0)
        pool.reserve(2, 1.0)
        assert pool.refund(3, 0.0, 2.0) == 3.0
        assert pool.used == 2
        assert pool.wait_for(9, 2.0) == pytest.approx(9.0)  # only the 1.0 entry is left

    def test_duplicate_stamps_lose_one_entry_per_refund(self) -> None:
        pool = SlidingWindowPool(10, 10.0)
        pool.reserve(2, 0.0)
        pool.reserve(2, 0.0)
        assert pool.refund(2, 0.0, 0.0) == 2.0
        assert pool.used == 2
        assert pool.refund(2, 0.0, 0.0) == 2.0
        assert pool.refund(2, 0.0, 0.0) == 0.0
        assert pool.used == 0

    def test_needs_both_stamp_and_cost_to_match(self) -> None:
        pool = SlidingWindowPool(10, 10.0)
        pool.reserve(2, 0.0)
        assert pool.refund(3, 0.0, 0.0) == 0.0
        assert pool.refund(2, 0.5, 0.0) == 0.0
        assert pool.used == 2

    def test_charge_at_the_boundary_still_refunds(self) -> None:
        pool = SlidingWindowPool(10, 10.0)
        pool.reserve(4, 0.0)
        assert pool.refund(4, 0.0, 10.0) == 4.0  # age == window still counts
        assert pool.used == 0

    def test_expired_charge_returns_nothing(self) -> None:
        pool = SlidingWindowPool(10, 10.0)
        pool.reserve(4, 0.0)
        pool.reserve(4, 10.001)
        assert pool.refund(4, 0.0, 10.001) == 0.0
        assert pool.used == 4

    def test_survives_a_same_kind_rebuild(self) -> None:
        pool = SlidingWindowPool(2, 10.0)
        pool.reserve(1, 0.0)
        rebuilt = carry_over(pool, "window", 5, 10.0, 0.0, 1.0)
        assert rebuilt.refund(1, 0.0, 1.0) == 1.0
        assert rebuilt.used == 0

    def test_bucket_carry_over_replaces_the_entry(self) -> None:
        old = TokenBucketPool(5, 1)
        old.evict(0.0)
        old.reserve(2, 0.0)
        pool = carry_over(old, "window", 5, 10.0, 0.0, 0.5)  # one charge of ceil(1.5) == 2
        assert pool.refund(2, 0.0, 0.5) == 0.0
        assert pool.used == 2


class TestBucketRefund:
    def test_returns_the_cost(self) -> None:
        pool = TokenBucketPool(10, 1)
        pool.evict(0.0)
        pool.reserve(4, 0.0)
        assert pool.refund(4, 0.0, 0.0) == 4.0
        assert pool.used == 0

    def test_credits_only_the_unrefilled_part(self) -> None:
        pool = TokenBucketPool(10, 1)
        pool.evict(0.0)
        pool.reserve(4, 0.0)
        pool.reserve(4, 2.0)  # deficit 6: 2 of the first charge refilled meanwhile
        assert pool.refund(4, 0.0, 2.0) == pytest.approx(2.0)
        assert pool.used == pytest.approx(4.0)

    def test_never_rises_above_capacity(self) -> None:
        pool = TokenBucketPool(10, 1)
        pool.evict(0.0)
        pool.reserve(4, 0.0)
        assert pool.refund(4, 0.0, 3.0) == pytest.approx(1.0)  # 3 of the 4 already refilled
        assert pool.used == 0
        assert pool.refund(4, 0.0, 3.0) == 0.0
        assert pool.used == 0
        assert pool.wait_for(10, 3.0) == 0.0

    def test_charge_lost_to_clamping_credits_nothing(self) -> None:
        pool = TokenBucketPool(10, 1)
        pool.evict(0.0)
        pool.reserve(5, 0.0)
        pool.evict(5.0)  # refilled to capacity: the first charge is gone
        pool.reserve(8, 5.0)
        assert pool.refund(5, 0.0, 5.0) == 0.0
        assert pool.used == 8


class TestNewPool:
    def test_bucket_refill_is_limit_over_margined_interval(self) -> None:
        pool = new_pool("token_bucket", 10, 5.0, 0.25)
        assert isinstance(pool, TokenBucketPool)
        assert pool.capacity == 10
        assert pool.refill_per_second == pytest.approx(10 / (5.0 * 1.25))

    def test_window_kind(self) -> None:
        assert isinstance(new_pool("window", 10, 5.0, 0.0), SlidingWindowPool)
