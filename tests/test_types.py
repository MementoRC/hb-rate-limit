from dataclasses import FrozenInstanceError
from decimal import Decimal
from types import SimpleNamespace

import pytest

from rate_limit import (
    LinkedLimitWeightPair,
    RateLimit,
    SlidingWindowPool,
    coerce_limit,
    scale_limit,
)


def test_defaults_and_equality() -> None:
    a = RateLimit("a", 5, 1)
    assert a.weight == 1
    assert a.linked_limits == ()
    assert a.time_interval == 1.0
    assert a == RateLimit("a", 5, 1.0)
    assert hash(a) == hash(RateLimit("a", 5, 1.0))


def test_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        RateLimit("a", 5, 1).limit = 6  # type: ignore[misc]


def test_linked_limits_become_tuple_of_pairs() -> None:
    legacy = SimpleNamespace(limit_id="x", weight=3)
    rl = RateLimit("a", 5, 1, linked_limits=[LinkedLimitWeightPair("y", 2), legacy])
    assert rl.linked_limits == (LinkedLimitWeightPair("y", 2), LinkedLimitWeightPair("x", 3))


def test_linked_order_matters_for_equality() -> None:
    p, q = LinkedLimitWeightPair("p"), LinkedLimitWeightPair("q")
    assert RateLimit("a", 5, 1, linked_limits=[p, q]) != RateLimit("a", 5, 1, linked_limits=[q, p])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit_id": "", "limit": 1, "time_interval": 1},
        {"limit_id": "a", "limit": 0, "time_interval": 1},
        {"limit_id": "a", "limit": 1.5, "time_interval": 1},
        {"limit_id": "a", "limit": True, "time_interval": 1},
        {"limit_id": "a", "limit": 1, "time_interval": 0},
        {"limit_id": "a", "limit": 1, "time_interval": float("inf")},
        {"limit_id": "a", "limit": 1, "time_interval": float("nan")},
        {"limit_id": "a", "limit": 1, "time_interval": "1"},
        {"limit_id": "a", "limit": 1, "time_interval": 1, "weight": 0},
    ],
)
def test_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RateLimit(**kwargs)  # type: ignore[arg-type]


def test_linked_weight_must_be_ge1() -> None:
    with pytest.raises(ValueError):
        LinkedLimitWeightPair("x", 0)


def test_per_minute_and_per_second() -> None:
    assert RateLimit.per_minute("m", 10, weight=2) == RateLimit("m", 10, 60, weight=2)
    assert RateLimit.per_second("s", 3) == RateLimit("s", 3, 1)


@pytest.mark.parametrize(
    ("raw", "pct", "expected"),
    [(10, "1", 10), (10, "0.5", 5), (10, "0.55", 5), (3, "0.1", 1), (1, "0.01", 1), (7, "0.3", 2)],
)
def test_scale_limit_floors_with_min_one(raw: int, pct: str, expected: int) -> None:
    assert scale_limit(raw, Decimal(pct)) == expected


def test_scaled_returns_new_object() -> None:
    rl = RateLimit("a", 10, 2, weight=3, linked_limits=[LinkedLimitWeightPair("b", 2)])
    scaled = rl.scaled(Decimal("0.5"))
    assert scaled == RateLimit("a", 5, 2, weight=3, linked_limits=[LinkedLimitWeightPair("b", 2)])
    assert rl.limit == 10


def test_coerce_limit_identity_and_structural() -> None:
    rl = RateLimit("a", 5, 1)
    assert coerce_limit(rl) is rl
    legacy = SimpleNamespace(
        limit_id="a",
        limit=5,
        time_interval=1.0,
        weight=1,
        linked_limits=[SimpleNamespace(limit_id="b", weight=2)],
    )
    assert coerce_limit(legacy) == RateLimit(
        "a", 5, 1, linked_limits=[LinkedLimitWeightPair("b", 2)]
    )


def test_sliding_window_pool_basics() -> None:
    pool = SlidingWindowPool(2, 10.0)
    assert pool.capacity == 2
    assert pool.reserve(1, 0.0) == 0.0
    assert pool.reserve(1, 1.0) == 0.0
    assert pool.used == 2
    assert pool.reserve(1, 2.0) == pytest.approx(8.0)
    assert pool.used == 2
    pool.evict(10.5)
    assert pool.used == 1


def test_sliding_window_pool_boundary_is_inclusive() -> None:
    pool = SlidingWindowPool(1, 10.0)
    pool.reserve(1, 0.0)
    pool.evict(10.0)
    assert pool.used == 1
    assert pool.wait_for(1, 10.0) > 0
    pool.evict(10.0, expired_through=10.0)
    assert pool.used == 0


def test_sliding_window_pool_margin_extends_window() -> None:
    pool = SlidingWindowPool(1, 10.0, margin_pct=0.1)
    pool.reserve(1, 0.0)
    assert pool.wait_for(1, 5.0) == pytest.approx(6.0)


def test_sliding_window_pool_weighted_wait_walks_deque() -> None:
    pool = SlidingWindowPool(10, 10.0)
    pool.reserve(4, 0.0)
    pool.reserve(4, 3.0)
    pool.reserve(2, 5.0)
    # need 5: free after the first two entries expire (4+4 removed -> 2 + 5 <= 10)
    assert pool.wait_for(5, 6.0) == pytest.approx(7.0)
    # need 10: everything must expire
    assert pool.wait_for(10, 6.0) == pytest.approx(9.0)
