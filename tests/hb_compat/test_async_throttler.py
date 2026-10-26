import copy
import logging
import math
from decimal import Decimal
from unittest.mock import patch

import pytest

from rate_limit.hb_compat.async_throttler import AsyncThrottler
from rate_limit.hb_compat.data_types import LinkedLimitWeightPair, RateLimit

METRICS_LOG_LEVEL = 21

TEST_PATH_URL = "/hummingbot"
TEST_POOL_ID = "TEST"
TEST_WEIGHTED_POOL_ID = "TEST_WEIGHTED"
TEST_WEIGHTED_TASK_1_ID = "/weighted_task_1"
TEST_WEIGHTED_TASK_2_ID = "/weighted_task_2"

logging.basicConfig(level=METRICS_LOG_LEVEL)


@pytest.fixture
def rate_limits() -> list[RateLimit]:
    return [
        RateLimit(limit_id=TEST_POOL_ID, limit=1, time_interval=5.0),
        RateLimit(
            limit_id=TEST_PATH_URL,
            limit=1,
            time_interval=5.0,
            linked_limits=[LinkedLimitWeightPair(TEST_POOL_ID)],
        ),
        RateLimit(limit_id=TEST_WEIGHTED_POOL_ID, limit=10, time_interval=5.0),
        RateLimit(
            limit_id=TEST_WEIGHTED_TASK_1_ID,
            limit=1000,
            time_interval=5.0,
            linked_limits=[LinkedLimitWeightPair(TEST_WEIGHTED_POOL_ID, 5)],
        ),
        RateLimit(
            limit_id=TEST_WEIGHTED_TASK_2_ID,
            limit=1000,
            time_interval=5.0,
            linked_limits=[LinkedLimitWeightPair(TEST_WEIGHTED_POOL_ID, 1)],
        ),
    ]


@pytest.fixture
def throttler(rate_limits: list[RateLimit]) -> AsyncThrottler:
    return AsyncThrottler(rate_limits=rate_limits)


def test_init_without_rate_limits_share_pct(throttler):
    assert throttler._retry_interval == 0.1
    assert len(throttler._rate_limits) == 5
    assert throttler._id_to_limit_map[TEST_POOL_ID].limit == 1
    assert throttler._id_to_limit_map[TEST_PATH_URL].limit == 1


def test_init_with_rate_limits_share_pct(rate_limits):
    rate_share_pct: Decimal = Decimal("55")
    AsyncThrottler(rate_limits=rate_limits, limits_share_percentage=rate_share_pct)

    rate_limits = rate_limits.copy()
    rate_limits.append(RateLimit(limit_id="ANOTHER_TEST", limit=10, time_interval=5))
    expected_limit = math.floor(Decimal("10") * rate_share_pct / Decimal("100"))

    throttler = AsyncThrottler(rate_limits=rate_limits, limits_share_percentage=rate_share_pct)
    assert throttler._retry_interval == 0.1
    assert len(throttler._rate_limits) == 6
    assert Decimal("1") == throttler._id_to_limit_map[TEST_POOL_ID].limit
    assert Decimal("1") == throttler._id_to_limit_map[TEST_PATH_URL].limit
    assert expected_limit == throttler._id_to_limit_map["ANOTHER_TEST"].limit


def test_get_related_limits(throttler):
    assert len(throttler._rate_limits) == 5

    rate_limit, related_limits = throttler.get_related_limits(TEST_POOL_ID)
    assert rate_limit.limit_id == TEST_POOL_ID
    assert len(related_limits) == 0

    rate_limit, related_limits = throttler.get_related_limits(TEST_PATH_URL)
    assert rate_limit.limit_id == TEST_PATH_URL
    assert len(related_limits) == 1


def test_has_limit_and_get_limit_return_stored_scaled_limit():
    throttler = AsyncThrottler(
        rate_limits=[RateLimit("known", limit=10, time_interval=5.0)],
        limits_share_percentage=Decimal("55"),
    )

    assert throttler.has_limit("known") is True
    assert throttler.has_limit("nope") is False
    assert throttler.get_limit("nope") is None
    stored = throttler.get_limit("known")
    assert stored is throttler._id_to_limit_map["known"]
    assert stored is not None
    assert stored.limit == 5


def test_rate_limits_assignment_rebuilds_id_to_limit_map():
    throttler = AsyncThrottler(
        rate_limits=[RateLimit("old", limit=10, time_interval=5.0)],
        limits_share_percentage=Decimal("50"),
    )

    throttler._rate_limits = [RateLimit("new", limit=10, time_interval=5.0)]

    assert list(throttler._id_to_limit_map) == ["new"]
    assert [limit.limit_id for limit in throttler._rate_limits] == ["new"]
    assert throttler._id_to_limit_map["new"].limit == 5
    assert throttler._rate_limits[0].limit == 10
    assert throttler._id_to_limit_map["new"] is not throttler._rate_limits[0]


def test_direct_rate_limits_assignment_scales_once_and_leaves_source_untouched():
    tier_limits = [
        RateLimit(limit_id="A", limit=100, time_interval=60),
        RateLimit(
            limit_id="B",
            limit=10,
            time_interval=1,
            linked_limits=[LinkedLimitWeightPair("A", 2)],
        ),
    ]
    snapshot = copy.deepcopy(tier_limits)
    throttler = AsyncThrottler(rate_limits=[], limits_share_percentage=Decimal("50"))

    throttler._rate_limits = tier_limits
    throttler._rate_limits = tier_limits

    assert throttler.get_limit("A").limit == 50
    assert throttler.get_limit("B").limit == 5
    assert set(throttler._id_to_limit_map) == {"A", "B"}
    for stored in throttler._rate_limits:
        assert throttler._id_to_limit_map[stored.limit_id] is not stored
        assert all(stored is not source for source in tier_limits)

    for source, before in zip(tier_limits, snapshot, strict=True):
        assert source.limit_id == before.limit_id
        assert source.limit == before.limit
        assert source.time_interval == before.time_interval
        assert source.weight == before.weight
        assert source.linked_limits == before.linked_limits


def test_rate_limits_getter_returns_fresh_raw_copies_and_deepcopy_works():
    throttler = AsyncThrottler(rate_limits=[RateLimit("a", limit=3, time_interval=1.0)])

    assert throttler._rate_limits is not throttler._rate_limits
    throttler.add_rate_limits([RateLimit("b", limit=2, time_interval=1.0)])
    assert [limit.limit_id for limit in throttler._rate_limits] == ["a", "b"]

    copied = copy.deepcopy(throttler._rate_limits)
    assert [limit.limit_id for limit in copied] == ["a", "b"]
    assert copied[0] is not throttler._rate_limits[0]


def test_raw_limits_hold_unscaled_copies_and_set_rate_limits_replaces_them():
    source = RateLimit("a", limit=10, time_interval=5.0)
    throttler = AsyncThrottler(rate_limits=[source], limits_share_percentage=Decimal("50"))

    assert throttler._raw_limits["a"].limit == 10
    assert throttler._raw_limits["a"] is not source
    assert throttler._id_to_limit_map["a"].limit == 5

    throttler.add_rate_limits([RateLimit("b", limit=8, time_interval=1.0)])
    assert throttler._raw_limits["b"].limit == 8
    assert throttler._id_to_limit_map["b"].limit == 4

    throttler.set_rate_limits([RateLimit("c", limit=6, time_interval=1.0)])
    assert list(throttler._raw_limits) == ["c"]


def test_add_rate_limits_identical_readd_is_noop_and_mismatch_keeps_existing_state_untouched():
    throttler = AsyncThrottler(rate_limits=[RateLimit("a", limit=10, time_interval=5.0)])
    stored = throttler._id_to_limit_map["a"]

    with patch.object(AsyncThrottler, "logger") as logger_mock:
        throttler.add_rate_limits([RateLimit("a", limit=10, time_interval=5.0)])
        assert logger_mock.return_value.warning.call_count == 0
        throttler.add_rate_limits([RateLimit("a", limit=99, time_interval=5.0)])
        assert logger_mock.return_value.warning.call_count == 1

    assert throttler._id_to_limit_map["a"] is stored
    assert stored.limit == 10
    assert throttler._raw_limits["a"].limit == 10
    assert len(throttler._rate_limits) == 1


def _weighted_throttler() -> AsyncThrottler:
    return AsyncThrottler(
        rate_limits=[
            RateLimit("pool", limit=10, time_interval=5.0),
            RateLimit("other", limit=10, time_interval=5.0),
            RateLimit(
                "ep",
                limit=100,
                time_interval=5.0,
                weight=2,
                linked_limits=[LinkedLimitWeightPair("pool", 1), LinkedLimitWeightPair("other", 1)],
            ),
        ]
    )


@pytest.mark.parametrize("bad_key", ["nope", "unregistered"])
def test_weights_unknown_key_raises_value_error(bad_key):
    throttler = AsyncThrottler(
        rate_limits=[
            RateLimit(
                "ep",
                limit=10,
                time_interval=1.0,
                linked_limits=[LinkedLimitWeightPair("unregistered", 1)],
            ),
            RateLimit("unrelated", limit=10, time_interval=1.0),
        ]
    )
    with pytest.raises(ValueError):
        throttler.execute_task("ep", weights={bad_key: 1})
    with pytest.raises(ValueError):
        throttler.execute_task("ep", weights={"unrelated": 1})


def test_weights_for_unknown_primary_raises_value_error():
    throttler = AsyncThrottler(rate_limits=[])
    with pytest.raises(ValueError):
        throttler.execute_task("nope", weights={"nope": 1})


@pytest.mark.parametrize("bad_weight", [0, -1, True, False, 1.5, "2"])
def test_weights_invalid_values_raise_value_error(bad_weight):
    throttler = _weighted_throttler()
    with pytest.raises(ValueError):
        throttler.execute_task("ep", weights={"pool": bad_weight})


def _scaled_by_id(throttler: AsyncThrottler) -> dict[str, int]:
    return {limit_id: limit.limit for limit_id, limit in throttler._id_to_limit_map.items()}


def test_limits_share_percentage_zero_none_and_hundred_are_identical():
    rate_limits = [
        RateLimit("a", limit=10, time_interval=1.0),
        RateLimit("b", limit=1, time_interval=1.0),
    ]
    full = _scaled_by_id(AsyncThrottler(rate_limits, limits_share_percentage=Decimal("100")))

    assert full == {"a": 10, "b": 1}
    assert _scaled_by_id(AsyncThrottler(rate_limits, limits_share_percentage=None)) == full
    assert _scaled_by_id(AsyncThrottler(rate_limits, limits_share_percentage=Decimal("0"))) == full
    assert _scaled_by_id(AsyncThrottler(rate_limits)) == full


def test_limits_share_percentage_scaling_is_floored_with_minimum_of_one():
    rate_limits = [
        RateLimit("a", limit=10, time_interval=1.0),
        RateLimit("b", limit=1, time_interval=1.0),
        RateLimit("c", limit=7, time_interval=1.0),
    ]
    throttler = AsyncThrottler(rate_limits, limits_share_percentage=Decimal("55"))

    assert _scaled_by_id(throttler) == {"a": 5, "b": 1, "c": 3}


async def test_unknown_limit_id_warn_state_is_per_throttler_instance():
    first = AsyncThrottler(rate_limits=[])
    second = AsyncThrottler(rate_limits=[])

    with patch.object(AsyncThrottler, "logger") as logger_mock:
        async with first.execute_task("nope"):
            pass
        async with second.execute_task("nope"):
            pass

    assert logger_mock.return_value.warning.call_count == 2
