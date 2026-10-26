from unittest.mock import patch

import pytest

from rate_limit.hb_compat.async_throttler import AsyncThrottler
from rate_limit.hb_compat.data_types import LinkedLimitWeightPair, RateLimit


def make(limit_id: str = "a", **overrides) -> RateLimit:
    fields = {"limit": 10, "time_interval": 5.0}
    fields.update(overrides)
    return RateLimit(limit_id, **fields)


def test_attach_connector_readd_keeps_connector_definition_and_warns_once():
    # Parent CandlesBase.attach_connector re-adds the candles feed limits to a connector throttler.
    throttler = AsyncThrottler(
        rate_limits=[RateLimit(limit_id="REQUEST_WEIGHT", limit=2400, time_interval=60)]
    )
    candles = [
        RateLimit(limit_id="REQUEST_WEIGHT", limit=1200, time_interval=60),
        RateLimit(
            limit_id="candles_ep",
            limit=1200,
            time_interval=60,
            linked_limits=[LinkedLimitWeightPair("REQUEST_WEIGHT", 2)],
        ),
    ]

    with patch.object(AsyncThrottler, "logger") as logger_mock:
        throttler.add_rate_limits(candles)
        throttler.add_rate_limits(candles)

    assert throttler.get_limit("REQUEST_WEIGHT").limit == 2400
    assert throttler.has_limit("candles_ep")
    warning = logger_mock.return_value.warning
    assert warning.call_count == 1
    assert "REQUEST_WEIGHT" in warning.call_args.args[0]


def add_capturing_warnings(throttler: AsyncThrottler, limits: list[RateLimit]) -> list[str]:
    with patch.object(AsyncThrottler, "logger") as logger_mock:
        throttler.add_rate_limits(limits)
    return [call.args[0] for call in logger_mock.return_value.warning.call_args_list]


def test_mismatched_readd_keeps_existing_and_warns_once_without_raising():
    throttler = AsyncThrottler(rate_limits=[make("a")])

    first = add_capturing_warnings(throttler, [make("a", limit=99)])
    second = add_capturing_warnings(throttler, [make("a", limit=99)])

    assert len(first) == 1
    assert "'a'" in first[0]
    assert "limit=10" in first[0]
    assert "limit=99" in first[0]
    assert "keeping the existing" in first[0]
    assert second == []
    assert throttler.get_limit("a").limit == 10


@pytest.mark.parametrize(
    "override",
    [
        {"limit": 11},
        {"time_interval": 6.0},
        {"weight": 2},
        {"linked_limits": [LinkedLimitWeightPair("p", 1)]},
    ],
)
def test_every_raw_field_difference_keeps_existing_and_warns(override):
    throttler = AsyncThrottler(rate_limits=[make("a")])

    warnings = add_capturing_warnings(throttler, [make("a", **override)])

    assert len(warnings) == 1
    assert throttler._raw_limits["a"].limit == 10
    assert throttler._raw_limits["a"].time_interval == 5.0
    assert throttler._raw_limits["a"].weight == 1
    assert throttler._raw_limits["a"].linked_limits == []


def test_mismatch_is_detected_against_raw_not_scaled_definition():
    from decimal import Decimal

    throttler = AsyncThrottler(rate_limits=[make("a")], limits_share_percentage=Decimal("50"))

    # identical raw: silent no-op even though stored is scaled
    assert add_capturing_warnings(throttler, [make("a")]) == []
    # equals the SCALED value, still a mismatch against the raw definition
    assert len(add_capturing_warnings(throttler, [make("a", limit=5)])) == 1
    assert throttler.get_limit("a").limit == 5  # 10 scaled by 50%, existing kept


def test_mismatch_keeps_existing_and_still_applies_new_ids_in_the_batch():
    throttler = AsyncThrottler(rate_limits=[make("a")])
    stored = throttler.get_limit("a")

    warnings = add_capturing_warnings(throttler, [make("new"), make("a", limit=99), make("other")])

    assert len(warnings) == 1
    assert throttler.has_limit("new")
    assert throttler.has_limit("other")
    assert throttler.get_limit("a") is stored
    assert stored.limit == 10
    assert [limit.limit_id for limit in throttler._rate_limits] == ["a", "new", "other"]
    assert list(throttler._raw_limits) == ["a", "new", "other"]


def test_conflicting_duplicates_inside_the_batch_keep_first_and_warn_once():
    throttler = AsyncThrottler(rate_limits=[])

    warnings = add_capturing_warnings(
        throttler, [make("x"), make("y"), make("x", limit=3), make("x", limit=4)]
    )

    assert len(warnings) == 1
    assert [limit.limit_id for limit in throttler._rate_limits] == ["x", "y"]
    assert throttler.get_limit("x").limit == 10
    assert throttler._raw_limits["x"].limit == 10


def test_identical_duplicates_inside_the_batch_are_collapsed():
    throttler = AsyncThrottler(rate_limits=[])

    throttler.add_rate_limits([make("x"), make("x")])

    assert [limit.limit_id for limit in throttler._rate_limits] == ["x"]


def test_linked_limit_order_does_not_cause_a_conflict():
    linked = [LinkedLimitWeightPair("p", 1), LinkedLimitWeightPair("q", 2)]
    throttler = AsyncThrottler(rate_limits=[make("a", linked_limits=linked)])

    throttler.add_rate_limits([make("a", linked_limits=list(reversed(linked)))])

    assert len(throttler._rate_limits) == 1


def test_set_rate_limits_and_setter_never_raise():
    throttler = AsyncThrottler(rate_limits=[make("a")])

    throttler.set_rate_limits([make("a", limit=99)])
    assert throttler.get_limit("a").limit == 99

    throttler._rate_limits = [make("a", limit=7)]
    assert throttler.get_limit("a").limit == 7
