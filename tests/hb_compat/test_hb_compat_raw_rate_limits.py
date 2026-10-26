"""Contract: ``_rate_limits`` exposes RAW definitions; effective limits stay scaled (#9)."""

import copy
import logging
from decimal import Decimal

import pytest

from rate_limit.hb_compat.async_throttler import AsyncThrottler
from rate_limit.hb_compat.data_types import RateLimit
from rate_limit.testing import ManualClock

LIMIT_ID = "orders"
INTERVAL = 60.0
RATE_LIMITS = [RateLimit(LIMIT_ID, limit=100, time_interval=INTERVAL)]


async def tick(_seconds: float) -> None:
    pass


def build(share: int) -> AsyncThrottler:
    return AsyncThrottler(
        rate_limits=[RateLimit(LIMIT_ID, limit=100, time_interval=INTERVAL)],
        limits_share_percentage=Decimal(share),
        clock=ManualClock(),
        sleep=tick,
    )


def effective(throttler: AsyncThrottler) -> float:
    limit = throttler.get_limit(LIMIT_ID)
    assert limit is not None
    assert throttler._id_to_limit_map[LIMIT_ID].limit == limit.limit
    return limit.limit


def raw(throttler: AsyncThrottler) -> float:
    (entry,) = [item for item in throttler._rate_limits if item.limit_id == LIMIT_ID]
    return entry.limit


def definition_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "different definition" in r.getMessage()]


def test_bitrue_style_round_trips_do_not_rescale_at_half_share():
    throttler = build(50)

    for _ in range(3):
        snapshot = copy.deepcopy(throttler._rate_limits)
        throttler.set_rate_limits(snapshot)

        assert effective(throttler) == 50
        assert raw(throttler) == 100


def test_direct_raw_assignment_scales_once_and_reads_back_raw():
    throttler = build(50)

    throttler._rate_limits = RATE_LIMITS

    assert effective(throttler) == 50
    assert raw(throttler) == 100


def test_readd_of_true_raw_value_after_round_trip_does_not_warn(
    caplog: pytest.LogCaptureFixture,
):
    throttler = build(50)
    throttler.set_rate_limits(copy.deepcopy(throttler._rate_limits))

    with caplog.at_level(logging.WARNING):
        throttler.add_rate_limits([RateLimit(LIMIT_ID, limit=100, time_interval=INTERVAL)])

    assert definition_warnings(caplog) == []


def test_edited_copy_is_scaled_once_and_reports_raw():
    throttler = build(50)
    edited = copy.deepcopy(throttler._rate_limits)
    edited[0].limit = 200

    throttler.set_rate_limits(edited)

    assert effective(throttler) == 100
    assert raw(throttler) == 200


def test_full_share_round_trip_is_identity():
    throttler = build(100)

    for _ in range(3):
        throttler.set_rate_limits(copy.deepcopy(throttler._rate_limits))

        assert effective(throttler) == 100
        assert raw(throttler) == 100


def test_raw_getter_entries_are_not_the_effective_limit_objects():
    throttler = build(50)

    (entry,) = throttler._rate_limits
    entry.limit = 999

    assert entry is not throttler._id_to_limit_map[LIMIT_ID]
    assert effective(throttler) == 50
