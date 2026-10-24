from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from rate_limit import (
    DEFAULT_418_BAN_SECONDS,
    DEFAULT_429_BACKOFF_BASE_SECONDS,
    DEFAULT_429_BACKOFF_CAP_SECONDS,
    RATE_LIMIT_STATUSES,
    BackoffPolicy,
    PauseState,
    RateLimitExceededError,
    parse_retry_after,
    wait_until,
)
from rate_limit.testing import ManualClock, assert_conforms_backoff


def test_statuses() -> None:
    assert frozenset({418, 429}) == RATE_LIMIT_STATUSES


@pytest.mark.parametrize(
    ("value", "expected"),
    [("7", 7.0), (" 2.5 ", 2.5), ("0", 0.0), ("-3", 0.0)],
)
def test_parse_delta_seconds(value: str, expected: float) -> None:
    assert parse_retry_after(value) == expected


@pytest.mark.parametrize("value", [None, "", "  ", "soon", "nan", "inf", "Mon, 99 Foo"])
def test_parse_garbage_is_none(value: str | None) -> None:
    assert parse_retry_after(value) is None


def test_parse_http_date_uses_injected_now() -> None:
    when = datetime(2030, 1, 1, 12, 0, 30, tzinfo=UTC)
    header = format_datetime(when, usegmt=True)
    epoch = when.timestamp() - 30
    assert parse_retry_after(header, now=lambda: epoch) == pytest.approx(30.0)
    assert parse_retry_after(header, now=lambda: epoch + 100) == 0.0


def test_parse_http_date_default_wall_clock() -> None:
    when = datetime.now(UTC) + timedelta(seconds=30)
    result = parse_retry_after(format_datetime(when, usegmt=True))
    assert result is not None
    assert 27 <= result <= 31


def test_parse_naive_http_date_is_utc() -> None:
    epoch = datetime(2030, 1, 1, tzinfo=UTC).timestamp()
    assert parse_retry_after("Tue, 01 Jan 2030 00:00:10 -0000", now=lambda: epoch) == pytest.approx(
        10.0
    )


class TestExceededError:
    def test_args_and_attributes(self) -> None:
        err = RateLimitExceededError("boom", status=429, retry_after=3.0, limit_id="x")
        assert err.args == ("boom",)
        assert str(err) == "boom"
        assert (err.status, err.retry_after, err.limit_id) == (429, 3.0, "x")

    def test_default_message_from_status(self) -> None:
        err = RateLimitExceededError(status=418)
        assert "418" in str(err)
        assert err.args == (str(err),)
        assert RateLimitExceededError().status is None
        assert "rate limited" in str(RateLimitExceededError())

    def test_not_an_oserror_and_no_slots(self) -> None:
        err = RateLimitExceededError("x")
        assert not isinstance(err, OSError)
        assert hasattr(err, "__dict__")

    def test_from_response_case_insensitive(self) -> None:
        err = RateLimitExceededError.from_response(429, {"retry-after": "4"}, "m")
        assert (err.status, err.retry_after, err.args) == (429, 4.0, ("m",))

    def test_from_response_without_headers(self) -> None:
        assert RateLimitExceededError.from_response(418).retry_after is None

    def test_from_response_returns_subclass(self) -> None:
        class MineError(RateLimitExceededError):
            pass

        assert isinstance(MineError.from_response(429), MineError)

    def test_dual_base_oserror_subclass(self) -> None:
        class EdgeError(OSError, RateLimitExceededError):
            def __init__(self, legacy: str, *, status: int) -> None:
                OSError.__init__(self, legacy)
                self.status = status

        err = EdgeError("legacy text", status=429)
        assert str(err) == "legacy text"
        assert err.args == ("legacy text",)
        assert err.errno is None
        assert isinstance(err, OSError)
        assert isinstance(err, RateLimitExceededError)


class TestBackoffPolicy:
    def test_conformance(self) -> None:
        assert_conforms_backoff(BackoffPolicy)

    def test_exponential_with_cap(self) -> None:
        policy = BackoffPolicy()
        delays = [policy.next_delay(429, None) for _ in range(10)]
        base = DEFAULT_429_BACKOFF_BASE_SECONDS
        assert delays[:4] == [base, 2 * base, 4 * base, 8 * base]
        assert delays[-1] == DEFAULT_429_BACKOFF_CAP_SECONDS

    def test_418_ban_default(self) -> None:
        assert BackoffPolicy().next_delay(418, None) == DEFAULT_418_BAN_SECONDS

    def test_exponent_clamp_at_streak_5000(self) -> None:
        policy = BackoffPolicy()
        policy.streak = 5000
        assert policy.next_delay(429, None) == DEFAULT_429_BACKOFF_CAP_SECONDS
        assert policy.streak == 5001

    def test_fixed(self) -> None:
        policy = BackoffPolicy.fixed(63)
        assert [policy.next_delay(429, None) for _ in range(5)] == [63] * 5
        assert policy.next_delay(418, None) == 63

    def test_in_burst_rules(self) -> None:
        policy = BackoffPolicy()
        assert policy.next_delay(429, None, in_burst=True) == 0.0
        assert policy.streak == 0
        assert policy.next_delay(429, 9.0, in_burst=True) == 9.0
        assert policy.streak == 1


async def test_wait_until_follows_extended_deadline() -> None:
    clock = ManualClock()
    deadline = [2.0]

    async def sleep(seconds: float) -> None:
        await clock.sleep(seconds)
        deadline[0] = 5.0

    await wait_until(lambda: deadline[0], clock, sleep)
    assert clock() == 5.0


def test_pause_state_defaults() -> None:
    state = PauseState()
    assert state.deadline == 0.0
    assert state.backoff.streak == 0
