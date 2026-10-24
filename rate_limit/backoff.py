"""Retry-After parsing, rate-limit errors and backoff policy."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from rate_limit.types import Clock, Sleep

# HTTP statuses meaning "stop sending": 429 rate limited, 418 IP banned (Binance).
RATE_LIMIT_STATUSES: frozenset[int] = frozenset({418, 429})

DEFAULT_429_BACKOFF_BASE_SECONDS = 5.0  # first pause; doubles per consecutive 429
DEFAULT_429_BACKOFF_CAP_SECONDS = 300.0
DEFAULT_418_BAN_SECONDS = 120.0
LONG_RETRY_AFTER_WARN_SECONDS = 3600.0  # honoured anyway, but flagged in the log

_MAX_EXPONENT = 62


def parse_retry_after(value: str | None, *, now: Callable[[], float] = time.time) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) into seconds.

    :param value: Raw header value
    :param now: Wall clock (epoch seconds) used for the HTTP-date form
    :return: Seconds to wait (negatives clamp to 0), or None when absent or unparseable
    """
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            moment = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        seconds = moment.timestamp() - now()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
    """Case-insensitive header lookup that also works for plain dicts."""
    for key, value in (headers or {}).items():
        if key.lower() == name.lower():
            return value
    return None


class RateLimitExceededError(Exception):
    """The remote side signalled a rate limit (HTTP 429/418 or an equivalent).

    Derives from ``Exception`` (not ``OSError``) on purpose; ``args == (message,)`` and
    ``status``/``retry_after``/``limit_id`` are plain attributes, so subclasses with an
    ``OSError`` co-base may bypass this ``__init__``.
    """

    status: int | None
    retry_after: float | None
    limit_id: str | None

    def __init__(
        self,
        message: str = "",
        *,
        status: int | None = None,
        retry_after: float | None = None,
        limit_id: str | None = None,
    ) -> None:
        if not message:
            message = (
                f"HTTP {status}: rate limited (retry_after={retry_after})"
                if status is not None
                else f"rate limited (retry_after={retry_after})"
            )
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.limit_id = limit_id

    @classmethod
    def from_response(
        cls,
        status: int,
        headers: Mapping[str, str] | None = None,
        message: str = "",
        *,
        now: Callable[[], float] = time.time,
    ) -> Self:
        """Build the error from a response status and its headers (``now``: wall clock)."""
        retry_after = parse_retry_after(_header(headers, "Retry-After"), now=now)
        return cls(message, status=status, retry_after=retry_after)


@dataclass(frozen=True)
class RateLimitVerdict:
    """Outcome of reporting a response or throttle signal."""

    rate_limited: bool
    status: int
    retry_after: float | None
    delay: float
    paused_all: bool


class BackoffPolicy:
    """Pause length for the next rate-limit signal, tracked per reported limit.

    ``Retry-After`` wins. Otherwise 418 pauses for ``ban_seconds`` and 429 backs off
    exponentially (``base``, ``2*base``, ... up to ``cap``) while 429s keep arriving; the
    streak resets on the next success.
    """

    def __init__(
        self,
        base_seconds: float = DEFAULT_429_BACKOFF_BASE_SECONDS,
        cap_seconds: float = DEFAULT_429_BACKOFF_CAP_SECONDS,
        ban_seconds: float = DEFAULT_418_BAN_SECONDS,
    ) -> None:
        self.base_seconds = base_seconds
        self.cap_seconds = cap_seconds
        self.ban_seconds = ban_seconds
        self.streak = 0

    @classmethod
    def fixed(cls, seconds: float) -> BackoffPolicy:
        """A constant cool-off: base == cap == ban == ``seconds``."""
        return cls(seconds, seconds, seconds)

    def next_delay(self, status: int, retry_after: float | None, in_burst: bool = False) -> float:
        """Register one rate-limit signal and return how long to pause.

        :param in_burst: The pool was already paused when this signal arrived, i.e. it
            belongs to requests that were in flight when the pause began. A non-418
            signal without ``Retry-After`` then neither escalates the streak nor extends
            the pause (0).
        """
        if in_burst and status != 418 and retry_after is None:
            return 0.0
        self.streak += 1
        if retry_after is not None:
            return retry_after
        if status == 418:
            return self.ban_seconds
        exponent = min(self.streak - 1, _MAX_EXPONENT)
        return min(self.cap_seconds, self.base_seconds * float(2**exponent))

    def reset(self) -> None:
        """Forget the streak (a request succeeded)."""
        self.streak = 0


@dataclass
class PauseState:
    """Pause deadline plus backoff streak, shared by everything on one limit."""

    deadline: float = 0.0
    backoff: BackoffPolicy = field(default_factory=BackoffPolicy)


async def wait_until(get_deadline: Callable[[], float], clock: Clock, sleep: Sleep) -> None:
    """Sleep until the (possibly extended meanwhile) deadline has passed."""
    while (remaining := get_deadline() - clock()) > 0:
        await sleep(remaining)


wait_out_deadline = wait_until
