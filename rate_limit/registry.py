"""Process-wide shared limiters, one per caller-chosen pool key."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rate_limit.backoff import BackoffPolicy
from rate_limit.errors import RateLimitPoolConflictError
from rate_limit.limiter import BoundLimiter, RateLimiter
from rate_limit.types import Clock, RateLimit, RateLimitLike, Sleep, coerce_limit

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class _Entry:
    limiter: RateLimiter
    raw: RateLimit
    owner: str | None


_LIMITERS: dict[str, _Entry] = {}
_REGISTRY_LOCK = threading.Lock()


def get_pool_limiter(
    pool_key: str,
    limit: RateLimitLike,
    *,
    clock: Clock = time.monotonic,
    sleep: Sleep = asyncio.sleep,
    owner: str | None = None,
    backoff_factory: Callable[[], BackoffPolicy] | None = None,
) -> BoundLimiter:
    """Return the process-wide limiter for ``pool_key``, creating it on first use.

    The first caller's clock, sleep and backoff factory win. The pool's single limit takes
    ``pool_key`` as its ``limit_id``. Callers sharing a budget pass the same key; use a
    non-reversible tag (never the API key itself) when keying by credentials.

    :raises RateLimitPoolConflictError: the pool exists with a different raw definition
    """
    given = coerce_limit(limit)
    requested = RateLimit(
        pool_key, given.limit, given.time_interval, given.weight, given.linked_limits
    )
    with _REGISTRY_LOCK:
        entry = _LIMITERS.get(pool_key)
        if entry is None:
            limiter = RateLimiter(
                [requested],
                clock=clock,
                sleep=sleep,
                backoff_factory=backoff_factory or BackoffPolicy,
            )
            entry = _LIMITERS[pool_key] = _Entry(limiter, requested, owner)
        elif entry.raw != requested:
            raise RateLimitPoolConflictError(pool_key, entry.raw, requested, entry.owner, owner)
    return entry.limiter.bind(pool_key)


def reset_pool_limiters() -> None:
    """Drop all registered limiters (test isolation)."""
    with _REGISTRY_LOCK:
        _LIMITERS.clear()
