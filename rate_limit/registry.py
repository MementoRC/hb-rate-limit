"""Process-wide shared limiters, one per caller-chosen pool key."""

from __future__ import annotations

import asyncio
import math
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
    margin: float


_LIMITERS: dict[str, _Entry] = {}
_REGISTRY_LOCK = threading.Lock()


def get_pool_limiter(
    pool_key: str,
    limit: RateLimitLike,
    *,
    clock: Clock = time.monotonic,
    sleep: Sleep = asyncio.sleep,
    wall_clock: Callable[[], float] = time.time,
    owner: str | None = None,
    backoff_factory: Callable[[], BackoffPolicy] | None = None,
    safety_margin_pct: float = 0.0,
) -> BoundLimiter:
    """Return the process-wide limiter for ``pool_key``, creating it on first use.

    The first caller's clock, sleep, wall clock and backoff factory win. The pool's single
    limit takes ``pool_key`` as its ``limit_id``. Callers sharing a budget pass the same key; use a
    non-reversible tag (never the API key itself) when keying by credentials.

    ``safety_margin_pct`` (default 0.0) widens the pool like :class:`RateLimiter`'s and is part
    of the pool's identity. hb_compat shims pass 0.05 for parity with hummingbot's
    ``AsyncThrottlerBase``.

    :raises RateLimitPoolConflictError: the pool exists with a different raw definition or margin
    :raises ValueError: creating a pool from a limit that has linked limits, or a bad margin
    """
    if not math.isfinite(safety_margin_pct) or safety_margin_pct < 0:
        raise ValueError(f"safety_margin_pct must be finite and >= 0, got {safety_margin_pct!r}")
    given = coerce_limit(limit)
    requested = RateLimit(
        pool_key,
        given.limit,
        given.time_interval,
        given.weight,
        given.linked_limits,
        kind=given.kind,
    )
    with _REGISTRY_LOCK:
        entry = _LIMITERS.get(pool_key)
        if entry is None:
            if requested.linked_limits:
                raise ValueError(
                    f"pool limiters are single-limit: {pool_key!r} cannot declare linked limits"
                )
            limiter = RateLimiter(
                [requested],
                clock=clock,
                sleep=sleep,
                wall_clock=wall_clock,
                backoff_factory=backoff_factory or BackoffPolicy,
                safety_margin_pct=safety_margin_pct,
            )
            entry = _LIMITERS[pool_key] = _Entry(limiter, requested, owner, safety_margin_pct)
        elif entry.raw != requested or entry.margin != safety_margin_pct:
            raise RateLimitPoolConflictError(
                pool_key,
                entry.raw,
                requested,
                entry.owner,
                owner,
                detail=(
                    f"Margins differ: safety_margin_pct {entry.margin} vs {safety_margin_pct}. "
                    if entry.margin != safety_margin_pct
                    else ""
                ),
            )
    return entry.limiter.bind(pool_key)


def reset_pool_limiters() -> None:
    """Drop all registered limiters (test isolation)."""
    with _REGISTRY_LOCK:
        _LIMITERS.clear()
