"""Legacy ``AsyncThrottler`` over :class:`rate_limit.RateLimiter`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rate_limit.hb_compat.async_request_context_base import AsyncRequestContextBase
from rate_limit.hb_compat.async_throttler_base import AsyncThrottlerBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rate_limit.limiter import RateLimiter


class AsyncRequestContext(AsyncRequestContextBase):
    """Acquires ``limit_id`` (with optional weight overrides) on the limiter."""

    def __init__(
        self, limiter: RateLimiter, limit_id: str, weights: Mapping[str, int] | None
    ) -> None:
        self._limiter = limiter
        self._limit_id = limit_id
        self._weights = weights

    async def acquire(self) -> None:
        await self._limiter.acquire(self._limit_id, self._weights)


class AsyncThrottler(AsyncThrottlerBase):
    """Delays calls as needed so they stay within the declared limits."""

    def execute_task(
        self, limit_id: str, *, weights: Mapping[str, int] | None = None
    ) -> AsyncRequestContext:
        """Return an async context gating entry on capacity for ``limit_id``.

        :param weights: optional per-call overrides for the primary and/or its registered
            linked ids (ints >= 1); other ids keep their declared weight.
        :raises ValueError: for a weights key outside the primary and its registered links,
            or a value that is not an int >= 1.
        """
        if weights:
            self._validate_weights(limit_id, weights)
        self._warn_unknown(limit_id)
        return AsyncRequestContext(self._limiter, limit_id, dict(weights) if weights else None)
