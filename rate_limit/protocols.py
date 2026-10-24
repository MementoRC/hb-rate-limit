"""Structural protocols for limiters and throttlers.

``runtime_checkable`` only checks member presence, not signatures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from contextlib import AbstractAsyncContextManager

    from rate_limit.backoff import BackoffPolicy
    from rate_limit.types import RateLimit, RateLimitLike


@runtime_checkable
class RestRateLimiterProtocol(Protocol):
    """Anything REST calls can ``await acquire()`` on before sending."""

    async def acquire(self) -> None:
        """Wait until a request may be sent, then record it."""
        ...


@runtime_checkable
class PausableRateLimiterProtocol(RestRateLimiterProtocol, Protocol):
    """A limiter that can be paused after a 429/418."""

    def pause(self, seconds: float) -> None:
        """Block every ``acquire`` for ``seconds`` (never shortens an existing pause)."""
        ...


@runtime_checkable
class PoolPauseProtocol(PausableRateLimiterProtocol, Protocol):
    """Optional extras used by backoff."""

    @property
    def backoff(self) -> BackoffPolicy: ...

    def is_paused(self) -> bool: ...

    def blocked_for(self) -> float: ...


@runtime_checkable
class ThrottlerProtocol(Protocol):
    """``execute_task`` is a plain method returning an async context manager."""

    def execute_task(self, limit_id: str) -> AbstractAsyncContextManager[Any]: ...


AsyncThrottlerProtocol = ThrottlerProtocol


@runtime_checkable
class WeightedThrottlerProtocol(ThrottlerProtocol, Protocol):
    """A throttler with weighted charges, a registry and pauses."""

    def execute_task(
        self, limit_id: str, *, weights: Mapping[str, int] | None = None
    ) -> AbstractAsyncContextManager[Any]: ...

    def has_limit(self, limit_id: str) -> bool: ...

    def get_limit(self, limit_id: str) -> RateLimit | None: ...

    def add_limits(self, limits: Iterable[RateLimitLike], *, owner: str | None = None) -> None: ...

    def pause(self, limit_id: str, seconds: float) -> None: ...

    def pause_all(self, seconds: float) -> None: ...
