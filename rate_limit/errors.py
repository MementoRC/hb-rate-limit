"""Errors raised by the registry and limiter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rate_limit.types import RateLimit


class RateLimitWeightError(ValueError):
    """A request's weight exceeds the limit of a pool it touches; it could never pass."""


class UnknownRateLimitError(KeyError):
    """``unknown="raise"``: the limit id (or a linked id) is not registered."""


class RateLimitPoolConflictError(ValueError):
    """Two registrations of one limit id / pool declare different raw definitions."""

    pool: str
    existing: RateLimit
    requested: RateLimit

    def __init__(
        self,
        pool: str,
        existing: RateLimit,
        requested: RateLimit,
        existing_owner: str | None = None,
        requested_owner: str | None = None,
        detail: str = "",
    ) -> None:
        by = f" (registered by {existing_owner})" if existing_owner else ""
        asker = f" by {requested_owner}" if requested_owner else ""
        super().__init__(
            f"Rate-limit pool {pool!r} already has limit {existing}{by}; "
            f"conflicting limit {requested} requested{asker}. "
            f"{detail}Everything sharing a pool must declare identical limits."
        )
        self.pool = pool
        self.existing = existing
        self.requested = requested
