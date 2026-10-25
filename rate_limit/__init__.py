"""Stdlib-only async rate limiting primitives for hummingbot sub-packages."""

from rate_limit.__about__ import __version__
from rate_limit.backoff import (
    DEFAULT_418_BAN_SECONDS,
    DEFAULT_429_BACKOFF_BASE_SECONDS,
    DEFAULT_429_BACKOFF_CAP_SECONDS,
    LONG_RETRY_AFTER_WARN_SECONDS,
    RATE_LIMIT_STATUSES,
    BackoffPolicy,
    PauseState,
    RateLimitExceededError,
    RateLimitVerdict,
    parse_retry_after,
    wait_out_deadline,
    wait_until,
)
from rate_limit.errors import (
    RateLimitPoolConflictError,
    RateLimitWeightError,
    UnknownRateLimitError,
)
from rate_limit.helpers import (
    note_rate_limited,
    note_request_succeeded,
    run_paced,
    run_paced_limit,
)
from rate_limit.keyed import KeyedCooldown
from rate_limit.limiter import BoundLimiter, Charge, RateLimiter
from rate_limit.pools import Pool, SlidingWindowPool, TokenBucketPool
from rate_limit.protocols import (
    AsyncThrottlerProtocol,
    PausableRateLimiterProtocol,
    PoolPauseProtocol,
    RestRateLimiterProtocol,
    ThrottlerProtocol,
    WeightedThrottlerProtocol,
)
from rate_limit.registry import get_pool_limiter, reset_pool_limiters
from rate_limit.types import (
    Clock,
    LinkedLimitWeightPair,
    PoolKind,
    RateLimit,
    RateLimitLike,
    Sleep,
    coerce_limit,
    scale_limit,
)

__all__ = [
    "DEFAULT_418_BAN_SECONDS",
    "DEFAULT_429_BACKOFF_BASE_SECONDS",
    "DEFAULT_429_BACKOFF_CAP_SECONDS",
    "LONG_RETRY_AFTER_WARN_SECONDS",
    "RATE_LIMIT_STATUSES",
    "AsyncThrottlerProtocol",
    "BackoffPolicy",
    "BoundLimiter",
    "Charge",
    "Clock",
    "KeyedCooldown",
    "LinkedLimitWeightPair",
    "PausableRateLimiterProtocol",
    "PauseState",
    "Pool",
    "PoolKind",
    "PoolPauseProtocol",
    "RateLimit",
    "RateLimitExceededError",
    "RateLimitLike",
    "RateLimitPoolConflictError",
    "RateLimitVerdict",
    "RateLimitWeightError",
    "RateLimiter",
    "RestRateLimiterProtocol",
    "SlidingWindowPool",
    "Sleep",
    "ThrottlerProtocol",
    "TokenBucketPool",
    "UnknownRateLimitError",
    "WeightedThrottlerProtocol",
    "__version__",
    "coerce_limit",
    "get_pool_limiter",
    "note_rate_limited",
    "note_request_succeeded",
    "parse_retry_after",
    "reset_pool_limiters",
    "run_paced",
    "run_paced_limit",
    "scale_limit",
    "wait_out_deadline",
    "wait_until",
]
