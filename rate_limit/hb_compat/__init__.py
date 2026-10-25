"""Drop-in for ``hummingbot.core.api_throttler`` running on :class:`rate_limit.RateLimiter`.

The legacy surface is kept: the mutable ``RateLimit``, ``_rate_limits`` /
``_id_to_limit_map`` / ``_raw_limits``, keep-existing ``add_rate_limits`` and the
``max(1, floor)`` share scaling. Accounting, pauses and backoff are the core's.

Intentional behaviour changes from legacy:

* a weight that can never fit raises ``RateLimitWeightError`` where legacy hung
* duplicate linked ids are summed, and the ``add_rate_limits`` conflict check compares
  linked limits as a multiset
* windows use the injected monotonic clock (no ``time.time``)
* a ``time_interval`` <= 0 (or non-finite) raises ``ValueError``
* no FIFO guarantee among waiters
* ``TaskLog`` is import-only; nothing records it
"""

from rate_limit.hb_compat.async_request_context_base import (
    MAX_CAPACITY_REACHED_WARNING_INTERVAL,
    AsyncRequestContextBase,
)
from rate_limit.hb_compat.async_throttler import AsyncRequestContext, AsyncThrottler
from rate_limit.hb_compat.async_throttler_base import AsyncThrottlerBase
from rate_limit.hb_compat.data_types import (
    DEFAULT_PATH,
    DEFAULT_WEIGHT,
    Limit,
    LinkedLimitWeightPair,
    RateLimit,
    RequestPath,
    RequestWeight,
    Seconds,
    TaskLog,
)

__all__ = [
    "DEFAULT_PATH",
    "DEFAULT_WEIGHT",
    "MAX_CAPACITY_REACHED_WARNING_INTERVAL",
    "AsyncRequestContext",
    "AsyncRequestContextBase",
    "AsyncThrottler",
    "AsyncThrottlerBase",
    "Limit",
    "LinkedLimitWeightPair",
    "RateLimit",
    "RequestPath",
    "RequestWeight",
    "Seconds",
    "TaskLog",
]
