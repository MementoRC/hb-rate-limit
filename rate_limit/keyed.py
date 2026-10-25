"""Per-key sliding windows: at most ``limit`` weight per ``time_interval`` for each key.

:class:`KeyedCooldown` is independent of :class:`~rate_limit.limiter.RateLimiter` and not atomic
with it. When a request needs both, acquire the RateLimiter limit first and the keyed cooldown
immediately before sending, so the key's stamp sits as close as possible to the real send.

Interactive Brokers, for example, allows no identical request within 15 seconds and at most six
requests per (contract, exchange, tickType) per 2 seconds::

    identical = KeyedCooldown(1, 15.0)
    per_tick = KeyedCooldown(6, 2.0)

    async with limiter.execute_task("ib_messages"):
        await per_tick.acquire((con_id, exchange, tick_type))
        await identical.acquire(request_key)
        await send(request)

Concurrency model: as for the limiter, one ``threading.Lock`` guards all accounting and is held
only for synchronous check-and-record steps; ``acquire`` sleeps outside it, and a cancelled
``acquire`` has recorded nothing. Memory stays bounded without sweeping every key: keys are kept
in least-recently-charged order, and each operation drops keys from the front while their
window is empty.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from rate_limit.errors import RateLimitWeightError
from rate_limit.pools import SlidingWindowPool
from rate_limit.types import check_weight

if TYPE_CHECKING:
    from collections.abc import Hashable

    from rate_limit.types import Clock, Sleep

__all__ = ["KeyedCooldown"]


class KeyedCooldown:
    """One sliding window per key, created on first charge and dropped once it is empty.

    Each key gets a :class:`SlidingWindowPool` with the limiter's inclusive boundary: a charge
    counts while ``age <= time_interval * (1 + safety_margin_pct)``.
    """

    def __init__(
        self,
        limit: int,
        time_interval: float,
        *,
        safety_margin_pct: float = 0.0,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        check_weight(limit, "limit")
        if (
            isinstance(time_interval, bool)
            or not isinstance(time_interval, int | float)
            or not math.isfinite(time_interval)
            or time_interval <= 0
        ):
            raise ValueError(f"time_interval must be a finite number > 0, got {time_interval!r}")
        if not math.isfinite(safety_margin_pct) or safety_margin_pct < 0:
            raise ValueError(
                f"safety_margin_pct must be finite and >= 0, got {safety_margin_pct!r}"
            )
        self._limit = limit
        self._interval = float(time_interval)
        self._margin = safety_margin_pct
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        # least recently charged first, so empty windows collect at the front
        self._pools: OrderedDict[Hashable, SlidingWindowPool] = OrderedDict()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def time_interval(self) -> float:
        return self._interval

    def _check(self, key: Hashable, weight: int) -> None:
        check_weight(weight)
        if weight > self._limit:
            raise RateLimitWeightError(f"weight {weight} exceeds limit {self._limit} of {key!r}")

    def _drop_idle(self, now: float) -> None:
        """Pop front keys whose window is empty at ``now`` (lock held).

        The front key was charged least recently, so once its window still counts something,
        every later key's does too.
        """
        pools = self._pools
        while pools:
            pool = pools[next(iter(pools))]
            pool.evict(now)
            if pool.used:
                return
            pools.popitem(last=False)

    def _try_reserve(
        self, key: Hashable, weight: int, deadline: float, record: bool
    ) -> tuple[float, float]:
        """One synchronous check-and-record step; returns ``(wait, now)``. Never awaits."""
        self._check(key, weight)
        with self._lock:
            now = self._clock()
            self._drop_idle(now)
            pool = self._pools.get(key)
            if pool is None:
                if not record:
                    return 0.0, now  # weight <= limit always fits an empty window
                pool = self._pools[key] = SlidingWindowPool(
                    self._limit, self._interval, self._margin
                )
            else:
                pool.evict(now, deadline if now >= deadline else -math.inf)
                wait = pool.wait_for(weight, now)
                if wait > 0.0 or not record:
                    return wait, now
                self._pools.move_to_end(key)
            pool.reserve(weight, now)
        return 0.0, now

    async def acquire(self, key: Hashable, weight: int = 1) -> None:
        """Wait until ``weight`` fits in ``key``'s window, then record it.

        Sleeps happen outside the lock and the call re-assesses after each; cancelling records
        nothing.
        """
        deadline = -math.inf
        while True:
            wait, now = self._try_reserve(key, weight, deadline, record=True)
            if wait == 0.0:
                return
            deadline = now + wait
            await self._sleep(wait)

    def try_acquire(self, key: Hashable, weight: int = 1) -> float:
        """Non-blocking: 0.0 = recorded, else seconds to wait (nothing recorded)."""
        return self._try_reserve(key, weight, -math.inf, record=True)[0]

    def blocked_for(self, key: Hashable, weight: int = 1) -> float:
        """Seconds until ``weight`` would fit in ``key``'s window; records nothing."""
        return self._try_reserve(key, weight, -math.inf, record=False)[0]

    def reset(self, key: Hashable | None = None) -> None:
        """Forget ``key``'s window, or every key's when ``key`` is None."""
        with self._lock:
            if key is None:
                self._pools.clear()
            else:
                self._pools.pop(key, None)

    def __len__(self) -> int:
        """Keys currently tracked (idle ones are dropped first)."""
        with self._lock:
            self._drop_idle(self._clock())
            return len(self._pools)
