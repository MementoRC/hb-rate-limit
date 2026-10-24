"""Pool accounting: what :class:`~rate_limit.limiter.RateLimiter` needs from a limit.

Pools are not thread-safe; the limiter calls them under its own lock.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Protocol


class Pool(Protocol):
    """Accounting for one limit."""

    @property
    def capacity(self) -> float:
        """Maximum cost the pool admits at once."""
        ...

    @property
    def used(self) -> float:
        """Cost currently counted against the pool."""
        ...

    def evict(self, now: float, expired_through: float = -math.inf) -> None:
        """Forget charges that no longer count at ``now``.

        ``expired_through`` is a deadline previously returned by ``wait_for`` that has
        now been reached: charges expiring at or before it are dropped too.
        """
        ...

    def wait_for(self, cost: int, now: float) -> float:
        """Seconds until ``cost`` fits (0.0 = fits now). Records nothing."""
        ...

    def reserve(self, cost: int, now: float) -> float:
        """Record ``cost`` and return 0.0 if it fits, else return the wait and record nothing."""
        ...


class TokenBucketPool(Pool, Protocol):
    """Interface for a continuous-refill pool (implementation planned for v0.2).

    ``capacity`` is the bucket size; ``refill_per_second`` the continuous fractional refill.
    """

    @property
    def refill_per_second(self) -> float: ...


class SlidingWindowPool:
    """Weighted sliding window with an inclusive boundary.

    A charge counts while ``age <= window * (1 + margin_pct)``; a request is refused
    only if ``used + cost > limit``.
    """

    def __init__(self, limit: int, window: float, margin_pct: float = 0.0) -> None:
        self._limit = limit
        self._window = window * (1.0 + margin_pct)
        self._history: deque[tuple[float, int]] = deque()
        self._used = 0

    @property
    def capacity(self) -> float:
        return self._limit

    @property
    def used(self) -> float:
        return self._used

    def evict(self, now: float, expired_through: float = -math.inf) -> None:
        window = self._window
        history = self._history
        while history:
            expiry = history[0][0] + window
            if expiry >= now and expiry > expired_through:
                break
            self._used -= history.popleft()[1]

    def wait_for(self, cost: int, now: float) -> float:
        if self._used + cost <= self._limit:
            return 0.0
        remaining = self._used
        for stamp, charge in self._history:
            remaining -= charge
            if remaining + cost <= self._limit:
                expiry = stamp + self._window
                wait = expiry - now
                if wait <= 0:
                    # At age == window the charge still counts: report the smallest real wait.
                    return math.nextafter(now, math.inf) - now
                while now + wait < expiry:  # float rounding: never undershoot the expiry
                    wait = math.nextafter(wait, math.inf)
                return wait
        return math.inf  # pragma: no cover - cost > limit is rejected before reaching a pool

    def reserve(self, cost: int, now: float) -> float:
        wait = self.wait_for(cost, now)
        if wait > 0:
            return wait
        self._history.append((now, cost))
        self._used += cost
        return 0.0
