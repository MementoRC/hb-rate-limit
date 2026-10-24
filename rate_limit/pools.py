"""Pool accounting: what :class:`~rate_limit.limiter.RateLimiter` needs from a limit.

Two implementations: :class:`SlidingWindowPool` (weighted window) and
:class:`TokenBucketPool` (continuous refill). Pools are not thread-safe; the limiter calls
them under its own lock.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rate_limit.types import PoolKind


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

    def rebuilt(
        self, limit: int, window: float, margin_pct: float, now: float
    ) -> SlidingWindowPool:
        """A pool with the new definition that carries this pool's history, evicted at ``now``."""
        pool = SlidingWindowPool(limit, window, margin_pct)
        pool._history = self._history
        pool._used = self._used
        pool.evict(now)
        return pool

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


class TokenBucketPool:
    """Continuous-refill bucket: starts full, refills ``refill_per_second`` up to ``capacity``.

    ``used`` is ``capacity - tokens`` as of the last ``evict``/``reserve``. Admission treats a
    shortfall below ``capacity * 1e-12`` as zero, so float rounding cannot turn a reached
    ``wait_for`` deadline into an endless chain of tiny waits.
    """

    _TOLERANCE = 1e-12  # of capacity: thousands of ulps, far below any real token

    def __init__(self, capacity: float, refill_per_second: float) -> None:
        for name, value in (("capacity", capacity), ("refill_per_second", refill_per_second)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
        self._capacity = capacity
        self._refill = refill_per_second
        self._slack = capacity * self._TOLERANCE
        self._tokens = float(capacity)
        self._last: float | None = None

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def refill_per_second(self) -> float:
        return self._refill

    @property
    def used(self) -> float:
        return self._capacity - self._tokens

    def rebuilt(self, capacity: float, refill_per_second: float, now: float) -> TokenBucketPool:
        """A bucket with the new definition that carries this pool's deficit, refilled to ``now``"""
        self.evict(now)
        return self._reseeded(capacity, refill_per_second, self.used, now)

    @staticmethod
    def _reseeded(
        capacity: float, refill_per_second: float, used: float, now: float
    ) -> TokenBucketPool:
        pool = TokenBucketPool(capacity, refill_per_second)
        pool._tokens = min(capacity, capacity - used)
        pool._last = now
        return pool

    def _projected(self, at: float) -> float:
        last = self._last
        if last is None or at <= last:
            return self._tokens
        return min(self._capacity, self._tokens + (at - last) * self._refill)

    def evict(self, now: float, expired_through: float = -math.inf) -> None:
        # ``expired_through`` needs no handling: the admission slack covers rounding at a deadline.
        if self._last is None or now > self._last:
            self._tokens = self._projected(now)
            self._last = now

    def wait_for(self, cost: int, now: float) -> float:
        base = now if self._last is None or now >= self._last else self._last  # never go back
        tokens = self._projected(base)
        if tokens + self._slack >= cost:
            return 0.0
        target = base + (cost - tokens) / self._refill
        while self._projected(target) + self._slack < cost:  # float rounding: never undershoot
            target = math.nextafter(target, math.inf)
        wait = target - now
        while now + wait < target:
            wait = math.nextafter(wait, math.inf)
        return wait

    def reserve(self, cost: int, now: float) -> float:
        wait = self.wait_for(cost, now)
        if wait > 0:
            return wait
        self.evict(now)
        self._tokens -= cost
        return 0.0


def new_pool(kind: PoolKind, limit: int, time_interval: float, margin_pct: float) -> Pool:
    """A fresh pool for a (scaled) limit; a bucket refills ``limit`` per margined interval."""
    if kind == "token_bucket":
        return TokenBucketPool(limit, _bucket_refill(limit, time_interval, margin_pct))
    return SlidingWindowPool(limit, time_interval, margin_pct)


def _bucket_refill(limit: int, time_interval: float, margin_pct: float) -> float:
    return limit / (time_interval * (1.0 + margin_pct))


def carry_over(
    old: Pool, kind: PoolKind, limit: int, time_interval: float, margin_pct: float, now: float
) -> Pool:
    """A pool of ``kind`` that keeps the usage ``old`` had counted, evicted or refilled to ``now``.

    Same kind rebuilds in place. Window to bucket becomes a deficit of ``old.used``; bucket to
    window becomes one charge of ``ceil(old.used)`` stamped at ``now`` (conservative: it only
    expires a full window later).
    """
    if kind == "token_bucket":
        refill = _bucket_refill(limit, time_interval, margin_pct)
        if isinstance(old, TokenBucketPool):
            return old.rebuilt(limit, refill, now)
        old.evict(now)
        return TokenBucketPool._reseeded(limit, refill, old.used, now)
    if isinstance(old, SlidingWindowPool):
        return old.rebuilt(limit, time_interval, margin_pct, now)
    old.evict(now)
    window = SlidingWindowPool(limit, time_interval, margin_pct)
    charge = math.ceil(old.used)
    if charge > 0:
        window._history.append((now, charge))
        window._used = charge
    return window
