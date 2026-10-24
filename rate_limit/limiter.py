"""Weighted sliding-window limiter over linked pools.

Concurrency model: all accounting lives behind one ``threading.Lock`` that is held only for
synchronous check-and-record steps. ``acquire`` never sleeps (or awaits) while holding it,
so waiters on any number of event loops and threads share one history without serialising
behind a sleeper. A cancelled ``acquire`` has recorded nothing.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from decimal import Decimal
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Literal

from rate_limit.backoff import (
    LONG_RETRY_AFTER_WARN_SECONDS,
    RATE_LIMIT_STATUSES,
    BackoffPolicy,
    PauseState,
    RateLimitExceededError,
    RateLimitVerdict,
)
from rate_limit.errors import (
    RateLimitPoolConflictError,
    RateLimitWeightError,
    UnknownRateLimitError,
)
from rate_limit.pools import Pool, carry_over, new_pool
from rate_limit.types import Clock, RateLimit, RateLimitLike, Sleep, check_weight, coerce_limit

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from contextlib import AbstractAsyncContextManager

logger = logging.getLogger(__name__)

CAPACITY_WARN_INTERVAL_SECONDS = 30.0
CAPACITY_WARN_FRACTION = 0.9

__all__ = ["BoundLimiter", "Clock", "RateLimiter", "Sleep"]


class RateLimiter:
    """Registry of limits plus the acquire / pause / backoff machinery."""

    def __init__(
        self,
        limits: Iterable[RateLimitLike] = (),
        *,
        limits_pct: Decimal = Decimal(1),
        safety_margin_pct: float = 0.0,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        wall_clock: Callable[[], float] = time.time,
        backoff_factory: Callable[[], BackoffPolicy] = BackoffPolicy,
        unknown: Literal["raise", "allow"] = "raise",
    ) -> None:
        if not isinstance(limits_pct, Decimal) or not (
            limits_pct.is_finite() and Decimal(0) < limits_pct <= Decimal(1)
        ):
            raise ValueError(f"limits_pct must be a Decimal with 0 < pct <= 1, got {limits_pct!r}")
        if not math.isfinite(safety_margin_pct) or safety_margin_pct < 0:
            raise ValueError(
                f"safety_margin_pct must be finite and >= 0, got {safety_margin_pct!r}"
            )
        if unknown not in ("raise", "allow"):
            raise ValueError(f"unknown must be 'raise' or 'allow', got {unknown!r}")
        self._pct = limits_pct
        self._margin = safety_margin_pct
        self._clock = clock
        self._sleep = sleep
        self._wall_clock = wall_clock  # only for HTTP-date Retry-After; ``clock`` may be monotonic
        self._backoff_factory = backoff_factory
        self._unknown_mode = unknown
        self._lock = threading.Lock()
        self._report_lock = threading.Lock()
        self._raw: dict[str, RateLimit] = {}
        self._scaled: dict[str, RateLimit] = {}
        self._owners: dict[str, str | None] = {}
        self._pools: dict[str, Pool] = {}
        self._states: dict[str, PauseState] = {}
        self._pause_all_until = 0.0
        self._unknown_warned: set[str] = set()
        self._capacity_warned: dict[str, float] = {}
        self.add_limits(limits)

    # ---- registry -------------------------------------------------------------------

    @property
    def limits_pct(self) -> Decimal:
        """Fraction of every declared limit this limiter may use."""
        return self._pct

    def add_limit(self, limit: RateLimitLike, *, owner: str | None = None) -> RateLimit:
        """Register one limit; returns the stored (scaled) limit."""
        raw = coerce_limit(limit)
        with self._lock:
            self._add_locked([raw], owner)
            return self._scaled[raw.limit_id]

    def add_limits(self, limits: Iterable[RateLimitLike], *, owner: str | None = None) -> None:
        """Register limits atomically.

        An identical raw re-add is a no-op; a differing one raises
        :class:`RateLimitPoolConflictError` and nothing is applied.
        """
        incoming = [coerce_limit(item) for item in limits]
        with self._lock:
            self._add_locked(incoming, owner)

    def _add_locked(self, incoming: list[RateLimit], owner: str | None) -> None:
        pending: dict[str, RateLimit] = {}
        for raw in incoming:
            known = pending.get(raw.limit_id) or self._raw.get(raw.limit_id)
            if known is None:
                pending[raw.limit_id] = raw
            elif known != raw:
                existing_owner: str | None
                if raw.limit_id in pending:
                    batch = "earlier in the same batch"
                    existing_owner = f"{owner}, {batch}" if owner else batch
                else:
                    existing_owner = self._owners.get(raw.limit_id)
                raise RateLimitPoolConflictError(raw.limit_id, known, raw, existing_owner, owner)
        for raw in pending.values():
            self._install(raw, owner)

    def set_limits(self, limits: Iterable[RateLimitLike]) -> None:
        """Replace all definitions; usage carries over for every retained id.

        The pool holds traffic the venue already counted, so it stays valid under a new
        limit, interval, weight, link or kind; it is dropped only when an id is removed.
        """
        incoming = {raw.limit_id: raw for raw in map(coerce_limit, limits)}
        with self._lock:
            now = self._clock()
            old_scaled, old_pools, old_owners = self._scaled, self._pools, self._owners
            self._raw, self._scaled, self._pools, self._owners = {}, {}, {}, {}
            for limit_id, raw in incoming.items():
                self._install(raw, None)
                scaled = self._scaled[limit_id]
                old_pool = old_pools.get(limit_id)
                if old_pool is not None:
                    self._pools[limit_id] = carry_over(
                        old_pool,
                        scaled.kind,
                        scaled.limit,
                        scaled.time_interval,
                        self._margin,
                        now,
                    )
                if old_scaled.get(limit_id) == scaled:
                    self._owners[limit_id] = old_owners.get(limit_id)

    def _install(self, raw: RateLimit, owner: str | None) -> None:
        scaled = raw.scaled(self._pct)
        self._raw[raw.limit_id] = raw
        self._scaled[raw.limit_id] = scaled
        self._owners[raw.limit_id] = owner
        self._pools[raw.limit_id] = new_pool(
            scaled.kind, scaled.limit, scaled.time_interval, self._margin
        )

    def has_limit(self, limit_id: str) -> bool:
        with self._lock:
            return limit_id in self._scaled

    def get_limit(self, limit_id: str) -> RateLimit | None:
        """The SCALED stored limit."""
        with self._lock:
            return self._scaled.get(limit_id)

    def get_raw_limit(self, limit_id: str) -> RateLimit | None:
        """The raw definition as registered."""
        with self._lock:
            return self._raw.get(limit_id)

    def get_owner(self, limit_id: str) -> str | None:
        """The owner recorded when the limit was registered."""
        with self._lock:
            return self._owners.get(limit_id)

    def limits(self) -> Mapping[str, RateLimit]:
        """Read-only snapshot of the scaled limits."""
        with self._lock:
            return MappingProxyType(dict(self._scaled))

    # ---- planning -------------------------------------------------------------------

    def _unknown(self, limit_id: str, detail: str, notes: list[str]) -> None:
        """Raise, or queue the warn-once message in ``notes`` (logged after the lock is freed)."""
        if self._unknown_mode == "raise":
            raise UnknownRateLimitError(f"{detail}: unknown rate limit {limit_id!r}")
        if limit_id not in self._unknown_warned:
            self._unknown_warned.add(limit_id)
            notes.append(f"{detail}: unknown rate limit {limit_id!r}; running unthrottled")

    def _plan(
        self, limit_id: str, weights: Mapping[str, int] | None, notes: list[str]
    ) -> list[tuple[str, int]] | None:
        """Resolve the charges for one call (lock held). ``None`` = run unthrottled."""
        primary = self._scaled.get(limit_id)
        declared = {limit_id}
        if primary is not None:
            declared.update(pair.limit_id for pair in primary.linked_limits)
        for key, value in (weights or {}).items():
            if key not in declared:
                raise ValueError(
                    f"weights key {key!r} is neither {limit_id!r} nor one of its linked limits"
                )
            check_weight(value, f"weights[{key!r}]")
        if primary is None:
            self._unknown(limit_id, "acquire", notes)
            return None
        charges: dict[str, int] = {limit_id: primary.weight}
        for pair in primary.linked_limits:
            if pair.limit_id in self._scaled:
                charges[pair.limit_id] = charges.get(pair.limit_id, 0) + pair.weight
            else:
                self._unknown(pair.limit_id, f"acquire {limit_id!r} (linked)", notes)
        for key, value in (weights or {}).items():
            if key in charges:
                charges[key] = value
        for charge_id, cost in charges.items():
            capacity = self._scaled[charge_id].limit
            if cost > capacity:
                raise RateLimitWeightError(
                    f"weight {cost} exceeds limit {capacity} of {charge_id!r}"
                )
        return list(charges.items())

    def _validate(self, limit_id: str, weights: Mapping[str, int] | None) -> bool:
        """Raise on a bad call; return False when the call runs unthrottled."""
        notes: list[str] = []
        try:
            with self._lock:
                return self._plan(limit_id, weights, notes) is not None
        finally:
            for message in notes:
                logger.warning(message)

    def _assess(
        self, plan: list[tuple[str, int]], now: float, through: float, record: bool
    ) -> float:
        """Evict, find the longest wait over the touched pools, and record if all fit."""
        pools = [(self._pools[charge_id], cost) for charge_id, cost in plan]
        wait = 0.0
        for pool, cost in pools:
            pool.evict(now, through)
            wait = max(wait, pool.wait_for(cost, now))
        if wait == 0.0 and record:
            for pool, cost in pools:
                pool.reserve(cost, now)
        return wait

    def _near_capacity(self, plan: list[tuple[str, int]], now: float) -> list[str]:
        messages = []
        for charge_id, _ in plan:
            pool = self._pools[charge_id]
            if pool.used < CAPACITY_WARN_FRACTION * pool.capacity:
                continue
            last = self._capacity_warned.get(charge_id)
            if last is None or now - last >= CAPACITY_WARN_INTERVAL_SECONDS:
                self._capacity_warned[charge_id] = now
                messages.append(
                    f"Rate limit {charge_id!r} almost reached ({pool.used:g}/{pool.capacity:g})"
                )
        return messages

    def _try_reserve(
        self, limit_id: str, weights: Mapping[str, int] | None, deadline: float, record: bool
    ) -> tuple[float, float, bool]:
        """One synchronous pause-check-and-record step; never awaits.

        Returns ``(wait, now, paused)``. When recording, an active pause short-circuits
        before the window is touched (``paused=True``, nothing recorded); unknown ids in
        "allow" mode skip the window but still honour pauses.
        """
        warnings: list[str] = []
        try:
            with self._lock:
                now = self._clock()
                plan = self._plan(limit_id, weights, warnings)
                pause_wait = max(0.0, self._paused_until_locked(limit_id) - now)
                if record and pause_wait > 0.0:
                    return pause_wait, now, True
                window_wait = 0.0
                if plan is not None:
                    through = deadline if now >= deadline else -math.inf
                    window_wait = self._assess(plan, now, through, record)
                    if record and window_wait == 0.0:
                        warnings.extend(self._near_capacity(plan, now))
        finally:
            for message in warnings:
                logger.warning(message)
        return max(pause_wait, window_wait), now, False

    # ---- acquire --------------------------------------------------------------------

    async def acquire(self, limit_id: str, weights: Mapping[str, int] | None = None) -> None:
        """Wait out pauses and the window, then record the charge.

        Sleeps happen outside every lock; cancelling records nothing.
        """
        deadline = -math.inf
        while True:
            wait, now, paused = self._try_reserve(limit_id, weights, deadline, record=True)
            if wait == 0.0:
                return
            if not paused:
                deadline = now + wait
            await self._sleep(wait)

    def execute_task(
        self, limit_id: str, *, weights: Mapping[str, int] | None = None
    ) -> AbstractAsyncContextManager[None]:
        """Async context manager that acquires on entry; exit is a no-op."""
        self._validate(limit_id, weights)
        return _TaskContext(self, limit_id, weights)

    def try_acquire(self, limit_id: str, weights: Mapping[str, int] | None = None) -> float:
        """Non-blocking: 0.0 = reserved, else seconds to wait (nothing recorded)."""
        return self._try_reserve(limit_id, weights, -math.inf, record=True)[0]

    def blocked_for(self, limit_id: str, weights: Mapping[str, int] | None = None) -> float:
        """Seconds until an acquire would pass (pause or window); records nothing."""
        return self._try_reserve(limit_id, weights, -math.inf, record=False)[0]

    def bind(self, limit_id: str, weights: Mapping[str, int] | None = None) -> BoundLimiter:
        """A single-limit view satisfying the pausable limiter protocols."""
        return BoundLimiter(self, limit_id, weights)

    # ---- pause + backoff ------------------------------------------------------------

    def _state(self, limit_id: str) -> PauseState:
        state = self._states.get(limit_id)
        if state is None:
            state = self._states[limit_id] = PauseState(backoff=self._backoff_factory())
        return state

    def _until(self, seconds: float) -> float:
        if not math.isfinite(seconds):
            raise ValueError(f"pause seconds must be finite, got {seconds!r}")
        return self._clock() + max(0.0, seconds)

    def pause(self, limit_id: str, seconds: float) -> None:
        """Block acquires touching ``limit_id`` (never shortens; unknown ids are recorded)."""
        until = self._until(seconds)
        with self._lock:
            state = self._state(limit_id)
            state.deadline = max(state.deadline, until)

    def pause_all(self, seconds: float) -> None:
        """Block every acquire on this limiter (never shortens); used for a 418 IP ban."""
        until = self._until(seconds)
        with self._lock:
            self._pause_all_until = max(self._pause_all_until, until)

    def _paused_until_locked(self, limit_id: str) -> float:
        ids = {limit_id}
        primary = self._scaled.get(limit_id)
        if primary is not None:
            ids.update(pair.limit_id for pair in primary.linked_limits)
        deadlines = (self._states[i].deadline for i in ids if i in self._states)
        return max([self._pause_all_until, *deadlines])

    def paused_until(self, limit_id: str) -> float:
        """Clock value the pause on ``limit_id`` (or a linked id) lasts to; 0.0 if never."""
        with self._lock:
            return self._paused_until_locked(limit_id)

    def is_paused(self, limit_id: str) -> bool:
        return self.paused_until(limit_id) > self._clock()

    def backoff(self, limit_id: str) -> BackoffPolicy:
        """The backoff policy (streak state) for ``limit_id``."""
        with self._lock:
            return self._state(limit_id).backoff

    def report_throttled(
        self, limit_id: str, retry_after: float | None = None, *, status: int | None = None
    ) -> RateLimitVerdict:
        """Apply the backoff policy to a rate-limit signal (``status=None`` counts as 429)."""
        if retry_after is not None:
            if not math.isfinite(retry_after):
                raise ValueError(f"retry_after must be finite, got {retry_after!r}")
            retry_after = max(0.0, retry_after)
        effective = 429 if status is None else status
        with self._report_lock:  # one burst-check/escalate/pause step at a time
            in_burst = self.is_paused(limit_id)
            delay = self.backoff(limit_id).next_delay(effective, retry_after, in_burst)
            if delay == 0.0 and in_burst:
                logger.debug("Rate limit signal on %r joins the active pause", limit_id)
                return RateLimitVerdict(True, effective, retry_after, 0.0, False)
            paused_all = effective == 418
            if paused_all:
                self.pause_all(delay)
            else:
                self.pause(limit_id, delay)
        note = (
            " (unusually long Retry-After, honoured)"
            if retry_after is not None and retry_after > LONG_RETRY_AFTER_WARN_SECONDS
            else ""
        )
        logger.warning(
            "Rate limited (status %s) on %s; pausing %s for %.1fs%s",
            effective,
            limit_id,
            "all limits" if paused_all else "it and its dependents",
            delay,
            note,
        )
        return RateLimitVerdict(True, effective, retry_after, delay, paused_all)

    def report_rate_limited(self, limit_id: str, exc: RateLimitExceededError) -> RateLimitVerdict:
        """Backoff hook for an exception-based flow."""
        return self.report_throttled(limit_id, exc.retry_after, status=exc.status)

    def report_response(
        self, limit_id: str, status: int, headers: Mapping[str, str] | None = None
    ) -> RateLimitVerdict:
        """Transport-agnostic hook; never raises for HTTP statuses."""
        if status in RATE_LIMIT_STATUSES:
            return self.report_rate_limited(
                limit_id,
                RateLimitExceededError.from_response(status, headers, now=self._wall_clock),
            )
        if 200 <= status < 300:
            self.report_success(limit_id)
        return RateLimitVerdict(False, status, None, 0.0, False)

    def report_success(self, limit_id: str) -> None:
        """Reset the backoff streak after a successful request."""
        with self._lock:
            state = self._states.get(limit_id)
            if state is not None:
                state.backoff.reset()


class _TaskContext:
    def __init__(
        self, limiter: RateLimiter, limit_id: str, weights: Mapping[str, int] | None
    ) -> None:
        self._limiter = limiter
        self._limit_id = limit_id
        self._weights = weights

    async def __aenter__(self) -> None:
        await self._limiter.acquire(self._limit_id, self._weights)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class BoundLimiter:
    """One limit id of a :class:`RateLimiter` (``acquire()/pause(s)/is_paused()/backoff``)."""

    def __init__(
        self, limiter: RateLimiter, limit_id: str, weights: Mapping[str, int] | None = None
    ) -> None:
        self._limiter = limiter
        self._limit_id = limit_id
        self._weights = weights

    @property
    def limiter(self) -> RateLimiter:
        return self._limiter

    @property
    def limit_id(self) -> str:
        return self._limit_id

    @property
    def backoff(self) -> BackoffPolicy:
        return self._limiter.backoff(self._limit_id)

    async def acquire(self) -> None:
        await self._limiter.acquire(self._limit_id, self._weights)

    def pause(self, seconds: float) -> None:
        self._limiter.pause(self._limit_id, seconds)

    def is_paused(self) -> bool:
        return self._limiter.is_paused(self._limit_id)

    def blocked_for(self) -> float:
        return self._limiter.blocked_for(self._limit_id, self._weights)
