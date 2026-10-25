"""Weighted sliding-window limiter over linked pools.

Concurrency model: all accounting lives behind one ``threading.Lock`` that is held only for
synchronous check-and-record steps. ``acquire`` / ``reserve`` never sleep (or await) while
holding it, so waiters on any number of event loops and threads share one history without
serialising behind a sleeper. A cancelled ``acquire`` has recorded nothing. A
:meth:`Charge.refund` takes the same lock and wakes nobody: waiters see the returned capacity
when they next re-assess.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
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
from rate_limit.pools import Pool, SlidingWindowPool, TokenBucketPool, carry_over, new_pool
from rate_limit.types import Clock, RateLimit, RateLimitLike, Sleep, check_weight, coerce_limit

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping
    from contextlib import AbstractAsyncContextManager

logger = logging.getLogger(__name__)

CAPACITY_WARN_INTERVAL_SECONDS = 30.0
CAPACITY_WARN_FRACTION = 0.9

__all__ = ["BoundLimiter", "Charge", "Clock", "RateLimiter", "Sleep"]


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
        max_retry_after_seconds: float | None = None,
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
        if max_retry_after_seconds is not None and (
            not math.isfinite(max_retry_after_seconds) or max_retry_after_seconds <= 0
        ):
            raise ValueError(
                f"max_retry_after_seconds must be None or finite and > 0, "
                f"got {max_retry_after_seconds!r}"
            )
        self._pct = limits_pct
        self._max_retry_after = max_retry_after_seconds
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
        # pool generation per id: new on install, removal, kind change or bucket refill-rate
        # change, kept by any other same-kind rebuild (see ``_keeps_charges``); drawn from one
        # counter so a removed then re-added id never reuses one
        self._generations: dict[str, int] = {}
        self._next_generation = itertools.count()
        self._states: dict[str, PauseState] = {}
        self._pause_all_until = 0.0
        # (loop, event) of every acquirer sleeping out a pause; ``unpause*`` sets them
        self._wakers: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()
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
            old_generations = self._generations
            self._raw, self._scaled, self._pools, self._owners = {}, {}, {}, {}
            self._generations = {}
            for limit_id, raw in incoming.items():
                self._install(raw, None)
                scaled = self._scaled[limit_id]
                old_pool = old_pools.get(limit_id)
                if old_pool is not None:
                    pool = self._pools[limit_id] = carry_over(
                        old_pool,
                        scaled.kind,
                        scaled.limit,
                        scaled.time_interval,
                        self._margin,
                        now,
                    )
                    if self._keeps_charges(old_pool, pool):
                        self._generations[limit_id] = old_generations[limit_id]
                if old_scaled.get(limit_id) == scaled:
                    self._owners[limit_id] = old_owners.get(limit_id)

    @staticmethod
    def _keeps_charges(old: Pool, new: Pool) -> bool:
        """Whether charges taken on ``old`` stay refundable on its rebuild ``new``.

        A window rebuild shares the history, and a refund removes an exact entry. A bucket
        refund assumes one refill rate since the charge, so a rate change (either way) ends it.
        """
        if isinstance(old, TokenBucketPool) and isinstance(new, TokenBucketPool):
            return old.refill_per_second == new.refill_per_second
        return isinstance(old, SlidingWindowPool) and isinstance(new, SlidingWindowPool)

    def _install(self, raw: RateLimit, owner: str | None) -> None:
        scaled = raw.scaled(self._pct)
        self._raw[raw.limit_id] = raw
        self._scaled[raw.limit_id] = scaled
        self._owners[raw.limit_id] = owner
        self._pools[raw.limit_id] = new_pool(
            scaled.kind, scaled.limit, scaled.time_interval, self._margin
        )
        self._generations[raw.limit_id] = next(self._next_generation)

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
    ) -> tuple[float, float, bool, Charge | None]:
        """One synchronous pause-check-and-record step; never awaits.

        Returns ``(wait, now, paused, charge)``. When recording, an active pause short-circuits
        before the window is touched (``paused=True``, nothing recorded); unknown ids in
        "allow" mode skip the window but still honour pauses. ``charge`` is the receipt once
        something was recorded (empty for an unthrottled call), else None.
        """
        warnings: list[str] = []
        try:
            with self._lock:
                now = self._clock()
                plan = self._plan(limit_id, weights, warnings)
                pause_wait = max(0.0, self._paused_until_locked(limit_id) - now)
                if record and pause_wait > 0.0:
                    return pause_wait, now, True, None
                window_wait = 0.0
                if plan is not None:
                    through = deadline if now >= deadline else -math.inf
                    window_wait = self._assess(plan, now, through, record)
                    if record and window_wait == 0.0:
                        warnings.extend(self._near_capacity(plan, now))
                charge = None
                if record and window_wait == 0.0:
                    charge = self._receipt(limit_id, plan or [], now)
        finally:
            for message in warnings:
                logger.warning(message)
        return max(pause_wait, window_wait), now, False, charge

    def _receipt(self, limit_id: str, plan: list[tuple[str, int]], now: float) -> Charge:
        """The receipt for ``plan`` just recorded at ``now`` (lock held)."""
        generations = {charge_id: self._generations[charge_id] for charge_id, _ in plan}
        return Charge(self, limit_id, dict(plan), generations, now)

    # ---- acquire --------------------------------------------------------------------

    async def acquire(self, limit_id: str, weights: Mapping[str, int] | None = None) -> None:
        """Wait out pauses and the window, then record the charge.

        :meth:`reserve` without the receipt: sleeps happen outside every lock, cancelling
        records nothing, and an unpause ends a sleep spent in a pause early.
        """
        await self.reserve(limit_id, weights)

    async def reserve(self, limit_id: str, weights: Mapping[str, int] | None = None) -> Charge:
        """Wait out pauses and the window, record the charge, and return its :class:`Charge`.

        Sleeps happen outside every lock; cancelling records nothing. A sleep spent in a pause
        ends early when :meth:`unpause` / :meth:`unpause_all` lifts it, and the call re-assesses.
        An unknown id in "allow" mode returns an empty charge, whose refund is a no-op.
        """
        deadline = -math.inf
        while True:
            wait, now, paused, charge = self._try_reserve(limit_id, weights, deadline, record=True)
            if charge is not None:
                return charge
            if paused:
                await self._sleep_unless_unpaused(limit_id, wait)
                continue
            deadline = now + wait
            await self._sleep(wait)

    async def _sleep_unless_unpaused(self, limit_id: str, wait: float) -> None:
        """Sleep ``wait`` seconds, returning early if an unpause happens meanwhile."""
        woken = asyncio.Event()
        waker = (asyncio.get_running_loop(), woken)
        with self._lock:
            self._wakers.add(waker)
        try:
            # registered first, so an unpause racing with this step is seen here or wakes us
            if self.paused_until(limit_id) > self._clock():
                await self._race(self._sleep(wait), woken)
        finally:
            with self._lock:
                self._wakers.discard(waker)

    @staticmethod
    async def _race(sleep: Awaitable[None], woken: asyncio.Event) -> None:
        """Await ``sleep`` or ``woken``, whichever ends first; both tasks are reaped."""
        sleeper = asyncio.ensure_future(sleep)
        waiter = asyncio.ensure_future(woken.wait())
        try:
            await asyncio.wait({sleeper, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if sleeper.done():
                sleeper.result()  # surface a failing sleep
        finally:
            sleeper.cancel()
            waiter.cancel()
            await asyncio.gather(sleeper, waiter, return_exceptions=True)

    def _wake_sleepers(self) -> None:
        with self._lock:
            wakers = list(self._wakers)
        for loop, event in wakers:
            with contextlib.suppress(RuntimeError):  # that loop has been closed
                loop.call_soon_threadsafe(event.set)

    def execute_task(
        self,
        limit_id: str,
        *,
        weights: Mapping[str, int] | None = None,
        refund_on_error: bool = False,
    ) -> AbstractAsyncContextManager[None]:
        """Async context manager that reserves on entry; exit is a no-op by default.

        With ``refund_on_error``, a body raising an :class:`Exception` refunds the charge before
        the exception propagates. A ``BaseException`` such as ``CancelledError`` keeps it: the
        request may already have gone out. Opt in only for bodies that fail before the request
        reaches the venue (signing, validation): an exception after sending (a 429, a response
        timeout, a parse error) means the venue already counted it, so a refund under-throttles;
        there, use :meth:`reserve` and refund explicitly.
        """
        self._validate(limit_id, weights)
        return _TaskContext(self, limit_id, weights, refund_on_error)

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
        """Block acquires touching ``limit_id``.

        Never shortens an existing pause (use :meth:`unpause` to lift one); unknown ids are
        recorded.
        """
        until = self._until(seconds)
        with self._lock:
            state = self._state(limit_id)
            state.deadline = max(state.deadline, until)

    def pause_all(self, seconds: float) -> None:
        """Block every acquire on this limiter; used for a 418 IP ban.

        Never shortens an existing pause (use :meth:`unpause_all` to lift it).
        """
        until = self._until(seconds)
        with self._lock:
            self._pause_all_until = max(self._pause_all_until, until)

    def unpause(self, limit_id: str) -> None:
        """Lift the pause set on ``limit_id`` and wake acquirers sleeping out a pause.

        A :meth:`pause_all` stays in force, and pool accounting is untouched. Unknown ids
        have nothing to lift.
        """
        with self._lock:
            state = self._states.get(limit_id)
            if state is not None:
                state.deadline = 0.0
        self._wake_sleepers()

    def unpause_all(self) -> None:
        """Lift :meth:`pause_all` and every per-id pause, waking sleeping acquirers.

        Pool accounting is untouched: capacity already spent stays spent.
        """
        with self._lock:
            self._pause_all_until = 0.0
            for state in self._states.values():
                state.deadline = 0.0
        self._wake_sleepers()

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
        """Apply the backoff policy to a rate-limit signal (``status=None`` counts as 429).

        A server-supplied ``retry_after`` above ``max_retry_after_seconds`` is clamped to it;
        delays the :class:`BackoffPolicy` derives itself are never capped.
        """
        if retry_after is not None:
            if not math.isfinite(retry_after):
                raise ValueError(f"retry_after must be finite, got {retry_after!r}")
            retry_after = max(0.0, retry_after)
            cap = self._max_retry_after
            if cap is not None and retry_after > cap:
                logger.warning(
                    "Retry-After %gs for %s exceeds max_retry_after_seconds; capping to %gs",
                    retry_after,
                    limit_id,
                    cap,
                )
                retry_after = cap
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


class Charge:
    """Receipt for one :meth:`RateLimiter.reserve`: the cost charged to each pool, and when.

    :meth:`refund` hands that capacity back, e.g. when signing failed and the request never went
    out. A refund does not wake waiters early: they pick up the returned capacity when they next
    re-assess.
    """

    def __init__(
        self,
        limiter: RateLimiter,
        limit_id: str,
        costs: Mapping[str, int],
        generations: Mapping[str, int],
        stamp: float,
    ) -> None:
        self._limiter = limiter
        self._limit_id = limit_id
        self._costs = MappingProxyType(dict(costs))
        self._generations = dict(generations)  # pool generation charged, per pool id
        self._stamp = stamp
        self._refunded = False

    @property
    def limit_id(self) -> str:
        return self._limit_id

    @property
    def costs(self) -> Mapping[str, int]:
        """Pool id -> cost charged; empty for an unknown id run unthrottled in "allow" mode."""
        return self._costs

    @property
    def stamp(self) -> float:
        """Clock time the charge was recorded at."""
        return self._stamp

    @property
    def refunded(self) -> bool:
        return self._refunded

    def refund(self) -> None:
        """Return the charge to every pool it touched; idempotent and thread-safe.

        A pool is skipped unless it is still the generation charged: removing its limit id
        (even if re-added later) or changing its kind via ``set_limits`` starts a new one, a
        same-kind rebuild does not. A window gives the charge back only while it still counts;
        a bucket returns only the part that cannot have refilled yet (see
        :meth:`TokenBucketPool.refund`). Waiters are not woken early: they pick up the returned
        capacity when they next re-assess.
        """
        limiter = self._limiter
        with limiter._lock:
            if self._refunded:
                return
            self._refunded = True
            now = limiter._clock()
            for pool_id, cost in self._costs.items():
                if limiter._generations.get(pool_id) == self._generations[pool_id]:
                    limiter._pools[pool_id].refund(cost, self._stamp, now)


class _TaskContext:
    def __init__(
        self,
        limiter: RateLimiter,
        limit_id: str,
        weights: Mapping[str, int] | None,
        refund_on_error: bool,
    ) -> None:
        self._limiter = limiter
        self._limit_id = limit_id
        self._weights = weights
        self._refund_on_error = refund_on_error
        self._charge: Charge | None = None

    async def __aenter__(self) -> None:
        self._charge = await self._limiter.reserve(self._limit_id, self._weights)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Exception only: after a BaseException (e.g. cancellation) the request may have gone out
        if self._refund_on_error and isinstance(exc, Exception) and self._charge is not None:
            self._charge.refund()


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

    async def reserve(self) -> Charge:
        return await self._limiter.reserve(self._limit_id, self._weights)

    def pause(self, seconds: float) -> None:
        self._limiter.pause(self._limit_id, seconds)

    def unpause(self) -> None:
        self._limiter.unpause(self._limit_id)

    def unpause_all(self) -> None:
        self._limiter.unpause_all()

    def is_paused(self) -> bool:
        return self._limiter.is_paused(self._limit_id)

    def blocked_for(self) -> float:
        return self._limiter.blocked_for(self._limit_id, self._weights)

    def paused_until(self) -> float:
        return self._limiter.paused_until(self._limit_id)

    def report_throttled(
        self, retry_after: float | None = None, *, status: int | None = None
    ) -> RateLimitVerdict:
        return self._limiter.report_throttled(self._limit_id, retry_after, status=status)
