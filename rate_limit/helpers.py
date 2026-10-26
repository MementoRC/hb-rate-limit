"""Pacing helpers for callers that hold a limiter protocol rather than a RateLimiter."""

from __future__ import annotations

import contextlib
import logging
import time
import weakref
from typing import TYPE_CHECKING, Any

from rate_limit.backoff import (
    LONG_RETRY_AFTER_WARN_SECONDS,
    BackoffPolicy,
    RateLimitExceededError,
    wait_out_deadline,
    wait_until,
)
from rate_limit.limiter import BoundLimiter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from rate_limit.limiter import RateLimiter
    from rate_limit.protocols import RestRateLimiterProtocol
    from rate_limit.types import Clock

logger = logging.getLogger(__name__)

__all__ = [
    "note_rate_limited",
    "note_request_succeeded",
    "run_paced",
    "run_paced_limit",
    "wait_out_deadline",
    "wait_until",
]

_FALLBACK_POLICIES: weakref.WeakKeyDictionary[Any, BackoffPolicy] = weakref.WeakKeyDictionary()
_UNPAUSABLE_WARNED: weakref.WeakSet[Any] = weakref.WeakSet()
# Limiters that cannot be weak-referenced: id -> the limiter itself. The strong reference
# keeps the id from being reused while the entry exists. Trade-off: up to 256 such limiters
# are held alive, and the 257th distinct one evicts the oldest, which may then warn again.
_UNPAUSABLE_WARNED_IDS: dict[int, Any] = {}
_UNPAUSABLE_WARNED_IDS_MAX = 256
# Pause deadlines the helper set itself, for limiters that have pause() but no is_paused().
_OWN_PAUSE_DEADLINES: weakref.WeakKeyDictionary[Any, float] = weakref.WeakKeyDictionary()
_CLOCK: Callable[[], float] = time.monotonic


def _backoff_for(limiter: Any) -> BackoffPolicy:
    """Return the limiter's backoff policy, attaching a per-instance one if it has none."""
    policy = getattr(limiter, "backoff", None)
    if isinstance(policy, BackoffPolicy):
        return policy
    try:
        return _FALLBACK_POLICIES.setdefault(limiter, BackoffPolicy())
    except TypeError:  # not weak-referenceable: no streak memory
        return BackoffPolicy()


def _first_unpausable_warning(limiter: Any) -> bool:
    """True the first time ``limiter`` is seen.

    Non-weak-referenceable limiters are tracked by identity in a bounded registry that
    holds them alive; an entry for the same id but a different object is stale.
    """
    try:
        if limiter in _UNPAUSABLE_WARNED:
            return False
        _UNPAUSABLE_WARNED.add(limiter)
    except TypeError:
        key = id(limiter)
        if _UNPAUSABLE_WARNED_IDS.get(key) is limiter:
            return False
        _UNPAUSABLE_WARNED_IDS.pop(key, None)
        _UNPAUSABLE_WARNED_IDS[key] = limiter
        if len(_UNPAUSABLE_WARNED_IDS) > _UNPAUSABLE_WARNED_IDS_MAX:
            del _UNPAUSABLE_WARNED_IDS[next(iter(_UNPAUSABLE_WARNED_IDS))]
    return True


def _in_own_pause(limiter: Any, clock: Clock | None = None) -> bool:
    """Whether a pause this helper set on ``limiter`` is still running."""
    try:
        deadline = _OWN_PAUSE_DEADLINES.get(limiter)
    except TypeError:  # not weak-referenceable: memoryless
        return False
    return deadline is not None and (clock or _CLOCK)() < deadline


def _remember_pause(limiter: Any, delay: float, clock: Clock | None = None) -> None:
    with contextlib.suppress(TypeError):  # not weak-referenceable: memoryless
        _OWN_PAUSE_DEADLINES[limiter] = max(
            _OWN_PAUSE_DEADLINES.get(limiter, 0.0), (clock or _CLOCK)() + delay
        )


def _limiter_label(limiter: Any) -> str:
    return str(
        getattr(limiter, "name", None)
        or getattr(limiter, "limit_id", None)
        or type(limiter).__name__
    )


def note_rate_limited(
    limiter: RestRateLimiterProtocol, exc: RateLimitExceededError, *, clock: Clock | None = None
) -> None:
    """Pause ``limiter`` after a rate-limit error and log one warning for the event.

    Limiters without ``pause()`` cannot back off: that is logged once and nothing else
    happens (the caller re-raises the error). A limiter with ``pause()`` but no
    ``is_paused()`` gets burst detection from the pause deadline this helper set itself,
    measured on ``clock`` (default ``time.monotonic``); pass the same clock for a given
    limiter on every call, since the remembered deadline is compared on it. If the limiter
    cannot be weak-referenced the helper is memoryless and every signal escalates.
    ``clock`` has no effect on a :class:`BoundLimiter`, which uses its own limiter's clock.
    """
    if isinstance(limiter, BoundLimiter):
        limiter.limiter.report_rate_limited(limiter.limit_id, exc)  # 418 pauses every limit
        return
    status = 429 if exc.status is None else exc.status
    pause = getattr(limiter, "pause", None)
    if not callable(pause):
        if _first_unpausable_warning(limiter):
            logger.warning(
                "Rate limiter %s cannot pause; no backoff after status %s",
                _limiter_label(limiter),
                status,
            )
        return
    is_paused = getattr(limiter, "is_paused", None)
    in_burst = bool(is_paused()) if callable(is_paused) else _in_own_pause(limiter, clock)
    delay = _backoff_for(limiter).next_delay(status, exc.retry_after, in_burst)
    if delay == 0.0 and in_burst:
        logger.debug("Status %s on %s joins the active pause", status, _limiter_label(limiter))
        return
    pause(delay)
    if not callable(is_paused):
        _remember_pause(limiter, delay, clock)
    note = (
        " (unusually long Retry-After, honoured)"
        if exc.retry_after is not None and exc.retry_after > LONG_RETRY_AFTER_WARN_SECONDS
        else ""
    )
    logger.warning(
        "Rate limited (status %s) on %s; pausing for %.1fs%s",
        status,
        _limiter_label(limiter),
        delay,
        note,
    )


def note_request_succeeded(limiter: RestRateLimiterProtocol) -> None:
    """Reset the backoff streak of ``limiter`` after a successful request."""
    policy = getattr(limiter, "backoff", None)
    if not isinstance(policy, BackoffPolicy):
        try:
            policy = _FALLBACK_POLICIES.get(limiter)
        except TypeError:
            return
    if policy is not None:
        policy.reset()


async def run_paced[T](
    limiter: RestRateLimiterProtocol,
    call: Callable[[], Awaitable[T]],
    *,
    clock: Clock | None = None,
) -> T:
    """Acquire once, run ``call``, and apply rate-limit backoff bookkeeping.

    A ``RateLimitExceededError`` pauses the limiter and propagates; success resets the
    backoff streak. ``clock`` is forwarded to :func:`note_rate_limited`.
    """
    await limiter.acquire()
    try:
        result = await call()
    except RateLimitExceededError as exc:
        note_rate_limited(limiter, exc, clock=clock)
        raise
    note_request_succeeded(limiter)
    return result


async def run_paced_limit[T](
    rl: RateLimiter,
    limit_id: str,
    call: Callable[[], Awaitable[T]],
    *,
    weights: Mapping[str, int] | None = None,
) -> T:
    """:func:`run_paced` against a multi-limit :class:`RateLimiter` (418 pauses all limits)."""
    await rl.acquire(limit_id, weights)
    try:
        result = await call()
    except RateLimitExceededError as exc:
        rl.report_rate_limited(limit_id, exc)
        raise
    rl.report_success(limit_id)
    return result
