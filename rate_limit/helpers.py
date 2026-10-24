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
_UNPAUSABLE_WARNED_IDS: set[int] = set()  # limiters that cannot be weak-referenced
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
    """True the first time ``limiter`` is seen; non-weak-referenceable ones are keyed by id."""
    try:
        if limiter in _UNPAUSABLE_WARNED:
            return False
        _UNPAUSABLE_WARNED.add(limiter)
    except TypeError:
        if id(limiter) in _UNPAUSABLE_WARNED_IDS:
            return False
        _UNPAUSABLE_WARNED_IDS.add(id(limiter))
    return True


def _in_own_pause(limiter: Any) -> bool:
    """Whether a pause this helper set on ``limiter`` is still running."""
    try:
        deadline = _OWN_PAUSE_DEADLINES.get(limiter)
    except TypeError:  # not weak-referenceable: memoryless
        return False
    return deadline is not None and _CLOCK() < deadline


def _remember_pause(limiter: Any, delay: float) -> None:
    with contextlib.suppress(TypeError):  # not weak-referenceable: memoryless
        _OWN_PAUSE_DEADLINES[limiter] = max(
            _OWN_PAUSE_DEADLINES.get(limiter, 0.0), _CLOCK() + delay
        )


def _limiter_label(limiter: Any) -> str:
    return str(
        getattr(limiter, "name", None)
        or getattr(limiter, "limit_id", None)
        or type(limiter).__name__
    )


def note_rate_limited(limiter: RestRateLimiterProtocol, exc: RateLimitExceededError) -> None:
    """Pause ``limiter`` after a rate-limit error and log one warning for the event.

    Limiters without ``pause()`` cannot back off: that is logged once and nothing else
    happens (the caller re-raises the error). A limiter with ``pause()`` but no
    ``is_paused()`` gets burst detection from the pause deadline this helper set itself
    (``time.monotonic``); if it cannot be weak-referenced the helper is memoryless and every
    signal escalates.
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
    in_burst = bool(is_paused()) if callable(is_paused) else _in_own_pause(limiter)
    delay = _backoff_for(limiter).next_delay(status, exc.retry_after, in_burst)
    if delay == 0.0 and in_burst:
        logger.debug("Status %s on %s joins the active pause", status, _limiter_label(limiter))
        return
    pause(delay)
    if not callable(is_paused):
        _remember_pause(limiter, delay)
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


async def run_paced[T](limiter: RestRateLimiterProtocol, call: Callable[[], Awaitable[T]]) -> T:
    """Acquire once, run ``call``, and apply rate-limit backoff bookkeeping.

    A ``RateLimitExceededError`` pauses the limiter and propagates; success resets the
    backoff streak.
    """
    await limiter.acquire()
    try:
        result = await call()
    except RateLimitExceededError as exc:
        note_rate_limited(limiter, exc)
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
