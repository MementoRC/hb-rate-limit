"""Pacing helpers for callers that hold a limiter protocol rather than a RateLimiter."""

from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING, Any

from rate_limit.backoff import (
    LONG_RETRY_AFTER_WARN_SECONDS,
    BackoffPolicy,
    RateLimitExceededError,
    wait_out_deadline,
    wait_until,
)

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


def _backoff_for(limiter: Any) -> BackoffPolicy:
    """Return the limiter's backoff policy, attaching a per-instance one if it has none."""
    policy = getattr(limiter, "backoff", None)
    if isinstance(policy, BackoffPolicy):
        return policy
    try:
        return _FALLBACK_POLICIES.setdefault(limiter, BackoffPolicy())
    except TypeError:  # not weak-referenceable: no streak memory
        return BackoffPolicy()


def _limiter_label(limiter: Any) -> str:
    return str(
        getattr(limiter, "name", None)
        or getattr(limiter, "limit_id", None)
        or type(limiter).__name__
    )


def note_rate_limited(limiter: RestRateLimiterProtocol, exc: RateLimitExceededError) -> None:
    """Pause ``limiter`` after a rate-limit error and log one warning for the event.

    Limiters without ``pause()`` cannot back off: that is logged once and nothing else
    happens (the caller re-raises the error).
    """
    status = 429 if exc.status is None else exc.status
    pause = getattr(limiter, "pause", None)
    if not callable(pause):
        try:
            if limiter not in _UNPAUSABLE_WARNED:
                _UNPAUSABLE_WARNED.add(limiter)
                logger.warning(
                    "Rate limiter %s cannot pause; no backoff after status %s",
                    _limiter_label(limiter),
                    status,
                )
        except TypeError:
            pass
        return
    is_paused = getattr(limiter, "is_paused", None)
    in_burst = bool(is_paused()) if callable(is_paused) else False
    delay = _backoff_for(limiter).next_delay(status, exc.retry_after, in_burst)
    if delay == 0.0 and in_burst:
        logger.debug("Status %s on %s joins the active pause", status, _limiter_label(limiter))
        return
    pause(delay)
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
