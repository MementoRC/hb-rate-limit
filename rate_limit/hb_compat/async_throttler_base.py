"""Legacy ``AsyncThrottlerBase``: mutable limit mirrors and pause API over a ``RateLimiter``."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

from rate_limit.limiter import RateLimiter
from rate_limit.types import coerce_limit, scale_limit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from rate_limit.backoff import RateLimitVerdict
    from rate_limit.hb_compat.async_request_context_base import AsyncRequestContextBase
    from rate_limit.hb_compat.data_types import RateLimit
    from rate_limit.types import Clock, Sleep


def _check_interval(limit: RateLimit) -> None:
    value = limit.time_interval
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"rate limit {limit.limit_id!r}: time_interval must be a finite number > 0, "
            f"got {value!r}"
        )


def _links(limit: RateLimit) -> Counter[tuple[str, int]]:
    return Counter((pair.limit_id, pair.weight) for pair in limit.linked_limits)


def _raw_definition_equal(a: RateLimit, b: RateLimit) -> bool:
    """Compare unscaled definitions; ``linked_limits`` compare as a multiset of (id, weight)."""
    return (
        a.limit == b.limit
        and a.time_interval == b.time_interval
        and a.weight == b.weight
        and _links(a) == _links(b)
    )


def _describe(limit: RateLimit) -> str:
    links = [(pair.limit_id, pair.weight) for pair in limit.linked_limits]
    return (
        f"limit={limit.limit}, time_interval={limit.time_interval}, "
        f"weight={limit.weight}, linked_limits={links}"
    )


class AsyncThrottlerBase(ABC):
    """Registry of scaled legacy limits plus pauses; accounting runs on a ``RateLimiter``."""

    _logger: ClassVar[logging.Logger | None] = None

    @classmethod
    def logger(cls) -> logging.Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(
        self,
        rate_limits: list[RateLimit],
        retry_interval: float = 0.1,
        safety_margin_pct: float | None = 0.05,
        limits_share_percentage: Decimal | None = None,
        *,
        clock: Clock = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """
        :param rate_limits: Limits to enforce.
        :param retry_interval: Kept for compatibility; waits are computed, not polled.
        :param safety_margin_pct: Extra fraction of each window kept occupied.
        :param limits_share_percentage: Percent of every limit this instance may use.
        :param clock: Monotonic clock for windows and pause deadlines.
        :param wall_clock: Epoch clock, used only to parse an HTTP-date ``Retry-After``.
        :param sleep: Awaitable sleep used while waiting.
        """
        self.limits_pct: Decimal = (limits_share_percentage or Decimal(100)) / 100
        self._retry_interval = retry_interval
        self._safety_margin_pct = 0.0 if safety_margin_pct is None else safety_margin_pct
        self._limiter = RateLimiter(
            limits_pct=Decimal(1),
            safety_margin_pct=self._safety_margin_pct,
            clock=clock,
            sleep=sleep,
            wall_clock=wall_clock,
            unknown="allow",
        )
        self._mirror_lock = threading.Lock()
        self._warned_conflict_ids: set[str] = set()
        self._warned_unknown_ids: set[str] = set()
        self._paused_ids: set[str] = set()
        self._paused_all = False
        self.set_rate_limits(rate_limits)

    # ---- limit registry -------------------------------------------------------------

    def _scale(self, limit: RateLimit) -> RateLimit:
        scaled = copy.deepcopy(limit)
        scaled.limit = scale_limit(limit.limit, self.limits_pct)
        return scaled

    def set_rate_limits(self, rate_limits: list[RateLimit]) -> None:
        """Replace the active set; the limits are deep-copied and scaled by ``limits_pct``."""
        with self._mirror_lock:
            for limit in rate_limits:
                _check_interval(limit)
            scaled = [self._scale(limit) for limit in rate_limits]
            core = {item.limit_id: coerce_limit(item) for item in scaled}
            self._limiter.set_limits(core.values())
            self._scaled_rate_limits = scaled
            self._raw_limits = {limit.limit_id: copy.deepcopy(limit) for limit in rate_limits}
            self._id_to_limit_map = {item.limit_id: item for item in scaled}

    @property
    def _rate_limits(self) -> list[RateLimit]:
        """The live list of scaled limits."""
        return self._scaled_rate_limits

    @_rate_limits.setter
    def _rate_limits(self, rate_limits: list[RateLimit]) -> None:
        self.set_rate_limits(rate_limits)

    def add_rate_limits(self, rate_limits: list[RateLimit]) -> None:
        """Add limits, keeping the existing definition of any id already registered.

        An identical re-add is a silent no-op; a differing one is ignored and warned about once
        per id per instance. Never raises for a conflict.
        """
        with self._mirror_lock:
            for limit in rate_limits:
                _check_interval(limit)
            staged: dict[str, tuple[RateLimit, RateLimit]] = {}
            for limit in rate_limits:
                known = staged[limit.limit_id][0] if limit.limit_id in staged else None
                existing = known or self._raw_limits.get(limit.limit_id)
                if existing is None:
                    staged[limit.limit_id] = (copy.deepcopy(limit), self._scale(limit))
                elif not _raw_definition_equal(existing, limit):
                    self._warn_definition_kept(existing, limit)
            core = [coerce_limit(scaled) for _, scaled in staged.values()]
            self._limiter.add_limits(core)
            for limit_id, (raw, scaled) in staged.items():
                self._raw_limits[limit_id] = raw
                self._scaled_rate_limits.append(scaled)
                self._id_to_limit_map[limit_id] = scaled

    def _warn_definition_kept(self, existing: RateLimit, rejected: RateLimit) -> None:
        if rejected.limit_id in self._warned_conflict_ids:
            return
        self._warned_conflict_ids.add(rejected.limit_id)
        self.logger().warning(
            f"Rate limit '{rejected.limit_id}' is already registered with a different definition; "
            f"keeping the existing one. Existing (kept): {_describe(existing)}. "
            f"New (rejected): {_describe(rejected)}."
        )

    def has_limit(self, limit_id: str) -> bool:
        return limit_id in self._id_to_limit_map

    def get_limit(self, limit_id: str) -> RateLimit | None:
        """The live stored (scaled) limit, or None; callers must not mutate it."""
        return self._id_to_limit_map.get(limit_id)

    def get_related_limits(
        self, limit_id: str
    ) -> tuple[RateLimit | None, list[tuple[RateLimit, int]]]:
        """Primary limit and its registered linked limits with weights."""
        primary = self._id_to_limit_map.get(limit_id)
        pairs = [] if primary is None else primary.linked_limits
        return primary, [
            (self._id_to_limit_map[pair.limit_id], pair.weight)
            for pair in pairs
            if pair.limit_id in self._id_to_limit_map
        ]

    def _warn_unknown(self, limit_id: str) -> None:
        """Warn once per unknown id and per instance, and mute the core's duplicate warning."""
        if limit_id in self._id_to_limit_map or limit_id in self._warned_unknown_ids:
            return
        self._warned_unknown_ids.add(limit_id)
        # RateLimiter exposes no public hook to silence its own warn-once; seeding its set is
        # the only way to keep a single warning per id (same distribution, tested together).
        self._limiter._unknown_warned.add(limit_id)
        self.logger().warning(f"Unknown rate limit id '{limit_id}': request will run unthrottled.")

    # ---- pauses ---------------------------------------------------------------------

    def pause(self, limit_id: str, seconds: float) -> None:
        """Block acquirers of ``limit_id`` and of ids linking to it; never shortens a pause."""
        self._limiter.pause(limit_id, seconds)
        self._paused_ids.add(limit_id)

    def pause_all(self, seconds: float) -> None:
        """Block every acquirer, registered or not; never shortens a pause."""
        self._limiter.pause_all(seconds)
        self._paused_all = True

    def unpause(self, limit_id: str) -> None:
        self._limiter.unpause(limit_id)

    def unpause_all(self) -> None:
        self._limiter.unpause_all()

    def paused_until(self, limit_id: str) -> float | None:
        """Monotonic deadline of the latest pause affecting ``limit_id``; None if never paused."""
        if not self._paused_all:
            primary = self._id_to_limit_map.get(limit_id)
            links = () if primary is None else primary.linked_limits
            if self._paused_ids.isdisjoint({limit_id, *(pair.limit_id for pair in links)}):
                return None
        return self._limiter.paused_until(limit_id)

    def report_response(
        self, limit_id: str, status: int, headers: Mapping[str, str] | None = None
    ) -> RateLimitVerdict:
        """Feed an HTTP outcome to the backoff policy: 429 pauses ``limit_id``, 418 pauses all."""
        verdict = self._limiter.report_response(limit_id, status, headers)
        if verdict.paused_all:
            self._paused_all = True
        elif verdict.rate_limited:
            self._paused_ids.add(limit_id)
        return verdict

    def _validate_weights(self, limit_id: str, weights: Mapping[str, int]) -> None:
        """Raise ValueError unless every key is the primary or a registered linked id and every
        value is an int >= 1 (bool rejected)."""
        primary = self._id_to_limit_map.get(limit_id)
        allowed: set[str] = set()
        if primary is not None:
            allowed.add(limit_id)
            allowed.update(
                pair.limit_id
                for pair in primary.linked_limits
                if pair.limit_id in self._id_to_limit_map
            )
        for key, weight in weights.items():
            if key not in allowed:
                raise ValueError(
                    f"weights key '{key}' is neither '{limit_id}' "
                    "nor a registered linked limit of it"
                )
            if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
                raise ValueError(f"weight for '{key}' must be an int >= 1, got {weight!r}")

    @abstractmethod
    def execute_task(
        self, limit_id: str, *, weights: Mapping[str, int] | None = None
    ) -> AsyncRequestContextBase:
        """Return an async context manager gating entry on capacity for ``limit_id``."""
        raise NotImplementedError
