"""Data model: limit declarations, scaling and structural typing."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Protocol

PoolKind = Literal["window", "token_bucket"]
Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


def _check_int_ge1(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} must be an int >= 1, got {value!r}")
    return value


def check_weight(value: object, what: str = "weight") -> int:
    """Validate a charge: an int >= 1 (bool is rejected)."""
    return _check_int_ge1(value, what)


@dataclass(frozen=True)
class LinkedLimitWeightPair:
    """A linked pool and the weight charged to it."""

    limit_id: str
    weight: int = 1

    def __post_init__(self) -> None:
        _check_int_ge1(self.weight, "linked weight")


@dataclass(frozen=True, eq=True, init=False)
class RateLimit:
    """At most ``limit`` weight units per ``time_interval`` seconds.

    ``kind="window"`` (default) is a sliding window. ``kind="token_bucket"`` is a bucket of
    capacity ``limit`` refilling ``limit / time_interval`` per second, so a connector's
    ``PoolSpec(name, capacity, refill_rate)`` maps to
    ``RateLimit(name, capacity, capacity / refill_rate, kind="token_bucket")``.
    """

    limit_id: str
    limit: int
    time_interval: float
    weight: int = 1
    linked_limits: tuple[LinkedLimitWeightPair, ...] = ()
    kind: PoolKind = "window"

    def __init__(
        self,
        limit_id: str,
        limit: int,
        time_interval: float,
        weight: int = 1,
        linked_limits: Sequence[Any] | None = None,
        *,
        kind: PoolKind = "window",
    ) -> None:
        if not isinstance(limit_id, str) or not limit_id:
            raise ValueError(f"limit_id must be a non-empty str, got {limit_id!r}")
        _check_int_ge1(limit, "limit")
        _check_int_ge1(weight, "weight")
        if (
            isinstance(time_interval, bool)
            or not isinstance(time_interval, int | float)
            or not math.isfinite(time_interval)
            or time_interval <= 0
        ):
            raise ValueError(f"time_interval must be a finite number > 0, got {time_interval!r}")
        if kind not in ("window", "token_bucket"):
            raise ValueError(f"kind must be 'window' or 'token_bucket', got {kind!r}")
        pairs = tuple(_as_pair(item) for item in (linked_limits or ()))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "limit_id", limit_id)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "time_interval", float(time_interval))
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "linked_limits", pairs)

    def scaled(self, pct: Decimal) -> RateLimit:
        """Return a copy with ``limit`` scaled by ``pct`` (see :func:`scale_limit`)."""
        return RateLimit(
            self.limit_id,
            scale_limit(self.limit, pct),
            self.time_interval,
            self.weight,
            self.linked_limits,
            kind=self.kind,
        )

    @classmethod
    def per_minute(cls, limit_id: str, limit: int, **kw: Any) -> RateLimit:
        """Declare ``limit`` per 60 seconds."""
        return cls(limit_id, limit, 60, **kw)

    @classmethod
    def per_second(cls, limit_id: str, limit: int, **kw: Any) -> RateLimit:
        """Declare ``limit`` per second."""
        return cls(limit_id, limit, 1, **kw)


def _as_pair(item: Any) -> LinkedLimitWeightPair:
    if isinstance(item, LinkedLimitWeightPair):
        return item
    return LinkedLimitWeightPair(item.limit_id, item.weight)


class RateLimitLike(Protocol):
    """Structural limit declaration; hummingbot's mutable ``RateLimit`` also fits.

    An optional ``kind`` attribute is honoured by :func:`coerce_limit` (default ``"window"``).
    """

    @property
    def limit_id(self) -> str: ...

    @property
    def limit(self) -> int: ...

    @property
    def time_interval(self) -> float: ...

    @property
    def weight(self) -> int: ...

    @property
    def linked_limits(self) -> Sequence[Any]: ...


def coerce_limit(obj: RateLimitLike) -> RateLimit:
    """Return ``obj`` as an immutable :class:`RateLimit` (identity if it already is one)."""
    if isinstance(obj, RateLimit):
        return obj
    return RateLimit(
        obj.limit_id,
        obj.limit,
        obj.time_interval,
        obj.weight,
        list(obj.linked_limits or ()),
        kind=getattr(obj, "kind", "window"),
    )


def scale_limit(raw: int, pct: Decimal) -> int:
    """Scale a raw limit: ``max(1, floor(raw * pct))``."""
    return max(1, math.floor(Decimal(str(raw)) * pct))
