"""Legacy ``hummingbot.core.api_throttler.data_types`` model (mutable ``RateLimit``)."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PATH = ""
DEFAULT_WEIGHT = 1

Limit = int  # calls per time interval
RequestPath = str
RequestWeight = int
Seconds = float


@dataclass
class LinkedLimitWeightPair:
    """Associates a secondary ``RateLimit`` (by id) with a per-call weight contribution."""

    limit_id: str
    weight: int = DEFAULT_WEIGHT


class RateLimit:
    """Call rate limit typical for an API endpoint."""

    def __init__(
        self,
        limit_id: str,
        limit: int,
        time_interval: float,
        weight: int = DEFAULT_WEIGHT,
        linked_limits: list[LinkedLimitWeightPair] | None = None,
    ) -> None:
        """
        :param limit_id: Unique identifier, usually the request path.
        :param limit: Total weight permitted within ``time_interval``.
        :param time_interval: Window length in seconds.
        :param weight: Weight of each call.
        :param linked_limits: Other limits charged by each call, with their weights.
        """
        self.limit_id = limit_id
        self.limit = limit
        self.time_interval = time_interval
        self.weight = weight
        self.linked_limits = linked_limits or []

    def __repr__(self) -> str:
        return (
            f"limit_id: {self.limit_id}, limit: {self.limit}, time interval: {self.time_interval}, "
            f"weight: {self.weight}, linked_limits: {self.linked_limits}"
        )


@dataclass
class TaskLog:
    """Legacy call record; kept for import compatibility only."""

    timestamp: float
    rate_limit: RateLimit
    weight: int
