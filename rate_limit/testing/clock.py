"""Deterministic clock and sleep for tests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class ManualClock:
    """A clock that only moves when told to; ``sleep`` advances it instead of waiting.

    :ivar sleeps: every ``sleep`` duration requested, in order
    :ivar on_sleep: optional hook called with the duration at the start of each ``sleep``
        (e.g. to assert that no lock is held while sleeping)
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self.on_sleep: Callable[[float], None] | None = None

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    async def sleep(self, seconds: float) -> None:
        start = self.now
        self.sleeps.append(seconds)
        if self.on_sleep is not None:
            self.on_sleep(seconds)
        await asyncio.sleep(0)  # yield so concurrent waiters interleave
        self.now = max(self.now, start + seconds)  # concurrent sleepers overlap, not sum
