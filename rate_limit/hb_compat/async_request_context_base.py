"""Legacy ``AsyncRequestContextBase``: the ``async with`` gate returned by ``execute_task``."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

MAX_CAPACITY_REACHED_WARNING_INTERVAL = 30.0


class AsyncRequestContextBase(ABC):
    """Async context manager that acquires capacity on entry."""

    logger = staticmethod(partial(logging.getLogger, __name__))

    @abstractmethod
    async def acquire(self) -> None:
        """Block until capacity is available, then record the call."""
        raise NotImplementedError

    async def __aenter__(self) -> AsyncRequestContextBase:
        await self.acquire()
        return self

    async def __aexit__(  # noqa: B027 - capacity frees as the window slides; nothing to release
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass
