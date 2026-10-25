"""Pytest plugin: ``pytest -p rate_limit.testing.pytest_plugin``.

Provides fixtures for consumer tests that drive the conformance kit:

* ``manual_clock``: a fresh :class:`ManualClock`
* ``reset_pool_limiters``: autouse isolation for the process-wide pool registry
* ``conformance_suites``: the ``assert_conforms_*`` functions as a namespace; ``token_bucket``,
  ``unpausable`` and ``retry_after_cap`` are opt-in (run them only if you claim the feature)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from rate_limit.registry import reset_pool_limiters as _reset
from rate_limit.testing import conformance
from rate_limit.testing.clock import ManualClock

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def manual_clock() -> ManualClock:
    return ManualClock()


@pytest.fixture(autouse=True)
def reset_pool_limiters() -> Iterator[None]:
    _reset()
    yield
    _reset()


@pytest.fixture
def conformance_suites() -> SimpleNamespace:
    return SimpleNamespace(
        rest_limiter=conformance.assert_conforms_rest_limiter,
        pausable=conformance.assert_conforms_pausable,
        throttler=conformance.assert_conforms_throttler,
        backoff=conformance.assert_conforms_backoff,
        token_bucket=conformance.assert_conforms_token_bucket,
        unpausable=conformance.assert_conforms_unpausable,
        retry_after_cap=conformance.assert_conforms_retry_after_cap,
    )
