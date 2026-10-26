import pytest

from rate_limit import helpers

pytest_plugins = ["rate_limit.testing.pytest_plugin"]


@pytest.fixture(autouse=True)
def _reset_helper_state() -> None:
    """Module-global helper state must not leak across tests."""
    helpers._UNPAUSABLE_WARNED.clear()
    helpers._UNPAUSABLE_WARNED_IDS.clear()
    helpers._FALLBACK_POLICIES.clear()
    helpers._OWN_PAUSE_DEADLINES.clear()
