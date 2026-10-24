import rate_limit
from rate_limit.__about__ import __version__


def test_version_is_str() -> None:
    assert isinstance(rate_limit.__version__, str)


def test_version_matches_about() -> None:
    assert rate_limit.__version__ == __version__
