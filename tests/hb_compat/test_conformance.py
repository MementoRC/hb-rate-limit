import pytest


@pytest.mark.skip(reason="N/A per spec §4.6: poll-based acquisition does not guarantee FIFO")
def test_fifo_conformance_checks():
    raise AssertionError("never runs")
