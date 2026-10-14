"""Shared pytest fixtures."""

import pytest
from pynwb.testing.mock.ecephys import mock_ElectricalSeries


@pytest.fixture
def electrical_series():
    """Return the mock_ElectricalSeries factory.

    The mock builds its own electrode table and accepts data, rate, and
    starting_time overrides.
    """
    return mock_ElectricalSeries
