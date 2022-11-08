"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4])
    ]
)
def test_daily_mean(test, expected):
    """Test that mean function works for arrays of zeros and integers."""
    from inflammation.models import daily_mean
    
    print(test, expected)
    npt.assert_array_equal(daily_mean(test), expected)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [5, 6])
    ]
)
def test_daily_max(test, expected):
    """Test that max function works for an array of zeros."""
    from inflammation.models import daily_max

    npt.assert_array_equal(daily_max(test), expected)