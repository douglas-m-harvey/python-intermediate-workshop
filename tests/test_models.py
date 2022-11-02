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


# def test_daily_max_integers():
#     """Test that max function works for an array of positive integers"""
#     from inflammation.models import daily_max

#     test_input = np.arange(1, 7).reshape((3, 2))
#     test_result = np.array([5, 6])

#     npt.assert_array_equal(daily_max(test_input), test_result)


# def test_daily_min_integers():
#     """Test that max function works for an array of random integers"""
#     from inflammation.models import daily_min

#     test_input = np.arange(1, 7).reshape((3, 2))
#     test_result = np.array([1, 2])

#     npt.assert_array_equal(daily_min(test_input), test_result)