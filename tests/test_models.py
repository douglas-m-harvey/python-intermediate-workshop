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
=======


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    #from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
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


@pytest.mark.parametrize(
    "test, expected, expect_raises", 
    [
        (np.zeros((3, 3)), np.zeros((3, 3)), None),
        (np.ones((3, 3)), np.ones((3, 3)), None),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None), 
        ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], ValueError)
    ]
 )
def test_patient_normalise(test, expected, expect_raises):
    """Test that normalisation works with a 3x3 array"""
    from inflammation.models import patient_normalise
    if expect_raises is not None: 
        with pytest.raises(expect_raises):
            npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal = 2)
    else: 
        npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal = 2)