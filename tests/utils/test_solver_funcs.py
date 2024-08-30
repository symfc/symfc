"""Tests of functions in solver_funcs."""

import numpy as np

from symfc.utils.solver_funcs import fit, get_batch_slice


def test_fit():
    """Test linear regression fit."""
    X = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 4.0, 8.0],
            [1.0, 4.0, 16.0, 64.0],
            [1.0, 6.0, 36.0, 216.0],
            [1.0, 8.0, 64.0, 512.0],
            [1.0, 10.0, 100.0, 1000.0],
        ]
    )
    y = np.array([14.1, 41.9, 179.2, 458.7, 931.0, 1642.6])
    coefs = fit(X, y)
    np.testing.assert_allclose(coefs, [3.18818393, 3.46030191, 6.13455974, 0.99131806])


def test_batch_slice():
    """Test get_batch_slice."""
    begin, end = get_batch_slice(n_data=500, batch_size=50)
    assert len(begin) == len(end) == 10
    diff = np.array(end) - np.array(begin)
    np.testing.assert_array_equal(diff, [50] * 10)

    begin, end = get_batch_slice(n_data=1000, batch_size=36)
    assert len(begin) == len(end) == 28
    diff = np.array(end) - np.array(begin)
    np.testing.assert_array_equal(diff[:-1], [36] * 27)
    assert diff[-1] == 28
