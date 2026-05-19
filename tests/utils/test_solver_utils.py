"""Tests of functions in solver_funcs."""

import numpy as np

from symfc.utils.solver_utils import calc_sum_xtx


def test_calc_sum_xtx():
    """Test calc_sum_xtx."""
    xtx = None
    X = np.random.random((5, 3))
    xtx = calc_sum_xtx(xtx, X, nbytes_threshold=0.0)
    np.testing.assert_allclose(xtx, X.T @ X)

    xtx = calc_sum_xtx(xtx, X, nbytes_threshold=0.0)
    np.testing.assert_allclose(xtx, X.T @ X * 2)
