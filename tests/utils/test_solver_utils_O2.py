"""Tests of functions in solvers."""

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.solver_utils_O2 import _reshape_nN33_nx_to_N3_n3nx


def test_reshape_O2():
    """Test reshape_nN33_nx_to_N3_n3nx in solver_O2."""
    N = 30
    n = 15
    nx = 23
    row = [365, 243, 123, 93, 832, 581]
    col = [21, 15, 17, 17, 3, 7]
    data = [1, 2, 4, 3, 6, 5]
    mat = csr_array((data, (row, col)), shape=(n * N * 9, nx), dtype=int)
    mat_reshape = _reshape_nN33_nx_to_N3_n3nx(mat, N, n)
    row_reshape, col_reshape = mat_reshape.nonzero()
    assert mat_reshape.shape == (90, 1035)
    np.testing.assert_array_equal(row_reshape, [7, 14, 30, 32, 39, 81])
    np.testing.assert_array_equal(col_reshape, [233, 168, 40, 113, 63, 15])
    np.testing.assert_array_equal(mat_reshape.data, [6, 5, 3, 1, 4, 2])
