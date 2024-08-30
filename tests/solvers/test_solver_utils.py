"""Tests of functions in solvers."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from symfc.solvers.solver_O2 import reshape_nN33_nx_to_N3_n3nx
from symfc.solvers.solver_O2O3 import (
    reshape_nNN333_nx_to_N3N3_n3nx,
    set_disps_N3N3,
)


def test_reshape_O2():
    """Test reshape_nN33_nx_to_N3_n3nx in solver_O2."""
    N = 30
    n = 15
    nx = 23
    row = [365, 243, 123, 93, 832, 581]
    col = [21, 15, 17, 17, 3, 7]
    data = [1, 2, 4, 3, 6, 5]
    mat = csr_array((data, (row, col)), shape=(n * N * 9, nx), dtype=int)
    mat_reshape = reshape_nN33_nx_to_N3_n3nx(mat, N, n)
    row_reshape, col_reshape = mat_reshape.nonzero()
    assert mat_reshape.shape == (90, 1035)
    np.testing.assert_array_equal(row_reshape, [7, 14, 30, 32, 39, 81])
    np.testing.assert_array_equal(col_reshape, [233, 168, 40, 113, 63, 15])
    np.testing.assert_array_equal(mat_reshape.data, [6, 5, 3, 1, 4, 2])


def test_reshape_O3():
    """Test reshape_nNN333_nx_to_N3N3_n3nx in solver_O2O3."""
    N = 30
    N = 4
    n = 2
    nx = 23
    row = [365, 243, 123, 93, 832, 581]
    col = [21, 15, 17, 17, 3, 7]
    data = [1, 2, 4, 3, 6, 5]
    mat = csr_array((data, (row, col)), shape=(n * N * N * 27, nx), dtype=int)

    mat_reshape = reshape_nNN333_nx_to_N3N3_n3nx(mat, N, n)
    row_reshape, col_reshape = mat_reshape.nonzero()
    assert mat_reshape.shape == (144, 138)
    np.testing.assert_array_equal(row_reshape, [21, 53, 60, 75, 125, 127])
    np.testing.assert_array_equal(col_reshape, [40, 99, 40, 15, 44, 118])
    np.testing.assert_array_equal(mat_reshape.data, [3, 5, 4, 2, 1, 6])


def test_disps_N3N3():
    """Test set_disps_N3N3 in solver_O2O3."""
    disps = np.array(
        [
            [-0.47312848, -0.48690453, 0.16071225, 0.46222049, 0.27704068, 0.17635471],
            [
                0.13874228,
                -0.07145154,
                0.48085802,
                -0.15023848,
                -0.28755176,
                -0.05748815,
            ],
            [0.06457948, -0.43326015, -0.07721428, -0.13029631, 0.01356546, -0.0166158],
        ]
    )
    disps_2nd = set_disps_N3N3(disps)
    assert disps_2nd.shape == (3, 36)
    assert np.sum(disps_2nd) == pytest.approx(0.3518406621303114)
    assert np.sum(np.abs(disps_2nd)) == pytest.approx(6.095152665184942)
    assert disps_2nd[0, 7] == pytest.approx(0.2370760213345209)
    assert disps_2nd[2, 13] == pytest.approx(0.033453870534942)
