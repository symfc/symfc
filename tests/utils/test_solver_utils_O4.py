"""Tests of functions in solvers O4."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from symfc.utils.solver_utils_O4 import (
    _reshape_nNNN3333_nx_to_N3N3N3_n3nx,
    set_disps_N3N3N3,
)


def test_reshape_O4():
    """Test _reshape_nNNN3333_nx_to_N3N3N3_n3nx."""
    N = 4
    n = 2
    nx = 23
    row = [365, 243, 123, 93, 832, 581]
    col = [21, 15, 17, 17, 3, 7]
    data = [1, 2, 4, 3, 6, 5]
    mat = csr_array((data, (row, col)), shape=(n * N * N * N * 81, nx), dtype=int)

    mat_reshape = _reshape_nNNN3333_nx_to_N3N3N3_n3nx(mat, N, n, n_batch=3)
    row_reshape, col_reshape = mat_reshape.nonzero()
    assert mat_reshape.shape == (1728, 138)
    np.testing.assert_array_equal(row_reshape, [9, 159, 171, 194, 203, 379])
    np.testing.assert_array_equal(col_reshape, [15, 17, 40, 44, 7, 3])
    np.testing.assert_array_equal(mat_reshape.data, [2, 3, 4, 1, 5, 6])


def test_disps_N3N3N3():
    """Test set_disps_N3N3N3."""
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
    disps_3rd = set_disps_N3N3N3(disps)
    assert disps_3rd.shape == (3, 216)
    assert np.sum(disps_3rd) == pytest.approx(-0.19262699606372016)
    assert np.sum(np.abs(disps_3rd)) == pytest.approx(10.511858648239011)
    assert disps_3rd[0, 7] == pytest.approx(-0.11216741761844945)
    assert disps_3rd[2, 13] == pytest.approx(0.0021604335631338757)
