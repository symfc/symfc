"""Tests of functions in utils_O3."""

import numpy as np

from symfc.utils.utils_O3 import (
    get_atomic_lat_trans_decompr_indices_O3,
    get_lat_trans_compr_matrix_O3,
    get_lat_trans_decompr_indices_O3,
)


def test_lat_trans(cell_spg_reps_bcc):
    """Test lat_trans_indices and lat_trans_compr_matrix."""
    _, trans_perms, _ = cell_spg_reps_bcc
    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    np.testing.assert_array_equal(atomic_decompr_idx, [0, 1, 2, 3, 3, 2, 1, 0])

    decompr_idx_from_atomic = (
        atomic_decompr_idx[:, None] * 27 + np.arange(27)[None, :]
    ).reshape(-1)
    np.testing.assert_array_equal(decompr_idx, decompr_idx_from_atomic)

    c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
    row, col = c_trans.nonzero()
    np.testing.assert_array_equal(decompr_idx, col)
    np.testing.assert_allclose(c_trans.data, [0.7071067811865475] * len(decompr_idx))
