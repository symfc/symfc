"""Tests of functions in utils_O4."""

import numpy as np

from symfc.utils.utils_O4 import (
    get_atomic_lat_trans_decompr_indices_O4,
    get_lat_trans_compr_matrix_O4,
    get_lat_trans_decompr_indices_O4,
)


def test_lat_trans(cell_spg_reps_bcc):
    """Test lat_trans_indices and lat_trans_compr_matrix."""
    _, trans_perms, _ = cell_spg_reps_bcc
    decompr_idx = get_lat_trans_decompr_indices_O4(trans_perms)
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
    np.testing.assert_array_equal(
        atomic_decompr_idx, [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0]
    )

    decompr_idx_from_atomic = (
        atomic_decompr_idx[:, None] * 81 + np.arange(81)[None, :]
    ).reshape(-1)
    np.testing.assert_array_equal(decompr_idx, decompr_idx_from_atomic)

    c_trans = get_lat_trans_compr_matrix_O4(trans_perms)
    row, col = c_trans.nonzero()
    np.testing.assert_array_equal(decompr_idx, col)
    np.testing.assert_allclose(c_trans.data, [0.7071067811865475] * len(decompr_idx))
