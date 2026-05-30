"""Tests of functions in utils_O2."""

import numpy as np

from symfc.utils.utils_O2 import (
    _get_atomic_lat_trans_decompr_indices,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
)


def test_lat_trans(cell_spg_reps_bcc):
    """Test lat_trans_indices and lat_trans_compr_matrix."""
    _, trans_perms, _ = cell_spg_reps_bcc
    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    np.testing.assert_array_equal(atomic_decompr_idx, [0, 1, 1, 0])

    decompr_idx_from_atomic = (
        atomic_decompr_idx[:, None] * 9 + np.arange(9)[None, :]
    ).reshape(-1)
    np.testing.assert_array_equal(decompr_idx, decompr_idx_from_atomic)

    N, n_lp = trans_perms.shape
    c_trans = get_lat_trans_compr_matrix(decompr_idx, N, n_lp)
    row, col = c_trans.nonzero()
    np.testing.assert_array_equal(decompr_idx, col)
    np.testing.assert_allclose(c_trans.data, [0.7071067811865475] * len(decompr_idx))
