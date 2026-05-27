"""Tests of functions in utils_O1."""

import numpy as np

from symfc.utils.utils_O1 import (
    _get_atomic_lat_trans_decompr_indices,
    get_lat_trans_decompr_indices,
)


def test_lat_trans_indices(cell_spg_reps_bcc):
    """Test lat_trans_indices."""
    _, trans_perms, _ = cell_spg_reps_bcc
    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    np.testing.assert_array_equal(decompr_idx, [0, 1, 2, 0, 1, 2])

    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    np.testing.assert_array_equal(atomic_decompr_idx, [0, 0])
