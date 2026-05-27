"""Tests of functions in utils_O2."""

import numpy as np
import pytest

from symfc.spg_reps import SpgRepsO2
from symfc.utils.utils_O2 import (
    _get_atomic_lat_trans_decompr_indices,
    get_compr_coset_projector_O2,
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


def test_coset_projector_O2(cell_spg_reps_bcc):
    """Test get_compr_coset_projector_O2."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
    spg_reps = SpgRepsO2(supercell)
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    coset = get_compr_coset_projector_O2(spg_reps, atomic_decompr_idx)
    assert coset.trace() == pytest.approx(2.0)
    assert np.sum(coset.data) == pytest.approx(6.0)
    for irow in [1, 2, 3, 5, 6, 7, 10, 11, 12, 14, 15, 16]:
        assert coset[[irow]].sum() == pytest.approx(0.0)
