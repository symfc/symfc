"""Tests of functions in utils_O2."""

import numpy as np
import pytest

from symfc.spg_reps import SpgRepsO2
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import (
    _get_atomic_lat_trans_decompr_indices,
    get_compr_coset_projector_O2,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
)


def structure_bcc():
    """Get bcc structure."""
    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    spg_reps = SpgRepsO2(supercell)
    trans_perms = spg_reps.translation_permutations
    return supercell, trans_perms, spg_reps


supercell, trans_perms, spg_reps = structure_bcc()


def test_lat_trans():
    """Test lat_trans_indices and lat_trans_compr_matrix."""
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


def test_coset_projector_O2():
    """Test get_compr_coset_projector_O2."""
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    coset = get_compr_coset_projector_O2(spg_reps, atomic_decompr_idx)
    assert coset.trace() == pytest.approx(2.0)
    assert np.sum(coset.data) == pytest.approx(6.0)
    for irow in [1, 2, 3, 5, 6, 7, 10, 11, 12, 14, 15, 16]:
        assert coset[[irow]].sum() == pytest.approx(0.0)
