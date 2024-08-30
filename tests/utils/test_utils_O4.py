"""Tests of functions in utils_O4."""

import numpy as np

from symfc.spg_reps import SpgRepsBase
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O4 import (
    get_atomic_lat_trans_decompr_indices_O4,
    get_lat_trans_compr_matrix_O4,
    get_lat_trans_decompr_indices_O4,
)


def structure_bcc():
    """Get bcc structure."""
    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    spg_reps = SpgRepsBase(supercell)
    trans_perms = spg_reps.translation_permutations
    return supercell, trans_perms


supercell, trans_perms = structure_bcc()


def test_lat_trans():
    """Test lat_trans_indices and lat_trans_compr_matrix."""
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
