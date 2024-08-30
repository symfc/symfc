"""Tests of functions in utils_O1."""

import numpy as np

from symfc.spg_reps import SpgRepsBase
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O1 import (
    _get_atomic_lat_trans_decompr_indices,
    get_lat_trans_decompr_indices,
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


def test_lat_trans_indices():
    """Test lat_trans_indices."""
    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    np.testing.assert_array_equal(decompr_idx, [0, 1, 2, 0, 1, 2])

    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    np.testing.assert_array_equal(atomic_decompr_idx, [0, 0])
