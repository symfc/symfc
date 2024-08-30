"""Tests of functions in utils_O2."""

import numpy as np

from symfc.spg_reps import SpgRepsBase
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import (
    _get_atomic_lat_trans_decompr_indices,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
)


def structure_CsCl():
    """Get CsCl structure."""
    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    spg_reps = SpgRepsBase(supercell)
    trans_perms = spg_reps.translation_permutations
    return supercell, trans_perms


supercell, trans_perms = structure_CsCl()


def test_lat_trans():
    """Test lat_trans_indices and lat_trans_compr_matrix."""
    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    decompr_idx_ref = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
    )
    np.testing.assert_array_equal(decompr_idx, decompr_idx_ref)

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
    np.testing.assert_allclose(
        c_trans.data, [0.7071067811865475] * len(decompr_idx_ref)
    )
