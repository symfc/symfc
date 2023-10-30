"""Tests of matrix manipulating functions."""

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from symfc.spg_reps import SpgReps
from symfc.utils import (
    get_indep_atoms_by_lat_trans,
    get_lat_trans_compr_indices,
    get_lat_trans_compr_matrix,
)


def test_get_indep_atoms_by_lattice_translation(ph_nacl_222: Phonopy):
    """Test get_indep_atoms_by_lattice_translation."""
    ph = ph_nacl_222
    sym_op_reps = SpgReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
    )
    trans_perms = sym_op_reps.translation_permutations
    assert trans_perms.shape == (32, 64)
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    np.testing.assert_array_equal(indep_atoms, [0, 32])


def test_get_lat_trans_compr_indices(cell_nacl_111: PhonopyAtoms):
    """Test get_lat_trans_compr_indices.

    The one dimensional array with row-size of compr-mat.
    Every element indicates column position.

    """
    unitcell = cell_nacl_111
    sym_op_reps = SpgReps(
        unitcell.cell.T,
        unitcell.scaled_positions.T,
        unitcell.numbers,
    )
    trans_perms = sym_op_reps.translation_permutations
    assert trans_perms.shape == (4, 8)
    compr_mat = get_lat_trans_compr_matrix(trans_perms).toarray()
    compr_idx = get_lat_trans_compr_indices(trans_perms)
    for r, c in enumerate(compr_idx):
        np.testing.assert_almost_equal(compr_mat[r, c], 0.5)
