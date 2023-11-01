"""Tests of matrix manipulating functions."""

import numpy as np
import pytest
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from symfc.spg_reps import SpgReps
from symfc.utils import (
    get_indep_atoms_by_lat_trans,
    get_lat_trans_compr_indices,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
)


def test_get_indep_atoms_by_lattice_translation(ph_nacl_222: Phonopy):
    """Test get_indep_atoms_by_lattice_translation."""
    ph = ph_nacl_222
    sym_op_reps = SpgReps(ph.supercell)
    trans_perms = sym_op_reps.translation_permutations
    assert trans_perms.shape == (32, 64)
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    np.testing.assert_array_equal(indep_atoms, [0, 32])


@pytest.mark.parametrize("shape", ["NN33", "N3,N3"])
def test_get_lat_trans_decompr_indices(cell_nacl_111: PhonopyAtoms, shape: str):
    """Test get_lat_trans_decompr_indices.

    The one dimensional array with row-size of compr-mat.
    Every element indicates column position.

    """
    unitcell = cell_nacl_111
    sym_op_reps = SpgReps(unitcell)
    trans_perms = sym_op_reps.translation_permutations
    assert trans_perms.shape == (4, 8)
    compr_mat = get_lat_trans_compr_matrix(trans_perms).toarray()
    decompr_idx = get_lat_trans_decompr_indices(trans_perms, shape=shape)
    if shape == "N3,N3":
        N = len(unitcell.numbers)
        decompr_idx = np.transpose(
            decompr_idx.reshape(N, 3, N, 3), axes=[0, 2, 1, 3]
        ).ravel()
    for r, c in enumerate(decompr_idx):
        np.testing.assert_almost_equal(compr_mat[r, c], 0.5)


def test_get_lat_trans_compr_indices(cell_nacl_111: PhonopyAtoms):
    """Test get_lat_trans_compr_indices.

    The two dimensional array (n_a * N * 9, n_lp) stores NN33 indices where
    compression matrix elements are non-zero.

    """
    unitcell = cell_nacl_111
    sym_op_reps = SpgReps(unitcell)
    trans_perms = sym_op_reps.translation_permutations
    assert trans_perms.shape == (4, 8)
    compr_mat = get_lat_trans_compr_matrix(trans_perms).toarray()
    compr_idx = get_lat_trans_compr_indices(trans_perms)
    for c, elem_idx in enumerate(compr_idx):
        for r in elem_idx:
            np.testing.assert_almost_equal(compr_mat[r, c], 0.5)
