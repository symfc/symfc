"""Tests of matrix manipulating functions."""

import numpy as np
from phonopy import Phonopy
from scipy.sparse import csr_array

from symfc.spg_reps import SpgReps
from symfc.utils import get_indep_atoms_by_lattice_translation, kron_c


def test_kron_c_NaCl_222(ph_nacl_222: Phonopy):
    """Test kron_c by its rank."""
    ph = ph_nacl_222
    sym_op_reps = SpgReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
    )
    natom = len(ph.supercell)
    size_sq = natom**2 * 9
    row, col, data = kron_c(sym_op_reps.representations, natom)
    proj_mat = csr_array((data, (row, col)), shape=(size_sq, size_sq), dtype="double")
    rank = np.rint(proj_mat.diagonal().sum()).astype(int)
    assert rank == 42


def test_get_indep_atoms_by_lattice_translation(ph_nacl_222: Phonopy):
    """Test get_indep_atoms_by_lattice_translation."""
    ph = ph_nacl_222
    sym_op_reps = SpgReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
    )
    trans_perms = sym_op_reps.translation_permutations
    indep_atoms = get_indep_atoms_by_lattice_translation(trans_perms)
    np.testing.assert_array_equal(indep_atoms, [0, 32])