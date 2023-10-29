"""Tests of matrix manipulating functions."""

import numpy as np
from phonopy import Phonopy

from symfc.spg_reps import SpgReps
from symfc.utils import get_indep_atoms_by_lat_trans


def test_get_indep_atoms_by_lattice_translation(ph_nacl_222: Phonopy):
    """Test get_indep_atoms_by_lattice_translation."""
    ph = ph_nacl_222
    sym_op_reps = SpgReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
    )
    trans_perms = sym_op_reps.translation_permutations
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    np.testing.assert_array_equal(indep_atoms, [0, 32])
