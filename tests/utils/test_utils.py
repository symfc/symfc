"""Tests of matrix manipulating functions."""

import numpy as np
from phonopy import Phonopy

from symfc.spg_reps import SpgRepsBase
from symfc.utils.utils import get_indep_atoms_by_lat_trans


def test_get_indep_atoms_by_lattice_translation(ph_nacl_222: Phonopy):
    """Test of get_indep_atoms_by_lattice_translation."""
    ph = ph_nacl_222
    sym_op_reps = SpgRepsBase(ph.supercell)
    trans_perms = sym_op_reps.translation_permutations
    assert trans_perms.shape == (32, 64)
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    np.testing.assert_array_equal(indep_atoms, [0, 32])
