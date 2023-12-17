"""Tests of matrix manipulating functions."""

import numpy as np
from phonopy import Phonopy
from phonopy.structure.cells import compute_all_sg_permutations

from symfc.spg_reps import SpgRepsBase
from symfc.utils.utils import compute_sg_permutations, get_indep_atoms_by_lat_trans


def test_get_indep_atoms_by_lattice_translation(ph_nacl_222: Phonopy):
    """Test of get_indep_atoms_by_lattice_translation."""
    ph = ph_nacl_222
    sym_op_reps = SpgRepsBase(ph.supercell)
    trans_perms = sym_op_reps.translation_permutations
    assert trans_perms.shape == (32, 64)
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    np.testing.assert_array_equal(indep_atoms, [0, 32])


def test_compute_sg_permutations(ph_gan_222: Phonopy):
    """Test compute_sg_permutations."""

    primitive_dataset = ph_gan_222.primitive_symmetry.dataset
    primitive = ph_gan_222.primitive
    perms = compute_sg_permutations(
        primitive.scaled_positions,
        primitive_dataset["rotations"],
        primitive_dataset["translations"],
        primitive.cell,
    )
    ref_perms = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [1, 0, 3, 2],
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [0, 1, 2, 3],
    ]
    np.testing.assert_array_equal(ref_perms, perms)

    dataset = ph_gan_222.symmetry.dataset
    supercell = ph_gan_222.supercell
    ref_perms_super = compute_all_sg_permutations(
        supercell.scaled_positions,
        dataset["rotations"],
        dataset["translations"],
        supercell.cell,
        1e-5,
    )
    perms_super = compute_sg_permutations(
        supercell.scaled_positions,
        dataset["rotations"],
        dataset["translations"],
        supercell.cell,
    )
    np.testing.assert_array_equal(ref_perms_super, perms_super)
