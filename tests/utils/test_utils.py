"""Tests of matrix manipulating functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from symfc.spg_reps import SpgRepsBase
from symfc.utils.utils import (
    SymfcAtoms,
    compute_sg_permutations,
    get_indep_atoms_by_lat_trans,
)

cwd = Path(__file__).parent


def test_get_indep_atoms_by_lattice_translation(
    ph_nacl_222: tuple[SymfcAtoms, np.ndarray, np.ndarray],
):
    """Test of get_indep_atoms_by_lattice_translation."""
    supercell, _, _ = ph_nacl_222
    sym_op_reps = SpgRepsBase(supercell)
    trans_perms = sym_op_reps.translation_permutations
    assert trans_perms.shape == (32, 64)
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    np.testing.assert_array_equal(indep_atoms, [0, 32])


def test_compute_sg_permutations(
    ph_gan_222: tuple[SymfcAtoms, np.ndarray, np.ndarray], cell_gan_111: SymfcAtoms
):
    """Test compute_sg_permutations."""
    pytest.importorskip("spglib", minversion="2.5")
    from spglib import spglib

    supercell, _, _ = ph_gan_222
    primitive = cell_gan_111
    dataset = spglib.get_symmetry_dataset(supercell.totuple())
    primitive_dataset = spglib.get_symmetry_dataset(primitive.totuple())
    perms = compute_sg_permutations(
        primitive.scaled_positions,
        primitive_dataset.rotations,
        primitive_dataset.translations,
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

    perms_super = compute_sg_permutations(
        supercell.scaled_positions,
        dataset.rotations,
        dataset.translations,
        supercell.cell,
    )
    # np.savetxt("perms_super.dat", perms_super, fmt="%d")
    perms_super_ref = np.loadtxt(cwd / ".." / "perms_super.dat", dtype=int)
    np.testing.assert_array_equal(perms_super_ref, perms_super)
