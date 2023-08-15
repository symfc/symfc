"""Tests of SpgReps class."""

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from symfc.spg_reps import SpgReps


def test_SpgReps_NaCl_111(cell_nacl_111: PhonopyAtoms):
    """Test SpgReps by its trace."""
    cell = cell_nacl_111
    sym_op_reps = SpgReps(
        cell.cell.T,
        cell.scaled_positions.T,
        cell.numbers,
    )

    reps = sym_op_reps.representations
    proj = np.zeros_like(reps[0])
    for rep in reps:
        proj += rep
    proj /= len(reps)
    for v in proj.toarray():
        print(v)
    assert np.rint(proj.trace()).astype(int) == 0
    len(proj.data) == 0


def test_SpgReps_NaCl_222(ph_nacl_222: Phonopy):
    """Test SpgReps by its trace."""
    ph = ph_nacl_222
    sym_op_reps = SpgReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
    )

    reps = sym_op_reps.representations
    proj = np.zeros_like(reps[0])
    for rep in reps:
        proj += rep
    proj /= len(reps)
    assert np.rint(proj.trace()).astype(int) == 0
    len(proj.data) == 0


def test_translation_permutations_NaCl_111(cell_nacl_111: PhonopyAtoms):
    """Test SpgReps.translation_permutations."""
    cell = cell_nacl_111
    sym_op_reps = SpgReps(
        cell.cell.T,
        cell.scaled_positions.T,
        cell.numbers,
    )
    trans_perms = sym_op_reps.translation_permutations
    # for v in trans_perms:
    #     print("[", ", ".join([f"{x}" for x in v]), "],")
    ref = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [3, 2, 1, 0, 7, 6, 5, 4],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [1, 0, 3, 2, 5, 4, 7, 6],
    ]
    np.testing.assert_array_equal(trans_perms, ref)
