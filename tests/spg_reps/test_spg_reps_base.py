"""Tests of SpgRepsBase class."""

from __future__ import annotations

import numpy as np

from symfc.spg_reps import SpgRepsBase
from symfc.utils.utils import SymfcAtoms


def test_translation_permutations_NaCl_111(cell_nacl_111: SymfcAtoms):
    """Test SpgReps.translation_permutations."""
    cell = cell_nacl_111
    sym_reps = SpgRepsBase(cell)
    trans_perms = sym_reps.translation_permutations
    # for v in trans_perms:
    #     print("[", ", ".join([f"{x}" for x in v]), "],")
    perms_ref = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [3, 2, 1, 0, 7, 6, 5, 4],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [1, 0, 3, 2, 5, 4, 7, 6],
    ]
    done = []
    for tperm in trans_perms:
        for i, perm in enumerate(perms_ref):
            if np.array_equal(tperm, perm):
                done.append(i)
                break
    np.testing.assert_array_equal(np.sort(done), [0, 1, 2, 3])


def test_translation_permutations_shape_GaN_222(
    ph_gan_222: tuple[SymfcAtoms, np.ndarray, np.ndarray],
):
    """Test SpgReps.translation_permutations."""
    supercell, _, _ = ph_gan_222
    sym_reps = SpgRepsBase(supercell)
    trans_perms = sym_reps.translation_permutations
    assert trans_perms.shape == (8, 32)
