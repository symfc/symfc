"""Tests of SpgReps class."""
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from symfc.spg_reps import SpgReps, SpgRepsO1


def test_translation_permutations_NaCl_111(cell_nacl_111: PhonopyAtoms):
    """Test SpgReps.translation_permutations."""
    cell = cell_nacl_111
    sym_reps = SpgReps(cell)
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


def test_translation_permutations_shape_GaN_222(ph_gan_222: Phonopy):
    """Test SpgReps.translation_permutations."""
    cell = ph_gan_222.supercell
    sym_reps = SpgReps(cell)
    trans_perms = sym_reps.translation_permutations
    assert trans_perms.shape == (8, 32)


def test_spg_reps_o1(cell_nacl_111: PhonopyAtoms):
    """Test of SpgRepsO1."""
    ref_r_reps = [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
    ]
    spg_reps_o1 = SpgRepsO1(cell_nacl_111)
    N = len(cell_nacl_111)
    for i, _ in enumerate(spg_reps_o1.unique_rotation_indices):
        s_r = spg_reps_o1.get_sigma1_rep(i).toarray()
        r_atoms = s_r @ np.arange(N, dtype=int)
        r_c = spg_reps_o1.r_reps[i]
        L = spg_reps_o1._lattice.T
        r = np.linalg.inv(L) @ r_c @ L
        r_pos = spg_reps_o1._positions @ r.T
        for j, pos in enumerate(spg_reps_o1._positions):
            diff = r_pos - pos
            diff -= np.rint(diff)
            idx = np.where(np.linalg.norm(diff @ L.T, axis=1) < 1e-10)[0][0]
            assert r_atoms[j] == idx

    np.testing.assert_array_almost_equal(
        ref_r_reps, [r.toarray().ravel() for r in spg_reps_o1.r_reps]
    )
