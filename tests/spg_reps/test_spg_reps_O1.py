"""Tests of SpgRepsO1 class."""

import numpy as np

from symfc.spg_reps import SpgRepsO1
from symfc.utils.utils import SymfcAtoms


def test_spg_reps_o1_get_sigma1_rep_NaCl111(cell_nacl_111: SymfcAtoms):
    """Test of SpgRepsO1.get_sigma1_rep."""
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


def test_spg_reps_o1_r_reps_NaCl111(cell_nacl_111: SymfcAtoms):
    """Test of SpgRepsO1.r_reps."""
    ref_r_reps = np.array(
        [
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
    )

    spg_reps_o1 = SpgRepsO1(cell_nacl_111)
    for r in spg_reps_o1.r_reps:
        r_ravel = r.toarray().ravel()
        ids = np.where(np.linalg.norm(ref_r_reps - r_ravel, axis=1) < 1e-5)[0]
        assert len(ids) == 1
