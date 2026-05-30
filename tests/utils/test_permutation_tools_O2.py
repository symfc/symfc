"""Tests of functions in permutation_tools_O2."""

import numpy as np
import pytest

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.permutation_tools_O2 import PermutationO2, _N3N3_to_NNand33
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices


def test_N3N3_to_NNand33():
    """Test N3N3_to_NNand33."""
    N = 3
    combs = np.array([[0, 1], [2, 4], [5, 8]])
    vecNN, vec33 = _N3N3_to_NNand33(combs, N)
    np.testing.assert_allclose(vecNN, [0, 1, 5])
    np.testing.assert_allclose(vec33, [1, 7, 8])


def test_permutationO2_1(cell_spg_reps_bcc):
    """Test PermutationO2."""
    _, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    perm2 = PermutationO2(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=None,
    ).run()
    c_pt = perm2.basis_set

    proj = c_pt @ c_pt.T
    assert proj.trace() == pytest.approx(12.0)
    assert proj.shape == (18, 18)
    proj_ref = np.zeros(proj.shape)
    proj_ref[([0, 4, 8, 9, 13, 17], [0, 4, 8, 9, 13, 17])] = 1.0
    for iset in [[1, 3], [2, 6], [5, 7], [10, 12], [11, 15], [14, 16]]:
        row, col = np.meshgrid(*[iset, iset])
        row = row.reshape(-1)
        col = col.reshape(-1)
        proj_ref[(row, col)] = 0.5
    np.testing.assert_allclose(proj.toarray(), proj_ref)


def test_permutationO2_2(cell_spg_reps_bcc):
    """Test PermutationO2."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    perm2 = PermutationO2(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=FCCutoff(supercell, cutoff=1),
    ).run()
    c_pt = perm2.basis_set

    proj = c_pt @ c_pt.T

    proj_ref = np.zeros(proj.shape)
    proj_ref[([0, 4, 8, 9, 13, 17], [0, 4, 8, 9, 13, 17])] = 1.0
    for iset in [[1, 3], [2, 6], [5, 7], [10, 12], [11, 15], [14, 16]]:
        row, col = np.meshgrid(*[iset, iset])
        row = row.reshape(-1)
        col = col.reshape(-1)
        proj_ref[(row, col)] = 0.5

    proj_ref_cutoff = np.zeros_like(proj_ref)
    proj_ref_cutoff[0:9, 0:9] = proj_ref[0:9, 0:9]
    assert proj.trace() == pytest.approx(6.0)
    assert proj.shape == (18, 18)
    np.testing.assert_allclose(proj.toarray(), proj_ref_cutoff)
