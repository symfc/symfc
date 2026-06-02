"""Tests of functions in permutation_tools_O3."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.permutation_tools_O3 import PermutationO3, _N3N3N3_to_NNNand333
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def test_N3N3N3_to_NNNand333():
    """Test N3N33_to_NNNand333."""
    N = 3
    combs = np.array([[0, 1, 2], [2, 4, 6], [3, 5, 8]])
    vecNNN, vec333 = _N3N3N3_to_NNNand333(combs, N)
    np.testing.assert_allclose(vecNNN, [0, 5, 14])
    np.testing.assert_allclose(vec333, [5, 21, 8])


def test_PermutationO3_1(cell_spg_reps_bcc):
    """Test PermutationO3."""
    _, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    perm3 = PermutationO3(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=None,
    ).run()
    c_pt = perm3.basis_set
    proj = c_pt @ c_pt.T
    assert proj.trace() == pytest.approx(28.0)
    assert proj.shape == (108, 108)
    assert len(proj.data) == 498
    assert np.count_nonzero(np.isclose(proj.data, 1)) == 3
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 3.0)) == 135
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 6.0)) == 360


def test_PermutationO3_2(cell_spg_reps_bcc):
    """Test PermutationO3."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    perm3 = PermutationO3(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=FCCutoff(supercell, cutoff=1),
    ).run()
    c_pt = perm3.basis_set
    proj = c_pt @ c_pt.T
    assert proj.trace() == pytest.approx(10.0)
    assert len(proj.data) == 93
    assert np.count_nonzero(np.isclose(proj.data, 1)) == 3
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 3.0)) == 54
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 6.0)) == 36


def test_PermutationO3_methods(cell_spg_reps_bcc):
    """Test methods in PermutationO3."""
    _, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    perm3 = PermutationO3(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=None,
    )

    mat = np.random.random((5, 5))
    cp1 = np.random.random((5, 6))
    cp2 = np.random.random((5, 3))
    cp = np.hstack((cp1, cp2))
    true = cp.T @ mat @ cp

    mat = csr_array(mat)
    cp1 = csr_array(cp1)
    cp2 = csr_array(cp2)
    perm3._cpt_array = [cp1, cp2]
    mat = perm3.blocked_triple_product(mat)
    np.testing.assert_allclose(mat.toarray(), true)

    assert perm3.col_shape == 9
    assert perm3.basis_set.shape == (5, 9)
    assert len(perm3.divided_basis_set) == 2
