"""Tests of functions in permutation_tools_O4."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.permutation_tools_O4 import PermutationO4, _N3N3N3N3_to_NNNNand3333
from symfc.utils.utils_O4 import get_atomic_lat_trans_decompr_indices_O4


def test_N3N3N3N3_to_NNNNand3333():
    """Test N3N33_to_NNNand333."""
    N = 3
    combs = np.array([[0, 1, 2, 5], [2, 4, 6, 8], [3, 4, 5, 8]])
    vecNNNN, vec3333 = _N3N3N3N3_to_NNNNand3333(combs, N)
    np.testing.assert_allclose(vecNNNN, [1, 17, 41])
    np.testing.assert_allclose(vec3333, [17, 65, 17])


def test_PermutationO4(cell_spg_reps_bcc):
    """Test PermutationO4."""
    _, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
    perm4 = PermutationO4(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=None,
    ).run()
    c_pt = perm4.basis_set
    proj = c_pt @ c_pt.T
    assert proj.trace() == pytest.approx(66.0)
    assert proj.shape == (648, 648)
    assert len(proj.data) == 8694
    assert np.count_nonzero(np.isclose(proj.data, 1)) == 3
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 3.0)) == 27
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 6.0)) == 216


def test_PermutationO4_2(cell_spg_reps_bcc):
    """Test PermutationO4."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
    perm4 = PermutationO4(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=FCCutoff(supercell, cutoff=2),
    ).run()
    c_pt = perm4.basis_set
    proj = c_pt @ c_pt.T
    assert proj.trace() == pytest.approx(66.0)
    assert len(proj.data) == 8694
    assert np.count_nonzero(np.isclose(proj.data, 1)) == 3
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 3.0)) == 27
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 6.0)) == 216


def test_PermutationO4_methods(cell_spg_reps_bcc):
    """Test methods in PermutationO4."""
    _, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
    perm4 = PermutationO4(
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
    perm4._cpt_array = [cp1, cp2]
    mat = perm4.blocked_triple_product(mat)
    np.testing.assert_allclose(mat.toarray(), true)

    assert perm4.col_shape == 9
    assert perm4.basis_set.shape == (5, 9)
    assert len(perm4.divided_basis_set) == 2


def test_PermutationO4_methods_2(cell_spg_reps_bcc):
    """Test methods in PermutationO4."""
    _, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
    perm4 = PermutationO4(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=None,
    )

    mat = np.random.random((5, 3))
    cp1 = np.random.random((6, 3))
    cp2 = np.random.random((6, 2))
    cp = np.hstack((cp1, cp2))
    true = cp @ mat

    mat = csr_array(mat)
    cp1 = csr_array(cp1)
    cp2 = csr_array(cp2)
    perm4._cpt_array = [cp1, cp2]
    mat = perm4.blocked_product(mat)
    np.testing.assert_allclose(mat.toarray(), true)

    assert perm4.col_shape == 5
    assert perm4.basis_set.shape == (6, 5)
    assert len(perm4.divided_basis_set) == 2
