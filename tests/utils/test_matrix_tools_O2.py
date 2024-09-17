"""Tests of functions in matrix_tools_O2."""

import numpy as np
import pytest
import scipy

from symfc.spg_reps import SpgRepsBase
from symfc.utils.matrix_tools_O2 import (
    N3N3_to_NNand33,
    compressed_projector_sum_rules_O2,
    projector_permutation_lat_trans_O2,
)
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import (
    _get_atomic_lat_trans_decompr_indices,
)


def structure_bcc():
    """Get bcc structure."""
    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    spg_reps = SpgRepsBase(supercell)
    trans_perms = spg_reps.translation_permutations
    return supercell, trans_perms


supercell, trans_perms = structure_bcc()


def test_N3N3_to_NNand33():
    """Test N3N3_to_NNand33."""
    N = 3
    combs = np.array([[0, 1], [2, 4], [5, 8]])
    vecNN, vec33 = N3N3_to_NNand33(combs, N)
    np.testing.assert_allclose(vecNN, [0, 1, 5])
    np.testing.assert_allclose(vec33, [1, 7, 8])


def test_projector_permutation_lat_trans_O2():
    """Test projector_permutation_lat_trans_O2."""
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    proj = projector_permutation_lat_trans_O2(
        trans_perms,
        atomic_decompr_idx,
        fc_cutoff=None,
    )
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


def test_compressed_projector_sum_rules_O2():
    """Test compressed_projector_sum_rules_O2."""
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    n_a_compress_mat = scipy.sparse.identity(18)
    proj = compressed_projector_sum_rules_O2(
        trans_perms,
        n_a_compress_mat,
        atomic_decompr_idx,
        fc_cutoff=None,
    )
    eigvals, _ = np.linalg.eigh(proj.toarray())
    assert proj.shape == (18, 18)
    assert np.count_nonzero(np.isclose(eigvals, 1.0)) == 9
