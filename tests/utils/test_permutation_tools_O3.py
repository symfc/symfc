"""Tests of functions in permutation_tools_O3."""

import numpy as np
import pytest

from symfc.spg_reps import SpgRepsBase
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.permutation_tools_O3 import (
    _N3N3N3_to_NNNand333,
    compr_permutation_lat_trans_O3,
)
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


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


def test_N3N3N3_to_NNNand333():
    """Test N3N33_to_NNNand333."""
    N = 3
    combs = np.array([[0, 1, 2], [2, 4, 6], [3, 5, 8]])
    vecNNN, vec333 = _N3N3N3_to_NNNand333(combs, N)
    np.testing.assert_allclose(vecNNN, [0, 5, 14])
    np.testing.assert_allclose(vec333, [5, 21, 8])


def test_projector_permutation_lat_trans_O3():
    """Test projector_permutation_lat_trans_O3."""
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    c_pt = compr_permutation_lat_trans_O3(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=None,
    )
    proj = c_pt @ c_pt.T
    assert proj.trace() == pytest.approx(28.0)
    assert proj.shape == (108, 108)
    assert len(proj.data) == 498
    assert np.count_nonzero(np.isclose(proj.data, 1)) == 3
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 3.0)) == 135
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 6.0)) == 360

    c_pt = compr_permutation_lat_trans_O3(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=FCCutoff(supercell, cutoff=1),
    )
    proj = c_pt @ c_pt.T
    assert proj.trace() == pytest.approx(10.0)
    assert len(proj.data) == 93
    assert np.count_nonzero(np.isclose(proj.data, 1)) == 3
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 3.0)) == 54
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 6.0)) == 36
