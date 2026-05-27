"""Tests of functions in permutation_tools_O4."""

import numpy as np
import pytest

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.permutation_tools_O4 import (
    _N3N3N3N3_to_NNNNand3333,
    compr_permutation_lat_trans_O4,
)
from symfc.utils.utils_O4 import get_atomic_lat_trans_decompr_indices_O4


def test_N3N3N3N3_to_NNNNand3333():
    """Test N3N33_to_NNNand333."""
    N = 3
    combs = np.array([[0, 1, 2, 5], [2, 4, 6, 8], [3, 4, 5, 8]])
    vecNNNN, vec3333 = _N3N3N3N3_to_NNNNand3333(combs, N)
    np.testing.assert_allclose(vecNNNN, [1, 17, 41])
    np.testing.assert_allclose(vec3333, [17, 65, 17])


def test_projector_permutation_lat_trans_O4(cell_spg_reps_bcc):
    """Test projector_permutation_lat_trans_O4."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
    c_pt = compr_permutation_lat_trans_O4(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=None,
    )
    proj = c_pt @ c_pt.T
    assert proj.trace() == pytest.approx(66.0)
    assert proj.shape == (648, 648)
    assert len(proj.data) == 8694
    assert np.count_nonzero(np.isclose(proj.data, 1)) == 3
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 3.0)) == 27
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 6.0)) == 216

    c_pt = compr_permutation_lat_trans_O4(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=FCCutoff(supercell, cutoff=2),
    )
    proj = c_pt @ c_pt.T
    assert proj.trace() == pytest.approx(66.0)
    assert len(proj.data) == 8694
    assert np.count_nonzero(np.isclose(proj.data, 1)) == 3
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 3.0)) == 27
    assert np.count_nonzero(np.isclose(proj.data, 1.0 / 6.0)) == 216
