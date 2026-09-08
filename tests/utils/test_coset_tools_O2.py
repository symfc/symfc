"""Tests of functions in coset_tools_O2."""

import numpy as np
import pytest

from symfc.spg_reps import SpgRepsO2
from symfc.utils.coset_tools_O2 import get_compr_coset_projector_O2
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices


def test_coset_projector_O2(cell_spg_reps_bcc):
    """Test get_compr_coset_projector_O2."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
    spg_reps = SpgRepsO2(supercell)
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    coset = get_compr_coset_projector_O2(spg_reps, atomic_decompr_idx)
    assert coset.trace() == pytest.approx(2.0)
    assert np.sum(coset.data) == pytest.approx(6.0)
    for irow in [1, 2, 3, 5, 6, 7, 10, 11, 12, 14, 15, 16]:
        assert coset[[irow]].sum() == pytest.approx(0.0)
