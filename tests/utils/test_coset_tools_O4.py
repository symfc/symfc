"""Tests of functions in coset_tools_O4."""

import numpy as np
import pytest

from symfc.spg_reps import SpgRepsO4
from symfc.utils.coset_tools_O4 import get_compr_coset_projector_O4
from symfc.utils.utils_O4 import get_atomic_lat_trans_decompr_indices_O4


def test_coset_projector_O4(cell_spg_reps_bcc):
    """Test get_compr_coset_projector_O4."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
    spg_reps = SpgRepsO4(supercell)
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
    coset = get_compr_coset_projector_O4(spg_reps, atomic_decompr_idx)
    assert coset.trace() == pytest.approx(32.0)
    assert np.sum(coset.data) == pytest.approx(168.0)
