"""Tests of SpgRepsO2 class."""

import numpy as np

from symfc.spg_reps import SpgRepsO2
from symfc.utils.utils import SymfcAtoms


def test_spg_reps_o2(cell_nacl_111: SymfcAtoms):
    """Test of SpgRepsO2."""
    spg_reps_o2 = SpgRepsO2(cell_nacl_111)
    trace_sum = 0
    for i, _ in enumerate(spg_reps_o2.unique_rotation_indices):
        trace_sum += np.count_nonzero(
            spg_reps_o2.get_sigma2_rep(i) == np.arange(64, dtype=int)
        )
    assert trace_sum == 960
