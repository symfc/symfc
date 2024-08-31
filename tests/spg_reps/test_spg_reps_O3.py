"""Tests of SpgRepsO3 class."""

import numpy as np

from symfc.spg_reps import SpgRepsO3
from symfc.utils.utils import SymfcAtoms


def test_spg_reps_o3(cell_nacl_111: SymfcAtoms):
    """Test of SpgRepsO3."""
    spg_reps_o3 = SpgRepsO3(cell_nacl_111)
    trace_sum = 0
    for i, _ in enumerate(spg_reps_o3.unique_rotation_indices):
        trace_sum += np.count_nonzero(
            spg_reps_o3.get_sigma3_rep(i) == np.arange(512, dtype=int)
        )
    assert trace_sum == 5760
