"""Tests of SpgRepsO4 class."""

import numpy as np

from symfc.spg_reps import SpgRepsO4
from symfc.utils.utils import SymfcAtoms


def test_spg_reps_o4(cell_nacl_111: SymfcAtoms):
    """Test of SpgRepsO4."""
    spg_reps_o4 = SpgRepsO4(cell_nacl_111)
    trace_sum = 0
    for i, _ in enumerate(spg_reps_o4.unique_rotation_indices):
        trace_sum += np.count_nonzero(
            spg_reps_o4.get_sigma4_rep(i) == np.arange(4096, dtype=int)
        )
    assert trace_sum == 39168
