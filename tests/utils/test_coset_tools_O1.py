"""Tests of functions in coset_tools_O1."""

import numpy as np
import pytest

from symfc.spg_reps import SpgRepsO1
from symfc.utils.coset_tools_O1 import get_compr_coset_projector_O1


def test_coset_projector_O1(cell_spg_reps_bcc):
    """Test get_compr_coset_projector_O1."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
    spg_reps = SpgRepsO1(supercell)
    coset = get_compr_coset_projector_O1(spg_reps)
    assert coset.trace() == pytest.approx(0.0)
    assert np.sum(coset.data) == pytest.approx(0.0)
