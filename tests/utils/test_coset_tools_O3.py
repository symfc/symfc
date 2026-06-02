"""Tests of functions in coset_tools_O3."""

import numpy as np
import pytest

from symfc.spg_reps import SpgRepsO3
from symfc.utils.coset_tools_O3 import get_compr_coset_projector_O3
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def structure_bcc_rev():
    """Get modified bcc structure."""
    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.49, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    spg_reps = SpgRepsO3(supercell)
    trans_perms = spg_reps.translation_permutations
    return supercell, trans_perms, spg_reps


def test_coset_projector_O3(cell_spg_reps_bcc):
    """Test get_compr_coset_projector_O3."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
    spg_reps = SpgRepsO3(supercell)
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    coset = get_compr_coset_projector_O3(spg_reps, atomic_decompr_idx)
    assert coset.trace() == pytest.approx(0.0)
    assert np.sum(coset.data) == pytest.approx(0.0)


def test_coset_projector_O3_rev():
    """Test get_compr_coset_projector_O3 with displaced structure."""
    supercell_rev, trans_perms_rev, spg_reps_rev = structure_bcc_rev()
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms_rev)
    coset = get_compr_coset_projector_O3(spg_reps_rev, atomic_decompr_idx)
    assert coset.trace() == pytest.approx(16.0)
    assert np.sum(coset.data) == pytest.approx(0.0)

    coset = get_compr_coset_projector_O3(
        spg_reps_rev,
        atomic_decompr_idx,
        fc_cutoff=FCCutoff(supercell_rev, cutoff=1),
    )
    assert coset.trace() == pytest.approx(4.0)
    assert np.sum(coset.data) == pytest.approx(0.0)
