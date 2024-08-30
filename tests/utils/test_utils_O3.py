"""Tests of functions in utils_O3."""

import numpy as np
import pytest

from symfc.spg_reps import SpgRepsO3
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O3 import (
    get_atomic_lat_trans_decompr_indices_O3,
    get_compr_coset_projector_O3,
    get_lat_trans_compr_matrix_O3,
    get_lat_trans_decompr_indices_O3,
)


def structure_bcc():
    """Get bcc structure."""
    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    spg_reps = SpgRepsO3(supercell)
    trans_perms = spg_reps.translation_permutations
    return supercell, trans_perms, spg_reps


def structure_bcc_rev():
    """Get modified bcc structure."""
    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.49, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    spg_reps = SpgRepsO3(supercell)
    trans_perms = spg_reps.translation_permutations
    return supercell, trans_perms, spg_reps


supercell, trans_perms, spg_reps = structure_bcc()


def test_lat_trans():
    """Test lat_trans_indices and lat_trans_compr_matrix."""
    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    np.testing.assert_array_equal(atomic_decompr_idx, [0, 1, 2, 3, 3, 2, 1, 0])

    decompr_idx_from_atomic = (
        atomic_decompr_idx[:, None] * 27 + np.arange(27)[None, :]
    ).reshape(-1)
    np.testing.assert_array_equal(decompr_idx, decompr_idx_from_atomic)

    c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
    row, col = c_trans.nonzero()
    np.testing.assert_array_equal(decompr_idx, col)
    np.testing.assert_allclose(c_trans.data, [0.7071067811865475] * len(decompr_idx))


def test_coset_projector_O3():
    """Test get_compr_coset_projector_O3."""
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    coset = get_compr_coset_projector_O3(spg_reps, atomic_decompr_idx)
    assert coset.trace() == pytest.approx(0.0)
    assert np.sum(coset.data) == pytest.approx(0.0)

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
