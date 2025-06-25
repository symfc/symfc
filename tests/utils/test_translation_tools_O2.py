"""Tests of functions in translation_tools_O2."""

import numpy as np
import pytest
import scipy

from symfc.spg_reps import SpgRepsBase
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.translation_tools_O2 import compressed_projector_sum_rules_O2
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices


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


def test_compressed_projector_sum_rules_O2():
    """Test compressed_projector_sum_rules_O2."""
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    n_a_compress_mat = scipy.sparse.identity(18)
    proj = compressed_projector_sum_rules_O2(
        trans_perms,
        n_a_compress_mat,
        atomic_decompr_idx,
        fc_cutoff=None,
    )
    eigvals, _ = np.linalg.eigh(proj.toarray())
    assert proj.shape == (18, 18)
    assert np.count_nonzero(np.isclose(eigvals, 1.0)) == 9

    proj = compressed_projector_sum_rules_O2(
        trans_perms,
        n_a_compress_mat,
        atomic_decompr_idx,
        fc_cutoff=FCCutoff(supercell, cutoff=1),
    )
    """If the cutoff implementation is changed, the trace value may also change."""
    assert proj.trace() == pytest.approx(13.5)
    assert len(proj.data) == 18
