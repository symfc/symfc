"""Tests of functions in translation_tools_O4."""

import numpy as np
import pytest
import scipy

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.translation_tools_O4 import compressed_projector_sum_rules_O4
from symfc.utils.utils_O4 import get_atomic_lat_trans_decompr_indices_O4


def test_compressed_projector_sum_rules_O4(cell_spg_reps_bcc):
    """Test compressed_projector_sum_rules_O4."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
    n_a_compress_mat = scipy.sparse.identity(648)
    proj = compressed_projector_sum_rules_O4(
        trans_perms,
        n_a_compress_mat,
        atomic_decompr_idx,
        fc_cutoff=None,
    )
    eigvals, _ = np.linalg.eigh(proj.toarray())
    assert proj.shape == (648, 648)
    assert np.count_nonzero(np.isclose(eigvals, 1.0)) == 324

    proj = compressed_projector_sum_rules_O4(
        trans_perms,
        n_a_compress_mat,
        atomic_decompr_idx,
        fc_cutoff=FCCutoff(supercell, cutoff=1),
    )
    """If the cutoff implementation is changed, the trace value may also change."""
    assert proj.trace() == pytest.approx(607.5)
    assert len(proj.data) == 648
