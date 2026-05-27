"""Tests of functions in translation_tools_O2."""

import numpy as np
import pytest
import scipy

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.translation_tools_O2 import compressed_projector_sum_rules_O2
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices


def test_compressed_projector_sum_rules_O2(cell_spg_reps_bcc):
    """Test compressed_projector_sum_rules_O2."""
    supercell, trans_perms, _ = cell_spg_reps_bcc
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
