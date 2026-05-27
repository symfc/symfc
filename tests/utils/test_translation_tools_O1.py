"""Tests of functions in translation_tools_O1."""

import numpy as np
import scipy

from symfc.utils.translation_tools_O1 import compressed_projector_sum_rules


def test_compressed_projector_sum_rules(cell_spg_reps_bcc):
    """Test compressed_projector_sum_rules."""
    _, trans_perms, _ = cell_spg_reps_bcc
    n_a_compress_mat = scipy.sparse.identity(6)
    proj = compressed_projector_sum_rules(n_a_compress_mat, trans_perms.shape[1])
    eigvals, _ = np.linalg.eigh(proj.toarray())
    assert proj.shape == (6, 6)
    assert np.count_nonzero(np.isclose(eigvals, 1.0)) == 3
