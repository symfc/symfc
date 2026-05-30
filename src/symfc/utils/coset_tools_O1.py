"""Utility functions for 1st order force constants."""

import numpy as np
from scipy.sparse import csr_array, kron

from symfc.spg_reps import SpgRepsO1
from symfc.utils.utils_O1 import _get_atomic_lat_trans_decompr_indices


def get_compr_coset_projector_O1(spg_reps: SpgRepsO1):
    """Return compressed projector of coset reps sum."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N * 3 // n_lp
    coset_reps_sum = csr_array(([], ([], [])), shape=(size, size), dtype="double")
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    C = csr_array(
        (
            np.ones(N, dtype=int),
            (np.arange(N, dtype=int), atomic_decompr_idx),
        ),
        shape=(N, N // n_lp),
    )
    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        mat = spg_reps.get_sigma1_rep(i)
        mat = mat @ C
        mat = C.T @ mat
        coset_reps_sum += kron(mat, spg_reps.r_reps[i] * factor)

    return coset_reps_sum
