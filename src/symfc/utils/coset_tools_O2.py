"""Coset functions for 2nd order force constants."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array, kron

from symfc.spg_reps import SpgRepsO2
from symfc.utils.coset_tools import kron_spg_reps
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.permutation_tools_O2 import PermutationO2
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices


def get_compr_coset_projector_O2(
    spg_reps: SpgRepsO2,
    atomic_decompr_idx: NDArray | None = None,
    fc_cutoff: FCCutoff | None = None,
    permutation: PermutationO2 | None = None,
) -> csr_array:
    """Return compr matrix of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**2 * 9 // n_lp if permutation is None else permutation.col_shape  # type: ignore
    coset_reps_sum = csr_array((size, size), dtype="double")

    if atomic_decompr_idx is None:
        _atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    else:
        _atomic_decompr_idx = atomic_decompr_idx

    if fc_cutoff is None:
        nonzero = None
        cols = _atomic_decompr_idx
    else:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc2()
        cols = _atomic_decompr_idx[nonzero]

    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    size_coset = N**2 // n_lp
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        perms = spg_reps.get_sigma2_rep(i, nonzero=nonzero)
        mat = kron_spg_reps(
            atomic_decompr_idx[perms],
            cols,
            spg_reps.r_reps[i],
            factor,
            size_coset,
        )
        if permutation is not None:
            mat = permutation.blocked_triple_product(mat, use_mkl=False)

        coset_reps_sum += mat

    return coset_reps_sum


def get_compr_coset_reps_sum(spg_reps: SpgRepsO2) -> csr_array:
    """Return compressed projector of coset reps sum.

    Deprecated.
    """
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**2 * 9 // n_lp
    coset_reps_sum = csr_array(([], ([], [])), shape=(size, size), dtype="double")
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    C = csr_array(
        (
            np.ones(N**2, dtype=int),
            (np.arange(N**2, dtype=int), atomic_decompr_idx),
        ),
        shape=(N**2, N**2 // n_lp),
    )
    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        mat = spg_reps.get_sigma2_rep(i)
        mat = mat @ C
        mat = C.T @ mat
        coset_reps_sum += kron(mat, spg_reps.r_reps[i] * factor)

    return coset_reps_sum
