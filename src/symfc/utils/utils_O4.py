"""Utility functions for 4th order force constants."""

import numpy as np
from scipy.sparse import csr_array, kron

from symfc.spg_reps import SpgRepsO4
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.utils import get_indep_atoms_by_lat_trans


def get_atomic_lat_trans_decompr_indices_O4(trans_perms: np.ndarray) -> np.ndarray:
    """Return indices to de-compress compressed matrix by atom-lat-trans-sym.

    This is atomic permutation only version of get_lat_trans_decompr_indices.

    Usage
    -----
    vec[indices] of shape (n_a*N*N*N,) gives an array of shape=(N**4,).
    1/sqrt(n_lp) must be multiplied manually after decompression.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattce points and
        atoms in supercell.

    Returns
    -------
    indices : ndarray
        Indices of n_a * N * N * N elements.
        shape=(N**4,), dtype='int_'.

    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_lp, N = trans_perms.shape
    size_row = N**4

    n = 0
    indices = np.zeros(size_row, dtype="int_")
    for i_patom in indep_atoms:
        index_shift_i = trans_perms[:, i_patom] * N**3
        for j in range(N):
            index_shift_j = index_shift_i + trans_perms[:, j] * N**2
            for k in range(N):
                index_shift_k = index_shift_j + trans_perms[:, k] * N
                for ll in range(N):
                    index_shift = index_shift_k + trans_perms[:, ll]
                    indices[index_shift] = n
                    n += 1
    assert n * n_lp == size_row
    return indices


def get_compr_coset_projector_O4(
    spg_reps: SpgRepsO4,
    fc_cutoff: FCCutoff = None,
    atomic_decompr_idx: np.ndarray = None,
    c_pt: csr_array = None,
    verbose: bool = False,
) -> csr_array:
    """Return compr projector of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**4 * 81 // n_lp if c_pt is None else c_pt.shape[1]
    coset_reps_sum = csr_array(([], ([], [])), shape=(size, size), dtype="double")

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)

    if fc_cutoff is None:
        nonzero = None
        size_data = N**4
    else:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc4()
        size_data = np.count_nonzero(nonzero)

    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        if verbose:
            print("Coset sum:", i + 1, "/", len(spg_reps.unique_rotation_indices))
        permutation = spg_reps.get_sigma4_rep(i, nonzero=nonzero)
        if nonzero is None:
            """Equivalent to mat = C.T @ spg_reps.get_sigma4_rep(i) @ C
            C: atomic_lat_trans_compr_mat, shape=(NNNN, NNNN/n_lp)"""
            mat = csr_array(
                (
                    np.ones(size_data, dtype="int_"),
                    (atomic_decompr_idx[permutation], atomic_decompr_idx),
                ),
                shape=(N**4 // n_lp, N**4 // n_lp),
                dtype="int_",
            )
        else:
            mat = csr_array(
                (
                    np.ones(size_data, dtype="int_"),
                    (atomic_decompr_idx[permutation], atomic_decompr_idx[nonzero]),
                ),
                shape=(N**4 // n_lp, N**4 // n_lp),
                dtype="int_",
            )

        mat = kron(mat, spg_reps.r_reps[i] * factor)
        if c_pt is not None:
            mat = c_pt.T @ mat @ c_pt

        coset_reps_sum += mat

    return coset_reps_sum
