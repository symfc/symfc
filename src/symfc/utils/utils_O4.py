"""Utility functions for 4th order force constants."""

import numpy as np
from scipy.sparse import csr_array

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
        shape=(n_l, N), where n_l and N are the numbers of lattice points and
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

    trans_perms = trans_perms.astype("int_")
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


def get_lat_trans_decompr_indices_O4(trans_perms: np.ndarray) -> np.ndarray:
    """Return indices to de-compress compressed matrix by lat-trans-sym.

    Usage
    -----
    vec[indices] of shape (n_a*N*N*N*81,) gives an array of shape=(N**4*81,).
    1/sqrt(n_lp) must be multiplied manually after decompression to mimic
    get_lat_trans_compr_matrix.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattice points and
        atoms in supercell.

    Returns
    -------
    indices : ndarray
        Indices of n_a * N * N * N * 81 elements.
        shape=(N^4*81,), dtype='int_'.

    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_a = len(indep_atoms)
    N = trans_perms.shape[1]
    n_lp = N // n_a
    size_row = 81 * N**4

    trans_perms = trans_perms.astype("int_")
    n = 0
    indices = np.zeros(size_row, dtype="int_")
    for i_patom in indep_atoms:
        index_shift_i = trans_perms[:, i_patom] * N**3 * 81
        for j in range(N):
            index_shift_j = index_shift_i + trans_perms[:, j] * N**2 * 81
            for k in range(N):
                index_shift_k = index_shift_j + trans_perms[:, k] * N * 81
                for ll in range(N):
                    index_shift = index_shift_k + trans_perms[:, ll] * 81
                    for ab in range(81):
                        indices[index_shift + ab] = n
                        n += 1
    assert n * n_lp == size_row
    return indices


def get_lat_trans_compr_matrix_O4(trans_perms):
    """Return lat trans compression matrix."""
    n_lp, N = trans_perms.shape
    decompr_idx = get_lat_trans_decompr_indices_O4(trans_perms)
    c_trans = _get_lat_trans_compr_matrix_O4(decompr_idx, N, n_lp)
    return c_trans


def _get_lat_trans_compr_matrix_O4(
    decompr_idx: np.ndarray, N: int, n_lp: int
) -> csr_array:
    """Return compression matrix by lattice translation symmetry.

    `decompr_idx` is obtained by `get_lat_trans_decompr_indices`.

    Matrix shape is (NNNN3333, n_a*NNN3333), where n_a is the number of independent
    atoms by lattice translation symmetry.

    Data order is (N, N, N, N, 3, 3, 3, 3, n_a, N, N, N, 3, 3, 3, 3)
    if it is in dense array.

    """
    NNNN81 = N**4 * 81
    compression_mat = csr_array(
        (
            np.full(NNNN81, 1 / np.sqrt(n_lp), dtype="double"),
            (np.arange(NNNN81, dtype="int_"), decompr_idx),
        ),
        shape=(NNNN81, NNNN81 // n_lp),
        dtype="double",
    )
    return compression_mat
