"""Utility functions for 2nd order force constants."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from .utils import get_indep_atoms_by_lat_trans


def _get_atomic_lat_trans_decompr_indices(trans_perms: NDArray) -> NDArray:
    """Return indices to de-compress compressed matrix by atom-lat-trans-sym.

    This is atomic permutation only version of get_lat_trans_decompr_indices.


    Usage
    -----
    vec[indices] of shape (n_a*N,) gives an array of shape=(N**2,).
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
        Indices of n_a * N elements.
        shape=(N^2*,), dtype='int_'.
    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_lp, N = trans_perms.shape
    size_row = N**2

    trans_perms = trans_perms.astype("int_")
    n = 0
    indices = np.zeros(size_row, dtype="int_")
    for i_patom in indep_atoms:
        index_shift_i = trans_perms[:, i_patom] * N
        for j in range(N):
            index_shift = index_shift_i + trans_perms[:, j]
            indices[index_shift] = n
            n += 1
    assert n * n_lp == size_row
    return indices


def get_lat_trans_decompr_indices(trans_perms: NDArray) -> NDArray:
    """Return indices to de-compress compressed matrix by lat-trans-sym.

    Usage
    -----
    vec[indices] of shape (n_a*N*9,) gives an array of shape=(N**2*9,).
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
        Indices of n_a * N9 elements.
        shape=(N^2*9,), dtype='int_'.

    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_a = len(indep_atoms)
    N = trans_perms.shape[1]
    n_lp = N // n_a
    size_row = (N * 3) ** 2

    trans_perms = trans_perms.astype("int_")
    n = 0
    indices = np.zeros(size_row, dtype="int_")
    for i_patom in indep_atoms:
        index_shift_i = trans_perms[:, i_patom] * N * 9
        for j in range(N):
            index_shift = index_shift_i + trans_perms[:, j] * 9
            for ab in range(9):
                indices[index_shift + ab] = n
                n += 1
    assert n * n_lp == size_row
    return indices


def get_lat_trans_compr_indices(trans_perms: NDArray) -> NDArray:
    """Return indices to compress matrix by lat-trans-sym.

    Usage
    -----
    vec[indices] of shape (N**2*9,) vec gives an array of shape=(n_a*N*9, n_lp).
    1/sqrt(n_lp) must be multiplied manually after compression to mimic
    get_lat_trans_compr_matrix.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'. shape=(n_l, N), where n_l and N are the numbers of lattice
        points and atoms in supercell.

    Returns
    -------
    indices : ndarray
        shape=(n_a*N9, n_lp), dtype='int_'.

    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_a = len(indep_atoms)
    N = trans_perms.shape[1]
    n_lp = N // n_a
    size_row = (N * 3) ** 2

    n = 0
    indices = np.zeros((n_a * N * 9, n_lp), dtype="int_")
    for i_patom in indep_atoms:
        for j in range(N):
            for ab in range(9):
                indices[n, :] = (
                    trans_perms[:, i_patom] * 9 * N + trans_perms[:, j] * 9 + ab
                )
                n += 1
    assert n * n_lp == size_row
    return indices


def get_lat_trans_compr_matrix(decompr_idx: NDArray, N: int, n_lp: int) -> csr_array:
    """Return compression matrix by lattice translation symmetry.

    `decompr_idx` is obtained by `get_lat_trans_decompr_indices`.

    Matrix shape is (NN33, n_a*N33), where n_a is the number of independent
    atoms by lattice translation symmetry.

    Data order is (N, N, 3, 3, n_a, N, 3, 3) if it is in dense array.

    """
    NN9 = N**2 * 9
    compression_mat = csr_array(
        (
            np.full(NN9, 1 / np.sqrt(n_lp), dtype="double"),
            (np.arange(NN9, dtype=int), decompr_idx),
        ),
        shape=(NN9, NN9 // n_lp),
        dtype="double",
    )
    return compression_mat


def get_lat_trans_compr_matrix_O2(trans_perms: NDArray):
    """Return lat trans compression matrix."""
    n_lp, N = trans_perms.shape
    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    c_trans = get_lat_trans_compr_matrix(decompr_idx, N, n_lp)
    return c_trans
