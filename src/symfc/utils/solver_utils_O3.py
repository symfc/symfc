"""Solver utility functions for O3."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from symfc.utils.solver_funcs import get_batch_slice


def _reshape_nNN333_nx_to_N3N3_n3nx(
    mat: NDArray, N: int, n: int, n_batch: int = 9
) -> csr_array:
    """Reorder and reshape a sparse matrix (nNN333,nx)->(N3N3,n3nx).

    Return reordered csr_matrix used for FC3.
    """
    _, nx = mat.shape
    NN33 = N**2 * 9
    n3nx = n * 3 * nx
    mat = mat.tocoo(copy=False)

    batch_size = len(mat.row) if len(mat.row) < n_batch else len(mat.row) // n_batch

    begin_batch, end_batch = get_batch_slice(len(mat.row), batch_size)
    for begin, end in zip(begin_batch, end_batch, strict=True):
        div, rem = np.divmod(mat.row[begin:end], 27 * N * N)
        mat.col[begin:end] += div * 3 * nx
        div, rem = np.divmod(rem, 27 * N)
        mat.row[begin:end] = div * 9 * N
        div, rem = np.divmod(rem, 27)
        mat.row[begin:end] += div * 3
        div, rem = np.divmod(rem, 9)
        mat.col[begin:end] += div * nx
        div, rem = np.divmod(rem, 3)
        mat.row[begin:end] += div * 3 * N + rem

    mat.resize((NN33, n3nx))
    mat = mat.tocsr(copy=False)
    return mat


def reshape_compr_mat_O3(
    compact_compress_mat_fc3: csr_array,
    atomic_decompr_idx_fc3: NDArray,
    N: int,
    atom_idx_begin: int,
    atom_idx_end: int,
) -> csr_array:
    """Reorder and reshape a sparse matrix (nNN333,nx)->(N3N3,n3nx).

    Return reordered csr_matrix used for FC3.
    """
    NN = N * N
    n_atom_batch = atom_idx_end - atom_idx_begin
    decompr_idx = (
        atomic_decompr_idx_fc3[atom_idx_begin * NN : atom_idx_end * NN, None] * 27
        + np.arange(27)[None, :]
    ).reshape(-1)
    compr_mat_fc3 = _reshape_nNN333_nx_to_N3N3_n3nx(
        compact_compress_mat_fc3[decompr_idx], N, n_atom_batch, n_batch=9
    )
    return compr_mat_fc3


def set_disps_N3N3(disps, sparse=False):
    """Calculate Kronecker products of displacements.

    Parameter
    ---------
    disps: shape=(n_supercell, N3)

    Return
    ------
    disps_2nd: shape=(n_supercell, N3N3)
    """
    n_supercell = disps.shape[0]
    disps_2nd = (disps[:, :, None] * disps[:, None, :]).reshape((n_supercell, -1))

    if sparse:
        return csr_array(disps_2nd)
    return disps_2nd


def _reshape(vec: NDArray, N: int, n: int) -> csr_array:
    """Reorder and reshape a numpy vector (nNN333)->(N3N3,n3).

    Return reordered numpy array used for iterative FC3 solver.
    """
    n3 = n * 3
    vec = np.array(vec)
    mat = vec.reshape(n, N, N, 3, 3, 3)
    return mat.transpose(1, 3, 2, 4, 0, 5).reshape((-1, n3))


def reshape_vec_O3(
    vec: csr_array,
    atomic_decompr_idx_fc3: NDArray,
    N: int,
    atom_idx_begin: int,
    atom_idx_end: int,
) -> csr_array:
    """Reorder and reshape a sparse matrix (nNN333,nx)->(N3N3,n3nx).

    Return reordered csr_matrix used for FC3.
    """
    NN = N * N
    n_atom_batch = atom_idx_end - atom_idx_begin
    decompr_idx = (
        atomic_decompr_idx_fc3[atom_idx_begin * NN : atom_idx_end * NN, None] * 27
        + np.arange(27)[None, :]
    ).reshape(-1)
    reshape_vec = _reshape(vec[decompr_idx], N, n_atom_batch)

    return reshape_vec


def dot_O3(
    compact_compress_mat_fc3: csr_array,
    atomic_decompr_idx_fc3: NDArray,
    prod_fc3: NDArray,
    N: int,
    atom_idx_begin: int,
    atom_idx_end: int,
) -> csr_array:
    """Reorder and reshape a sparse matrix (nNN333,nx)->(N3N3,n3nx).

    Return reordered csr_matrix used for FC3.
    """
    NN = N * N
    n_atom_batch = atom_idx_end - atom_idx_begin
    decompr_idx = (
        atomic_decompr_idx_fc3[atom_idx_begin * NN : atom_idx_end * NN, None] * 27
        + np.arange(27)[None, :]
    ).reshape(-1)

    compr_mat = compact_compress_mat_fc3[decompr_idx]
    prod_fc3 = prod_fc3.reshape(N, 3, N, 3, n_atom_batch, 3)
    vec = prod_fc3.transpose(4, 0, 2, 5, 1, 3).reshape(-1)
    return compr_mat.T @ vec
