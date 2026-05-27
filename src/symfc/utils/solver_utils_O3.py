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


def slice_compact_compress_mat_O3(
    compact_compress_mat_fc3: csr_array,
    atomic_decompr_idx_fc3: NDArray,
    N: int,
    atom_idx_begin: int,
    atom_idx_end: int,
):
    """Slice compact compresstion matrix."""
    NN = N * N
    decompr_idx_fc3 = (
        atomic_decompr_idx_fc3[atom_idx_begin * NN : atom_idx_end * NN, None] * 27
        + np.arange(27)[None, :]
    ).reshape(-1)
    compr_mat_fc3 = compact_compress_mat_fc3[decompr_idx_fc3]
    return (decompr_idx_fc3, compr_mat_fc3)


def calc_predictions_O3(
    compact_compress_mat_fc3: csr_array,
    decompr_idx_fc3: NDArray,
    N: int,
    coefs: NDArray,
    dispN3N3: NDArray,
):
    """Calculate predicted forces used in iterative solver.

    pred3 = X3 @ coefs are calculated,
    where X3 = displacements @ compress_mat @ compress_eigvecs.

    Return
    ------
    pred3: Predicted forces, shape=(n_supercell * n_atom_batch * 3)
    """
    NN33 = N * N * 9
    prod = compact_compress_mat_fc3 @ coefs
    prod = prod[decompr_idx_fc3].reshape(-1, N, N, 3, 3, 3)
    prod = prod.transpose(1, 4, 2, 5, 0, 3).reshape(NN33, -1)
    pred3 = (dispN3N3 @ prod).reshape(-1)
    return pred3


def calc_gradients_O3(
    sliced_compact_compress_mat_fc3: csr_array,
    N: int,
    error: NDArray,
    dispN3N3: NDArray,
):
    """Calculate gradients used in iterative solver.

    grad = X3.T @ errors are calculated,
    where X3 = displacements @ compress_mat @ compress_eigvecs.
    Errors must be ([X2, X3] @ coeffs23 - forces) when using both FC2 and FC3.

    Return
    ------
    grad: Gradients of loss function with respect to coefficients, shape=(n_compr_fc3,)
    """
    n_supercell = dispN3N3.shape[0]
    prod = dispN3N3.T @ error.reshape((n_supercell, -1))
    prod = prod.reshape(N, 3, N, 3, -1, 3)
    prod = prod.transpose(4, 0, 2, 5, 1, 3).reshape(-1)
    grad3 = sliced_compact_compress_mat_fc3.T @ prod
    return grad3
