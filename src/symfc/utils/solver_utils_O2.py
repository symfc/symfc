"""Solver utility functions for O2."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from symfc.utils.solver_funcs import get_batch_slice


def _reshape_nN33_nx_to_N3_n3nx(mat, N: int, n: int, n_batch: int = 1) -> csr_array:
    """Reorder and reshape a sparse matrix (nN33,nx)->(N3,n3nx).

    mat : csr_array

    Return reordered csr_matrix used for FC2.
    """
    _, nx = mat.shape
    N3 = N * 3
    n3nx = n * 3 * nx
    mat = mat.tocoo(copy=False)

    begin_batch, end_batch = get_batch_slice(len(mat.row), len(mat.row) // n_batch)
    for begin, end in zip(begin_batch, end_batch, strict=True):
        div, rem = np.divmod(mat.row[begin:end], 9 * N)
        mat.col[begin:end] += div * 3 * nx
        div, rem = np.divmod(rem, 9)
        mat.row[begin:end] = div * 3
        div, rem = np.divmod(rem, 3)
        mat.col[begin:end] += div * nx
        mat.row[begin:end] += rem

    mat.resize((N3, n3nx))
    mat = mat.tocsr(copy=False)
    return mat


def reshape_compr_mat_O2(
    compact_compress_mat_fc2: csr_array,
    atomic_decompr_idx_fc2: NDArray,
    N: int,
    atom_idx_begin: int,
    atom_idx_end: int,
) -> csr_array:
    """Reorder and reshape a sparse matrix (nN33,nx)->(N3,n3nx).

    Return reordered csr_matrix used for FC2.
    """
    n_atom_batch = atom_idx_end - atom_idx_begin
    decompr_idx = (
        atomic_decompr_idx_fc2[atom_idx_begin * N : atom_idx_end * N, None] * 9
        + np.arange(9)[None, :]
    ).reshape(-1)
    compr_mat_fc2 = _reshape_nN33_nx_to_N3_n3nx(
        compact_compress_mat_fc2[decompr_idx],
        N,
        n_atom_batch,
    )
    return compr_mat_fc2


def slice_compact_compress_mat_O2(
    compact_compress_mat_fc2: csr_array,
    atomic_decompr_idx_fc2: NDArray,
    N: int,
    atom_idx_begin: int,
    atom_idx_end: int,
):
    """Slice compact compresstion matrix."""
    decompr_idx_fc2 = (
        atomic_decompr_idx_fc2[atom_idx_begin * N : atom_idx_end * N, None] * 9
        + np.arange(9)[None, :]
    ).reshape(-1)
    compr_mat_fc2 = compact_compress_mat_fc2[decompr_idx_fc2]
    return (decompr_idx_fc2, compr_mat_fc2)


def calc_predictions_O2(
    compact_compress_mat_fc2: csr_array,
    decompr_idx_fc2: NDArray,
    N: int,
    coefs: NDArray,
    disps: NDArray,
):
    """Calculate predicted forces used in iterative solver.

    pred2 = X2 @ coefs are calculated,
    where X2 = displacements @ compress_mat @ compress_eigvecs.

    Return
    ------
    pred2: Predicted forces, shape=(n_supercell * n_atom_batch * 3)
    """
    N3 = N * 3
    prod = compact_compress_mat_fc2 @ coefs
    prod = prod[decompr_idx_fc2].reshape(-1, N, 3, 3)
    prod = prod.transpose(1, 3, 0, 2).reshape(N3, -1)
    pred2 = (disps @ prod).reshape(-1)
    return pred2


def calc_gradients_O2(
    sliced_compact_compress_mat_fc2: csr_array,
    N: int,
    error: NDArray,
    disps: NDArray,
):
    """Calculate gradients used in iterative solver.

    grad = X2.T @ errors are calculated,
    where X2 = displacements @ compress_mat @ compress_eigvecs.
    Errors must be ([X2, X3] @ coefs23 - forces) when using both FC2 and FC3.

    Return
    ------
    grad: Gradients of loss function with respect to coefficients, shape=(n_compr_fc2,)
    """
    n_supercell = disps.shape[0]
    prod = disps.T @ error.reshape((n_supercell, -1))
    prod = prod.reshape(N, 3, -1, 3)
    prod = prod.transpose(2, 0, 3, 1).reshape(-1)
    grad2 = sliced_compact_compress_mat_fc2.T @ prod
    return grad2
