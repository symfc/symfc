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
