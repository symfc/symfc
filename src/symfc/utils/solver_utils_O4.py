"""Solver utility functions for O3."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from symfc.utils.solver_funcs import get_batch_slice


def _reshape_nNNN3333_nx_to_N3N3N3_n3nx(
    mat: NDArray, N: int, n: int, n_batch: int = 36
) -> csr_array:
    """Reorder and reshape a sparse matrix (nNNN3333,nx)->(N3N3N3,n3nx).

    Return reordered csr_matrix used for FC4.
    """
    _, nx = mat.shape
    NNN333 = N**3 * 27
    n3nx = n * 3 * nx
    mat = mat.tocoo(copy=False)

    begin_batch, end_batch = get_batch_slice(len(mat.row), len(mat.row) // n_batch)
    for begin, end in zip(begin_batch, end_batch, strict=True):
        div, rem = np.divmod(mat.row[begin:end], 81 * N * N * N)
        mat.col[begin:end] += div * 3 * nx
        div, rem = np.divmod(rem, 81 * N * N)
        mat.row[begin:end] = div * 27 * N * N
        div, rem = np.divmod(rem, 81 * N)
        mat.row[begin:end] += div * 9 * N
        div, rem = np.divmod(rem, 81)
        mat.row[begin:end] += div * 3
        div, rem = np.divmod(rem, 27)
        mat.col[begin:end] += div * nx
        div, rem = np.divmod(rem, 9)
        mat.row[begin:end] += div * 9 * N * N
        div, rem = np.divmod(rem, 3)
        mat.row[begin:end] += div * 3 * N + rem

    mat.resize((NNN333, n3nx))
    mat = mat.tocsr(copy=False)
    return mat


def reshape_compr_mat_O4(
    compact_compress_mat_fc4: csr_array,
    atomic_decompr_idx_fc4: NDArray,
    N: int,
    atom_idx_begin: int,
    atom_idx_end: int,
):
    """Reorder and reshape a sparse matrix (nNNN3333,nx)->(N3N3N3,n3nx).

    Return reordered csr_matrix used for FC4.
    """
    NNN = N * N * N
    n_atom_batch = atom_idx_end - atom_idx_begin

    decompr_idx = (
        atomic_decompr_idx_fc4[atom_idx_begin * NNN : atom_idx_end * NNN, None] * 81
        + np.arange(81)[None, :]
    ).reshape(-1)
    compr_mat_fc4 = _reshape_nNNN3333_nx_to_N3N3N3_n3nx(
        compact_compress_mat_fc4[decompr_idx],
        N,
        n_atom_batch,
    )
    return compr_mat_fc4


def set_disps_N3N3N3(disps, sparse=True, disps_N3N3=None):
    """Calculate Kronecker products of displacements.

    Parameter
    ---------
    disps: shape=(n_supercell, N3)

    Return
    ------
    disps_3rd: shape=(n_supercell, N3N3N3)
    """
    n_supercell = disps.shape[0]
    if disps_N3N3 is not None:
        disps_3rd = (disps_N3N3[:, :, None] * disps[:, None, :]).reshape(
            (n_supercell, -1)
        )
    else:
        disps_3rd = (
            disps[:, :, None, None] * disps[:, None, :, None] * disps[:, None, None, :]
        ).reshape((n_supercell, -1))

    if sparse:
        return csr_array(disps_3rd)
    return disps_3rd
