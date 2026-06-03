"""Permutation utility functions."""

from dataclasses import dataclass

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.sparse import csr_array

from symfc.eig_solvers.graph import connected_components

SCIPY_SPARSE_DATA_LIMIT = 2147483647


def _eliminate_zero_elements(
    perm_decompr_idx: np.ndarray, nonzero: np.ndarray
) -> np.ndarray:
    """Eliminate zero elements and reindex orbit indices."""
    size_full = len(perm_decompr_idx)
    if not np.all(nonzero):
        perm_decompr_idx = perm_decompr_idx[nonzero]
        nonzero_map = np.ones(size_full, dtype="int_") * -1
        nonzero_map[nonzero] = np.arange(len(perm_decompr_idx))
        perm_decompr_idx = nonzero_map[perm_decompr_idx]
    return perm_decompr_idx


def find_groups_perm_decompr_indices(
    perm_decompr_idx: np.ndarray, verbose: bool = False
):
    """Find groups perm_decompr_idx."""
    nonzero = perm_decompr_idx != -1
    perm_decompr_idx = _eliminate_zero_elements(perm_decompr_idx, nonzero)

    size1 = len(perm_decompr_idx)
    perm_lat_trans_graph = csr_array(
        (np.ones(size1, dtype=bool), (np.arange(size1), perm_decompr_idx)),
        shape=(size1, size1),
        dtype=bool,
    )

    if len(perm_lat_trans_graph.data) < SCIPY_SPARSE_DATA_LIMIT:
        if verbose:
            print("Use scipy connected_components.", flush=True)
        n_col, cols = scipy.sparse.csgraph.connected_components(perm_lat_trans_graph)
        key, cnt = np.unique(cols, return_counts=True)
    else:
        if verbose:
            print("Use symfc connected_components.", flush=True)
        perm_lat_trans_graph += perm_lat_trans_graph.T
        group = connected_components(perm_lat_trans_graph, verbose=verbose)
        cols = np.ones(perm_decompr_idx.shape, dtype="int_") * -1
        cnt = []
        for col_id, v in group.items():
            cols[v] = col_id
            cnt.append(len(v))

    rows = np.where(nonzero)[0]
    uniq_cnt, indices = np.unique(cnt[cols], return_inverse=True)

    permutation_matrices = [
        PermutationMatrix(value=np.reciprocal(np.sqrt(cnt))) for cnt in uniq_cnt
    ]
    for i, idx in enumerate(indices):
        permutation_matrices[idx].rows.append(rows[i])
        permutation_matrices[idx].cols.append(cols[i])

    for mat in permutation_matrices:
        mat.rows = np.array(mat.rows)
        mat.cols = np.array(mat.cols)
        mat.n_cols = len(np.unique(mat.cols))
        print(len(mat.rows), len(mat.cols), mat.value)

    return permutation_matrices


@dataclass
class PermutationMatrix:
    """Dataclass for permutation matrix."""

    rows: list | NDArray | None = None
    cols: list | NDArray | None = None
    value: float = 1.0
    n_cols: int = 0

    def __post_init__(self):
        """Init method."""
        self.rows = []
        self.cols = []


def construct_basis_from_perm_decompr_indices(
    perm_decompr_idx: np.ndarray, verbose: bool = False
):
    """Transform perm_decompr_idx into basis matrix.

    Parameters
    ----------
    perm_decompr_idx: Decompression indices of lattice translation basis
                      using permutations.
    Return
    ------
    c_pt: Compressed basis matrix for permutations and lattice translations.
          c_pt = eigh(C_trans.T @ C_perm @ C_perm.T @ C_trans)
    """
    if verbose:
        print("Construct permutation basis matrix.", flush=True)

    permutation_matrices = find_groups_perm_decompr_indices(
        perm_decompr_idx, verbose=verbose
    )
    return permutation_matrices
