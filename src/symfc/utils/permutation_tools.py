"""Permutation utility functions."""

from typing import Optional

import numpy as np
import scipy
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools import FCCutoff


def get_entire_combinations(n: int, r: int):
    """Return numpy array of combinations.

    combinations = np.array(
       list(itertools.combinations(range(n), r)), dtype=int
    )
    """
    combs = np.ones((r, n - r + 1), dtype=int)
    combs[0] = np.arange(n - r + 1)
    for j in range(1, r):
        reps = (n - r + j) - combs[j - 1]
        combs = np.repeat(combs, reps, axis=1)
        ind = np.add.accumulate(reps)
        combs[j, ind[:-1]] = 1 - reps[1:]
        combs[j, 0] = j
        combs[j] = np.add.accumulate(combs[j])
    return combs.T


def get_combinations(
    natom: int,
    order: int,
    fc_cutoff: Optional[FCCutoff] = None,
    indep_atoms: Optional[np.ndarray] = None,
):
    """Return numpy array of FC index combinations."""
    if fc_cutoff is None:
        combinations = get_entire_combinations(3 * natom, order)
    else:
        if order == 2:
            combinations = fc_cutoff.combinations2()
        elif order == 3:
            """Combinations can be divided using fc_cut.combiations3(i)."""
            combinations = fc_cutoff.combinations3_all()
        elif order == 4:
            combinations = fc_cutoff.combinations4_all()
        else:
            raise NotImplementedError(
                "Combinations are implemented only for 2 <= order <= 4."
            )

    if indep_atoms is not None:
        nonzero = np.zeros(combinations.shape[0], dtype=bool)
        atom_indices = combinations[:, 0] // 3
        for i in indep_atoms:
            nonzero[atom_indices == i] = True
        combinations = combinations[nonzero]
    return combinations


def _eliminate_zero_elements(
    perm_decompr_idx: np.ndarray, nonzero: np.ndarray
) -> np.ndarray:
    """Eliminate zero elements and reindex orbit indexes."""
    size_full = len(perm_decompr_idx)
    if not np.all(nonzero):
        perm_decompr_idx = perm_decompr_idx[nonzero]
        nonzero_map = np.ones(size_full, dtype="int") * -1
        nonzero_map[nonzero] = np.arange(len(perm_decompr_idx))
        perm_decompr_idx = nonzero_map[perm_decompr_idx]
    return perm_decompr_idx


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

    size_full = len(perm_decompr_idx)
    nonzero = perm_decompr_idx != -1
    perm_decompr_idx = _eliminate_zero_elements(perm_decompr_idx, nonzero)

    size1 = len(perm_decompr_idx)
    perm_lat_trans_graph = csr_array(
        (np.ones(size1, dtype=bool), (np.arange(size1), perm_decompr_idx)),
        shape=(size1, size1),
        dtype=bool,
    )

    n_col, cols = scipy.sparse.csgraph.connected_components(perm_lat_trans_graph)
    key, cnt = np.unique(cols, return_counts=True)
    values = np.reciprocal(np.sqrt(cnt))

    rows = np.where(nonzero)[0]
    c_pt = csr_array(
        (values[cols], (rows, cols)),
        shape=(size_full, n_col),
        dtype="double",
    )
    return c_pt
