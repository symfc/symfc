"""Matrix utility functions."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools import FCCutoff


def get_entire_combinations(n, r):
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


def permutation_dot_lat_trans(
    combinations_ijk: np.ndarray,
    combinations_abc: np.ndarray,
    atomic_decompr_idx: np.ndarray,
    n_perms: int,
    n_perms_group: int,
    n_lp: int,
    order: int,
    natom: int,
) -> csr_array:
    """Return C_perm.T @ C_trans for permulation symmetry rules."""
    assert combinations_ijk.shape[0] == combinations_abc.shape[0]

    sum_abc = 3**order
    sum_ijkabc = natom**order * sum_abc
    n_combinations = combinations_ijk.shape[0] // n_perms
    n_perms_sym = n_perms // n_perms_group
    c_pt = csr_array(
        (
            np.full(n_combinations * n_perms, 1 / np.sqrt(n_perms_sym * n_lp)),
            (
                np.repeat(np.arange(n_combinations * n_perms_group), n_perms_sym),
                atomic_decompr_idx[combinations_ijk] * sum_abc + combinations_abc,
            ),
        ),
        shape=(n_combinations * n_perms_group, sum_ijkabc // n_lp),
        dtype="double",
    )
    return c_pt
