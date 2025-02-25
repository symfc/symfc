"""Matrix utility functions."""

import numpy as np
from scipy.sparse import csr_array


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
