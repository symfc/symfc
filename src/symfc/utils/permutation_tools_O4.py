"""Permutation utility functions for 4th order force constants."""

import itertools
from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.permutation_tools import (
    construct_basis_from_perm_decompr_indices,
    get_combinations,
)
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O4 import get_atomic_lat_trans_decompr_indices_O4


def _N3N3N3N3_to_NNNNand3333(
    combs: np.ndarray, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Transform index order."""
    vecNNNN, vec3333 = np.divmod(combs[:, 0], 3)
    vecNNNN *= N**3
    vec3333 *= 27
    div, mod = np.divmod(combs[:, 1], 3)
    vecNNNN += div * N**2
    vec3333 += mod * 9
    div, mod = np.divmod(combs[:, 2], 3)
    vecNNNN += div * N
    vec3333 += mod * 3
    div, mod = np.divmod(combs[:, 3], 3)
    vecNNNN += div
    vec3333 += mod
    return vecNNNN, vec3333


def compr_permutation_lat_trans_O4(
    trans_perms: np.ndarray,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    n_batch: Optional[int] = None,
    verbose: bool = False,
) -> csr_array:
    r"""Build a compression matrix for permutation rules compressed by C_trans.

    This calculates C_(trans,perm) without allocating C_trans and C_perm.
    Batch calculations are used to reduce memory allocation.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattce points and
        atoms in supercell.
    fc_cutoff : FCCutoff class object. Default is None.

    Return
    ------
    c_pt: Compressed basis matrix for permutations and lattice translations.
          c_pt = eigh(C_trans.T @ C_perm @ C_perm.T @ C_trans)
          shape: (NNNN3333//n_lp, n_basis_pt).
          n_basis_pt denotes the size of basis for permutations and
          lattice translations.

    Algorithm
    ---------
    1. Calculate combinations {(i, a), (j, b), (k, c)}.
    2. Apply permutations to these combinations and calculate the orbits of
       {(i, a), (j, b), (k, c)} related to each other by permutations.
    3. Calculate the lattice translation basis indices of the orbits,
       and represent the lattice translation indices related to each other
       by a representative in perm_decompr_idx.
    """
    n_lp, natom = trans_perms.shape
    NNNN81 = natom**4 * 81
    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)

    perm_decompr_idx = np.ones(NNNN81 // n_lp, dtype="int") * -1
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)

    # order = 1
    combinations = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)
    perms = [[0, 0, 0, 0]]
    perm_decompr_idx = _update_perm_decompr_indices(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        perm_decompr_idx,
        n_perms_group=1,
        n_batch=1,
        verbose=verbose,
    )

    # order = 2
    combinations = get_combinations(
        natom, order=2, fc_cutoff=fc_cutoff, indep_atoms=indep_atoms
    )
    perms = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ]
    perm_decompr_idx = _update_perm_decompr_indices(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        perm_decompr_idx,
        n_perms_group=2,
        n_batch=1,
        verbose=verbose,
    )

    # order = 3
    if n_batch is None:
        n_batch3 = 1 if natom <= 128 else int(round((natom / 128) ** 2))

    combinations = get_combinations(
        natom, order=3, fc_cutoff=fc_cutoff, indep_atoms=indep_atoms
    )
    perms = [
        [0, 0, 1, 2],
        [0, 0, 2, 1],
        [0, 1, 0, 2],
        [0, 2, 0, 1],
        [0, 1, 2, 0],
        [0, 2, 1, 0],
        [1, 0, 0, 2],
        [2, 0, 0, 1],
        [1, 0, 2, 0],
        [2, 0, 1, 0],
        [1, 2, 0, 0],
        [2, 1, 0, 0],
        [1, 1, 0, 2],
        [1, 1, 2, 0],
        [1, 0, 1, 2],
        [1, 2, 1, 0],
        [1, 0, 2, 1],
        [1, 2, 0, 1],
        [0, 1, 1, 2],
        [2, 1, 1, 0],
        [0, 1, 2, 1],
        [2, 1, 0, 1],
        [0, 2, 1, 1],
        [2, 0, 1, 1],
        [2, 2, 1, 0],
        [2, 2, 0, 1],
        [2, 1, 2, 0],
        [2, 0, 2, 1],
        [2, 1, 0, 2],
        [2, 0, 1, 2],
        [1, 2, 2, 0],
        [0, 2, 2, 1],
        [1, 2, 0, 2],
        [0, 2, 1, 2],
        [1, 0, 2, 2],
        [0, 1, 2, 2],
    ]
    perm_decompr_idx = _update_perm_decompr_indices(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        perm_decompr_idx,
        n_perms_group=3,
        n_batch=n_batch3,
        verbose=verbose,
    )

    # order = 4
    if n_batch is None:
        n_batch4 = 1 if natom <= 16 else int(round((natom / 16) ** 2))

    combinations = get_combinations(
        natom, order=4, fc_cutoff=fc_cutoff, indep_atoms=indep_atoms
    )
    perms = np.array(list(itertools.permutations(range(4))))
    perm_decompr_idx = _update_perm_decompr_indices(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        perm_decompr_idx,
        n_perms_group=1,
        n_batch=n_batch4,
        verbose=verbose,
    )
    c_pt = construct_basis_from_perm_decompr_indices(perm_decompr_idx, verbose=verbose)
    return c_pt


def _update_perm_decompr_indices(
    combinations: np.ndarray,
    permutations: Union[np.ndarray, list],
    atomic_decompr_idx: np.ndarray,
    trans_perms: np.ndarray,
    perm_decompr_idx: np.ndarray,
    n_perms_group: int = 1,
    n_batch: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Apply permutations to lattice translation basis.

    Return
    ------
    perm_decompr_idx: Updated decompression indices of lattice translation basis
                      using permutations.
    """
    n_lp, natom = trans_perms.shape
    n_comb = combinations.shape[0]
    n_perms = len(permutations)
    n_perms_sym = n_perms // n_perms_group
    for begin, end in zip(*get_batch_slice(n_comb, n_comb // n_batch)):
        if verbose:
            print("Permutation basis:", str(end) + "/" + str(n_comb), flush=True)
        combs_perm = combinations[begin:end][:, permutations].reshape((-1, 4))
        combs_perm, combs3333 = _N3N3N3N3_to_NNNNand3333(combs_perm, natom)
        decompr_idx_combs_perm = atomic_decompr_idx[combs_perm] * 81 + combs3333
        decompr_idx_combs_perm = decompr_idx_combs_perm.reshape(-1, n_perms_sym)
        for orbit_components in decompr_idx_combs_perm.T:
            perm_decompr_idx[orbit_components] = decompr_idx_combs_perm[:, 0]
    return perm_decompr_idx
