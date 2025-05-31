"""Permutation utility functions for 3rd order force constants."""

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
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def _N3N3N3_to_NNNand333(combs: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Transform index order."""
    vecNNN, vec333 = np.divmod(combs[:, 0], 3)
    vecNNN *= N**2
    vec333 *= 9
    div, mod = np.divmod(combs[:, 1], 3)
    vecNNN += div * N
    vec333 += mod * 3
    div, mod = np.divmod(combs[:, 2], 3)
    vecNNN += div
    vec333 += mod
    return vecNNN, vec333


def compr_permutation_lat_trans_O3(
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
          shape: (NNN333//n_lp, n_basis_pt).
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
    NNN27 = natom**3 * 27
    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    if n_batch is None:
        n_batch = 1 if natom <= 128 else int(round((natom / 128) ** 2))

    perm_decompr_idx = np.ones(NNN27 // n_lp, dtype="int") * -1
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)

    # order = 1
    combinations = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)
    perms = [[0, 0, 0]]
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
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
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
    combinations = get_combinations(
        natom, order=3, fc_cutoff=fc_cutoff, indep_atoms=indep_atoms
    )
    perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    perm_decompr_idx = _update_perm_decompr_indices(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        perm_decompr_idx,
        n_perms_group=1,
        n_batch=n_batch,
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
        combs_perm = combinations[begin:end][:, permutations].reshape((-1, 3))
        combs_perm, combs333 = _N3N3N3_to_NNNand333(combs_perm, natom)
        decompr_idx_combs_perm = atomic_decompr_idx[combs_perm] * 27 + combs333
        decompr_idx_combs_perm = decompr_idx_combs_perm.reshape(-1, n_perms_sym)
        for orbit_components in decompr_idx_combs_perm.T:
            perm_decompr_idx[orbit_components] = decompr_idx_combs_perm[:, 0]
    return perm_decompr_idx
