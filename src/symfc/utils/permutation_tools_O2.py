"""Permutation utility functions for 3rd order force constants."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.matrix_tools import get_combinations
from symfc.utils.permutation_tools import construct_basis_from_orbits
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices


def _N3N3_to_NNand33(combs: np.ndarray, N: int) -> np.ndarray:
    """Transform index order."""
    vecNN, vec33 = np.divmod(combs[:, 0], 3)
    vecNN *= N
    vec33 *= 3
    div, mod = np.divmod(combs[:, 1], 3)
    vecNN += div
    vec33 += mod
    return vecNN, vec33


def compr_permutation_lat_trans_O2(
    trans_perms: np.ndarray,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    n_batch: Optional[int] = None,
    verbose: bool = False,
) -> csr_array:
    """Build a compression matrix for permutation rules compressed by C_trans.

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
    Compressed basis matrix for permutation
    C_pt = eigh(C_trans.T @ C_perm @ C_perm.T @ C_trans)
    """
    n_lp, natom = trans_perms.shape
    NN9 = natom**2 * 9
    if atomic_decompr_idx is None:
        atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    orbits = np.ones(NN9 // n_lp, dtype="int") * -1
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)

    # order = 1
    combinations = np.array([[i, i] for i in range(3 * natom)], dtype=int)
    perms = [[0, 0]]
    orbits = _update_orbits_from_combinations(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        orbits,
        n_perms_group=1,
        n_batch=1,
        verbose=verbose,
    )

    # order = 2
    combinations = get_combinations(
        natom, order=2, fc_cutoff=fc_cutoff, indep_atoms=indep_atoms
    )
    perms = [[0, 1], [1, 0]]
    orbits = _update_orbits_from_combinations(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        orbits,
        n_perms_group=1,
        n_batch=1,
        verbose=verbose,
    )
    if verbose:
        print("Construct basis matrix for permutations", flush=True)
    c_pt = construct_basis_from_orbits(orbits)
    return c_pt


def _update_orbits_from_combinations(
    combinations: np.ndarray,
    permutations: np.ndarray,
    atomic_decompr_idx: np.ndarray,
    trans_perms: np.ndarray,
    orbits: np.ndarray,
    n_perms_group: int = 1,
    n_batch: int = 1,
    verbose: bool = False,
) -> csr_array:
    """Construct projector of permutation and lattice translation."""
    n_lp, natom = trans_perms.shape
    n_comb = combinations.shape[0]
    n_perms = len(permutations)
    n_perms_sym = n_perms // n_perms_group
    for begin, end in zip(*get_batch_slice(n_comb, n_comb // n_batch)):
        if verbose:
            print("Permutation basis:", str(end) + "/" + str(n_comb), flush=True)
        combs_perm = combinations[begin:end][:, permutations].reshape((-1, 2))
        combs_perm, combs33 = _N3N3_to_NNand33(combs_perm, natom)
        cols = atomic_decompr_idx[combs_perm] * 9 + combs33
        cols = cols.reshape(-1, n_perms_sym)
        for c in cols.T:
            orbits[c] = cols[:, 0]
    return orbits
