"""Permutation utility functions for 3rd order force constants."""

from typing import Optional

import numpy as np
import scipy
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.matrix_tools import get_combinations
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def _N3N3N3_to_NNNand333(combs: np.ndarray, N: int) -> np.ndarray:
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
    Compressed projector for permutation
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    if n_batch is None:
        n_batch = 1 if natom <= 128 else int(round((natom / 128) ** 2))

    orbits = np.arange(NNN27 // n_lp)
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)

    # order = 1
    combinations = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)
    perms = [[0, 0, 0]]
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
    perms = [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
    orbits = _update_orbits_from_combinations(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        orbits,
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
    orbits = _update_orbits_from_combinations(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        orbits,
        n_perms_group=1,
        n_batch=n_batch,
        verbose=verbose,
    )

    if verbose:
        print("Construct basis matrix for permutations", flush=True)
    c_pt = _orbits_to_basis(orbits)
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
        combs_perm = combinations[begin:end][:, permutations].reshape((-1, 3))
        combs_perm, combs333 = _N3N3N3_to_NNNand333(combs_perm, natom)
        cols = atomic_decompr_idx[combs_perm] * 27 + combs333
        cols = cols.reshape(-1, n_perms_sym)
        for c in cols.T:
            orbits[c] = cols[:, 0]
    return orbits


def _orbits_to_basis(orbits: np.ndarray):
    """Transform orbits into basis matrix."""
    size1 = len(orbits)
    orbits = csr_array(
        (np.ones(size1, dtype=bool), (np.arange(size1), orbits)),
        shape=(size1, size1),
        dtype=bool,
    )

    n_col, cols = scipy.sparse.csgraph.connected_components(orbits)
    key, cnt = np.unique(cols, return_counts=True)
    values = np.reciprocal(np.sqrt(cnt))

    c_pt = csr_array(
        (values[cols], (np.arange(size1), cols)),
        shape=(size1, n_col),
        dtype="double",
    )
    return c_pt
