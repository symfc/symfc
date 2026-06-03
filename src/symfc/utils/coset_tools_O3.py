"""Coset calculations for 3rd order force constants."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_array, kron

from symfc.spg_reps.spg_reps_O3 import SpgRepsO3
from symfc.utils.coset_tools import kron_spg_reps
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.permutation_tools_O3 import PermutationO3
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def get_compr_coset_projector_O3(
    spg_reps: SpgRepsO3,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    permutation: Optional[PermutationO3] = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> csr_array:
    """Return compr matrix of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**3 * 27 // n_lp if permutation is None else permutation.col_shape  # type: ignore

    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    nonzero_indep_atom = np.zeros(N**3, dtype=bool)
    atom_indices = np.arange(N**3) // N**2
    for i in indep_atoms:
        nonzero_indep_atom[atom_indices == i] = True

    if fc_cutoff is None:
        nonzero = nonzero_indep_atom
    else:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc3()
        nonzero = nonzero & nonzero_indep_atom
    cols = atomic_decompr_idx[nonzero]

    n_cosets = min([int(np.sqrt(len(spg_reps.unique_rotation_indices))), 4])
    cosets = [csr_array(([], ([], [])), shape=(size, size), dtype="double")] * n_cosets

    factor = 1 / len(spg_reps.unique_rotation_indices)
    size_coset = N**3 // n_lp
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        """Calculate mat = C.T @ spg_reps.get_sigma3_rep(i) @ C
            and mat = kron(mat, spg_reps.r_reps[i] * factor).tocsr().
            C: atomic_lat_trans_compr_mat, shape=(NNN, NNN/n_lp).
        """
        if verbose:
            n_rot = len(spg_reps.unique_rotation_indices)
            print("Coset sum:", i + 1, "/", n_rot, flush=True)
        perms = spg_reps.get_sigma3_rep(i, nonzero=nonzero)
        mat = kron_spg_reps(
            atomic_decompr_idx[perms],
            cols,
            spg_reps.r_reps[i],
            factor,
            size_coset,
        )
        if permutation is not None:
            mat = permutation.blocked_triple_product(mat, use_mkl=use_mkl)

        cosets[i % n_cosets] += mat
    return sum(cosets)  # type: ignore


def get_compr_coset_projector_O3_stable(
    spg_reps: SpgRepsO3,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    permutation: Optional[PermutationO3] = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> csr_array:
    """Return compr matrix of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**3 * 27 // n_lp if permutation is None else permutation.col_shape  # type: ignore

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    if fc_cutoff is None:
        nonzero = None
        size_data = N**3
        col = atomic_decompr_idx
    else:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc3()
        size_data = np.count_nonzero(nonzero)
        col = atomic_decompr_idx[nonzero]

    n_cosets = min([int(np.sqrt(len(spg_reps.unique_rotation_indices))), 4])
    cosets = [csr_array(([], ([], [])), shape=(size, size), dtype="double")] * n_cosets

    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        if verbose:
            n_rot = len(spg_reps.unique_rotation_indices)
            print("Coset sum:", i + 1, "/", n_rot, flush=True)

        perms = spg_reps.get_sigma3_rep(i, nonzero=nonzero)
        """Equivalent to mat = C.T @ spg_reps.get_sigma3_rep(i) @ C
           C: atomic_lat_trans_compr_mat, shape=(NNN, NNN/n_lp)"""
        mat = csr_array(
            (
                np.ones(size_data, dtype="int_"),
                (atomic_decompr_idx[perms], col),
            ),
            shape=(N**3 // n_lp, N**3 // n_lp),
            dtype="int_",
        )
        mat = kron(mat, spg_reps.r_reps[i] * factor).tocsr()
        if permutation is not None:
            mat = permutation.blocked_triple_product(mat, use_mkl=use_mkl)

        cosets[i % n_cosets] += mat
    return sum(cosets)  # type: ignore
