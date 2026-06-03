"""Coset functions for 4th order force constants."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_array, kron

from symfc.spg_reps import SpgRepsO4
from symfc.utils.coset_tools import kron_spg_reps
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.permutation_tools_O4 import PermutationO4
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O4 import get_atomic_lat_trans_decompr_indices_O4


def get_compr_coset_projector_O4(
    spg_reps: SpgRepsO4,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    permutation: Optional[PermutationO4] = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> csr_array:
    """Return compr projector of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**4 * 81 // n_lp if permutation is None else permutation.col_shape  # type: ignore

    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)

    nonzero_indep_atom = np.zeros(N**4, dtype=bool)
    atom_indices = np.arange(N**4) // N**3
    for i in indep_atoms:
        nonzero_indep_atom[atom_indices == i] = True

    if fc_cutoff is None:
        nonzero = nonzero_indep_atom
    else:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc4()
        nonzero = nonzero & nonzero_indep_atom
    cols = atomic_decompr_idx[nonzero]

    n_cosets = min([int(np.sqrt(len(spg_reps.unique_rotation_indices))), 4])
    cosets = [csr_array(([], ([], [])), shape=(size, size), dtype="double")] * n_cosets

    factor = 1 / len(spg_reps.unique_rotation_indices)
    size_coset = N**4 // n_lp
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        """Calculate mat = C.T @ spg_reps.get_sigma3_rep(i) @ C
            and mat = kron(mat, spg_reps.r_reps[i] * factor).tocsr().
            C: atomic_lat_trans_compr_mat, shape=(NNN, NNN/n_lp).
        """
        if verbose:
            n_rot = len(spg_reps.unique_rotation_indices)
            print("Coset sum:", i + 1, "/", n_rot, flush=True)
        perms = spg_reps.get_sigma4_rep(i, nonzero=nonzero)
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


def get_compr_coset_projector_O4_stable(
    spg_reps: SpgRepsO4,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    permutation: Optional[PermutationO4] = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> csr_array:
    """Return compr projector of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**4 * 81 // n_lp if permutation is None else permutation.col_shape  # type: ignore

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)

    if fc_cutoff is None:
        nonzero = None
        size_data = N**4
        col = atomic_decompr_idx
    else:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc4()
        size_data = np.count_nonzero(nonzero)
        col = atomic_decompr_idx[nonzero]

    n_cosets = min([int(np.sqrt(len(spg_reps.unique_rotation_indices))), 4])
    cosets = [csr_array(([], ([], [])), shape=(size, size), dtype="double")] * n_cosets

    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        if verbose:
            n_rot = len(spg_reps.unique_rotation_indices)
            print("Coset sum:", i + 1, "/", n_rot, flush=True)

        perms = spg_reps.get_sigma4_rep(i, nonzero=nonzero)
        """Equivalent to mat = C.T @ spg_reps.get_sigma4_rep(i) @ C
        C: atomic_lat_trans_compr_mat, shape=(NNNN, NNNN/n_lp)"""
        mat = csr_array(
            (
                np.ones(size_data, dtype="int_"),
                (atomic_decompr_idx[perms], col),
            ),
            shape=(N**4 // n_lp, N**4 // n_lp),
            dtype="int_",
        )
        mat = kron(mat, spg_reps.r_reps[i] * factor).tocsr()
        if permutation is not None:
            mat = permutation.blocked_triple_product(mat, use_mkl=use_mkl)

        cosets[i % n_cosets] += mat
    return sum(cosets)  # type: ignore
