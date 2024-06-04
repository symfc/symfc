"""Utility functions for 1st order force constants."""

import numpy as np
from scipy.sparse import csr_array, kron

from symfc.spg_reps import SpgRepsO1

from .utils import get_indep_atoms_by_lat_trans


def get_lat_trans_decompr_indices(trans_perms: np.ndarray) -> np.ndarray:
    """Return indices to de-compress compressed matrix by lat-trans-sym.

    Usage
    -----
    vec[indices] of shape (n_a*3,) gives an array of shape=(N*3,).
    1/sqrt(n_lp) must be multiplied manually after decompression to mimic
    get_lat_trans_compr_matrix.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattce points and
        atoms in supercell.

    Returns
    -------
    indices : ndarray
        Indices of n_a * 3 elements.
        shape=(N*3,), dtype='int_'.

    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_a = len(indep_atoms)
    N = trans_perms.shape[1]
    n_lp = N // n_a
    size_row = N * 3

    n = 0
    indices = np.zeros(size_row, dtype="int_")
    for i_patom in indep_atoms:
        index_shift = trans_perms[:, i_patom] * 3
        for a in range(3):
            indices[index_shift + a] = n
            n += 1
    assert n * n_lp == size_row
    return indices


def get_lat_trans_compr_indices(trans_perms: np.ndarray) -> np.ndarray:
    """Return indices to compress matrix by lat-trans-sym.

    Usage
    -----
    vec[indices] of shape (N*3,) vec gives an array of shape=(n_a*3, n_lp).
    1/sqrt(n_lp) must be multiplied manually after compression to mimic
    get_lat_trans_compr_matrix.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'. shape=(n_l, N), where n_l and N are the numbers of lattce
        points and atoms in supercell.

    Returns
    -------
    indices : ndarray
        shape=(n_a*N9, n_lp), dtype='int_'.

    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_a = len(indep_atoms)
    N = trans_perms.shape[1]
    n_lp = N // n_a
    size_row = N * 3

    n = 0
    indices = np.zeros((n_a * 3, n_lp), dtype="int_")
    for i_patom in indep_atoms:
        for a in range(3):
            indices[n, :] = trans_perms[:, i_patom] * 3 + a
            n += 1
    assert n * n_lp == size_row
    return indices


def get_lat_trans_compr_matrix(decompr_idx: np.ndarray, N: int, n_lp: int) -> csr_array:
    """Return compression matrix by lattice translation symmetry.

    `decompr_idx` is obtained by `get_lat_trans_decompr_indices`.

    Matrix shape is (N3, n_a*3), where n_a is the number of independent
    atoms by lattice translation symmetry.

    Data order is (N, 3, n_a, 3) if it is in dense array.

    """
    N3 = N * 3
    compression_mat = csr_array(
        (
            np.full(N3, 1 / np.sqrt(n_lp), dtype="double"),
            (np.arange(N3, dtype=int), decompr_idx),
        ),
        shape=(N3, N3 // n_lp),
        dtype="double",
    )
    return compression_mat


def _get_atomic_lat_trans_decompr_indices(trans_perms: np.ndarray) -> np.ndarray:
    """Return indices to de-compress compressed matrix by atom-lat-trans-sym.

    This is atomic permutation only version of get_lat_trans_decompr_indices.

    Usage
    -----
    vec[indices] of shape (n_a,) gives an array of shape=(N,).
    1/sqrt(n_lp) must be multiplied manually after decompression.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattce points and
        atoms in supercell.

    Returns
    -------
    indices : ndarray
        Indices of n_a * N elements.
        shape=(N^2*,), dtype='int_'.

    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_lp, N = trans_perms.shape
    size_row = N

    n = 0
    indices = np.zeros(size_row, dtype="int_")
    for i_patom in indep_atoms:
        index_shift = trans_perms[:, i_patom]
        indices[index_shift] = n
        n += 1
    assert n * n_lp == size_row
    return indices


def get_compr_coset_reps_sum(spg_reps: SpgRepsO1):
    """Return compressed projector of coset reps sum."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N * 3 // n_lp
    coset_reps_sum = csr_array(([], ([], [])), shape=(size, size), dtype="double")
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
    C = csr_array(
        (
            np.ones(N, dtype=int),
            (np.arange(N, dtype=int), atomic_decompr_idx),
        ),
        shape=(N, N // n_lp),
    )
    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        mat = spg_reps.get_sigma1_rep(i)
        mat = mat @ C
        mat = C.T @ mat
        coset_reps_sum += kron(mat, spg_reps.r_reps[i] * factor)

    return coset_reps_sum
