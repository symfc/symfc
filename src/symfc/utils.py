"""Functions to handle matrix indices."""
import itertools
from typing import Literal

import numpy as np
from scipy.sparse import coo_array

from symfc.spg_reps import SpgReps


def to_serial(i: int, a: int, j: int, b: int, natom: int) -> int:
    """Return NN33-1D index."""
    return (i * 9 * natom) + (j * 9) + (a * 3) + b


def get_spg_projector(
    spg_reps: SpgReps,
    natom: int,
    compression_mat: coo_array,
) -> coo_array:
    """Compute compact spg projector matrix.

    This computes C.T @ spg_proj (kron_c) @ perm_proj @ C,
    where C is ``compression_mat``.

    """
    coset_reps_sum = kron_sum_c(spg_reps, compression_mat)
    C_perm = get_perm_compr_matrix(natom)
    perm = C_perm.T @ compression_mat
    perm = perm.T @ perm
    return coset_reps_sum @ perm


def get_indep_atoms_by_lat_trans(trans_perms: np.ndarray) -> np.ndarray:
    """Return independent atoms by lattice translation symmetry.

    Parameters
    ----------
    trans_perms : np.ndarray
        Atom indices after lattice translations.
        shape=(lattice_translations, supercell_atoms)

    Returns
    -------
    np.ndarray
        Independent atoms.
        shape=(n_indep_atoms_by_lattice_translation,), dtype=int

    """
    unique_atoms: list[int] = []
    assert np.array_equal(trans_perms[0, :], range(trans_perms.shape[1]))
    for i, perms in enumerate(trans_perms.T):
        is_found = False
        for j in unique_atoms:
            if j in perms:
                is_found = True
                break
        if not is_found:
            unique_atoms.append(i)
    return np.array(unique_atoms, dtype=int)


def kron_sum_c(
    spg_reps: SpgReps,
    C: coo_array,
):
    """Compute sum_r kron(r, r) / N_r in NN33 order in C.

    Sum of kron(r, r) are computed for unique r.

    Parameters
    ----------
    reps : list[coo_array]
        Symmetry operation representations in 3Nx3N.
    natom : int
        Number of atoms in supercell.
    C : coo_array
        Compression matrix.

    """
    mat_sum = coo_array(([], ([], [])), shape=(C.shape[1], C.shape[1]), dtype="double")
    for i, _ in enumerate(spg_reps.representations):
        mat = spg_reps.get_fc2_rep(i)
        mat = mat @ C
        mat = C.T @ mat
        mat_sum += mat
    mat_sum /= len(spg_reps.representations)

    return mat_sum


def get_lat_trans_compr_matrix(trans_perms: np.ndarray) -> coo_array:
    """Return compression matrix by lattice translation symmetry.

    Matrix shape is (NN33, n_a*N33), where n_a is the number of independent
    atoms by lattice translation symmetry.

    Data order is (N, N, 3, 3, n_a, N, 3, 3) if it is in dense array.

    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattce points and
        atoms in supercell.

    """
    col, row, data = [], [], []
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_a = len(indep_atoms)
    N = trans_perms.shape[1]
    n_lp = N // n_a
    val = 1.0 / np.sqrt(n_lp)
    size_row = (N * 3) ** 2

    n = 0
    for i_patom in indep_atoms:
        for j in range(N):
            for a, b in itertools.product(range(3), range(3)):
                for i_trans, j_trans in zip(trans_perms[:, i_patom], trans_perms[:, j]):
                    data.append(val)
                    col.append(n)
                    row.append(to_serial(i_trans, a, j_trans, b, N))
                n += 1

    assert n * n_lp == size_row
    return coo_array((data, (row, col)), shape=(size_row, n), dtype="double")


def get_lat_trans_compr_indices(trans_perms: np.ndarray) -> np.ndarray:
    """Return indices to compress matrix by lat-trans-sym.

    Usage
    -----
    vec[indices] of shape (N**2*9,) vec gives an array of shape=(n_a*N*9, n_lp).
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
    size_row = (N * 3) ** 2

    n = 0
    indices = np.zeros((n_a * N * 9, n_lp), dtype="int_")
    nums = np.zeros(n_a * N * 9, dtype="int_")
    for i_patom in indep_atoms:
        for j in range(N):
            for a, b in itertools.product(range(3), range(3)):
                for i_trans, j_trans in zip(trans_perms[:, i_patom], trans_perms[:, j]):
                    indices[n, nums[n]] = to_serial(i_trans, a, j_trans, b, N)
                    nums[n] += 1
                n += 1
    assert n * n_lp == size_row
    return indices


def get_lat_trans_decompr_indices(
    trans_perms: np.ndarray, shape: Literal["NN33", "N3,N3"] = "NN33"
) -> np.ndarray:
    """Return indices to de-compress compressed matrix by lat-trans-sym.

    Usage
    -----
    vec[indices] of shape (n_a*N*9,) gives an array of shape=(N**2*9,).
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
        Indices of n_a * N9 elements.
        shape=(N^2*9,) or (N3, N3)
        dtype='int_'.

    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_a = len(indep_atoms)
    N = trans_perms.shape[1]
    n_lp = N // n_a
    size_row = (N * 3) ** 2

    n = 0
    if shape == "NN33":
        indices = np.zeros(size_row, dtype="int_")
        for i_patom in indep_atoms:
            index_shift_i = trans_perms[:, i_patom] * N * 9
            for j in range(N):
                index_shift = index_shift_i + trans_perms[:, j] * 9
                for ab in range(9):
                    for shift in index_shift:
                        indices[shift + ab] = n
                    n += 1
    elif shape == "N3,N3":
        indices = np.zeros((N * 3, N * 3), dtype="int_")
        for i_patom in indep_atoms:
            index_shift_i = trans_perms[:, i_patom] * 3
            for j in range(N):
                index_shift_j = trans_perms[:, j] * 3
                for a, b in np.ndindex((3, 3)):
                    for i_t, j_t in zip(index_shift_i, index_shift_j):
                        indices[i_t + a, j_t + b] = n
                    n += 1
    else:
        raise RuntimeError("This should not happen.")
    assert n * n_lp == size_row
    return indices


def get_perm_compr_matrix(natom: int) -> coo_array:
    """Return compression matrix by permutation symmetry.

    Matrix shape is (NN33,(N*3)(N*3+1)/2).
    Non-zero only ijab and jiba column elements for ijab rows.
    Rows upper right NN33 matrix elements are selected for rows.

    """
    col, row, data = [], [], []
    val = np.sqrt(2) / 2
    size_row = natom**2 * 9

    n = 0
    for ia, jb in itertools.combinations_with_replacement(range(natom * 3), 2):
        i_i = ia // 3
        i_a = ia % 3
        i_j = jb // 3
        i_b = jb % 3
        col.append(n)
        row.append(to_serial(i_i, i_a, i_j, i_b, natom))
        if i_i == i_j and i_a == i_b:
            data.append(1)
        else:
            data.append(val)
            col.append(n)
            row.append(to_serial(i_j, i_b, i_i, i_a, natom))
            data.append(val)
        n += 1
    if (natom * 3) % 2 == 1:
        assert (natom * 3) * ((natom * 3 + 1) // 2) == n, f"{natom}, {n}"
    else:
        assert ((natom * 3) // 2) * (natom * 3 + 1) == n, f"{natom}, {n}"
    return coo_array((data, (row, col)), shape=(size_row, n), dtype="double")
