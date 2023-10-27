"""Functions to handle matrix indices."""
import itertools

import numpy as np
from scipy.sparse import coo_array

import symfc._symfc as symfcc


def to_serial(i: int, a: int, j: int, b: int, natom: int) -> int:
    """Return NN33-1D index."""
    return (i * 9 * natom) + (j * 9) + (a * 3) + b


def convert_basis_set_matrix_form(basis_set) -> list[np.ndarray]:
    """Convert basis set to matrix form."""
    b_mat_all = []
    for b in basis_set:
        b_seq = b.transpose((0, 2, 1, 3))
        b_mat = b_seq.reshape(
            (b_seq.shape[0] * b_seq.shape[1], b_seq.shape[2] * b_seq.shape[3])
        )
        b_mat_all.append(b_mat)
    return b_mat_all


def kron_sum_c(
    reps: list[coo_array],
    natom: int,
    C: coo_array,
):
    """Compute sum_r kron(r, r) / N_r in NN33 order in C.

    Sum of kron(r, r) are computed for unique r. Difference from kron_c is that
    in this function, coo_array is created for each r.

    Parameters
    ----------
    reps : list[coo_array]
        Symmetry operation representations in 3Nx3N.
    natom : int
        Number of atoms in supercell.
    C : coo_array
        Compression matrix.

    Note
    ----
    At some version of scipy, dtype of coo_array.col and coo_array.row changed.
    Here the dtype is assumed 'intc' (<1.11) or 'int_' (>=1.11).

    """
    row_dtype = reps[0].row.dtype
    col_dtype = reps[0].col.dtype
    data_dtype = reps[0].data.dtype
    assert row_dtype in (np.dtype("intc"), np.dtype("int_"))
    assert reps[0].row.flags.contiguous
    assert col_dtype in (np.dtype("intc"), np.dtype("int_"))
    assert reps[0].col.flags.contiguous
    assert data_dtype is np.dtype("double")
    assert reps[0].data.flags.contiguous

    kron_sum = coo_array(([], ([], [])), shape=(C.shape[1], C.shape[1]), dtype="double")
    for rmat in reps:
        kron = _kron_each_c(rmat, natom, row_dtype, col_dtype, data_dtype)
        kron = kron @ C
        kron = C.T @ kron
        kron_sum += kron
    kron_sum /= len(reps)

    return kron_sum


def _kron_each_c(
    rmat: coo_array,
    natom: int,
    row_dtype: np.dtype,
    col_dtype: np.dtype,
    data_dtype: np.dtype,
):
    size_sq = (3 * natom) ** 2
    size = rmat.row.shape[0] ** 2
    row = np.zeros(size, dtype=row_dtype)
    col = np.zeros(size, dtype=col_dtype)
    data = np.zeros(size, dtype=data_dtype)
    args = (row, col, data, rmat.row, rmat.col, rmat.data, 3 * natom)
    if col_dtype is np.dtype("intc") and row_dtype is np.dtype("intc"):
        symfcc.kron_nn33_int(*args)
    elif col_dtype is np.dtype("int_") and row_dtype is np.dtype("int_"):
        symfcc.kron_nn33_long(*args)
    else:
        raise RuntimeError("Incompatible data type of rows and cols of coo_array.")
    return coo_array((data, (row, col)), shape=(size_sq, size_sq), dtype="double")


def get_compression_spg_proj(
    reps: list[coo_array],
    natom: int,
    compression_mat: coo_array,
) -> coo_array:
    """Compute compact spg projector matrix.

    This computes C.T @ spg_proj (kron_c) @ perm_proj @ C,
    where C is ``compression_mat``.

    """
    coset_reps_sum = kron_sum_c(
        reps,
        natom,
        compression_mat,
    )
    # lattice translation and index permutation symmetry are projected.
    C_perm = _get_permutation_compression_matrix(natom)
    perm = C_perm.T @ compression_mat
    perm = C_perm @ perm
    perm = compression_mat.T @ perm

    return coset_reps_sum @ perm


def get_indep_atoms_by_lattice_translation(trans_perms: np.ndarray) -> np.ndarray:
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
    unique_atoms = []
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


def get_lattice_translation_compression_matrix(trans_perms: np.ndarray) -> coo_array:
    """Return compression matrix by lattice translation symmetry.

    Matrix shape is (NN33, n_a*N33), where n_a is the number of independent
    atoms by lattice translation symmetry.

    """
    col, row, data = [], [], []
    indep_atoms = get_indep_atoms_by_lattice_translation(trans_perms)
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


def _get_permutation_compression_matrix(natom: int) -> coo_array:
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
