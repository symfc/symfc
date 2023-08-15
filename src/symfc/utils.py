"""Functions to handle matrix indices."""
import itertools
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import coo_array, csr_array

import symfc._symfc as symfcc


def to_serial(i: int, a: int, j: int, b: int, natom: int) -> int:
    """Return NN33-1D index."""
    return (i * 9 * natom) + (j * 9) + (a * 3) + b


def transform_n3n3_serial_to_nn33_serial(n3n3_serial, natom) -> int:
    """Convert N3N3-1D index to NN33-1D index."""
    return to_serial(*_transform_n3n3_serial(n3n3_serial, natom))


def convert_basis_sets_matrix_form(basis_sets) -> list[np.ndarray]:
    """Convert basis sets to matrix form."""
    b_mat_all = []
    for b in basis_sets:
        b_seq = b.transpose((0, 2, 1, 3))
        b_mat = b_seq.reshape(
            (b_seq.shape[0] * b_seq.shape[1], b_seq.shape[2] * b_seq.shape[3])
        )
        b_mat_all.append(b_mat)
    return b_mat_all


def kron_c(
    reps: list[coo_array], natom: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute kron(r, r) in NN33 order in C.

    See the details about this method in _step1_kron_py_for_c.

    Note
    ----
    At some version of scipy, dtype of coo_array.col and coo_array.row changed.
    Here the dtype is assumed 'intc' (<1.11) or 'int_' (>=1.11).

    """
    size = 0
    for rmat in reps:
        size += rmat.row.shape[0] ** 2
    row_dtype = reps[0].row.dtype
    col_dtype = reps[0].col.dtype
    data_dtype = reps[0].data.dtype
    row = np.zeros(size, dtype=row_dtype)
    col = np.zeros(size, dtype=col_dtype)
    data = np.zeros(size, dtype=data_dtype)
    assert row_dtype is np.dtype("intc") or row_dtype is np.dtype("int_")
    assert reps[0].row.flags.contiguous
    assert col_dtype is np.dtype("intc") or col_dtype is np.dtype("int_")
    assert reps[0].col.flags.contiguous
    assert data_dtype is np.dtype("double")
    assert reps[0].data.flags.contiguous
    i_shift = 0
    for rmat in reps:
        if col_dtype is np.dtype("intc") and row_dtype is np.dtype("intc"):
            symfcc.kron_nn33_int(
                row[i_shift:],
                col[i_shift:],
                data[i_shift:],
                rmat.row,
                rmat.col,
                rmat.data,
                3 * natom,
            )
        elif col_dtype is np.dtype("int_") and row_dtype is np.dtype("int_"):
            symfcc.kron_nn33_long(
                row[i_shift:],
                col[i_shift:],
                data[i_shift:],
                rmat.row,
                rmat.col,
                rmat.data,
                3 * natom,
            )
        else:
            raise RuntimeError("Incompatible data type of rows and cols of coo_array.")
        i_shift += rmat.row.shape[0] ** 2
    data /= len(reps)

    return row, col, data


def kron_sum_c(
    reps: list[coo_array],
    natom: int,
    C: coo_array,
    rotations: Optional[np.ndarray] = None,
    translation_indices: Optional[np.ndarray] = None,
):
    """Compute sum_r kron(r, r) / N_r in NN33 order in C.

    When ``rotations`` and ``translation_indices`` are given
        1. Sum of kron(r, r) for unique r's are computed.
        2. Sum of kron(r, r) for r=identity are computed.
        3. Product of 1 and 2 are computed.
    Otherwise
        Sum of kron(r, r) are computed. Difference from kron_c is that in this
        function, coo_array is created for each r.

    Parameters
    ----------
    reps : list[coo_array]
        Symmetry operation representations in 3Nx3N.
    natom : int
        Number of atoms in supercell.
    C : coo_array
        Compression matrix.
    rotations : np.ndarray
        Rotation matrices of all symmetry operations. Optional.
    translation_indices : np.ndarray
        Indices of pure translation operations in all symmetry operations.
        Optional.

    Note
    ----
    At some version of scipy, dtype of coo_array.col and coo_array.row changed.
    Here the dtype is assumed 'intc' (<1.11) or 'int_' (>=1.11).

    """
    row_dtype = reps[0].row.dtype
    col_dtype = reps[0].col.dtype
    data_dtype = reps[0].data.dtype
    assert row_dtype is np.dtype("intc") or row_dtype is np.dtype("int_")
    assert reps[0].row.flags.contiguous
    assert col_dtype is np.dtype("intc") or col_dtype is np.dtype("int_")
    assert reps[0].col.flags.contiguous
    assert data_dtype is np.dtype("double")
    assert reps[0].data.flags.contiguous

    if rotations is None or translation_indices is None:
        # Sum over all operations.
        kron_sum = coo_array(
            ([], ([], [])), shape=(C.shape[1], C.shape[1]), dtype="double"
        )
        for i, rmat in enumerate(reps):
            kron = _kron_each_c(rmat, natom, row_dtype, col_dtype, data_dtype)
            kron = kron @ C
            kron = C.T @ kron
            kron_sum += kron
        kron_sum /= len(reps)
    else:
        # Sum over unique r operations (r's of coset representatives)
        kron_sum_rot = coo_array(
            ([], ([], [])), shape=(C.shape[1], C.shape[1]), dtype="double"
        )
        unique_rotations: list[np.ndarray] = []
        for i, rmat in enumerate(reps):
            if rotations is not None:
                is_found = False
                for r in unique_rotations:
                    if np.array_equal(r, rotations[i]):
                        is_found = True
                        break
                if is_found:
                    continue
                else:
                    unique_rotations.append(rotations[i])

            kron = _kron_each_c(rmat, natom, row_dtype, col_dtype, data_dtype)
            kron = kron @ C
            kron = C.T @ kron
            kron_sum_rot += kron

        # Sum over all r=identity.
        kron_sum_trans = coo_array(
            ([], ([], [])), shape=(C.shape[1], C.shape[1]), dtype="double"
        )
        eye3 = np.eye(3, dtype=int)
        num_latp = 0
        for i in translation_indices:
            rmat = reps[i]
            if np.array_equal(eye3, rotations[i]):
                num_latp += 1
                kron = _kron_each_c(rmat, natom, row_dtype, col_dtype, data_dtype)
                kron = kron @ C
                kron = C.T @ kron
                kron_sum_trans += kron

        # Product of coset representatives and translation group.
        kron_sum = kron_sum_trans @ kron_sum_rot
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
    rotations: Optional[np.ndarray] = None,
    translation_indices: Optional[np.ndarray] = None,
) -> coo_array:
    """Compute compact spg projector matrix.

    This computes perm_mat.T @ spg_proj (kron_c) @ perm_mat.

    """
    C = compression_mat
    # row, col, data = kron_c(reps, natom)
    # size_sq = (3 * natom) ** 2
    # proj_mat = coo_array((data, (row, col)), shape=(size_sq, size_sq), dtype="double")
    # proj_mat = proj_mat @ C
    # proj_mat = C.T @ proj_mat
    proj_mat = kron_sum_c(
        reps, natom, C, rotations=rotations, translation_indices=translation_indices
    )
    return proj_mat


def get_projector_constraints(
    natom: int, with_permutation: bool = True, with_translation: bool = True
) -> csr_array:
    """Construct matrices of sum rule and permutation."""
    size_sq = (3 * natom) ** 2
    C = _get_projector_constraints_array(
        natom, with_permutation=with_permutation, with_translation=with_translation
    )
    Cinv = scipy.sparse.linalg.inv((C.T).dot(C))
    proj = scipy.sparse.eye(size_sq) - (C.dot(Cinv)).dot(C.T)
    return proj


def get_projector_sum_rule(natom) -> coo_array:
    """Return sum rule constraint projector.

    Equivalent to C below,

    A = get_projector_constraints_sum_rule_array(natom)
    C = scipy.sparse.eye(size_sq) - (A @ A.T) / self._natom

    """
    size_sq = 9 * natom * natom
    block = np.tile(np.eye(9, dtype=int), (natom, natom))
    csr = coo_array(block)
    row1, col1 = csr.nonzero()
    size = row1.shape[0]
    row = np.zeros(size * natom, dtype=row1.dtype)
    col = np.zeros(size * natom, dtype=col1.dtype)
    data = np.ones(size * natom, dtype=float) / natom
    if row1.dtype is np.dtype("intc") and col1.dtype is np.dtype("intc"):
        symfcc.projector_sum_rule_int(row, col, row1, col1, natom)
    elif row1.dtype is np.dtype("int_") and col1.dtype is np.dtype("int_"):
        symfcc.projector_sum_rule_long(row, col, natom)
    else:
        raise RuntimeError("Incompatible data type of rows and cols of coo_array.")
    # for i in range(natom):
    #     row[size * i : size * (i + 1)] = row1 + 9 * natom * i
    #     col[size * i : size * (i + 1)] = col1 + 9 * natom * i
    C = coo_array((data, (row, col)), shape=(size_sq, size_sq))
    proj = scipy.sparse.eye(size_sq) - C
    return proj


def get_projector_permutations(natom: int) -> coo_array:
    """Return permutation constraint projector."""
    size = 3 * natom
    size_sq = size**2
    row, col, data = [], [], []
    for ia, jb in itertools.combinations(range(size), 2):
        i, a = ia // 3, ia % 3
        j, b = jb // 3, jb % 3
        id1 = to_serial(i, a, j, b, natom)
        id2 = to_serial(j, b, i, a, natom)
        row += [id1, id2, id1, id2]
        col += [id1, id2, id2, id1]
        data += [0.5, 0.5, -0.5, -0.5]
    C = coo_array((data, (row, col)), shape=(size_sq, size_sq))
    proj = scipy.sparse.eye(size_sq) - C
    return proj


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


def _transform_n3n3_serial(serial_id: int, natom: int) -> tuple[int, int, int, int]:
    """Decode 1D index to (N, 3, N, 3) indices."""
    b = serial_id % 3
    j = (serial_id // 3) % natom
    a = (serial_id // (3 * natom)) % 3
    i = serial_id // (9 * natom)
    return i, a, j, b, natom


def _get_projector_constraints_array(
    natom: int, with_permutation: bool = True, with_translation: bool = True
) -> csr_array:
    size_sq = (3 * natom) ** 2
    n, row, col, data = 0, [], [], []
    # sum rules
    if with_translation:
        n = _get_projector_constraints_sum_rule(row, col, data, natom, n)

    # permutation
    if with_permutation:
        n = _get_projector_constraints_permutations(row, col, data, natom, n)

    # Temporary fix
    # scipy.sparse.linalg.inv (finally splu) doesn't accept
    # "int_" (or list[int]) row and col values at scipy 1.11.1.
    dtype = "intc"
    row = np.array(row, dtype=dtype)
    col = np.array(col, dtype=dtype)
    return csr_array((data, (row, col)), shape=(size_sq, n))


def _get_projector_constraints_sum_rule(
    row: list, col: list, data: list, natom: int, n: int
) -> int:
    """Sparse array data of sum rule constraints.

    Each column contains N of 1 and others are zero.
    shape=((3N)**2, 9N)

    """
    for i in range(natom):
        for alpha, beta in itertools.product(range(3), range(3)):
            for j in range(natom):
                row.append(to_serial(i, alpha, j, beta, natom))
                col.append(n)
                data.append(1.0)
            n += 1
    return n


def _get_projector_constraints_permutations(
    row: list, col: list, data: list, natom: int, n: int
) -> int:
    """Sparse array data of permutation constraints.

    Each colum contains one 1 and one -1, and others are all zero.
    Diagonal elements in (NN33, NN33) representation are zero.
    shape=((3N)**2, 3N(3N-1))

    """
    for ia, jb in itertools.combinations(range(natom * 3), 2):
        i, a = ia // 3, ia % 3
        j, b = jb // 3, jb % 3
        id1 = to_serial(i, a, j, b, natom)
        id2 = to_serial(j, b, i, a, natom)
        row += [id1, id2]
        col += [n, n]
        data += [1, -1]
        n += 1
    return n
