"""Functions to handle matrix indices."""
import itertools

import numpy as np
import scipy
from scipy.sparse import csr_array

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


def kron_c(reps, natom) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    row_dtype = reps[0].row.dtype
    col_dtype = reps[0].col.dtype
    assert row_dtype is np.dtype("intc") or row_dtype is np.dtype("int_")
    assert reps[0].row.flags.contiguous
    assert col_dtype is np.dtype("intc") or col_dtype is np.dtype("int_")
    assert reps[0].col.flags.contiguous
    assert data_dtype == np.dtype("double")
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


def get_projector_sum_rule(natom) -> csr_array:
    """Return sum rule constraint projector.

    Equivalent to C below,

    A = get_projector_constraints_sum_rule_array(natom)
    C = scipy.sparse.eye(size_sq) - (A @ A.T) / self._natom

    """
    size_sq = 9 * natom * natom
    row, col, data = [], [], []
    block = np.tile(np.eye(9), (natom, natom))
    csr = csr_array(block)
    for i in range(natom):
        row1, col1 = csr.nonzero()
        row += (row1 + 9 * natom * i).tolist()
        col += (col1 + 9 * natom * i).tolist()
        data += (csr.data / natom).tolist()
    C = csr_array((data, (row, col)), shape=(size_sq, size_sq))
    proj = scipy.sparse.eye(size_sq) - C
    return proj


def get_projector_permutations(natom: int) -> csr_array:
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
    C = csr_array((data, (row, col)), shape=(size_sq, size_sq))
    proj = scipy.sparse.eye(size_sq) - C
    return proj


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
