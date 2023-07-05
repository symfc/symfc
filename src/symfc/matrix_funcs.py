"""Functions to handle matrix indices."""
import numpy as np

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


def _transform_n3n3_serial(serial_id: int, natom: int) -> tuple[int, int, int, int]:
    """Decode 1D index to (N, 3, N, 3) indices."""
    b = serial_id % 3
    j = (serial_id // 3) % natom
    a = (serial_id // (3 * natom)) % 3
    i = serial_id // (9 * natom)
    return i, a, j, b, natom
