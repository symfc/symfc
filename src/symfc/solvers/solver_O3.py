"""3rd order force constants solver."""

import time
from typing import Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import fit, get_batch_slice, solve_linear_equation
from symfc.utils.utils_O2 import get_perm_compr_matrix


def run_solver_dense_O3(disps, forces, compress_mat, compress_eigvecs):
    """Estimate coeffs. in X @ coeffs = y.

    X: features (calculated from Kronecker products of displacements)
                (n_samples * N3, N_basis)
    y: observations (forces), (n_samples * N3)

    """
    N3 = disps.shape[1]
    N = N3 // 3

    t0 = time.time()
    full_basis = compress_mat @ compress_eigvecs
    n_basis = full_basis.shape[1]
    full_basis = full_basis.T.reshape((n_basis, N, N, N, 3, 3, 3))
    t1 = time.time()
    print(" elapsed time (compute full basis) =", t1 - t0)

    X, y = _get_training_from_full_basis_set(disps, forces, full_basis)
    coefs = fit(X, y)
    t2 = time.time()
    print(" elapsed time (solve fc3)          =", t2 - t1)

    return coefs


def run_solver_sparse_O3(
    disps, forces, compress_mat, compress_eigvecs, batch_size=200, use_mkl=False
):
    """Estimating coeffs. in X @ coeffs = y by solving normal equation.

    (X.T @ X) @ coeffs = X.T @ y
    X = displacements @ compress_mat @ compress_eigvecs
    Matrix reshapings are appropriately applied.
    X: features (n_samples * N3, N_basis)
    y: observations (forces), (n_samples * N3)

    """
    XTX, XTy = _get_training_exact(
        disps,
        forces,
        compress_mat,
        compress_eigvecs,
        batch_size=batch_size,
        use_mkl=use_mkl,
    )
    coefs = solve_linear_equation(XTX, XTy)
    return coefs


def csr_NNN333_to_NN33N3(mat, N):
    """Reorder row indices in a sparse matrix (NNN333->NN33N3).

    Return reordered csr_matrix.

    """
    NNN333, nx = mat.shape
    mat = mat.tocoo()
    row = _NNN333_to_NN33N3(mat.row, N)
    mat = csr_array((mat.data, (row, mat.col)), shape=(NNN333, nx))
    return mat


def csr_NNN333_to_NN33na3(compress_mat: csr_array, N: int, n_a: Optional[int] = None):
    """Reorder row indices in a sparse matrix.

    (n,N,N,3,3,3) -> (N,3,N,3,n,3)
    (i,j,k,a,b,c) -> (j,b,k,c,i,a)

    Return reordered csr_matrix.

    n can be any integer, but it is supporsed to be either N or n_a, where n_a
    is the number of atoms in primitive cell.

    """
    compress_mat_coo = compress_mat.tocoo()
    data = compress_mat_coo.data
    row = compress_mat_coo.row
    col = compress_mat_coo.col

    if n_a is None:
        _N = N
    else:
        _N = n_a

    NN333 = N * _N * 27
    N333 = _N * 27
    N33 = _N * 9
    N3 = _N * 3
    conversion_array = np.array(
        [
            j * NN333 + k * N333 + m * N33 + n * N3 + i * 3 + l
            for i, j, k, l, m, n in np.ndindex((_N, N, N, 3, 3, 3))
        ],
        dtype=int,
    )
    new_row = conversion_array[row]
    return csr_array((data, (new_row, col)), shape=compress_mat.shape).reshape(
        N * N * 3 * 3, -1
    )


def set_2nd_disps(disps, sparse=True):
    """Calculate Kronecker products of displacements.

    disps: (n_samples, N3)
    disps_2nd: (n_samples, NN33)

    """
    N = disps.shape[1] // 3
    disps_2nd = np.zeros((disps.shape[0], 9 * (N**2)))
    for i, u_vec in enumerate(disps):
        u2 = np.kron(u_vec, u_vec).reshape((N, 3, N, 3))
        disps_2nd[i] = u2.transpose((0, 2, 1, 3)).reshape(-1)

    if sparse:
        return csr_array(disps_2nd)
    return disps_2nd


def _NNN333_to_NN33N3(row, N):
    """Reorder row indices in a sparse matrix (NNN333->NN33N3)."""
    i, rem = np.divmod(row, 27 * (N**2))
    j, rem = np.divmod(rem, 27 * N)
    k, rem = np.divmod(rem, 27)
    a, rem = np.divmod(rem, 9)
    b, c = np.divmod(rem, 3)

    vec = i * 27 * (N**2)
    vec += j * 27 * N
    vec += k * 3
    vec += a * 9 * N
    vec += b * 3 * N + c
    return vec


def _get_training_from_full_basis_set(disps, forces, full_basis):
    """Return training dataset (X, y).

    The dataset is transformed from displacements, forces, and full basis-set.

    disps: (n_samples, N3)
    forces: (n_samples, N3)
    full_basis: (n_basis, N, N, N, 3, 3, 3)

    X: features (calculated from Kronecker products of displacements)
                (n_samples * N3, N_basis)
    y: observations (forces), (n_samples * N3)

    X = displacements @ compress_mat @ compress_eigvecs

    displacements: (n_samples, NN33)
    compress_mat: (NNN333, n_compr)
    compress_eigvecs: (n_compr, n_basis)
    Matrix reshapings are appropriately applied to compress_mat
    and its products.

    """
    N3 = full_basis.shape[1] * 3
    NN33 = N3 * N3
    n_basis = full_basis.shape[0]

    disps_reshape = set_2nd_disps(disps, sparse=False)

    full_basis = full_basis.transpose((1, 2, 4, 5, 3, 6, 0)).reshape((NN33, -1))
    X = -0.5 * np.dot(disps_reshape, full_basis)
    X = X.reshape((-1, n_basis))
    y = forces.reshape(-1)
    return X, y


def _get_training_exact(
    disps, forces, compress_mat, compress_eigvecs, batch_size=200, use_mkl=False
):
    r"""Calculate X.T @ X and X.T @ y.

    X = displacements @ compress_mat @ compress_eigvecs

    displacements: (n_samples, NN33)
    compress_mat: (NNN333, n_compr)
    compress_eigvecs: (n_compr, n_basis)
    Matrix reshapings are appropriately applied to compress_mat
    and its products.

    X.T @ X and X.T @ y are sequentially calculated using divided dataset.
        X.T @ X = \sum_i X_i.T @ X_i
        X.T @ y = \sum_i X_i.T @ y_i (i: batch index)

    """
    N3 = disps.shape[1]
    N = N3 // 3
    NN33 = N3 * N3
    n_compr = compress_mat.shape[1]

    t1 = time.time()
    c_perm_fc2 = get_perm_compr_matrix(N)
    compress_mat = csr_NNN333_to_NN33N3(compress_mat, N).reshape((NN33, -1)).tocsr()
    compress_mat = -0.5 * (c_perm_fc2.T @ compress_mat)
    t2 = time.time()
    print(" precond. compress_mat (for fc3):", t2 - t1)

    sparse_disps = True if use_mkl else False
    XTX = np.zeros((n_compr, n_compr), dtype=float)
    XTy = np.zeros(n_compr, dtype=float)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)
    for begin, end in zip(begin_batch, end_batch):
        t01 = time.time()
        disps_batch = set_2nd_disps(disps[begin:end], sparse=sparse_disps)
        disps_batch = disps_batch @ c_perm_fc2
        X = dot_product_sparse(
            disps_batch, compress_mat, use_mkl=use_mkl, dense=True
        ).reshape((-1, n_compr))
        y_batch = forces[begin:end].reshape(-1)
        XTX += X.T @ X
        XTy += X.T @ y_batch
        t02 = time.time()
        print(" solver_block:", end, ":, t =", t02 - t01)

    XTX = compress_eigvecs.T @ XTX @ compress_eigvecs
    XTy = compress_eigvecs.T @ XTy
    t3 = time.time()
    print(" (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):", t3 - t2)
    return XTX, XTy
