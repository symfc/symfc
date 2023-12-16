"""Solver of 2nd and 3rd order force constants simultaneously."""
import time

import numpy as np

from symfc.utils.eig_tools import dot_product_sparse

from .solver_funcs import get_batch_slice, solve_linear_equation
from .solver_O2 import get_training_from_full_basis
from .solver_O3 import csr_NNN333_to_NN33N3, set_2nd_disps


def get_training_exact(
    disps,
    forces,
    compress_mat_fc2,
    compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    batch_size=200,
    use_mkl=False,
):
    r"""Calculate X.T @ X and X.T @ y.

    X = displacements @ compress_mat @ compress_eigvecs
    X = np.hstack([X_fc2, X_fc3])

    displacements (fc2): (n_samples, N3)
    displacements (fc3): (n_samples, NN33)
    compress_mat_fc2: (NN33, n_compr)
    compress_mat_fc3: (NNN333, n_compr_fc3)
    compress_eigvecs_fc2: (n_compr_fc2, n_basis_fc2)
    compress_eigvecs_fc3: (n_compr_fc3, n_basis_fc3)
    Matrix reshapings are appropriately applied to compress_mat
    and its products.

    X.T @ X and X.T @ y are sequentially calculated using divided dataset.
        X.T @ X = \sum_i X_i.T @ X_i
        X.T @ y = \sum_i X_i.T @ y_i (i: batch index)

    """
    N3 = disps.shape[1]
    N = N3 // 3
    NN33 = N3 * N3
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    n_basis_fc3 = compress_eigvecs_fc3.shape[1]
    n_compr_fc3 = compress_mat_fc3.shape[1]
    n_basis = n_basis_fc2 + n_basis_fc3

    t1 = time.time()
    full_basis_fc2 = compress_mat_fc2 @ compress_eigvecs_fc2
    X2, y_all = get_training_from_full_basis(
        disps, forces, full_basis_fc2.T.reshape((n_basis_fc2, N, N, 3, 3))
    )
    t2 = time.time()
    print(" training data (fc2):    ", t2 - t1)

    t1 = time.time()
    compress_mat_fc3 = (
        -0.5 * csr_NNN333_to_NN33N3(compress_mat_fc3, N).reshape((NN33, -1)).tocsr()
    )
    t2 = time.time()
    print(" reshape (compr_mat_fc3):", t2 - t1)

    sparse_disps = True if use_mkl else False
    mat33 = np.zeros((n_compr_fc3, n_compr_fc3), dtype=float)
    mat23 = np.zeros((n_basis_fc2, n_compr_fc3), dtype=float)
    mat3y = np.zeros(n_compr_fc3, dtype=float)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)
    for begin, end in zip(begin_batch, end_batch):
        t01 = time.time()
        disps_batch = set_2nd_disps(disps[begin:end], sparse=sparse_disps)
        X3 = dot_product_sparse(
            disps_batch, compress_mat_fc3, use_mkl=use_mkl, dense=True
        ).reshape((-1, n_compr_fc3))
        y_batch = forces[begin:end].reshape(-1)
        mat23 += X2[begin * N3 : end * N3].T @ X3
        mat33 += X3.T @ X3
        mat3y += X3.T @ y_batch
        t02 = time.time()
        print(" solver_block:", end, ":, t =", t02 - t01)

    XTX = np.zeros((n_basis, n_basis), dtype=float)
    XTy = np.zeros(n_basis, dtype=float)
    XTX[:n_basis_fc2, :n_basis_fc2] = X2.T @ X2
    XTX[:n_basis_fc2, n_basis_fc2:] = mat23 @ compress_eigvecs_fc3
    XTX[n_basis_fc2:, :n_basis_fc2] = XTX[:n_basis_fc2, n_basis_fc2:].T
    XTX[n_basis_fc2:, n_basis_fc2:] = (
        compress_eigvecs_fc3.T @ mat33 @ compress_eigvecs_fc3
    )
    XTy[:n_basis_fc2] = X2.T @ y_all
    XTy[n_basis_fc2:] = compress_eigvecs_fc3.T @ mat3y

    t3 = time.time()
    print(" (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):", t3 - t2)
    return XTX, XTy


def run_solver_sparse_O2O3(
    disps,
    forces,
    compress_mat_fc2,
    compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    batch_size=200,
    use_mkl=False,
):
    """Estimate coeffs. in X @ coeffs = y.

    X_fc2 = displacements_fc2 @ compress_mat_fc2 @ compress_eigvecs_fc2
    X_fc3 = displacements_fc3 @ compress_mat_fc3 @ compress_eigvecs_fc3
    X = np.hstack([X_fc2, X_fc3])

    Matrix reshapings are appropriately applied.
    X: features (n_samples * N3, N_basis_fc2 + N_basis_fc3)
    y: observations (forces), (n_samples * N3)

    """
    XTX, XTy = get_training_exact(
        disps,
        forces,
        compress_mat_fc2,
        compress_mat_fc3,
        compress_eigvecs_fc2,
        compress_eigvecs_fc3,
        batch_size=batch_size,
        use_mkl=use_mkl,
    )
    coefs = solve_linear_equation(XTX, XTy)
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    coefs_fc2, coefs_fc3 = coefs[:n_basis_fc2], coefs[n_basis_fc2:]
    return coefs_fc2, coefs_fc3
