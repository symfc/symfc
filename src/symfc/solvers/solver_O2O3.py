"""Solver of 2nd and 3rd order force constants simultaneously."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.basis_sets import FCBasisSetO2, FCBasisSetO3
from symfc.solvers.solver_O2 import reshape_nN33_nx_to_N3_n3nx
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import get_batch_slice, solve_linear_equation
from symfc.utils.utils_O2 import (
    _get_atomic_lat_trans_decompr_indices,
    get_perm_compr_matrix,
)
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3

from .solver_base import FCSolverBase
from .solver_O2 import get_training_from_full_basis
from .solver_O3 import csr_NNN333_to_NN33N3, set_2nd_disps


class FCSolverO2O3(FCSolverBase):
    """Simultaneous second and third order force constants solver."""

    def __init__(
        self,
        basis_set: Sequence[FCBasisSetO2, FCBasisSetO3],
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        super().__init__(basis_set, use_mkl=use_mkl, log_level=log_level)

    def solve(
        self, displacements: np.ndarray, forces: np.ndarray, batch_size: int = 200
    ) -> FCSolverO2O3:
        """Solve force constants.

        Note
        ----
        self._coefs = (coefs_fc2, coefs_fc3)

        Parameters
        ----------
        displacements : ndarray
            Displacements of atoms in Cartesian coordinates. shape=(n_snapshot,
            N, 3), dtype='double'
        forces : ndarray
            Forces of atoms in Cartesian coordinates. shape=(n_snapshot, N, 3),
            dtype='double'

        Returns
        -------
        ndarray
            Force constants. shape=(n_a, N, 3, 3) or (N, N, 3, 3). See
            `is_compact_fc` parameter. dtype='double', order='C'

        """
        n_data = forces.shape[0]
        f = forces.reshape(n_data, -1)
        d = displacements.reshape(n_data, -1)

        fc2_basis: FCBasisSetO2 = self._basis_set[0]
        fc3_basis: FCBasisSetO3 = self._basis_set[1]
        compress_mat_fc2 = fc2_basis.compression_matrix
        basis_set_fc2 = fc2_basis.basis_set
        compress_mat_fc3 = fc3_basis.compression_matrix
        basis_set_fc3 = fc3_basis.basis_set

        self._coefs = run_solver_sparse_O2O3(
            d,
            f,
            compress_mat_fc2,
            compress_mat_fc3,
            basis_set_fc2,
            basis_set_fc3,
            batch_size=batch_size,
            use_mkl=self._use_mkl,
            verbose=self._log_level > 0,
        )
        return self

    @property
    def full_fc(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return full force constants.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            shape=(N, N, 3, 3), dtype='double', order='C'
            shape=(N, N, N, 3, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("full")

    @property
    def compact_fc(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return full force constants.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            shape=(n_a, N, 3, 3), dtype='double', order='C'
            shape=(n_a, N, N, 3, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("compact")

    def _recover_fcs(
        self, comp_mat_type: str = Literal["full", "compact"]
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self._coefs is None:
            return None

        fc2_basis: FCBasisSetO2 = self._basis_set[0]
        fc3_basis: FCBasisSetO3 = self._basis_set[1]
        if comp_mat_type == "full":
            comp_mat_fc2 = fc2_basis.compression_matrix
            comp_mat_fc3 = fc3_basis.compression_matrix
        elif comp_mat_type == "compact":
            comp_mat_fc2 = fc2_basis.compact_compression_matrix
            comp_mat_fc3 = fc3_basis.compact_compression_matrix
        else:
            raise ValueError("Invalid comp_mat_type.")

        N = self._natom
        fc2 = fc2_basis.basis_set @ self._coefs[0]
        fc2 = np.array(
            (comp_mat_fc2 @ fc2).reshape((-1, N, 3, 3)), dtype="double", order="C"
        )
        fc3 = fc3_basis.basis_set @ self._coefs[1]
        fc3 = np.array(
            (comp_mat_fc3 @ fc3).reshape((-1, N, N, 3, 3, 3)),
            dtype="double",
            order="C",
        )
        return fc2, fc3


def get_training_exact(
    disps: np.ndarray,
    forces: np.ndarray,
    compress_mat_fc2,
    compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    batch_size: int = 200,
    use_mkl: bool = False,
    verbose: bool = True,
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
    if verbose:
        print(" training data (fc2):    ", t2 - t1)

    t1 = time.time()
    c_perm_fc2 = get_perm_compr_matrix(N)
    compress_mat_fc3 = (
        csr_NNN333_to_NN33N3(compress_mat_fc3, N).reshape((NN33, -1)).tocsr()
    )
    compress_mat_fc3 = -0.5 * (c_perm_fc2.T @ compress_mat_fc3)
    t2 = time.time()
    if verbose:
        print(" precond. compress_mat (for fc3):", t2 - t1)

    sparse_disps = True if use_mkl else False
    mat33 = np.zeros((n_compr_fc3, n_compr_fc3), dtype=float)
    mat23 = np.zeros((n_basis_fc2, n_compr_fc3), dtype=float)
    mat3y = np.zeros(n_compr_fc3, dtype=float)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)
    for begin, end in zip(begin_batch, end_batch):
        t01 = time.time()
        disps_batch = set_2nd_disps(disps[begin:end], sparse=sparse_disps)
        disps_batch = disps_batch @ c_perm_fc2
        X3 = dot_product_sparse(
            disps_batch, compress_mat_fc3, use_mkl=use_mkl, dense=True
        ).reshape((-1, n_compr_fc3))
        y_batch = forces[begin:end].reshape(-1)
        mat23 += X2[begin * N3 : end * N3].T @ X3
        mat33 += X3.T @ X3
        mat3y += X3.T @ y_batch
        t02 = time.time()
        if verbose:
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
    if verbose:
        print(" (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):", t3 - t2)
    return XTX, XTy


def run_solver_sparse_O2O3(
    disps: np.ndarray,
    forces: np.ndarray,
    compress_mat_fc2,
    compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    batch_size: int = 200,
    use_mkl: bool = False,
    verbose: bool = True,
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
        verbose=verbose,
    )
    coefs = solve_linear_equation(XTX, XTy)
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    coefs_fc2, coefs_fc3 = coefs[:n_basis_fc2], coefs[n_basis_fc2:]
    return coefs_fc2, coefs_fc3


def _NNN333_to_NN33N3_core(row, N):
    """Reorder row indices in a sparse matrix (NNN333->NN33N3)."""
    # i
    div, rem = np.divmod(row, 27 * (N**2))
    row_update = div * 27 * (N**2)
    # j
    div, rem = np.divmod(rem, 27 * N)
    row_update += div * 27 * N
    # k
    div, rem = np.divmod(rem, 27)
    row_update += div * 3
    # a
    div, rem = np.divmod(rem, 9)
    row_update += div * 9 * N
    # b, c
    div, rem = np.divmod(rem, 3)
    row_update += div * 3 * N + rem
    return row_update


def _NNN333_to_NN33N3(row, N, n_batch=10):
    """Reorder row indices in a sparse matrix (NNN333->NN33N3).

    This function uses divided row index sets.
    """
    batch_size = len(row) // n_batch
    begin_batch, end_batch = get_batch_slice(len(row), batch_size)
    for begin, end in zip(begin_batch, end_batch):
        row[begin:end] = _NNN333_to_NN33N3_core(row[begin:end], N)
    return row


def reshape_compress_mat(mat, N, n_batch=10):
    """Reorder row indices in a sparse matrix (NNN333->NN33N3).

    This function involves reshaping the sparse matrix
    into another sparse matrix (NN33N3,Nx) -> (NN33, N3Nx).

    Return reordered csr_matrix.
    """
    NNN333, nx = mat.shape
    mat = mat.tocoo()
    mat.row = _NNN333_to_NN33N3(mat.row, N, n_batch=n_batch)

    NN33 = (N**2) * 9
    N3 = N * 3
    """
    # reshape: (NN33N3,Nx) -> (NN33, N3Nx)
    mat.row, rem = np.divmod(mat.row, N3)
    mat.col += rem * nx
    """
    batch_size = len(mat.row) // n_batch
    begin_batch, end_batch = get_batch_slice(len(mat.row), batch_size)
    for begin, end in zip(begin_batch, end_batch):
        mat.row[begin:end], rem = np.divmod(mat.row[begin:end], N3)
        mat.col[begin:end] += rem * nx

    return csr_array((mat.data, (mat.row, mat.col)), shape=(NN33, N3 * nx))


def get_training(
    disps,
    forces,
    compress_mat_fc2,
    compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    batch_size=100,
    use_mkl=False,
    compress_perm_fc2=False,
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
    """
    compress_mat_fc3 = (
        csr_NNN333_to_NN33N3(compress_mat_fc3, N).reshape((NN33, -1)).tocsr()
    )
    """
    """peak memory part (when batch size is less than nearly 50)"""
    compress_mat_fc3 = -0.5 * reshape_compress_mat(compress_mat_fc3, N)
    if compress_perm_fc2:
        c_perm_fc2 = get_perm_compr_matrix(N)
        compress_mat_fc3 = dot_product_sparse(
            c_perm_fc2.T, compress_mat_fc3, use_mkl=use_mkl
        )

    t2 = time.time()
    print(" precond. compress_mat (for fc3):", t2 - t1)

    sparse_disps = True if use_mkl else False
    mat33 = np.zeros((n_compr_fc3, n_compr_fc3), dtype=float)
    mat23 = np.zeros((n_basis_fc2, n_compr_fc3), dtype=float)
    mat3y = np.zeros(n_compr_fc3, dtype=float)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)
    for begin, end in zip(begin_batch, end_batch):
        t01 = time.time()
        """peak memory part (when batch size is more than nearly 50)"""
        disps_batch = set_2nd_disps(disps[begin:end], sparse=sparse_disps)
        if compress_perm_fc2:
            disps_batch = disps_batch @ c_perm_fc2

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


def get_training_no_sum_rule_basis(
    disps,
    forces,
    compress_mat_fc2,
    compress_mat_fc3,
    compress_eigvecs_fc2,
    batch_size=100,
    use_mkl=False,
    compress_perm_fc2=False,
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
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    n_compr_fc3 = compress_mat_fc3.shape[1]
    n_basis = n_basis_fc2 + n_compr_fc3

    t1 = time.time()
    full_basis_fc2 = compress_mat_fc2 @ compress_eigvecs_fc2
    X2, y_all = get_training_from_full_basis(
        disps, forces, full_basis_fc2.T.reshape((n_basis_fc2, N, N, 3, 3))
    )
    t2 = time.time()
    print(" training data (fc2):    ", t2 - t1)

    t1 = time.time()
    """peak memory part (when batch size is less than nearly 50)"""
    compress_mat_fc3 = -0.5 * reshape_compress_mat(compress_mat_fc3, N)
    if compress_perm_fc2:
        c_perm_fc2 = get_perm_compr_matrix(N)
        compress_mat_fc3 = dot_product_sparse(
            c_perm_fc2.T, compress_mat_fc3, use_mkl=use_mkl
        )
    t2 = time.time()
    print(" precond. compress_mat (for fc3):", t2 - t1)

    sparse_disps = True if use_mkl else False
    mat33 = np.zeros((n_compr_fc3, n_compr_fc3), dtype=float)
    mat23 = np.zeros((n_basis_fc2, n_compr_fc3), dtype=float)
    mat3y = np.zeros(n_compr_fc3, dtype=float)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)
    for begin, end in zip(begin_batch, end_batch):
        t01 = time.time()
        """peak memory part (when batch size is more than 50)"""
        disps_batch = set_2nd_disps(disps[begin:end], sparse=sparse_disps)
        if compress_perm_fc2:
            disps_batch = disps_batch @ c_perm_fc2

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
    XTX[:n_basis_fc2, n_basis_fc2:] = mat23
    XTX[n_basis_fc2:, :n_basis_fc2] = XTX[:n_basis_fc2, n_basis_fc2:].T
    XTX[n_basis_fc2:, n_basis_fc2:] = mat33
    XTy[:n_basis_fc2] = X2.T @ y_all
    XTy[n_basis_fc2:] = mat3y

    t3 = time.time()
    print(" (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):", t3 - t2)
    return XTX, XTy


def run_solver_O2O3(
    disps,
    forces,
    compress_mat_fc2,
    compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    batch_size=100,
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
    XTX, XTy = get_training(
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


def run_solver_O2O3_no_sum_rule_basis(
    disps,
    forces,
    compress_mat_fc2,
    compress_mat_fc3,
    compress_eigvecs_fc2,
    batch_size=100,
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
    XTX, XTy = get_training_no_sum_rule_basis(
        disps,
        forces,
        compress_mat_fc2,
        compress_mat_fc3,
        compress_eigvecs_fc2,
        batch_size=batch_size,
        use_mkl=use_mkl,
    )
    coefs = solve_linear_equation(XTX, XTy)
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    coefs_fc2, coefs_fc3 = coefs[:n_basis_fc2], coefs[n_basis_fc2:]
    return coefs_fc2, coefs_fc3


def set_disps_NN33(disps, sparse=True):
    """Calculate Kronecker products of displacements.

    Parameter
    ---------
    disps: shape=(n_supercell, N3)

    Return
    ------
    disps_2nd: shape=(n_supercell, NN33)
    """
    n_supercell = disps.shape[0]
    N = disps.shape[1] // 3
    disps_2nd = (disps[:, :, None] * disps[:, None, :]).reshape((-1, N, 3, N, 3))
    disps_2nd = disps_2nd.transpose((0, 1, 3, 2, 4)).reshape((n_supercell, -1))
    """
    N = disps.shape[1] // 3
    disps_2nd = np.zeros((disps.shape[0], 9 * (N**2)))
    for i, u_vec in enumerate(disps):
        u2 = np.kron(u_vec, u_vec).reshape((N, 3, N, 3))
        disps_2nd[i] = u2.transpose((0, 2, 1, 3)).reshape(-1)
    """
    if sparse:
        return csr_array(disps_2nd)
    return disps_2nd


def reshape_nNN333_nx_to_NN33_n3nx(mat, N, n, n_batch=9):
    """Reorder and reshape a sparse matrix (nNN333,nx)->(NN33,n3nx).

    Return reordered csr_matrix used for FC3.
    """
    _, nx = mat.shape
    NN33 = N**2 * 9
    n3nx = n * 3 * nx
    mat = mat.tocoo(copy=False)

    begin_batch, end_batch = get_batch_slice(len(mat.row), len(mat.row) // n_batch)
    for begin, end in zip(begin_batch, end_batch):
        div, rem = np.divmod(mat.row[begin:end], 27 * N * N)
        mat.col[begin:end] += div * 3 * nx
        div, rem = np.divmod(rem, 27 * N)
        mat.row[begin:end] = div * 9 * N
        div, rem = np.divmod(rem, 27)
        mat.row[begin:end] += div * 9
        div, rem = np.divmod(rem, 9)
        mat.col[begin:end] += div * nx
        div, rem = np.divmod(rem, 3)
        mat.row[begin:end] += div * 3 + rem

    mat.resize((NN33, n3nx))
    mat = mat.tocsr(copy=False)
    return mat


def prepare_normal_equation_O2O3(
    disps,
    forces,
    compact_compress_mat_fc2,
    compact_compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    trans_perms,
    atomic_decompr_idx_fc2=None,
    atomic_decompr_idx_fc3=None,
    batch_size=100,
    use_mkl=False,
):
    r"""Calculate X.T @ X and X.T @ y.

    X = displacements @ compress_mat @ compress_eigvecs
    X = np.hstack([X_fc2, X_fc3])

    displacements (fc2): (n_samples, N3)
    displacements (fc3): (n_samples, NN33)
    compact_compress_mat_fc2: (n_aN33, n_compr)
    compact_compress_mat_fc3: (n_aNN333, n_compr_fc3)
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
    NN = N * N
    NNN333 = 27 * N**3

    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    n_basis_fc3 = compress_eigvecs_fc3.shape[1]
    n_compr_fc2 = compact_compress_mat_fc2.shape[1]
    n_compr_fc3 = compact_compress_mat_fc3.shape[1]
    n_basis = n_basis_fc2 + n_basis_fc3

    n_lp, _ = trans_perms.shape
    if atomic_decompr_idx_fc2 is None:
        atomic_decompr_idx_fc2 = _get_atomic_lat_trans_decompr_indices(trans_perms)
    if atomic_decompr_idx_fc3 is None:
        atomic_decompr_idx_fc3 = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    NNN333_unit = 27 * 256**3
    n_batch = min((NNN333 // NNN333_unit + 1) * (n_compr_fc3 // 30000 + 1), N)
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    mat22 = np.zeros((n_compr_fc2, n_compr_fc2), dtype=float)
    mat23 = np.zeros((n_compr_fc2, n_compr_fc3), dtype=float)
    mat33 = np.zeros((n_compr_fc3, n_compr_fc3), dtype=float)
    mat2y = np.zeros(n_compr_fc2, dtype=float)
    mat3y = np.zeros(n_compr_fc3, dtype=float)

    t_all1 = time.time()
    const_fc2 = -1.0 / np.sqrt(n_lp)
    const_fc3 = -0.5 / np.sqrt(n_lp)
    compact_compress_mat_fc2 *= const_fc2
    compact_compress_mat_fc3 *= const_fc3
    for begin_i, end_i in zip(begin_batch_atom, end_batch_atom):
        print("Solver_atoms:", begin_i + 1, "--", end_i, "/", N)
        n_atom_batch = end_i - begin_i

        t1 = time.time()
        decompr_idx = (
            atomic_decompr_idx_fc2[begin_i * N : end_i * N, None] * 9
            + np.arange(9)[None, :]
        ).reshape(-1)
        compr_mat_fc2 = reshape_nN33_nx_to_N3_n3nx(
            compact_compress_mat_fc2[decompr_idx],
            N,
            n_atom_batch,
        )

        decompr_idx = (
            atomic_decompr_idx_fc3[begin_i * NN : end_i * NN, None] * 27
            + np.arange(27)[None, :]
        ).reshape(-1)
        compr_mat_fc3 = reshape_nNN333_nx_to_NN33_n3nx(
            compact_compress_mat_fc3[decompr_idx],
            N,
            n_atom_batch,
        )
        t2 = time.time()
        print("Solver_compr_matrix_reshape:, t =", "{:.3f}".format(t2 - t1))

        for begin, end in zip(begin_batch, end_batch):
            t1 = time.time()
            X2 = dot_product_sparse(
                disps[begin:end],
                compr_mat_fc2,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc2))
            X3 = dot_product_sparse(
                set_disps_NN33(disps[begin:end], sparse=False),
                compr_mat_fc3,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc3))

            y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)
            mat22 += X2.T @ X2
            mat23 += X2.T @ X3
            mat33 += X3.T @ X3
            mat2y += X2.T @ y
            mat3y += X3.T @ y
            t2 = time.time()
            print("Solver_block:", end, ":, t =", "{:.3f}".format(t2 - t1))

    XTX = np.zeros((n_basis, n_basis), dtype=float)
    XTy = np.zeros(n_basis, dtype=float)
    XTX[:n_basis_fc2, :n_basis_fc2] = (
        compress_eigvecs_fc2.T @ mat22 @ compress_eigvecs_fc2
    )
    XTX[:n_basis_fc2, n_basis_fc2:] = (
        compress_eigvecs_fc2.T @ mat23 @ compress_eigvecs_fc3
    )
    XTX[n_basis_fc2:, :n_basis_fc2] = XTX[:n_basis_fc2, n_basis_fc2:].T
    XTX[n_basis_fc2:, n_basis_fc2:] = (
        compress_eigvecs_fc3.T @ mat33 @ compress_eigvecs_fc3
    )
    XTy[:n_basis_fc2] = compress_eigvecs_fc2.T @ mat2y
    XTy[n_basis_fc2:] = compress_eigvecs_fc3.T @ mat3y

    compact_compress_mat_fc2 /= const_fc2
    compact_compress_mat_fc3 /= const_fc3
    t_all2 = time.time()
    print(
        " (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):",
        "{:.3f}".format(t_all2 - t_all1),
    )
    return XTX, XTy


def run_solver_O2O3_update(
    disps,
    forces,
    compact_compress_mat_fc2,
    compact_compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    trans_perms,
    atomic_decompr_idx_fc2=None,
    atomic_decompr_idx_fc3=None,
    batch_size=100,
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
    XTX, XTy = prepare_normal_equation_O2O3(
        disps,
        forces,
        compact_compress_mat_fc2,
        compact_compress_mat_fc3,
        compress_eigvecs_fc2,
        compress_eigvecs_fc3,
        trans_perms,
        atomic_decompr_idx_fc2=atomic_decompr_idx_fc2,
        atomic_decompr_idx_fc3=atomic_decompr_idx_fc3,
        batch_size=batch_size,
        use_mkl=use_mkl,
    )
    coefs = solve_linear_equation(XTX, XTy)
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    coefs_fc2, coefs_fc3 = coefs[:n_basis_fc2], coefs[n_basis_fc2:]
    return coefs_fc2, coefs_fc3
