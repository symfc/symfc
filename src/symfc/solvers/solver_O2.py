"""2nd order force constants solver."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.basis_sets import FCBasisSetO2
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import get_batch_slice, solve_linear_equation
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices

from .solver_base import FCSolverBase


class FCSolverO2(FCSolverBase):
    """Third order force constants solver."""

    def __init__(
        self,
        basis_set: FCBasisSetO2,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        super().__init__(basis_set, use_mkl=use_mkl, log_level=log_level)

    @property
    def full_fc(self) -> Optional[np.ndarray]:
        """Return full force constants.

        Returns
        -------
        np.ndarray
            shape=(N, N, 3, 3), dtype='double', order='C'

        """
        N = self._natom
        fc = self._basis_set.basis_set @ self._coefs
        return np.array(
            (self._basis_set.compression_matrix @ fc).reshape((-1, N, 3, 3)),
            dtype="double",
            order="C",
        )

    @property
    def compact_fc(self) -> Optional[np.ndarray]:
        """Return full force constants.

        Returns
        -------
        np.ndarray
            shape=(n_a, N, 3, 3), dtype='double', order='C'

        """
        N = self._natom
        fc = self._basis_set.basis_set @ self._coefs
        return np.array(
            (self._basis_set.compact_compression_matrix @ fc).reshape((-1, N, 3, 3)),
            dtype="double",
            order="C",
        )

    def solve(
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
    ) -> FCSolverO2:
        """Solve coefficients of basis set from displacements and forces.

        Parameters
        ----------
        displacements : ndarray
            Displacements of atoms in Cartesian coordinates.
            shape=(n_snapshot, N, 3), dtype='double'
        forces : ndarray
            Forces of atoms in Cartesian coordinates.
            shape=(n_snapshot, N, 3), dtype='double'

        Returns
        -------
        self : FCSolverO2

        """
        n_data = forces.shape[0]
        f = forces.reshape(n_data, -1)
        d = displacements.reshape(n_data, -1)
        self._coefs = run_solver_sparse_O2(
            d,
            f,
            self._basis_set.compression_matrix,
            self._basis_set.basis_set,
            verbose=self._log_level > 0,
        )
        return self


def run_solver_sparse_O2(
    disps, forces, compress_mat, compress_eigvecs, batch_size=200, verbose=True
):
    """Estimating coeffs. in X @ coeffs = y by solving normal equation.

    (X.T @ X) @ coeffs = X.T @ y

    X = displacements @ compress_mat @ compress_eigvecs
    Matrix reshapings are appropriately applied.
    X: features (n_samples * N3, N_basis)
    y: observations (forces), (n_samples * N3)

    """
    XTX, XTy = _get_training(
        disps,
        forces,
        compress_mat,
        compress_eigvecs,
        batch_size=batch_size,
        verbose=verbose,
    )
    coefs = solve_linear_equation(XTX, XTy)
    return coefs


def _get_training(
    disps,
    forces,
    compress_mat,
    compress_eigvecs,
    batch_size=200,
    use_mkl=False,
    verbose=True,
):
    r"""Calculate X.T @ X and X.T @ y.

    X = displacements @ compress_mat @ compress_eigvecs

    displacements: (n_samples, N33)
    compress_mat: (NN33, n_compr)
    compress_eigvecs: (n_compr, n_basis)
    Matrix reshapings are appropriately applied to compress_mat
    and its products.

    X.T @ X and X.T @ y are sequentially calculated using divided dataset.
        X.T @ X = \sum_i X_i.T @ X_i
        X.T @ y = \sum_i X_i.T @ y_i (i: batch index)

    """
    N3 = disps.shape[1]
    N = N3 // 3
    n_basis = compress_eigvecs.shape[1]
    n_compr = compress_mat.shape[1]

    t1 = time.time()
    compress_mat = _csr_NN33_to_N3N3(compress_mat, N).reshape((N3, -1)).tocsr()
    t2 = time.time()
    if verbose:
        print(" reshape(compr):   ", t2 - t1, flush=True)

    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)
    XTX = np.zeros((n_basis, n_basis), dtype=float)
    XTy = np.zeros(n_basis, dtype=float)
    for begin, end in zip(begin_batch, end_batch):
        t01 = time.time()
        disps_batch = csr_array(disps[begin:end])
        X = dot_product_sparse(
            disps_batch, compress_mat, use_mkl=use_mkl, dense=True
        ).reshape((-1, n_compr))
        X = X @ compress_eigvecs
        y = -forces[begin:end].reshape(-1)
        XTX += X.T @ X
        XTy += X.T @ y
        t02 = time.time()
        if verbose:
            print(" solver_block:", end, ":, t =", t02 - t01, flush=True)
    t3 = time.time()
    if verbose:
        print(
            " (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):",
            t3 - t2,
            flush=True,
        )
    return XTX, XTy


def _NN33_to_N3N3(row, N):
    """Reorder row indices in a sparse matrix (NN33->N3N3)."""
    i, rem = np.divmod(row, 9 * N)
    j, rem = np.divmod(rem, 9)
    a, b = np.divmod(rem, 3)

    vec = i * 9 * N
    vec += j * 3
    vec += a * 3 * N + b
    return vec


def _csr_NN33_to_N3N3(mat, N):
    """Reorder row indices in a sparse matrix (NN33->N3N3).

    Return reordered csr_array.
    """
    NN33, nx = mat.shape
    mat = mat.tocoo()
    row = _NN33_to_N3N3(mat.row, N)
    mat = csr_array((mat.data, (row, mat.col)), shape=(NN33, nx))
    return mat


def reshape_nN33_nx_to_N3_n3nx(mat, N, n, n_batch=1):
    """Reorder and reshape a sparse matrix (nN33,nx)->(N3,n3nx).

    Return reordered csr_matrix used for FC2.
    """
    _, nx = mat.shape
    N3 = N * 3
    n3nx = n * 3 * nx
    mat = mat.tocoo(copy=False)

    begin_batch, end_batch = get_batch_slice(len(mat.row), len(mat.row) // n_batch)
    for begin, end in zip(begin_batch, end_batch):
        div, rem = np.divmod(mat.row[begin:end], 9 * N)
        mat.col[begin:end] += div * 3 * nx
        div, rem = np.divmod(rem, 9)
        mat.row[begin:end] = div * 3
        div, rem = np.divmod(rem, 3)
        mat.col[begin:end] += div * nx
        mat.row[begin:end] += rem

    mat.resize((N3, n3nx))
    mat = mat.tocsr(copy=False)
    return mat


def prepare_normal_equation_O2(
    disps,
    forces,
    compact_compress_mat_fc2,
    compress_eigvecs_fc2,
    trans_perms,
    atomic_decompr_idx_fc2=None,
    batch_size=100,
    use_mkl=False,
):
    r"""Calculate X.T @ X and X.T @ y.

    X = displacements @ compress_mat @ compress_eigvecs

    displacements (fc2): (n_samples, N3)
    compact_compress_mat_fc2: (n_aN33, n_compr)
    compress_eigvecs_fc2: (n_compr_fc2, n_basis_fc2)
    Matrix reshapings are appropriately applied to compress_mat
    and its products.

    X.T @ X and X.T @ y are sequentially calculated using divided dataset.
    X.T @ X = \sum_i X_i.T @ X_i
    X.T @ y = \sum_i X_i.T @ y_i (i: batch index)
    """
    N3 = disps.shape[1]
    N = N3 // 3
    n_compr_fc2 = compact_compress_mat_fc2.shape[1]

    n_lp, _ = trans_perms.shape
    if atomic_decompr_idx_fc2 is None:
        atomic_decompr_idx_fc2 = _get_atomic_lat_trans_decompr_indices(trans_perms)

    n_batch = 1
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    mat22 = np.zeros((n_compr_fc2, n_compr_fc2), dtype=float)
    mat2y = np.zeros(n_compr_fc2, dtype=float)

    t_all1 = time.time()
    const_fc2 = -1.0 / np.sqrt(n_lp)
    compact_compress_mat_fc2 *= const_fc2
    for begin_i, end_i in zip(begin_batch_atom, end_batch_atom):
        print("Solver_atoms:", begin_i + 1, "--", end_i, "/", N, flush=True)
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
        t2 = time.time()
        print("Solver_compr_matrix_reshape:, t =", "{:.3f}".format(t2 - t1), flush=True)

        for begin, end in zip(begin_batch, end_batch):
            t1 = time.time()
            X2 = dot_product_sparse(
                disps[begin:end],
                compr_mat_fc2,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc2))
            y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)
            mat22 += X2.T @ X2
            mat2y += X2.T @ y
            t2 = time.time()
            print("Solver_block:", end, ":, t =", "{:.3f}".format(t2 - t1), flush=True)

    XTX = compress_eigvecs_fc2.T @ mat22 @ compress_eigvecs_fc2
    XTy = compress_eigvecs_fc2.T @ mat2y

    compact_compress_mat_fc2 /= const_fc2
    t_all2 = time.time()
    print(
        " (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):",
        "{:.3f}".format(t_all2 - t_all1),
        flush=True,
    )
    return XTX, XTy


def run_solver_O2_update(
    disps,
    forces,
    compact_compress_mat_fc2,
    compress_eigvecs_fc2,
    trans_perms,
    atomic_decompr_idx_fc2=None,
    batch_size=100,
    use_mkl=False,
):
    """Estimate coeffs. in X @ coeffs = y.

    X = displacements_fc2 @ compress_mat_fc2 @ compress_eigvecs_fc2

    Matrix reshapings are appropriately applied.
    X: features (n_samples * N3, N_basis_fc2)
    y: observations (forces), (n_samples * N3)

    """
    XTX, XTy = prepare_normal_equation_O2(
        disps,
        forces,
        compact_compress_mat_fc2,
        compress_eigvecs_fc2,
        trans_perms,
        atomic_decompr_idx_fc2=atomic_decompr_idx_fc2,
        batch_size=batch_size,
        use_mkl=use_mkl,
    )
    coefs = solve_linear_equation(XTX, XTy)
    return coefs
