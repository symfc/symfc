"""2nd order force constants solver."""

import time
from typing import Optional

import numpy as np
from scipy.sparse import csc_array, csr_array

from symfc.basis_sets import FCBasisSetO2Base
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.utils_O2 import get_lat_trans_decompr_indices

from .solver_base import FCSolverBase
from .solver_funcs import fit, get_batch_slice, solve_linear_equation


class FCSolverO2(FCSolverBase):
    """Second order force constants solver."""

    def __init__(
        self,
        fc_basis_set: FCBasisSetO2Base,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        super().__init__(
            fc_basis_set,
            use_mkl=use_mkl,
            log_level=log_level,
        )

    def solve(
        self, displacements: np.ndarray, forces: np.ndarray, is_compact_fc=True
    ) -> Optional[np.ndarray]:
        """Solve force constants.

        Parameters
        ----------
        displacements : ndarray
            Displacements of atoms in Cartesian coordinates. shape=(n_snapshot,
            N, 3), dtype='double'
        forces : ndarray
            Forces of atoms in Cartesian coordinates. shape=(n_snapshot, N, 3),
            dtype='double'
        is_compact_fc : bool
            Shape of force constants array is (n_a, N, 3, 3) if True or (N, N,
            3, 3) if False.

        Returns
        -------
        ndarray
            Force constants.
            shape=(n_a, N, 3, 3) or (N, N, 3, 3). See `is_compact_fc` parameter.
            dtype='double', order='C'

        """
        N = self._natom
        if self._fc_basis_set.basis_set is None:
            return None
        assert displacements.shape == forces.shape
        compr_fc = self._fc_basis_set.basis_set @ self._get_basis_coeff(
            displacements, forces
        )
        if is_compact_fc:
            n_lp = self._fc_basis_set.translation_permutations.shape[0]
            n_a = N // n_lp
            if isinstance(self._fc_basis_set.compact_compression_matrix, int):
                return compr_fc.reshape(n_a, N, 3, 3)
            else:
                fc = dot_product_sparse(
                    self._fc_basis_set.compact_compression_matrix,
                    csr_array(compr_fc.reshape(-1, 1)),
                    use_mkl=self._use_mkl,
                )
                return fc.toarray().reshape(n_a, N, 3, 3)
        else:
            if isinstance(self._fc_basis_set.compact_compression_matrix, int):
                return compr_fc[self._fc_basis_set.decompression_indices].reshape(
                    N, N, 3, 3
                )
            else:
                fc = dot_product_sparse(
                    self._fc_basis_set.compression_matrix,
                    csr_array(compr_fc.reshape(-1, 1)),
                    use_mkl=self._use_mkl,
                )
                return fc.toarray().reshape(N, N, 3, 3)

    def _get_basis_coeff(
        self, displacements: np.ndarray, forces: np.ndarray
    ) -> Optional[np.ndarray]:
        N = self._natom
        if self._fc_basis_set.basis_set is None:
            return None
        n_snapshot = displacements.shape[0]
        disps = displacements.reshape(n_snapshot, N * 3)
        d_basis = np.zeros(
            (n_snapshot * N * 3, self._fc_basis_set.basis_set.shape[1]),
            dtype="double",
            order="C",
        )

        if isinstance(self._fc_basis_set.compact_compression_matrix, int):
            self._compute_with_trans_perms(d_basis, disps)
        else:
            self._compute_with_compression_matrix(d_basis, disps)
        if self._log_level:
            print("Computing product of displacements and basis set...")
        if self._log_level:
            print("Solving basis-set coefficients...")
        coeff = -(np.linalg.pinv(d_basis) @ forces.ravel())
        return coeff

    def _compute_with_compression_matrix(self, d_basis: np.ndarray, disps: np.ndarray):
        N = self._natom
        full_basis_set = dot_product_sparse(
            self._fc_basis_set.compression_matrix,
            csc_array(self._fc_basis_set.basis_set),
            use_mkl=self._use_mkl,
        )
        for i_basis_set in range(full_basis_set.shape[1]):
            vec = np.transpose(
                full_basis_set[:, [i_basis_set]].toarray().reshape(N, N, 3, 3),
                (0, 2, 1, 3),
            ).reshape(N * 3, N * 3)
            d_basis[:, i_basis_set] = (disps @ vec).ravel()

    def _compute_with_trans_perms(self, d_basis: np.ndarray, disps: np.ndarray):
        N = self._natom
        trans_perms = self._fc_basis_set.translation_permutations
        decompr_idx = get_lat_trans_decompr_indices(trans_perms)
        decompr_array = np.transpose(
            decompr_idx.reshape(N, N, 3, 3), (0, 2, 1, 3)
        ).reshape(N * 3, N * 3)
        for i, vec in enumerate(self._fc_basis_set.basis_set.T):
            d_basis[:, i] = (disps @ vec[decompr_array]).ravel()


def get_training_from_full_basis(disps, forces, full_basis):
    """Return training dataset (X, y).

    The dataset is transformed from displacements, forces, and full basis-set.

    disps: (n_samples, N3)
    forces: (n_samples, N3)
    full_basis: (n_basis, N, N, 3, 3)

    X: features (calculated from displacements),(n_samples * N3, N_basis)
    y: observations (forces), (n_samples * N3)

    Note
    ----
    #algorithm 1
    X = []
    for disp_st, force_st in zip(disps, forces):
        disp_reshape = disp_st.reshape(-1)
        X_tmp = []
        for basis in full_basis:
            Bb = basis.transpose((0,2,1,3)).reshape(N3,N3)
            X_tmp.append(-np.dot(Bb, disp_reshape))
        X_tmp = np.array(X_tmp).T
        X.append(X_tmp)
    X = np.vstack(X)

    #algorithm 2
    disps_reshape = disps.reshape((n_samples, N3))
    X = np.zeros((N3*n_samples, n_basis))
    for i, basis in enumerate(full_basis):
        Bb = basis.transpose((0,2,1,3)).reshape(N3,N3)
        prod = - np.dot(disps_reshape, Bb)
        X[:,i] = np.ravel(prod)

    """
    N3 = full_basis.shape[1] * 3
    n_basis = full_basis.shape[0]
    n_samples = disps.shape[0]

    disps_reshape = disps.reshape((n_samples, N3))
    full_basis = full_basis.transpose((1, 3, 2, 4, 0)).reshape((N3, -1))
    X = -np.dot(disps_reshape, full_basis)
    X = X.reshape((-1, n_basis))
    y = forces.reshape(-1)

    return X, y


def run_solver_dense_O2(disps, forces, compress_mat, compress_eigvecs):
    """Estimating coeffs. in X @ coeffs = y using least squares.

    X: features (calculated from displacements), (n_samples * N3, N_basis)
    y: observations (forces), (n_samples * N3)

    """
    N3 = disps.shape[1]
    N = N3 // 3

    full_basis = compress_mat @ compress_eigvecs
    n_basis = full_basis.shape[1]
    coefs = _solve(disps, forces, full_basis.T.reshape((n_basis, N, N, 3, 3)))
    return coefs


def run_solver_sparse_O2(disps, forces, compress_mat, compress_eigvecs, batch_size=200):
    """Estimating coeffs. in X @ coeffs = y by solving normal equation.

    (X.T @ X) @ coeffs = X.T @ y

    X = displacements @ compress_mat @ compress_eigvecs
    Matrix reshapings are appropriately applied.
    X: features (n_samples * N3, N_basis)
    y: observations (forces), (n_samples * N3)

    """
    XTX, XTy = _get_training(
        disps, forces, compress_mat, compress_eigvecs, batch_size=batch_size
    )
    coefs = solve_linear_equation(XTX, XTy)
    return coefs


def _get_training(
    disps, forces, compress_mat, compress_eigvecs, batch_size=200, use_mkl=False
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
    print(" reshape(compr):   ", t2 - t1)

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
        print(" solver_block:", end, ":, t =", t02 - t01)
    t3 = time.time()
    print(" (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):", t3 - t2)
    return XTX, XTy


def _solve(disps, forces, full_basis):
    """Solve force constants."""
    X, y = get_training_from_full_basis(disps, forces, full_basis)
    coefs = fit(X, y)
    return coefs


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
