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

from .solver_base import FCSolverBase


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
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
        batch_size: int = 100,
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
        compress_mat_fc2 = fc2_basis.compact_compression_matrix
        basis_set_fc2 = fc2_basis.basis_set
        compress_mat_fc3 = fc3_basis.compact_compression_matrix
        basis_set_fc3 = fc3_basis.basis_set

        atomic_decompr_idx_fc2 = fc2_basis.atomic_decompr_idx
        atomic_decompr_idx_fc3 = fc3_basis.atomic_decompr_idx

        self._coefs = run_solver_O2O3(
            d,
            f,
            compress_mat_fc2,
            compress_mat_fc3,
            basis_set_fc2,
            basis_set_fc3,
            atomic_decompr_idx_fc2,
            atomic_decompr_idx_fc3,
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


def set_disps_N3N3(disps, sparse=False):
    """Calculate Kronecker products of displacements.

    Parameter
    ---------
    disps: shape=(n_supercell, N3)

    Return
    ------
    disps_2nd: shape=(n_supercell, N3N3)
    """
    n_supercell = disps.shape[0]
    disps_2nd = (disps[:, :, None] * disps[:, None, :]).reshape((n_supercell, -1))

    if sparse:
        return csr_array(disps_2nd)
    return disps_2nd


def reshape_nNN333_nx_to_N3N3_n3nx(mat, N, n, n_batch=9):
    """Reorder and reshape a sparse matrix (nNN333,nx)->(N3N3,n3nx).

    Return reordered csr_matrix used for FC3.
    """
    _, nx = mat.shape
    NN33 = N**2 * 9
    n3nx = n * 3 * nx
    mat = mat.tocoo(copy=False)

    batch_size = len(mat.row) if len(mat.row) < n_batch else len(mat.row) // n_batch

    begin_batch, end_batch = get_batch_slice(len(mat.row), batch_size)
    for begin, end in zip(begin_batch, end_batch):
        div, rem = np.divmod(mat.row[begin:end], 27 * N * N)
        mat.col[begin:end] += div * 3 * nx
        div, rem = np.divmod(rem, 27 * N)
        mat.row[begin:end] = div * 9 * N
        div, rem = np.divmod(rem, 27)
        mat.row[begin:end] += div * 3
        div, rem = np.divmod(rem, 9)
        mat.col[begin:end] += div * nx
        div, rem = np.divmod(rem, 3)
        mat.row[begin:end] += div * 3 * N + rem

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
    atomic_decompr_idx_fc2,
    atomic_decompr_idx_fc3,
    batch_size=100,
    use_mkl=False,
    verbose=False,
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

    n_compr_fc2 = compact_compress_mat_fc2.shape[1]
    n_compr_fc3 = compact_compress_mat_fc3.shape[1]

    n_batch = (N // 256 + 1) * (n_compr_fc3 // 30000 + 1)
    n_batch = min(N, n_batch)
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    mat22 = np.zeros((n_compr_fc2, n_compr_fc2), dtype=float)
    mat23 = np.zeros((n_compr_fc2, n_compr_fc3), dtype=float)
    mat33 = np.zeros((n_compr_fc3, n_compr_fc3), dtype=float)
    mat2y = np.zeros(n_compr_fc2, dtype=float)
    mat3y = np.zeros(n_compr_fc3, dtype=float)

    t_all1 = time.time()
    const_fc2 = -1.0
    const_fc3 = -0.5
    compact_compress_mat_fc2 *= const_fc2
    compact_compress_mat_fc3 *= const_fc3
    for begin_i, end_i in zip(begin_batch_atom, end_batch_atom):
        if verbose:
            print("-----", flush=True)
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

        decompr_idx = (
            atomic_decompr_idx_fc3[begin_i * NN : end_i * NN, None] * 27
            + np.arange(27)[None, :]
        ).reshape(-1)
        compr_mat_fc3 = reshape_nNN333_nx_to_N3N3_n3nx(
            compact_compress_mat_fc3[decompr_idx],
            N,
            n_atom_batch,
        )
        t2 = time.time()
        if verbose:
            print(
                "Time (Solver_compr_matrix_reshape):",
                "{:.3f}".format(t2 - t1),
                flush=True,
            )

        for begin, end in zip(begin_batch, end_batch):
            t1 = time.time()
            X2 = dot_product_sparse(
                disps[begin:end],
                compr_mat_fc2,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc2))
            X3 = dot_product_sparse(
                set_disps_N3N3(disps[begin:end], sparse=False),
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
            if verbose:
                print("Solver_block:", end, "/", disps.shape[0], flush=True)
                print(" - Time:", "{:.3f}".format(t2 - t1), flush=True)

    if verbose:
        print("Solver:", "Calculate X.T @ X and X.T @ y", flush=True)

    mat23 = compress_eigvecs_fc2.T @ mat23 @ compress_eigvecs_fc3
    XTX = np.block(
        [
            [compress_eigvecs_fc2.T @ mat22 @ compress_eigvecs_fc2, mat23],
            [mat23.T, compress_eigvecs_fc3.T @ mat33 @ compress_eigvecs_fc3],
        ]
    )
    XTy = np.hstack([compress_eigvecs_fc2.T @ mat2y, compress_eigvecs_fc3.T @ mat3y])

    compact_compress_mat_fc2 /= const_fc2
    compact_compress_mat_fc3 /= const_fc3
    t_all2 = time.time()
    if verbose:
        print(
            "Time (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):",
            "{:.3f}".format(
                t_all2 - t_all1,
            ),
        )
    return XTX, XTy


def run_solver_O2O3(
    disps,
    forces,
    compact_compress_mat_fc2,
    compact_compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    atomic_decompr_idx_fc2,
    atomic_decompr_idx_fc3,
    batch_size=100,
    use_mkl=False,
    verbose=False,
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
        atomic_decompr_idx_fc2,
        atomic_decompr_idx_fc3,
        batch_size=batch_size,
        use_mkl=use_mkl,
        verbose=verbose,
    )
    coefs = solve_linear_equation(XTX, XTy)
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    coefs_fc2, coefs_fc3 = coefs[:n_basis_fc2], coefs[n_basis_fc2:]

    return coefs_fc2, coefs_fc3
