"""Solver of 2nd, 3rd and 4th order force constants simultaneously."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal, Optional, Union, cast

import numpy as np
from scipy.sparse import csr_array

from symfc.basis_sets import FCBasisSetO2, FCBasisSetO3, FCBasisSetO4
from symfc.eig_solvers.matrix import (
    BlockMatrixNode,
    block_matrix_sandwich_sym,
    link_block_matrix_nodes,
    root_block_matrix,
)
from symfc.solvers.solver_O2 import reshape_nN33_nx_to_N3_n3nx
from symfc.solvers.solver_O2O3 import reshape_nNN333_nx_to_N3N3_n3nx, set_disps_N3N3
from symfc.utils.solver_funcs import get_batch_slice, solve_linear_equation
from symfc.utils.solver_utils import calc_sum_xtx

try:
    from symfc.utils.matrix import dot_product_sparse
except ImportError:
    pass

from .solver_base import FCSolverBase


class FCSolverO2O3O4(FCSolverBase):
    """Simultaneous second, third and fourth order force constants solver."""

    def __init__(
        self,
        basis_set: Sequence[Union[FCBasisSetO2, FCBasisSetO3, FCBasisSetO4]],
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        basis_set : Sequence of (FCBasisSetO2, FCBasisSetO3, FCBasisSetO4)
            First element must be FCBasisSetO2, second must be FCBasisSetO3,
            and third must be FCBasisSetO4.
        use_mkl : bool, optional
            Use MKL if True. Default is False.
        log_level : int, optional
            Logging level. Default is 0.

        """
        if len(basis_set) != 3:
            raise ValueError("basis_set must contain exactly 3 elements")
        if not isinstance(basis_set[0], FCBasisSetO2):
            raise TypeError("First element must be FCBasisSetO2")
        if not isinstance(basis_set[1], FCBasisSetO3):
            raise TypeError("Second element must be FCBasisSetO3")
        if not isinstance(basis_set[2], FCBasisSetO4):
            raise TypeError("Third element must be FCBasisSetO4")
        self._basis_set: Sequence[Union[FCBasisSetO2, FCBasisSetO3, FCBasisSetO4]]
        super().__init__(basis_set, use_mkl=use_mkl, log_level=log_level)

    def solve(
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
        batch_size: int = 36,
    ) -> FCSolverO2O3O4:
        """Solve force constants.

        Note
        ----
        self._coefs = (coefs_fc2, coefs_fc3, coefs_fc4)

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
            Force constants. shape=(n_a, N, N, N, 3, 3, 3, 3). See
            `is_compact_fc` parameter. dtype='double', order='C'

        """
        n_data = forces.shape[0]
        f = forces.reshape(n_data, -1)
        d = displacements.reshape(n_data, -1)

        fc2_basis: FCBasisSetO2 = cast(FCBasisSetO2, self._basis_set[0])
        fc3_basis: FCBasisSetO3 = cast(FCBasisSetO3, self._basis_set[1])
        fc4_basis: FCBasisSetO4 = cast(FCBasisSetO4, self._basis_set[2])

        self._coefs = run_solver_O2O3O4(
            d,
            f,
            fc2_basis,
            fc3_basis,
            fc4_basis,
            batch_size=batch_size,
            use_mkl=self._use_mkl,
            verbose=self._log_level > 0,
        )
        return self

    @property
    def full_fc(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return full force constants.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            shape=(N, N, 3, 3), dtype='double', order='C'
            shape=(N, N, N, 3, 3, 3), dtype='double', order='C'
            shape=(N, N, N, N, 3, 3, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("full")

    @property
    def compact_fc(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return full force constants.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            shape=(n_a, N, 3, 3), dtype='double', order='C'
            shape=(n_a, N, N, 3, 3, 3), dtype='double', order='C'
            shape=(n_a, N, N, N, 3, 3, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("compact")

    def _recover_fcs(
        self, comp_mat_type: Literal["full", "compact"]
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._coefs is None:
            return None

        fc2_basis: FCBasisSetO2 = cast(FCBasisSetO2, self._basis_set[0])
        fc3_basis: FCBasisSetO3 = cast(FCBasisSetO3, self._basis_set[1])
        fc4_basis: FCBasisSetO4 = cast(FCBasisSetO4, self._basis_set[2])
        if comp_mat_type == "full":
            comp_mat_fc2 = fc2_basis.compression_matrix
            comp_mat_fc3 = fc3_basis.compression_matrix
            comp_mat_fc4 = fc4_basis.compression_matrix
        elif comp_mat_type == "compact":
            comp_mat_fc2 = fc2_basis.compact_compression_matrix
            comp_mat_fc3 = fc3_basis.compact_compression_matrix
            comp_mat_fc4 = fc4_basis.compact_compression_matrix
        else:
            raise ValueError("Invalid comp_mat_type.")

        N = self._natom
        fc2 = fc2_basis.blocked_basis_set @ self._coefs[0]
        fc2 = np.array(
            (comp_mat_fc2 @ fc2).reshape((-1, N, 3, 3)), dtype="double", order="C"
        )
        fc3 = fc3_basis.blocked_basis_set @ self._coefs[1]
        fc3 = np.array(
            (comp_mat_fc3 @ fc3).reshape((-1, N, N, 3, 3, 3)),
            dtype="double",
            order="C",
        )
        fc4 = fc4_basis.blocked_basis_set @ self._coefs[2]
        fc4 = np.array(
            (comp_mat_fc4 @ fc4).reshape((-1, N, N, N, 3, 3, 3, 3)),
            dtype="double",
            order="C",
        )

        return fc2, fc3, fc4


def set_disps_N3N3N3(disps, sparse=True, disps_N3N3=None):
    """Calculate Kronecker products of displacements.

    Parameter
    ---------
    disps: shape=(n_supercell, N3)

    Return
    ------
    disps_3rd: shape=(n_supercell, N3N3N3)
    """
    n_supercell = disps.shape[0]
    if disps_N3N3 is not None:
        disps_3rd = (disps_N3N3[:, :, None] * disps[:, None, :]).reshape(
            (n_supercell, -1)
        )
    else:
        disps_3rd = (
            disps[:, :, None, None] * disps[:, None, :, None] * disps[:, None, None, :]
        ).reshape((n_supercell, -1))

    if sparse:
        return csr_array(disps_3rd)
    return disps_3rd


def reshape_nNNN3333_nx_to_N3N3N3_n3nx(mat, N, n, n_batch=36):
    """Reorder and reshape a sparse matrix (nNNN3333,nx)->(N3N3N3,n3nx).

    Return reordered csr_matrix used for FC4.
    """
    _, nx = mat.shape
    NNN333 = N**3 * 27
    n3nx = n * 3 * nx
    mat = mat.tocoo(copy=False)

    begin_batch, end_batch = get_batch_slice(len(mat.row), len(mat.row) // n_batch)
    for begin, end in zip(begin_batch, end_batch, strict=True):
        div, rem = np.divmod(mat.row[begin:end], 81 * N * N * N)
        mat.col[begin:end] += div * 3 * nx
        div, rem = np.divmod(rem, 81 * N * N)
        mat.row[begin:end] = div * 27 * N * N
        div, rem = np.divmod(rem, 81 * N)
        mat.row[begin:end] += div * 9 * N
        div, rem = np.divmod(rem, 81)
        mat.row[begin:end] += div * 3
        div, rem = np.divmod(rem, 27)
        mat.col[begin:end] += div * nx
        div, rem = np.divmod(rem, 9)
        mat.row[begin:end] += div * 9 * N * N
        div, rem = np.divmod(rem, 3)
        mat.row[begin:end] += div * 3 * N + rem

    mat.resize((NNN333, n3nx))
    mat = mat.tocsr(copy=False)
    return mat


def _get_linked_compress_eigvecs(
    compress_eigvecs_fc2: BlockMatrixNode,
    compress_eigvecs_fc3: BlockMatrixNode,
    compress_eigvecs_fc4: BlockMatrixNode,
):
    """Return linked compressed eigenvectors."""
    shape2 = compress_eigvecs_fc2.shape
    shape3 = compress_eigvecs_fc3.shape
    shape4 = compress_eigvecs_fc4.shape

    _ = link_block_matrix_nodes(
        compress_eigvecs_fc3,
        compress_eigvecs_fc2,
        rows=np.arange(shape2[0], shape2[0] + shape3[0]),
        col_begin=shape2[1],
    )
    _ = link_block_matrix_nodes(
        compress_eigvecs_fc4,
        compress_eigvecs_fc3,
        rows=np.arange(shape2[0] + shape3[0], shape2[0] + shape3[0] + shape4[0]),
        col_begin=shape2[1] + shape3[1],
    )

    shape = (shape2[0] + shape3[0] + shape4[0], shape2[1] + shape3[1] + shape4[1])
    compress_eigvecs = root_block_matrix(shape=shape, first_child=compress_eigvecs_fc4)
    return compress_eigvecs


def prepare_normal_equation_O2O3O4(
    disps: np.ndarray,
    forces: np.ndarray,
    fc2_basis: FCBasisSetO2,
    fc3_basis: FCBasisSetO3,
    fc4_basis: FCBasisSetO4,
    batch_size: int = 36,
    use_mkl: bool = False,
    verbose: bool = False,
):
    r"""Calculate X.T @ X and X.T @ y.

    X = displacements @ compress_mat @ compress_eigvecs
    X = np.hstack([X_fc2, X_fc3, X_fc4])

    displacements (fc2): (n_samples, N3)
    displacements (fc3): (n_samples, NN33)
    displacements (fc4): (n_samples, NNN333)
    compact_compress_mat_fc2: (n_aN33, n_compr)
    compact_compress_mat_fc3: (n_aNN333, n_compr_fc3)
    compact_compress_mat_fc4: (n_aNNN3333, n_compr_fc4)
    compress_eigvecs_fc2: (n_compr_fc2, n_basis_fc2)
    compress_eigvecs_fc3: (n_compr_fc3, n_basis_fc3)
    compress_eigvecs_fc4: (n_compr_fc4, n_basis_fc4)
    Matrix reshapings are appropriately applied to compress_mat
    and its products.

    X.T @ X and X.T @ y are sequentially calculated using divided dataset.
    X.T @ X = \sum_i X_i.T @ X_i
    X.T @ y = \sum_i X_i.T @ y_i (i: batch index)
    """
    N3 = disps.shape[1]
    N = N3 // 3
    NN = N**2
    NNN = N**3

    compact_compress_mat_fc2 = fc2_basis.compact_compression_matrix
    compact_compress_mat_fc3 = fc3_basis.compact_compression_matrix
    compact_compress_mat_fc4 = fc4_basis.compact_compression_matrix
    atomic_decompr_idx_fc2 = fc2_basis.atomic_decompr_idx
    atomic_decompr_idx_fc3 = fc3_basis.atomic_decompr_idx
    atomic_decompr_idx_fc4 = fc4_basis.atomic_decompr_idx

    n_compr_fc2 = compact_compress_mat_fc2.shape[1]
    n_compr_fc3 = compact_compress_mat_fc3.shape[1]
    n_compr_fc4 = compact_compress_mat_fc4.shape[1]

    n_batch = (n_compr_fc3 // 10000 + n_compr_fc4 // 5000 + 1) * (N // 20 + 1)
    n_batch = min(N, n_batch)
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    n_compr = n_compr_fc2 + n_compr_fc3 + n_compr_fc4
    matx = np.zeros((n_compr, n_compr), dtype=float)
    maty = np.zeros(n_compr, dtype=float)

    t_all1 = time.time()
    const_fc2 = -1.0
    const_fc3 = -0.5
    const_fc4 = -1.0 / 6.0
    compact_compress_mat_fc2 *= const_fc2
    compact_compress_mat_fc3 *= const_fc3
    compact_compress_mat_fc4 *= const_fc4
    for begin_i, end_i in zip(begin_batch_atom, end_batch_atom, strict=True):
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

        decompr_idx = (
            atomic_decompr_idx_fc4[begin_i * NNN : end_i * NNN, None] * 81
            + np.arange(81)[None, :]
        ).reshape(-1)
        compr_mat_fc4 = reshape_nNNN3333_nx_to_N3N3N3_n3nx(
            compact_compress_mat_fc4[decompr_idx],
            N,
            n_atom_batch,
        )
        t2 = time.time()
        if verbose:
            time_pr = "{:.3f}".format(t2 - t1)
            print("Time (Solver_compr_matrix_reshape):", time_pr, flush=True)

        for begin, end in zip(begin_batch, end_batch, strict=True):
            if verbose:
                print("Solver_block:", end, "/", disps.shape[0], flush=True)
            t1 = time.time()
            X = np.zeros((n_atom_batch * 3 * (end - begin), n_compr))
            X[:, :n_compr_fc2] = dot_product_sparse(
                disps[begin:end],
                compr_mat_fc2,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc2))
            disps_N3N3 = set_disps_N3N3(disps[begin:end], sparse=False)
            X[:, n_compr_fc2 : n_compr_fc2 + n_compr_fc3] = dot_product_sparse(
                disps_N3N3,
                compr_mat_fc3,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc3))
            X[:, n_compr_fc2 + n_compr_fc3 :] = dot_product_sparse(
                set_disps_N3N3N3(disps[begin:end], sparse=False, disps_N3N3=disps_N3N3),
                compr_mat_fc4,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc4))
            y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)

            matx = calc_sum_xtx(matx, X, verbose=verbose)
            maty += X.T @ y
            t2 = time.time()
            if verbose:
                print(" - Time:", "{:.3f}".format(t2 - t1), flush=True)

    compress_eigvecs = _get_linked_compress_eigvecs(
        fc2_basis.blocked_basis_set,
        fc3_basis.blocked_basis_set,
        fc4_basis.blocked_basis_set,
    )

    if verbose:
        print("Solver:", "Calculate X.T @ X and X.T @ y", flush=True)
    XTX = block_matrix_sandwich_sym(compress_eigvecs, matx)
    XTy = compress_eigvecs.T @ maty

    fc2_basis.blocked_basis_set.reset_indices()
    fc3_basis.blocked_basis_set.reset_indices()
    fc4_basis.blocked_basis_set.reset_indices()

    compact_compress_mat_fc2 /= const_fc2
    compact_compress_mat_fc3 /= const_fc3
    compact_compress_mat_fc4 /= const_fc4
    t_all2 = time.time()
    if verbose:
        header = "Time (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):"
        print(header, "{:.3f}".format(t_all2 - t_all1), flush=True)
    return XTX, XTy


def run_solver_O2O3O4(
    disps: np.ndarray,
    forces: np.ndarray,
    fc2_basis: FCBasisSetO2,
    fc3_basis: FCBasisSetO3,
    fc4_basis: FCBasisSetO4,
    batch_size: int = 36,
    use_mkl: bool = False,
    verbose: bool = False,
):
    """Estimate coeffs. in X @ coeffs = y.

    X_fc2 = displacements_fc2 @ compress_mat_fc2 @ compress_eigvecs_fc2
    X_fc3 = displacements_fc3 @ compress_mat_fc3 @ compress_eigvecs_fc3
    X_fc4 = displacements_fc4 @ compress_mat_fc4 @ compress_eigvecs_fc4
    X = np.hstack([X_fc2, X_fc3, X_fc4])

    Matrix reshapings are appropriately applied.
    X: features (n_samples * N3, N_basis_fc2 + N_basis_fc3 + N_basis_fc4)
    y: observations (forces), (n_samples * N3)

    """
    XTX, XTy = prepare_normal_equation_O2O3O4(
        disps,
        forces,
        fc2_basis,
        fc3_basis,
        fc4_basis,
        batch_size=batch_size,
        use_mkl=use_mkl,
        verbose=verbose,
    )
    coefs = solve_linear_equation(XTX, XTy)
    n_basis_fc2 = fc2_basis.blocked_basis_set.shape[1]
    n_basis_fc3 = fc3_basis.blocked_basis_set.shape[1]
    coefs_fc2, coefs_fc3, coefs_fc4 = (
        coefs[:n_basis_fc2],
        coefs[n_basis_fc2 : n_basis_fc2 + n_basis_fc3],
        coefs[n_basis_fc2 + n_basis_fc3 :],
    )

    return coefs_fc2, coefs_fc3, coefs_fc4
