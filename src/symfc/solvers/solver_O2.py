"""2nd order force constants solver."""

from __future__ import annotations

import time
from typing import Literal, Optional

import numpy as np

from symfc.basis_sets import FCBasisSetO2
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import get_batch_slice, solve_linear_equation

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

    def solve(
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
        batch_size: int = 100,
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

        fc2_basis: FCBasisSetO2 = self._basis_set
        compress_mat_fc2 = fc2_basis.compact_compression_matrix
        basis_set_fc2 = fc2_basis.basis_set

        atomic_decompr_idx_fc2 = fc2_basis.atomic_decompr_idx

        self._coefs = run_solver_O2(
            d,
            f,
            compress_mat_fc2,
            basis_set_fc2,
            atomic_decompr_idx_fc2,
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
        np.ndarray
            shape=(N, N, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("full")

    @property
    def compact_fc(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return full force constants.

        Returns
        -------
        np.ndarray
            shape=(n_a, N, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("compact")

    def _recover_fcs(
        self, comp_mat_type: str = Literal["full", "compact"]
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self._coefs is None:
            return None

        fc2_basis: FCBasisSetO2 = self._basis_set
        if comp_mat_type == "full":
            comp_mat_fc2 = fc2_basis.compression_matrix
        elif comp_mat_type == "compact":
            comp_mat_fc2 = fc2_basis.compact_compression_matrix
        else:
            raise ValueError("Invalid comp_mat_type.")

        N = self._natom
        fc2 = fc2_basis.basis_set @ self._coefs
        fc2 = np.array(
            (comp_mat_fc2 @ fc2).reshape((-1, N, 3, 3)), dtype="double", order="C"
        )
        return fc2


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
    atomic_decompr_idx_fc2,
    batch_size=100,
    use_mkl=False,
    verbose=False,
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

    n_batch = 1
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    mat22 = np.zeros((n_compr_fc2, n_compr_fc2), dtype=float)
    mat2y = np.zeros(n_compr_fc2, dtype=float)

    t_all1 = time.time()
    const_fc2 = -1.0
    compact_compress_mat_fc2 *= const_fc2
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
            y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)
            mat22 += X2.T @ X2
            mat2y += X2.T @ y
            t2 = time.time()
            if verbose:
                print("Solver_block:", end, "/", disps.shape[0], flush=True)
                print(" - Time:", "{:.3f}".format(t2 - t1), flush=True)

    if verbose:
        print("Solver:", "Calculate X.T @ X and X.T @ y", flush=True)
    XTX = compress_eigvecs_fc2.T @ mat22 @ compress_eigvecs_fc2
    XTy = compress_eigvecs_fc2.T @ mat2y

    compact_compress_mat_fc2 /= const_fc2
    t_all2 = time.time()
    if verbose:
        print(
            " (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):",
            "{:.3f}".format(t_all2 - t_all1),
            flush=True,
        )
    return XTX, XTy


def run_solver_O2(
    disps,
    forces,
    compact_compress_mat_fc2,
    compress_eigvecs_fc2,
    atomic_decompr_idx_fc2,
    batch_size=100,
    use_mkl=False,
    verbose=False,
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
        atomic_decompr_idx_fc2,
        batch_size=batch_size,
        use_mkl=use_mkl,
        verbose=verbose,
    )
    coefs = solve_linear_equation(XTX, XTy)
    return coefs
