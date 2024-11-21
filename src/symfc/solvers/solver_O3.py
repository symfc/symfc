"""Solver of 3rd order force constants."""

from __future__ import annotations

import time
from typing import Literal, Optional

import numpy as np

from symfc.basis_sets import FCBasisSetO3
from symfc.solvers.solver_O2O3 import reshape_nNN333_nx_to_N3N3_n3nx, set_disps_N3N3
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import get_batch_slice, solve_linear_equation

from .solver_base import FCSolverBase


class FCSolverO3(FCSolverBase):
    """Third order force constants solver."""

    def __init__(
        self,
        basis_set: FCBasisSetO3,
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
    ) -> FCSolverO3:
        """Solve force constants.

        Note
        ----
        self._coefs = (coefs_fc3)

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
        self : FCSolverO3

        """
        n_data = forces.shape[0]
        f = forces.reshape(n_data, -1)
        d = displacements.reshape(n_data, -1)

        fc3_basis: FCBasisSetO3 = self._basis_set
        compress_mat_fc3 = fc3_basis.compact_compression_matrix
        basis_set_fc3 = fc3_basis.basis_set

        atomic_decompr_idx_fc3 = fc3_basis.atomic_decompr_idx

        self._coefs = run_solver_O3(
            d,
            f,
            compress_mat_fc3,
            basis_set_fc3,
            atomic_decompr_idx_fc3,
            batch_size=batch_size,
            use_mkl=self._use_mkl,
            verbose=self._log_level > 0,
        )
        return self

    @property
    def full_fc(self) -> Optional[np.ndarray]:
        """Return full force constants.

        Returns
        -------
        np.ndarray
            shape=(N, N, N, 3, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("full")

    @property
    def compact_fc(self) -> Optional[np.ndarray]:
        """Return full force constants.

        Returns
        -------
        np.ndarray
            shape=(n_a, N, N, 3, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("compact")

    def _recover_fcs(
        self, comp_mat_type: str = Literal["full", "compact"]
    ) -> Optional[np.ndarray]:
        if self._coefs is None:
            return None

        fc3_basis: FCBasisSetO3 = self._basis_set
        if comp_mat_type == "full":
            comp_mat_fc3 = fc3_basis.compression_matrix
        elif comp_mat_type == "compact":
            comp_mat_fc3 = fc3_basis.compact_compression_matrix
        else:
            raise ValueError("Invalid comp_mat_type.")

        N = self._natom
        fc3 = fc3_basis.basis_set @ self._coefs
        fc3 = np.array(
            (comp_mat_fc3 @ fc3).reshape((-1, N, N, 3, 3, 3)),
            dtype="double",
            order="C",
        )
        return fc3


def prepare_normal_equation_O3(
    disps,
    forces,
    compact_compress_mat_fc3,
    compress_eigvecs_fc3,
    atomic_decompr_idx_fc3,
    batch_size=100,
    use_mkl=False,
    verbose=False,
):
    r"""Calculate X.T @ X and X.T @ y.

    X = (displacements x displacements) @ compress_mat @ compress_eigvecs

    displacements (fc3): (n_samples, NN33)
    compact_compress_mat_fc3: (n_aNN333, n_compr_fc3)
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
    n_compr_fc3 = compact_compress_mat_fc3.shape[1]

    n_batch = (N // 256 + 1) * (n_compr_fc3 // 30000 + 1)
    n_batch = min(N, n_batch)
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    mat33 = np.zeros((n_compr_fc3, n_compr_fc3), dtype=float)
    mat3y = np.zeros(n_compr_fc3, dtype=float)

    t_all1 = time.time()
    const_fc3 = -0.5
    compact_compress_mat_fc3 *= const_fc3
    for begin_i, end_i in zip(begin_batch_atom, end_batch_atom):
        if verbose:
            print("-----", flush=True)
            print("Solver_atoms:", begin_i + 1, "--", end_i, "/", N, flush=True)
        n_atom_batch = end_i - begin_i

        t1 = time.time()
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
            X3 = dot_product_sparse(
                set_disps_N3N3(disps[begin:end], sparse=False),
                compr_mat_fc3,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc3))
            y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)
            mat33 += X3.T @ X3
            mat3y += X3.T @ y
            t2 = time.time()
            if verbose:
                print("Solver_block:", end, "/", disps.shape[0], flush=True)
                print(" - Time:", "{:.3f}".format(t2 - t1), flush=True)

    if verbose:
        print("Solver:", "Calculate X.T @ X and X.T @ y", flush=True)
    XTX = compress_eigvecs_fc3.T @ mat33 @ compress_eigvecs_fc3
    XTy = compress_eigvecs_fc3.T @ mat3y

    compact_compress_mat_fc3 /= const_fc3
    t_all2 = time.time()
    if verbose:
        print(
            "Time (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):",
            "{:.3f}".format(t_all2 - t_all1),
            flush=True,
        )
    return XTX, XTy


def run_solver_O3(
    disps,
    forces,
    compact_compress_mat_fc3,
    compress_eigvecs_fc3,
    atomic_decompr_idx_fc3,
    batch_size=100,
    use_mkl=False,
    verbose=False,
):
    """Estimate coeffs. in X @ coeffs = y.

    X = displacements_fc3 @ compress_mat_fc3 @ compress_eigvecs_fc3

    Matrix reshapings are appropriately applied.
    X: features (n_samples * N3, N_basis_fc3)
    y: observations (forces), (n_samples * N3)

    """
    XTX, XTy = prepare_normal_equation_O3(
        disps,
        forces,
        compact_compress_mat_fc3,
        compress_eigvecs_fc3,
        atomic_decompr_idx_fc3,
        batch_size=batch_size,
        use_mkl=use_mkl,
        verbose=verbose,
    )
    coefs = solve_linear_equation(XTX, XTy)
    return coefs
