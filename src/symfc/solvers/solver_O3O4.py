"""Solver of 3rd and 4th order force constants simultaneously."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np

from symfc.basis_sets import FCBasisSetO3, FCBasisSetO4
from symfc.solvers.solver_O2O3 import reshape_nNN333_nx_to_N3N3_n3nx, set_disps_N3N3
from symfc.solvers.solver_O2O3O4 import (
    reshape_nNNN3333_nx_to_N3N3N3_n3nx,
    set_disps_N3N3N3,
)
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import get_batch_slice, solve_linear_equation

from .solver_base import FCSolverBase


class FCSolverO3O4(FCSolverBase):
    """Simultaneous third and fourth order force constants solver."""

    def __init__(
        self,
        basis_set: Sequence[FCBasisSetO3, FCBasisSetO4],
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        super().__init__(basis_set, use_mkl=use_mkl, log_level=log_level)

    def solve(
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
        batch_size: int = 36,
    ) -> FCSolverO3O4:
        """Solve force constants.

        Note
        ----
        self._coefs = (coefs_fc3, coefs_fc4)

        Parameters
        ----------
        displacements : ndarray
            Displacements of atoms in Cartesian coordinates.
            shape=(n_snapshot, N, 3), dtype='double'
        forces : ndarray
            Forces of atoms in Cartesian coordinates. shape=(n_snapshot, N, 3),
            dtype='double'

        Returns
        -------
        self : FCSolverO3O4

        """
        n_data = forces.shape[0]
        f = forces.reshape(n_data, -1)
        d = displacements.reshape(n_data, -1)

        fc3_basis: FCBasisSetO3 = self._basis_set[0]
        fc4_basis: FCBasisSetO4 = self._basis_set[1]
        compress_mat_fc3 = fc3_basis.compact_compression_matrix
        basis_set_fc3 = fc3_basis.basis_set
        compress_mat_fc4 = fc4_basis.compact_compression_matrix
        basis_set_fc4 = fc4_basis.basis_set

        atomic_decompr_idx_fc3 = fc3_basis.atomic_decompr_idx
        atomic_decompr_idx_fc4 = fc4_basis.atomic_decompr_idx

        self._coefs = run_solver_O3O4(
            d,
            f,
            compress_mat_fc3,
            compress_mat_fc4,
            basis_set_fc3,
            basis_set_fc4,
            atomic_decompr_idx_fc3,
            atomic_decompr_idx_fc4,
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
            shape=(N, N, N, 3, 3, 3), dtype='double', order='C'
            shape=(N, N, N, N, 3, 3, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("full")

    @property
    def compact_fc(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return full force constants.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            shape=(n_a, N, N, 3, 3, 3), dtype='double', order='C'
            shape=(n_a, N, N, N, 3, 3, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("compact")

    def _recover_fcs(
        self, comp_mat_type: str = Literal["full", "compact"]
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self._coefs is None:
            return None

        fc3_basis: FCBasisSetO3 = self._basis_set[0]
        fc4_basis: FCBasisSetO4 = self._basis_set[1]
        if comp_mat_type == "full":
            comp_mat_fc3 = fc3_basis.compression_matrix
            comp_mat_fc4 = fc4_basis.compression_matrix
        elif comp_mat_type == "compact":
            comp_mat_fc3 = fc3_basis.compact_compression_matrix
            comp_mat_fc4 = fc4_basis.compact_compression_matrix
        else:
            raise ValueError("Invalid comp_mat_type.")

        N = self._natom
        fc3 = fc3_basis.basis_set @ self._coefs[0]
        fc3 = np.array(
            (comp_mat_fc3 @ fc3).reshape((-1, N, N, 3, 3, 3)),
            dtype="double",
            order="C",
        )
        fc4 = fc4_basis.basis_set @ self._coefs[1]
        fc4 = np.array(
            (comp_mat_fc4 @ fc4).reshape((-1, N, N, N, 3, 3, 3, 3)),
            dtype="double",
            order="C",
        )

        return fc3, fc4


def prepare_normal_equation_O3O4(
    disps,
    forces,
    compact_compress_mat_fc3,
    compact_compress_mat_fc4,
    compress_eigvecs_fc3,
    compress_eigvecs_fc4,
    atomic_decompr_idx_fc3,
    atomic_decompr_idx_fc4,
    batch_size=36,
    use_mkl=False,
    verbose=False,
):
    r"""Calculate X.T @ X and X.T @ y.

    X = displacements @ compress_mat @ compress_eigvecs
    X = np.hstack([X_fc3, X_fc4])

    displacements (fc3): (n_samples, NN33)
    displacements (fc4): (n_samples, NNN333)
    compact_compress_mat_fc3: (n_aNN333, n_compr_fc3)
    compact_compress_mat_fc4: (n_aNNN3333, n_compr_fc4)
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

    n_compr_fc3 = compact_compress_mat_fc3.shape[1]
    n_compr_fc4 = compact_compress_mat_fc4.shape[1]

    n_batch = (n_compr_fc3 // 10000 + n_compr_fc4 // 5000 + 1) * (N // 50 + 1)
    n_batch = min(N, n_batch)
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    mat33 = np.zeros((n_compr_fc3, n_compr_fc3), dtype=float)
    mat34 = np.zeros((n_compr_fc3, n_compr_fc4), dtype=float)
    mat44 = np.zeros((n_compr_fc4, n_compr_fc4), dtype=float)
    mat3y = np.zeros(n_compr_fc3, dtype=float)
    mat4y = np.zeros(n_compr_fc4, dtype=float)

    t_all1 = time.time()
    const_fc3 = -0.5
    const_fc4 = -1.0 / 6.0
    compact_compress_mat_fc3 *= const_fc3
    compact_compress_mat_fc4 *= const_fc4
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
            print(
                "Time (Solver_compr_matrix_reshape):",
                "{:.3f}".format(t2 - t1),
                flush=True,
            )

        for begin, end in zip(begin_batch, end_batch):
            t1 = time.time()
            disps_N3N3 = set_disps_N3N3(disps[begin:end], sparse=False)
            X3 = dot_product_sparse(
                disps_N3N3,
                compr_mat_fc3,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc3))
            X4 = dot_product_sparse(
                set_disps_N3N3N3(disps[begin:end], sparse=False, disps_N3N3=disps_N3N3),
                compr_mat_fc4,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc4))

            y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)
            mat33 += X3.T @ X3
            mat34 += X3.T @ X4
            mat44 += X4.T @ X4
            mat3y += X3.T @ y
            mat4y += X4.T @ y
            t2 = time.time()
            if verbose:
                print("Solver_block:", end, "/", disps.shape[0], flush=True)
                print(" - Time:", "{:.3f}".format(t2 - t1), flush=True)

    if verbose:
        print("Solver:", "Calculate X.T @ X and X.T @ y", flush=True)

    mat33 = compress_eigvecs_fc3.T @ mat33 @ compress_eigvecs_fc3
    mat34 = compress_eigvecs_fc3.T @ mat34 @ compress_eigvecs_fc4
    mat44 = compress_eigvecs_fc4.T @ mat44 @ compress_eigvecs_fc4
    mat3y = compress_eigvecs_fc3.T @ mat3y
    mat4y = compress_eigvecs_fc4.T @ mat4y

    XTX = np.block([[mat33, mat34], [mat34.T, mat44]])
    XTy = np.hstack([mat3y, mat4y])

    compact_compress_mat_fc3 /= const_fc3
    compact_compress_mat_fc4 /= const_fc4
    t_all2 = time.time()
    if verbose:
        print(
            "Time (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):",
            "{:.3f}".format(t_all2 - t_all1),
            flush=True,
        )
    return XTX, XTy


def run_solver_O3O4(
    disps,
    forces,
    compact_compress_mat_fc3,
    compact_compress_mat_fc4,
    compress_eigvecs_fc3,
    compress_eigvecs_fc4,
    atomic_decompr_idx_fc3,
    atomic_decompr_idx_fc4,
    batch_size=36,
    use_mkl=False,
    verbose=False,
):
    """Estimate coeffs. in X @ coeffs = y.

    X_fc3 = displacements_fc3 @ compress_mat_fc3 @ compress_eigvecs_fc3
    X_fc4 = displacements_fc4 @ compress_mat_fc4 @ compress_eigvecs_fc4
    X = np.hstack([X_fc3, X_fc4])

    Matrix reshapings are appropriately applied.
    X: features (n_samples * N3, N_basis_fc3 + N_basis_fc4)
    y: observations (forces), (n_samples * N3)

    """
    XTX, XTy = prepare_normal_equation_O3O4(
        disps,
        forces,
        compact_compress_mat_fc3,
        compact_compress_mat_fc4,
        compress_eigvecs_fc3,
        compress_eigvecs_fc4,
        atomic_decompr_idx_fc3,
        atomic_decompr_idx_fc4,
        batch_size=batch_size,
        use_mkl=use_mkl,
        verbose=verbose,
    )
    coefs = solve_linear_equation(XTX, XTy)
    n_basis_fc3 = compress_eigvecs_fc3.shape[1]
    coefs_fc3, coefs_fc4 = coefs[:n_basis_fc3], coefs[n_basis_fc3:]

    return coefs_fc3, coefs_fc4
