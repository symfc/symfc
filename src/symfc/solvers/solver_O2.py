"""2nd order force constants solver."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from symfc.basis_sets import FCBasisSetO2
from symfc.eig_solvers.matrix import block_matrix_sandwich_sym
from symfc.utils.solver_funcs import get_batch_slice, solve_linear_equation
from symfc.utils.solver_utils_O2 import reshape_compr_mat_O2

try:
    from symfc.utils.matrix import dot_product_sparse
except ImportError:
    pass

from numpy.typing import NDArray

from .solver_base import FCSolverBase


class FCSolverO2(FCSolverBase):
    """Second order force constants solver."""

    def __init__(
        self,
        basis_set: FCBasisSetO2,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._basis_set: FCBasisSetO2
        super().__init__(basis_set, use_mkl=use_mkl, log_level=log_level)

    def solve(
        self,
        displacements: NDArray,
        forces: NDArray,
        batch_size: int = 100,
        use_sparse_disps: bool = False,
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

        fc2_basis = self._basis_set
        XTX, XTy = prepare_normal_equation_O2(
            d,
            f,
            fc2_basis,
            batch_size=batch_size,
            use_sparse_disps=False,
            use_mkl=self._use_mkl,
            verbose=self._log_level > 0,
        )
        self._coefs = solve_linear_equation(XTX, XTy)
        return self

    @property
    def full_fc(self) -> NDArray | None:
        """Return full force constants.

        Returns
        -------
        np.ndarray
            shape=(N, N, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("full")

    @property
    def compact_fc(self) -> NDArray | None:
        """Return full force constants.

        Returns
        -------
        np.ndarray
            shape=(n_a, N, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("compact")

    def _recover_fcs(self, comp_mat_type: Literal["full", "compact"]) -> NDArray | None:
        fc2_basis = self._basis_set

        if self._coefs is None or fc2_basis.basis_set is None:
            return None

        if comp_mat_type == "full":
            comp_mat_fc2 = fc2_basis.compression_matrix
        elif comp_mat_type == "compact":
            comp_mat_fc2 = fc2_basis.compact_compression_matrix
        else:
            raise ValueError("Invalid comp_mat_type.")

        N = self._natom
        fc2 = fc2_basis.blocked_basis_set @ self._coefs
        fc2 = np.array(
            (comp_mat_fc2 @ fc2).reshape((-1, N, 3, 3)), dtype="double", order="C"
        )
        return fc2


def prepare_normal_equation_O2(
    disps: NDArray,
    forces: NDArray,
    fc2_basis: FCBasisSetO2,
    batch_size: int = 100,
    use_sparse_disps: bool = False,
    use_mkl: bool = False,
    verbose: bool = False,
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

    compact_compress_mat_fc2 = fc2_basis.compact_compression_matrix
    atomic_decompr_idx_fc2 = fc2_basis.atomic_decompr_idx
    n_compr_fc2 = compact_compress_mat_fc2.shape[1]  # type: ignore

    n_batch = max(N // 10, 1)
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    mat22 = np.zeros((n_compr_fc2, n_compr_fc2), dtype=float)
    mat2y = np.zeros(n_compr_fc2, dtype=float)

    t_all1 = time.time()
    const_fc2 = -1.0
    compact_compress_mat_fc2 *= const_fc2
    for begin_i, end_i in zip(begin_batch_atom, end_batch_atom, strict=True):
        if verbose:
            print("-----", flush=True)
            print("Solver_atoms:", begin_i + 1, "--", end_i, "/", N, flush=True)

        t1 = time.time()
        compr_mat_fc2 = reshape_compr_mat_O2(
            compact_compress_mat_fc2, atomic_decompr_idx_fc2, N, begin_i, end_i
        )
        t2 = time.time()
        if verbose:
            time_pr = "{:.3f}".format(t2 - t1)
            print("Time (Solver_compr_matrix_reshape):", time_pr, flush=True)

        for begin, end in zip(begin_batch, end_batch, strict=True):
            if verbose:
                print("Solver_block:", end, "/", disps.shape[0], flush=True)
            t1 = time.time()
            X2 = dot_product_sparse(
                disps[begin:end],
                compr_mat_fc2,
                use_mkl=use_mkl,
                dense=not use_sparse_disps,
            ).reshape((-1, n_compr_fc2))
            y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)
            mat22 += X2.T @ X2
            mat2y += X2.T @ y
            t2 = time.time()
            if verbose:
                print(" - Time:", "{:.3f}".format(t2 - t1), flush=True)
            del X2

    if verbose:
        print("Solver:", "Calculate X.T @ X and X.T @ y", flush=True)

    compress_eigvecs_fc2 = fc2_basis.blocked_basis_set
    XTX = block_matrix_sandwich_sym(compress_eigvecs_fc2, mat22)
    XTy = compress_eigvecs_fc2.T @ mat2y

    compact_compress_mat_fc2 /= const_fc2
    t_all2 = time.time()
    if verbose:
        header = " (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):"
        print(header, "{:.3f}".format(t_all2 - t_all1), flush=True)
    return XTX, XTy
