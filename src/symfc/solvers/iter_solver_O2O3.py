"""Solver of 2nd and 3rd order force constants simultaneously."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Union, cast

import numpy as np

from symfc.basis_sets import FCBasisSetO2, FCBasisSetO3
from symfc.eig_solvers.matrix import (
    BlockMatrixNode,
    link_block_matrix_nodes,
    root_block_matrix,
)
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.solver_utils_O2 import reshape_compr_mat_O2
from symfc.utils.solver_utils_O3 import (
    dot_O3,
    reshape_compr_mat_O3,
    reshape_vec_O3,
    set_disps_N3N3,
)

try:
    from symfc.utils.matrix import dot_product_sparse
except ImportError:
    pass

from .solver_O2O3 import FCSolverO2O3


class FCIterSolverO2O3(FCSolverO2O3):
    """Simultaneous second and third order force constants solver."""

    def __init__(
        self,
        basis_set: Sequence[Union[FCBasisSetO2, FCBasisSetO3]],
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        basis_set : Sequence of (FCBasisSetO2, FCBasisSetO3)
            First element must be FCBasisSetO2 and second must be FCBasisSetO3.
        use_mkl : bool, optional
            Use MKL if True. Default is False.
        log_level : int, optional
            Logging level. Default is 0.

        """
        super().__init__(basis_set, use_mkl=use_mkl, log_level=log_level)

    def solve(
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
        batch_size: int = 100,
    ) -> FCIterSolverO2O3:
        """Solve force constants using a gradient solver.

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

        fc2_basis: FCBasisSetO2 = cast(FCBasisSetO2, self._basis_set[0])
        fc3_basis: FCBasisSetO3 = cast(FCBasisSetO3, self._basis_set[1])

        coefs = solve_adam_O2O3(
            d,
            f,
            fc2_basis,
            fc3_basis,
            batch_size=batch_size,
            use_mkl=self._use_mkl,
            verbose=self._log_level > 0,
        )
        n_basis_fc2 = fc2_basis.blocked_basis_set.shape[1]
        self._coefs = coefs[:n_basis_fc2], coefs[n_basis_fc2:]

        return self


def _get_linked_compress_eigvecs(
    compress_eigvecs_fc2: BlockMatrixNode,
    compress_eigvecs_fc3: BlockMatrixNode,
):
    """Return linked compressed eigenvectors."""
    shape2 = compress_eigvecs_fc2.shape
    shape3 = compress_eigvecs_fc3.shape

    _ = link_block_matrix_nodes(
        compress_eigvecs_fc3,
        compress_eigvecs_fc2,
        rows=np.arange(shape2[0], shape2[0] + shape3[0]),
        col_begin=shape2[1],
    )
    shape = (shape2[0] + shape3[0], shape2[1] + shape3[1])
    compress_eigvecs = root_block_matrix(shape=shape, first_child=compress_eigvecs_fc3)
    return compress_eigvecs


def solve_sgd_O2O3(
    disps: np.ndarray,
    forces: np.ndarray,
    fc2_basis: FCBasisSetO2,
    fc3_basis: FCBasisSetO3,
    batch_size: int = 100,
    n_epoch: int = 1000,
    use_mkl: bool = False,
    verbose: bool = False,
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

    compact_compress_mat_fc2 = fc2_basis.compact_compression_matrix
    compact_compress_mat_fc3 = fc3_basis.compact_compression_matrix
    atomic_decompr_idx_fc2 = fc2_basis.atomic_decompr_idx
    atomic_decompr_idx_fc3 = fc3_basis.atomic_decompr_idx

    if compact_compress_mat_fc2 is None or compact_compress_mat_fc3 is None:
        raise ValueError(
            "Compression matrices or basis sets are not set. "
            "Call run() method to compute them."
        )

    n_compr_fc2 = compact_compress_mat_fc2.shape[1]  # type: ignore
    n_compr_fc3 = compact_compress_mat_fc3.shape[1]  # type: ignore

    n_batch = (N // 100 + 1) * (n_compr_fc3 // 20000 + 1)
    n_batch = min(N, n_batch)
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    n_compr = n_compr_fc2 + n_compr_fc3

    t_all1 = time.time()
    const_fc2 = -1.0
    const_fc3 = -0.5
    compact_compress_mat_fc2 *= const_fc2
    compact_compress_mat_fc3 *= const_fc3

    coefs = np.ones(n_compr)
    learning_rate = 100

    rmse_prev = 1e10
    for i_epoch in range(n_epoch):
        if verbose:
            print("-----", flush=True)
            print("Epoch:", i_epoch + 1, flush=True)

        error_all = []
        for begin_i, end_i in zip(begin_batch_atom, end_batch_atom, strict=True):
            if verbose:
                print("-----", flush=True)
                print("Solver_atoms:", begin_i + 1, "--", end_i, "/", N, flush=True)
            n_atom_batch = end_i - begin_i

            t1 = time.time()
            compr_mat_fc2 = reshape_compr_mat_O2(
                compact_compress_mat_fc2, atomic_decompr_idx_fc2, N, begin_i, end_i
            )
            compr_mat_fc3 = reshape_compr_mat_O3(
                compact_compress_mat_fc3, atomic_decompr_idx_fc3, N, begin_i, end_i
            )
            t2 = time.time()
            if verbose:
                time_pr = "{:.3f}".format(t2 - t1)
                print("Time (Solver_compr_matrix_reshape):", time_pr, flush=True)

            # TODO: Use structure index permutation.
            t1 = time.time()
            for begin, end in zip(begin_batch, end_batch, strict=True):
                if verbose:
                    print("Solver_block:", end, "/", disps.shape[0], flush=True)
                X = np.zeros((n_atom_batch * 3 * (end - begin), n_compr))
                ta = time.time()
                X[:, :n_compr_fc2] = dot_product_sparse(
                    disps[begin:end],
                    compr_mat_fc2,
                    use_mkl=use_mkl,
                    dense=True,
                ).reshape((-1, n_compr_fc2))
                X[:, n_compr_fc2:] = dot_product_sparse(
                    set_disps_N3N3(disps[begin:end], sparse=False),
                    compr_mat_fc3,
                    use_mkl=use_mkl,
                    dense=True,
                ).reshape((-1, n_compr_fc3))
                tb = time.time()
                y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)

                error = X @ coefs - y
                grad = X.T @ error
                coefs -= learning_rate * grad
                error_all.extend(error)
                tc = time.time()
                print(tb - ta, tc - tb)

            t2 = time.time()
            if verbose:
                print(" - Time:", "{:.3f}".format(t2 - t1), flush=True)

        error_all = np.array(error_all)
        rmse = np.sqrt(np.mean(error_all**2))
        if verbose:
            print("RMSE:", rmse, flush=True)

        if np.abs(rmse - rmse_prev) < 1e-8:
            break
        rmse_prev = rmse

    compress_eigvecs = _get_linked_compress_eigvecs(
        fc2_basis.blocked_basis_set,
        fc3_basis.blocked_basis_set,
    )
    coefs = compress_eigvecs.T @ coefs

    fc2_basis.blocked_basis_set.reset_indices()
    fc3_basis.blocked_basis_set.reset_indices()
    compact_compress_mat_fc2 /= const_fc2
    compact_compress_mat_fc3 /= const_fc3
    t_all2 = time.time()
    if verbose:
        header = "Time (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):"
        print(header, "{:.3f}".format(t_all2 - t_all1), flush=True)
    return coefs


def solve_adam_batch_atom_O2O3(
    disps: np.ndarray,
    forces: np.ndarray,
    fc2_basis: FCBasisSetO2,
    fc3_basis: FCBasisSetO3,
    batch_size: int = 100,
    n_epoch: int = 1000,
    beta1: float = 0.9,
    beta2: float = 0.999,
    tol_rmse: float = 1e-10,
    use_mkl: bool = False,
    verbose: bool = False,
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

    compact_compress_mat_fc2 = fc2_basis.compact_compression_matrix
    compact_compress_mat_fc3 = fc3_basis.compact_compression_matrix
    atomic_decompr_idx_fc2 = fc2_basis.atomic_decompr_idx
    atomic_decompr_idx_fc3 = fc3_basis.atomic_decompr_idx

    if compact_compress_mat_fc2 is None or compact_compress_mat_fc3 is None:
        raise ValueError(
            "Compression matrices or basis sets are not set. "
            "Call run() method to compute them."
        )

    n_compr_fc2 = compact_compress_mat_fc2.shape[1]  # type: ignore
    n_compr_fc3 = compact_compress_mat_fc3.shape[1]  # type: ignore

    n_batch = (N // 128 + 1) * (n_compr_fc3 // 20000 + 1)
    n_batch = min(N, n_batch)
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    n_compr = n_compr_fc2 + n_compr_fc3

    t_all1 = time.time()
    const_fc2 = -1.0
    const_fc3 = -0.5
    compact_compress_mat_fc2 *= const_fc2
    compact_compress_mat_fc3 *= const_fc3

    coefs = np.ones(n_compr)
    learning_rate = 100

    directions_prev = None
    magnitudes_prev = None
    rmse_prev = 1e10
    for i_epoch in range(n_epoch):
        if verbose:
            print("-----", flush=True)
            print("Epoch:", i_epoch + 1, flush=True)

        error_all = []
        for begin_i, end_i in zip(begin_batch_atom, end_batch_atom, strict=True):
            if verbose:
                print("-----", flush=True)
                print("Solver_atoms:", begin_i + 1, "--", end_i, "/", N, flush=True)
            n_atom_batch = end_i - begin_i

            t1 = time.time()
            compr_mat_fc2 = reshape_compr_mat_O2(
                compact_compress_mat_fc2, atomic_decompr_idx_fc2, N, begin_i, end_i
            )
            compr_mat_fc3 = reshape_compr_mat_O3(
                compact_compress_mat_fc3, atomic_decompr_idx_fc3, N, begin_i, end_i
            )
            t2 = time.time()
            if verbose:
                time_pr = "{:.3f}".format(t2 - t1)
                print("Time (Solver_compr_matrix_reshape):", time_pr, flush=True)

            # TODO: Use structure index permutation.
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
                X[:, n_compr_fc2:] = dot_product_sparse(
                    set_disps_N3N3(disps[begin:end], sparse=False),
                    compr_mat_fc3,
                    use_mkl=use_mkl,
                    dense=True,
                ).reshape((-1, n_compr_fc3))
                y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)

                error = X @ coefs - y
                error_all.extend(error)

                grad = X.T @ error
                if directions_prev is None:
                    directions = grad
                    magnitudes = grad**2
                else:
                    directions = beta1 * directions_prev + (1 - beta1) * grad
                    magnitudes = beta2 * magnitudes_prev + (1 - beta2) * (grad**2)
                normalized_directions = directions / np.sqrt(magnitudes)
                coefs -= learning_rate * normalized_directions

                directions_prev = directions
                magnitudes_prev = magnitudes

            t2 = time.time()
            if verbose:
                print(" - Time:", "{:.3f}".format(t2 - t1), flush=True)

        error_all = np.array(error_all)
        rmse = np.sqrt(np.mean(error_all**2))
        if verbose:
            print("RMSE:", rmse, flush=True)

        if np.abs(rmse - rmse_prev) < tol_rmse:
            break
        rmse_prev = rmse

    compress_eigvecs = _get_linked_compress_eigvecs(
        fc2_basis.blocked_basis_set,
        fc3_basis.blocked_basis_set,
    )
    coefs = compress_eigvecs.T @ coefs

    fc2_basis.blocked_basis_set.reset_indices()
    fc3_basis.blocked_basis_set.reset_indices()
    compact_compress_mat_fc2 /= const_fc2
    compact_compress_mat_fc3 /= const_fc3
    t_all2 = time.time()
    if verbose:
        header = "Time (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):"
        print(header, "{:.3f}".format(t_all2 - t_all1), flush=True)
    return coefs


def solve_adam_O2O3(
    disps: np.ndarray,
    forces: np.ndarray,
    fc2_basis: FCBasisSetO2,
    fc3_basis: FCBasisSetO3,
    batch_size: int = 100,
    n_epoch: int = 1000,
    beta1: float = 0.9,
    beta2: float = 0.999,
    tol_rmse: float = 1e-10,
    use_mkl: bool = False,
    verbose: bool = False,
):
    r"""Solve normal equations using Adam.

    X = displacements @ compress_mat @ compress_eigvecs
    X = np.hstack([X_fc2, X_fc3])

    displacements (fc2): (n_samples, N3)
    displacements (fc3): (n_samples, NN33)
    compact_compress_mat_fc2: (n_aN33, n_compr_fc2)
    compact_compress_mat_fc3: (n_aNN333, n_compr_fc3)
    compress_eigvecs_fc2: (n_compr_fc2, n_basis_fc2)
    compress_eigvecs_fc3: (n_compr_fc3, n_basis_fc3)
    Matrix reshapings are appropriately applied to compress_mat
    and its products.
    """
    N3 = disps.shape[1]
    N = N3 // 3

    compact_compress_mat_fc2 = fc2_basis.compact_compression_matrix
    compact_compress_mat_fc3 = fc3_basis.compact_compression_matrix
    atomic_decompr_idx_fc2 = fc2_basis.atomic_decompr_idx
    atomic_decompr_idx_fc3 = fc3_basis.atomic_decompr_idx

    if compact_compress_mat_fc2 is None or compact_compress_mat_fc3 is None:
        raise ValueError(
            "Compression matrices or basis sets are not set. "
            "Call run() method to compute them."
        )

    n_compr_fc2 = compact_compress_mat_fc2.shape[1]  # type: ignore
    n_compr_fc3 = compact_compress_mat_fc3.shape[1]  # type: ignore

    # n_batch = (N // 128 + 1) * (n_compr_fc3 // 20000 + 1)
    # n_batch = min(N, n_batch)
    # begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    n_compr = n_compr_fc2 + n_compr_fc3

    t_all1 = time.time()
    const_fc2 = -1.0
    const_fc3 = -0.5
    compact_compress_mat_fc2 *= const_fc2
    compact_compress_mat_fc3 *= const_fc3

    coefs = np.ones(n_compr)
    learning_rate = 1000

    directions_prev = None
    magnitudes_prev = None

    t1 = time.time()
    begin_i, end_i = 0, N
    compr_mat_fc2 = reshape_compr_mat_O2(
        compact_compress_mat_fc2, atomic_decompr_idx_fc2, N, begin_i, end_i
    )
    # compr_mat_fc3 = reshape_compr_mat_O3(
    #     compact_compress_mat_fc3, atomic_decompr_idx_fc3, N, begin_i, end_i
    # )
    n_atom_batch = end_i - begin_i
    t2 = time.time()
    if verbose:
        time_pr = "{:.3f}".format(t2 - t1)
        print("Time (Solver_compr_matrix_reshape):", time_pr, flush=True)

    rmse_prev = 1e10
    for i_epoch in range(n_epoch):
        if verbose:
            print("-----", flush=True)
            print("Epoch:", i_epoch + 1, flush=True)

        error_all = []
        t1 = time.time()
        for begin, end in zip(begin_batch, end_batch, strict=True):
            if verbose:
                print("Solver_block:", end, "/", disps.shape[0], flush=True)

            dispN3N3 = set_disps_N3N3(disps[begin:end], sparse=False)
            y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)

            X2 = dot_product_sparse(
                disps[begin:end],
                compr_mat_fc2,
                use_mkl=use_mkl,
                dense=True,
            ).reshape((-1, n_compr_fc2))
            pred2 = X2 @ coefs[:n_compr_fc2]

            vec1 = (compact_compress_mat_fc3 @ coefs[n_compr_fc2:]).reshape(-1, 1)
            mat1 = reshape_vec_O3(vec1, atomic_decompr_idx_fc3, N, begin_i, end_i)
            pred3 = dispN3N3 @ mat1
            pred3 = pred3.reshape(-1)

            error = pred2 + pred3 - y
            error_all.extend(error)

            grad2 = X2.T @ error
            mat1 = dispN3N3.T @ error.reshape((-1, n_atom_batch * 3))
            grad3 = dot_O3(
                compact_compress_mat_fc3,
                atomic_decompr_idx_fc3,
                mat1,
                N,
                begin_i,
                end_i,
            )
            grad = np.concatenate([grad2, grad3])

            if directions_prev is not None:
                directions = beta1 * directions_prev + (1 - beta1) * grad
                magnitudes = beta2 * magnitudes_prev + (1 - beta2) * (grad**2)
            else:
                directions = grad
                magnitudes = grad**2

            normalized_directions = directions / np.sqrt(magnitudes)
            coefs -= learning_rate * normalized_directions

            directions_prev = directions
            magnitudes_prev = magnitudes

        t2 = time.time()
        if verbose:
            print(" - Time:", "{:.3f}".format(t2 - t1), flush=True)

        error_all = np.array(error_all)
        rmse = np.sqrt(np.mean(error_all**2))
        if verbose:
            print("RMSE:", rmse, flush=True)

        if np.abs(rmse) < 1e-5:
            break
        if np.abs(rmse - rmse_prev) < tol_rmse:
            break

        rmse_prev = rmse

    compress_eigvecs = _get_linked_compress_eigvecs(
        fc2_basis.blocked_basis_set,
        fc3_basis.blocked_basis_set,
    )
    coefs = compress_eigvecs.T @ coefs

    fc2_basis.blocked_basis_set.reset_indices()
    fc3_basis.blocked_basis_set.reset_indices()
    compact_compress_mat_fc2 /= const_fc2
    compact_compress_mat_fc3 /= const_fc3
    t_all2 = time.time()
    if verbose:
        header = "Time (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):"
        print(header, "{:.3f}".format(t_all2 - t_all1), flush=True)
    return coefs
