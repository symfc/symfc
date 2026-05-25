"""Solver of 2nd and 3rd order force constants simultaneously."""

from __future__ import annotations

import time
import copy
from collections.abc import Sequence
from typing import Union, cast

import numpy as np

from symfc.basis_sets import FCBasisSetO2, FCBasisSetO3
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.solver_utils_O2 import reshape_compr_mat_O2
from symfc.utils.solver_utils_O3 import (
    calc_gradients_O3,
    calc_predictions_O3,
    set_disps_N3N3,
    slice_compact_compress_mat_O3,
)

try:
    from symfc.utils.matrix import dot_product_sparse
except ImportError:
    pass

from .solver_O2O3 import FCSolverO2O3, _get_linked_compress_eigvecs


class FCGradSolverO2O3(FCSolverO2O3):
    """Simultaneous second and third order force constants solver using gradients."""

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
    ) -> FCGradSolverO2O3:
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


def solve_adam_O2O3(
    disps: np.ndarray,
    forces: np.ndarray,
    fc2_basis: FCBasisSetO2,
    fc3_basis: FCBasisSetO3,
    batch_size: int = 100,
    n_epochs: int = 10000,
    beta1: float = 0.95,
    beta2: float = 0.999,
    tol_rmse: float = 1e-10,
    gtol_fc2: float = 1e-5,
    gtol_fc3: float = 1e-8,
    eps_grad: float = 1e-15,
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

    n_batch = (N // 10 + 1) * (n_compr_fc3 // 20000 + 1)
    n_batch = min(N, n_batch)
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    n_compr = n_compr_fc2 + n_compr_fc3
    coefs = np.zeros(n_compr)

    t_all1 = time.time()
    const_fc2 = -1.0
    const_fc3 = -0.5
    compact_compress_mat_fc2 *= const_fc2
    compact_compress_mat_fc3 *= const_fc3

    grad_prev, magn_prev, coefs_prev = None, None, None
    rmse = np.inf
    for i_epoch in range(n_epochs):
        learning_rate = min(rmse * 1e4 / np.sqrt(i_epoch + 1), 1)
        learning_rate = np.round(learning_rate, 5)

        if verbose:
            print("-----", flush=True)
            print("Epoch:", i_epoch + 1, flush=True)
            print("- Learning rate:", learning_rate, flush=True)

        t1 = time.time()

        error_all = []
        converge = False

        order_atom = np.arange(len(begin_batch_atom))
        np.random.shuffle(order_atom)
        for i_atom in order_atom:
            begin_i = begin_batch_atom[i_atom]
            end_i = end_batch_atom[i_atom]
            if verbose:
                print("-----", flush=True)
                print("Solver_atoms:", begin_i + 1, "--", end_i, "/", N, flush=True)

            compr_mat_fc2 = reshape_compr_mat_O2(
                compact_compress_mat_fc2, atomic_decompr_idx_fc2, N, begin_i, end_i
            )
            decompr_idx_fc3, compr_mat_fc3 = slice_compact_compress_mat_O3(
                compact_compress_mat_fc3, atomic_decompr_idx_fc3, N, begin_i, end_i
            )

            order_supercell = np.arange(len(begin_batch))
            np.random.shuffle(order_supercell)
            for i_supercell in order_supercell:
                begin = begin_batch[i_supercell]
                end = end_batch[i_supercell]
                # if verbose:
                #     print("Solver_block:", end, "/", disps.shape[0], flush=True)

                dispN3N3 = set_disps_N3N3(disps[begin:end], sparse=False)
                y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)

                # Calculate pred = [X2, X3] @ coefs.
                X2 = dot_product_sparse(
                    disps[begin:end],
                    compr_mat_fc2,
                    use_mkl=use_mkl,
                    dense=True,
                ).reshape((-1, n_compr_fc2))
                pred2 = X2 @ coefs[:n_compr_fc2]
                pred3 = calc_predictions_O3(
                    compact_compress_mat_fc3,
                    decompr_idx_fc3,
                    N,
                    coefs[n_compr_fc2:],
                    dispN3N3,
                )
                error = pred2 + pred3 - y
                error_all.extend(error)

                # Calculate grad = [X2, X3].T @ error.
                grad2 = X2.T @ error
                grad3 = calc_gradients_O3(
                    compr_mat_fc3,
                    N,
                    error,
                    dispN3N3,
                )
                grad = np.concatenate([grad2, grad3])

                if grad_prev is not None:
                    magn = beta2 * magn_prev + (1 - beta2) * (grad**2)
                    grad = beta1 * grad_prev + (1 - beta1) * grad
                else:
                    magn = grad**2

                agrad2 = np.max(np.abs(grad2)) 
                agrad3 = np.max(np.abs(grad3))
                if verbose:
                    print(" - Max gradient (FC2):", "{:.5e}".format(agrad2), flush=True)
                    print(" - Max gradient (FC3):", "{:.5e}".format(agrad3), flush=True)

                if agrad2 < gtol_fc2 and agrad3 < gtol_fc3:
                    converge = True
                    break

                # if agrad2 > gtol_fc2:
                #     converge = False
                # if agrad3 > gtol_fc3:
                #     converge = False

                # TODO: Tune epsilon
                magn_sqrt = np.sqrt(magn)
                magn_sqrt[magn_sqrt < eps_grad] = np.inf
                coefs -= learning_rate * (grad / magn_sqrt)
                # coefs -= learning_rate * (grad / (np.sqrt(magn) + eps_grad))
                # coefs -= learning_rate * (grad / (np.sqrt(magn)))

                grad_prev, magn_prev = grad, magn

        t2 = time.time()

        if verbose:
            error_all = np.array(error_all)
            rmse_forces = np.sqrt(np.mean(error_all**2))
            # agrad2 = np.max(np.abs(grad[:n_compr_fc2])) 
            # agrad3 = np.max(np.abs(grad[n_compr_fc2:]))
            print(" - Time:        ", "{:.3f}".format(t2 - t1), flush=True)
            print(" - RMSE (Force):", "{:.5e}".format(rmse_forces), flush=True)
            # print(" - Max gradient (FC2):", "{:.5e}".format(agrad2), flush=True)
            # print(" - Max gradient (FC3):", "{:.5e}".format(agrad3), flush=True)

        if converge:
            if verbose:
                print("Gradient:", np.max(np.abs(grad)), flush=True)
            break

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
