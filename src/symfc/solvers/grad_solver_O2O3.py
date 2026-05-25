"""Solver of 2nd and 3rd order force constants simultaneously."""

from __future__ import annotations

import time
import copy
from collections.abc import Sequence
from typing import Union, cast

import numpy as np

from symfc.basis_sets import FCBasisSetO2, FCBasisSetO3
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.solver_utils_O2 import (
    slice_compact_compress_mat_O2,
    calc_predictions_O2, 
    calc_gradients_O2,
)
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
    beta: float = 0.95,
    gtol_fc2: float = 1e-5,
    gtol_fc3: float = 1e-8,
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
    beta2 = beta ** 2 / (beta ** 2 + (1 - beta) **2)
    eps_grad = min(gtol_fc2, gtol_fc3)

    average_force = np.average(np.linalg.norm(forces.reshape((-1, 3)), axis=1))
    gtol_fc2 *= (average_force / 0.2) ** 2 
    gtol_fc3 *= (average_force / 0.2) ** 3

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

    grad_prev, magn_prev = np.zeros(n_compr), np.zeros(n_compr)
    converge = False
    for i_epoch in range(n_epochs):
        t1 = time.time()

        if verbose:
            print("-----", flush=True)
            print("Epoch:", i_epoch + 1, flush=True)

        rate = np.zeros(n_compr)
        rate2 = max(min(3 / np.sqrt(i_epoch + 1), 1), 1e-5)
        rate3 = max(min(30 / np.sqrt(i_epoch + 1), 10), 1e-3)
        rate[:n_compr_fc2] = rate2
        rate[n_compr_fc2:] = rate3
        if verbose:
            print("- Learning rate (FC2):", "{:.5f}".format(rate2), flush=True)
            print("- Learning rate (FC3):", "{:.5f}".format(rate3), flush=True)

        error_all = []
        order_atom = np.arange(len(begin_batch_atom))
        np.random.shuffle(order_atom)
        for i_atom in order_atom:
            begin_i, end_i = begin_batch_atom[i_atom], end_batch_atom[i_atom]
            decompr_idx_fc2, compr_mat_fc2 = slice_compact_compress_mat_O2(
                compact_compress_mat_fc2, atomic_decompr_idx_fc2, N, begin_i, end_i
            )
            decompr_idx_fc3, compr_mat_fc3 = slice_compact_compress_mat_O3(
                compact_compress_mat_fc3, atomic_decompr_idx_fc3, N, begin_i, end_i
            )

            order_supercell = np.arange(len(begin_batch))
            np.random.shuffle(order_supercell)
            for i_supercell in order_supercell:
                begin, end = begin_batch[i_supercell], end_batch[i_supercell]
                y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)

                # Calculate pred = [X2, X3] @ coefs.
                pred2 = calc_predictions_O2(
                    compact_compress_mat_fc2,
                    decompr_idx_fc2,
                    N,
                    coefs[:n_compr_fc2],
                    disps[begin:end],
                )
 
                dispN3N3 = set_disps_N3N3(disps[begin:end], sparse=False)
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
                grad2 = calc_gradients_O2(
                    compr_mat_fc2,
                    N,
                    error,
                    disps[begin:end],
                )
                grad3 = calc_gradients_O3(
                    compr_mat_fc3,
                    N,
                    error,
                    dispN3N3,
                )

                grad_trial = np.concatenate([grad2, grad3])
                magn = beta2 * magn_prev + (1 - beta2) * (grad_trial**2)
                grad = beta * grad_prev + (1 - beta) * grad_trial

                agrad2 = np.max(np.abs(grad[:n_compr_fc2])) 
                agrad3 = np.max(np.abs(grad[n_compr_fc2:]))
                if agrad2 < gtol_fc2 and agrad3 < gtol_fc3:
                    converge = True
                    break

                magn_sqrt = np.sqrt(magn)
                magn_sqrt[magn_sqrt < eps_grad] = np.inf
                coefs -= rate * grad / magn_sqrt

                grad_prev, magn_prev = grad, magn

        t2 = time.time()
        if verbose:
            error_all = np.array(error_all)
            rmse_forces = np.sqrt(np.mean(error_all**2))
            print("- Time:              ", "{:.3f}".format(t2 - t1), "s", flush=True)
            print("- RMSE (Force):      ", "{:.5e}".format(rmse_forces), flush=True)
            agrad2 = np.max(np.abs(grad[:n_compr_fc2])) 
            print("- Max gradient (FC2):", "{:.5e}".format(agrad2), flush=True)
            agrad3 = np.max(np.abs(grad[n_compr_fc2:]))
            print("- Max gradient (FC3):", "{:.5e}".format(agrad3), flush=True)

        if converge:
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
