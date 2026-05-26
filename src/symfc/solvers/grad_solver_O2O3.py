"""Solver of 2nd and 3rd order force constants simultaneously."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Union, cast

import numpy as np

from symfc.basis_sets import FCBasisSetO2, FCBasisSetO3
from symfc.utils.solver_funcs import (
    calc_gradient_stats,
    get_batch_slice,
    shuffle_batch_order,
    update_coefs_adam,
    update_gradients_adam,
)
from symfc.utils.solver_utils_O2 import (
    calc_gradients_O2,
    calc_predictions_O2,
    slice_compact_compress_mat_O2,
)
from symfc.utils.solver_utils_O3 import (
    calc_gradients_O3,
    calc_predictions_O3,
    set_disps_N3N3,
    slice_compact_compress_mat_O3,
)

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
    gtol_fc2: float = 1e-9,
    gtol_fc3: float = 1e-12,
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
    and its products.
    """
    N3 = disps.shape[1]
    N = N3 // 3
    beta2 = beta**2 / (beta**2 + (1 - beta) ** 2)

    # TODO: Check gtol in various systems.
    average_force = np.average(np.linalg.norm(forces.reshape((-1, 3)), axis=1))
    gtol_fc2 *= (average_force / 1.0) ** 2
    gtol_fc3 *= (average_force / 1.0) ** 3
    eps_grad = min(gtol_fc2, gtol_fc3)

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
    rate_const = 1.0
    log_grads_fc2, log_grads_fc3 = [], []
    for i_epoch in range(n_epochs):
        t1 = time.time()

        if verbose:
            print("-----", flush=True)
            print("Epoch:", i_epoch + 1, flush=True)

        rate = np.zeros(n_compr)
        rate2 = max(100 / np.sqrt(i_epoch + 1), 1e-4) * rate_const
        rate3 = max(1000 / np.sqrt(i_epoch + 1), 1e-3) * rate_const
        rate[:n_compr_fc2] = rate2
        rate[n_compr_fc2:] = rate3
        if verbose:
            print("- Learning rate (FC2):", "{:.5f}".format(rate2), flush=True)
            print("- Learning rate (FC3):", "{:.5f}".format(rate3), flush=True)

        error_all = []
        for i_atom in shuffle_batch_order(len(begin_batch_atom)):
            begin_i, end_i = begin_batch_atom[i_atom], end_batch_atom[i_atom]
            if n_compr > 10000 and verbose:
                print("- Solver_atoms:", begin_i + 1, "--", end_i, "/", N, flush=True)

            decompr_idx_fc2, compr_mat_fc2 = slice_compact_compress_mat_O2(
                compact_compress_mat_fc2, atomic_decompr_idx_fc2, N, begin_i, end_i
            )
            decompr_idx_fc3, compr_mat_fc3 = slice_compact_compress_mat_O3(
                compact_compress_mat_fc3, atomic_decompr_idx_fc3, N, begin_i, end_i
            )
            for i_supercell in shuffle_batch_order(len(begin_batch)):
                begin, end = begin_batch[i_supercell], end_batch[i_supercell]
                y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)
                n_data = len(y)
                disps_batch = disps[begin:end]
                dispN3N3 = set_disps_N3N3(disps_batch, sparse=False)

                # Calculate pred = [X2, X3] @ coefs.
                pred2 = calc_predictions_O2(
                    compact_compress_mat_fc2,
                    decompr_idx_fc2,
                    N,
                    coefs[:n_compr_fc2],
                    disps_batch,
                )
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
                grad2 = calc_gradients_O2(compr_mat_fc2, N, error, disps_batch)
                grad3 = calc_gradients_O3(compr_mat_fc3, N, error, dispN3N3)
                grad_trial = np.concatenate([grad2, grad3]) / n_data

                grad, magn = update_gradients_adam(
                    grad_trial, grad_prev, magn_prev, beta, beta2
                )
                grad2_ave, grad2_max = calc_gradient_stats(grad[:n_compr_fc2])
                grad3_ave, grad3_max = calc_gradient_stats(grad[n_compr_fc2:])
                if (
                    grad2_ave < gtol_fc2
                    and grad2_max < gtol_fc2 * 10
                    and grad3_ave < gtol_fc3
                    and grad3_max < gtol_fc3 * 10
                ):
                    converge = True
                    break

                coefs = update_coefs_adam(coefs, grad, magn, rate, eps_grad)
                grad_prev, magn_prev = grad, magn

        t2 = time.time()
        if verbose:
            error_all = np.array(error_all)
            rmse_forces = np.sqrt(np.mean(error_all**2))
            print("- Time:              ", "{:.3f}".format(t2 - t1), "s", flush=True)
            print("- RMSE (Force):      ", "{:.5e}".format(rmse_forces), flush=True)
            print("- Max gradient (FC2):", "{:.5e}".format(grad2_max), flush=True)
            print("- Ave gradient (FC2):", "{:.5e}".format(grad2_ave), flush=True)
            print("- Max gradient (FC3):", "{:.5e}".format(grad3_max), flush=True)
            print("- Ave gradient (FC3):", "{:.5e}".format(grad3_ave), flush=True)

        if converge:
            break

        log_grads_fc2.append(grad2_ave)
        log_grads_fc3.append(grad3_ave)
        n_sl = 5
        if len(log_grads_fc2) > n_sl:
            grad2_slice = log_grads_fc2[-n_sl:]
            grad3_slice = log_grads_fc3[-n_sl:]
            d2 = np.sum(np.diff(grad2_slice / np.average(grad2_slice)))
            d3 = np.sum(np.diff(grad3_slice / np.average(grad3_slice)))
            if d2 > -0.001 and d3 > -0.001:
                if rate_const > 1e-4:
                    rate_const *= 0.1
                    log_grads_fc2, log_grads_fc3 = [], []
                else:
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
