"""Solver of 2nd order force constants using gradients."""

from __future__ import annotations

import time

import numpy as np

from symfc.basis_sets import FCBasisSetO2
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

from .solver_O2 import FCSolverO2


class FCGradSolverO2(FCSolverO2):
    """Second order force constants solver using gradients."""

    def __init__(
        self,
        basis_set: FCBasisSetO2,
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
        self._basis_set: FCBasisSetO2
        super().__init__(basis_set, use_mkl=use_mkl, log_level=log_level)

    def solve(
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
        batch_size: int = 100,
    ) -> FCGradSolverO2:
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
        self : FCGradSolverO2

        """
        n_data = forces.shape[0]
        f = forces.reshape(n_data, -1)
        d = displacements.reshape(n_data, -1)

        fc2_basis = self._basis_set
        coefs = solve_adam_O2(
            d,
            f,
            fc2_basis,
            batch_size=batch_size,
            use_mkl=self._use_mkl,
            verbose=self._log_level > 0,
        )
        self._coefs = coefs
        return self


def solve_adam_O2(
    disps: np.ndarray,
    forces: np.ndarray,
    fc2_basis: FCBasisSetO2,
    batch_size: int = 100,
    n_epochs: int = 10000,
    beta: float = 0.95,
    gtol_fc2: float = 1e-11,
    use_mkl: bool = False,
    verbose: bool = False,
):
    r"""Solve normal equations using Adam.

    X = displacements @ compress_mat @ compress_eigvecs

    displacements (fc2): (n_samples, N3)
    compact_compress_mat_fc2: (n_aN33, n_compr_fc2)
    compress_eigvecs_fc2: (n_compr_fc2, n_basis_fc2)
    """
    N3 = disps.shape[1]
    N = N3 // 3
    beta2 = beta**2 / (beta**2 + (1 - beta) ** 2)

    average_force = np.average(np.linalg.norm(forces.reshape((-1, 3)), axis=1))
    gtol_fc2 *= (average_force / 1.0) ** 2
    eps_grad = gtol_fc2

    compact_compress_mat_fc2 = fc2_basis.compact_compression_matrix
    atomic_decompr_idx_fc2 = fc2_basis.atomic_decompr_idx

    if compact_compress_mat_fc2 is None:
        raise ValueError(
            "Compression matrices or basis sets are not set. "
            "Call run() method to compute them."
        )

    n_compr_fc2 = compact_compress_mat_fc2.shape[1]  # type: ignore

    n_batch = N // 10 + 1
    begin_batch_atom, end_batch_atom = get_batch_slice(N, N // n_batch)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    coefs = np.zeros(n_compr_fc2)

    t_all1 = time.time()
    const_fc2 = -1.0
    compact_compress_mat_fc2 *= const_fc2

    grad_prev, magn_prev = np.zeros(n_compr_fc2), np.zeros(n_compr_fc2)
    converge = False
    rate_const = 1.0
    log_grads_fc2 = []
    for i_epoch in range(n_epochs):
        t1 = time.time()
        if verbose:
            print("-----", flush=True)
            print("Epoch:", i_epoch + 1, flush=True)

        rate = max(100 / np.sqrt(i_epoch + 1), 1e-4) * rate_const
        if verbose:
            print("- Learning rate (FC2):", "{:.5f}".format(rate), flush=True)

        error_all = []
        for i_atom in shuffle_batch_order(len(begin_batch_atom)):
            begin_i, end_i = begin_batch_atom[i_atom], end_batch_atom[i_atom]
            decompr_idx_fc2, compr_mat_fc2 = slice_compact_compress_mat_O2(
                compact_compress_mat_fc2, atomic_decompr_idx_fc2, N, begin_i, end_i
            )
            for i_supercell in shuffle_batch_order(len(begin_batch)):
                begin, end = begin_batch[i_supercell], end_batch[i_supercell]
                y = forces[begin:end, begin_i * 3 : end_i * 3].reshape(-1)
                n_data = len(y)

                # Calculate pred = X2 @ coefs
                pred2 = calc_predictions_O2(
                    compact_compress_mat_fc2,
                    decompr_idx_fc2,
                    N,
                    coefs,
                    disps[begin:end],
                )
                error = pred2 - y
                error_all.extend(error)

                # Calculate grad = X2.T @ error.
                grad_trial = calc_gradients_O2(
                    compr_mat_fc2,
                    N,
                    error,
                    disps[begin:end],
                )
                grad_trial /= n_data

                grad, magn = update_gradients_adam(
                    grad_trial, grad_prev, magn_prev, beta, beta2
                )
                grad_ave, grad_max = calc_gradient_stats(grad)
                if grad_ave < gtol_fc2 and grad_max < gtol_fc2 * 10:
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
            print("- Max gradient (FC2):", "{:.5e}".format(grad_max), flush=True)
            print("- Ave gradient (FC2):", "{:.5e}".format(grad_ave), flush=True)

        if converge:
            break

        log_grads_fc2.append(grad_ave)
        n_sl = 5
        if len(log_grads_fc2) > n_sl:
            grad2_slice = log_grads_fc2[-n_sl:]
            d2 = np.sum(np.diff(grad2_slice / np.average(grad2_slice)))
            if d2 > -0.001:
                if rate_const > 1e-4:
                    rate_const *= 0.1
                    log_grads_fc2 = []
                else:
                    break

    compress_eigvecs = fc2_basis.blocked_basis_set
    coefs = compress_eigvecs.T @ coefs

    compact_compress_mat_fc2 /= const_fc2
    t_all2 = time.time()
    if verbose:
        header = "Time (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):"
        print(header, "{:.3f}".format(t_all2 - t_all1), flush=True)
    return coefs
