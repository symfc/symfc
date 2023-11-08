"""Force constants solvers."""
from typing import Optional

import numpy as np

from symfc.utils import get_lat_trans_decompr_indices


class FCSolverO2:
    """Second order force constants solver."""

    def __init__(
        self,
        basis_set: np.ndarray,
        translation_permutations: np.ndarray,
        log_level: int = 0,
    ):
        """Init method."""
        self._basis_set = basis_set
        self._translation_permutations = translation_permutations
        self._log_level = log_level

        _, self._natom = self._translation_permutations.shape

    def solve(
        self, displacements: np.ndarray, forces: np.ndarray, is_compact_fc=True
    ) -> Optional[np.ndarray]:
        """Solve force constants.

        Parameters
        ----------
        displacements : ndarray
            Displacements of atoms in Cartesian coordinates. shape=(n_snapshot,
            N, 3), dtype='double'
        forces : ndarray
            Forces of atoms in Cartesian coordinates. shape=(n_snapshot, N, 3),
            dtype='double'
        is_compact_fc : bool
            Shape of force constants array is (n_a, N, 3, 3) if True or (N, N,
            3, 3) if False.

        Returns
        -------
        ndarray
            Force constants.
            shape=(n_a, N, 3, 3) or (N, N, 3, 3). See `is_compact_fc` parameter.
            dtype='double', order='C'

        """
        trans_perms = self._translation_permutations
        N = self._natom
        decompr_idx = get_lat_trans_decompr_indices(trans_perms)
        if self._basis_set is None:
            return None
        assert displacements.shape == forces.shape
        fc = self._basis_set @ self._get_basis_coeff(displacements, forces, decompr_idx)
        if is_compact_fc:
            return fc.reshape(-1, N, 3, 3)
        else:
            return fc[decompr_idx].reshape(N, N, 3, 3)

    def _get_basis_coeff(
        self, displacements: np.ndarray, forces: np.ndarray, decompr_idx: np.ndarray
    ) -> Optional[np.ndarray]:
        N = self._natom
        if self._basis_set is None:
            return None
        decompr_array = np.transpose(
            decompr_idx.reshape(N, N, 3, 3), (0, 2, 1, 3)
        ).reshape(N * 3, N * 3)
        n_snapshot = displacements.shape[0]
        disps = displacements.reshape(n_snapshot, N * 3)
        d_basis = np.zeros(
            (n_snapshot * N * 3, self._basis_set.shape[1]), dtype="double", order="C"
        )
        if self._log_level:
            print("Computing product of displacements and basis set...")
        for i, vec in enumerate(self._basis_set.T):
            d_basis[:, i] = (disps @ vec[decompr_array]).ravel()
        if self._log_level:
            print("Solving basis-set coefficients...")
        coeff = -(np.linalg.pinv(d_basis) @ forces.ravel())
        return coeff
