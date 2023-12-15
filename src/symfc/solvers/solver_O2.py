"""2nd order force constants solver."""
from typing import Optional, Union

import numpy as np
from scipy.sparse import csc_array, csr_array

from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.utils_O2 import get_lat_trans_decompr_indices

from .solver_base import FCSolverBase


class FCSolverO2(FCSolverBase):
    """Second order force constants solver."""

    def __init__(
        self,
        basis_set: np.ndarray,
        translation_permutations: np.ndarray,
        compression_matrix: Optional[Union[csr_array, csc_array]] = None,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        super().__init__(
            basis_set,
            translation_permutations,
            compression_matrix=compression_matrix,
            use_mkl=use_mkl,
            log_level=log_level,
        )

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
        N = self._natom
        if self._basis_set is None:
            return None
        assert displacements.shape == forces.shape
        fc = self._basis_set @ self._get_basis_coeff(displacements, forces)
        if is_compact_fc:
            return fc.reshape(-1, N, 3, 3)
        else:
            if self._compression_matrix is None:
                trans_perms = self._translation_permutations
                decompr_idx = get_lat_trans_decompr_indices(trans_perms)
                return fc[decompr_idx].reshape(N, N, 3, 3)
            else:
                fc = dot_product_sparse(
                    self._compression_matrix,
                    csr_array(fc.reshape(-1, 1)),
                    use_mkl=self._use_mkl,
                )
                return fc.toarray().reshape(N, N, 3, 3)

    def _get_basis_coeff(
        self, displacements: np.ndarray, forces: np.ndarray
    ) -> Optional[np.ndarray]:
        N = self._natom
        if self._basis_set is None:
            return None
        n_snapshot = displacements.shape[0]
        disps = displacements.reshape(n_snapshot, N * 3)
        d_basis = np.zeros(
            (n_snapshot * N * 3, self._basis_set.shape[1]), dtype="double", order="C"
        )

        if self._compression_matrix is None:
            self._compute_with_trans_perms(d_basis, disps)
        else:
            self._compute_with_compression_matrix(d_basis, disps)
        if self._log_level:
            print("Computing product of displacements and basis set...")
        if self._log_level:
            print("Solving basis-set coefficients...")
        coeff = -(np.linalg.pinv(d_basis) @ forces.ravel())
        return coeff

    def _compute_with_compression_matrix(self, d_basis: np.ndarray, disps: np.ndarray):
        N = self._natom
        full_basis_set = dot_product_sparse(
            self._compression_matrix,
            csc_array(self._basis_set),
            use_mkl=self._use_mkl,
        )
        for i_basis_set in range(full_basis_set.shape[1]):
            vec = np.transpose(
                full_basis_set[:, [i_basis_set]].toarray().reshape(N, N, 3, 3),
                (0, 2, 1, 3),
            ).reshape(N * 3, N * 3)
            d_basis[:, i_basis_set] = (disps @ vec).ravel()

    def _compute_with_trans_perms(self, d_basis: np.ndarray, disps: np.ndarray):
        N = self._natom
        trans_perms = self._translation_permutations
        decompr_idx = get_lat_trans_decompr_indices(trans_perms)
        decompr_array = np.transpose(
            decompr_idx.reshape(N, N, 3, 3), (0, 2, 1, 3)
        ).reshape(N * 3, N * 3)
        for i, vec in enumerate(self._basis_set.T):
            d_basis[:, i] = (disps @ vec[decompr_array]).ravel()
