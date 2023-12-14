"""3rd order force constants solver."""
from typing import Optional, Union

import numpy as np
from scipy.sparse import csc_array, csr_array

from symfc.utils.eig_tools import dot_product_sparse

from .solver_base import FCSolverBase


class FCSolverO3(FCSolverBase):
    """Third order force constants solver."""

    def __init__(
        self,
        basis_set: Union[np.ndarray, csr_array],
        translation_permutations: np.ndarray,
        log_level: int = 0,
    ):
        """Init method."""
        super().__init__(basis_set, translation_permutations, log_level=log_level)

    def solve(
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
        compress_mat: Union[csr_array, csc_array],
        is_compact_fc=True,
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
            Shape of force constants array is (n_a, N, N, 3, 3, 3) if True or
            (M, N, N, 3, 3, 3) if False.

        Returns
        -------
        ndarray
            Force constants. shape=(n_a, N, 3, 3) or (N, N, 3, 3). See
            `is_compact_fc` parameter. dtype='double', order='C'

        """
        N = self._natom
        if self._basis_set is None:
            return None
        assert displacements.shape == forces.shape

        mat = self._get_basis_mat(displacements, compress_mat)
        mat = np.linalg.pinv(mat)
        coeff = -2 * (mat @ forces.ravel())
        fc = dot_product_sparse(compress_mat, self._basis_set @ coeff)
        return fc.reshape(N, N, N, 3, 3, 3)

    def _get_basis_mat(self, displacements, compress_mat):
        d_compr_mat = self._get_d_compr_mat(displacements, compress_mat)
        mat = self._get_mat(d_compr_mat)
        return mat

    def _get_d_compr_mat(self, displacements, compress_mat):
        N = self._natom
        n_snapshot = displacements.shape[0]
        disps = displacements
        disp_disps = csr_array(
            np.array(
                [
                    np.transpose(
                        np.outer(d_snapshot.ravel(), d_snapshot.ravel()).reshape(
                            N, 3, N, 3
                        ),
                        (0, 2, 1, 3),
                    )
                    for d_snapshot in disps
                ],
                dtype="double",
            ).reshape(n_snapshot, -1)
        )
        compress_mat = self._get_compress_mat_NN33N3(compress_mat)
        print("disp, compress", disp_disps.shape, compress_mat.shape)
        d_compr_mat = dot_product_sparse(disp_disps, compress_mat).toarray()
        d_compr_mat = d_compr_mat.reshape(-1, self._basis_set.shape[0])
        print("d_compr_mat", d_compr_mat.shape, type(d_compr_mat))
        return d_compr_mat

    def _get_mat(self, d_compr_mat):
        print("basis_set", self._basis_set.shape)
        mat = dot_product_sparse(d_compr_mat, self._basis_set)
        print("mat", mat.shape, type(mat))
        return mat

    def _get_compress_mat_NN33N3(self, compress_mat):
        N = self._natom
        compress_mat_coo = compress_mat.tocoo()
        data = compress_mat_coo.data
        row = compress_mat_coo.row
        col = compress_mat_coo.col

        NN333 = N * N * 27
        N333 = N * 27
        N33 = N * 9
        N3 = N * 3
        conversion_array = np.array(
            [
                i * NN333 + j * N333 + l * N33 + m * N3 + k * 3 + n
                for i, j, k, l, m, n in np.ndindex((N, N, N, 3, 3, 3))
            ],
            dtype=int,
        )
        new_row = conversion_array[row]
        return csr_array((data, (new_row, col)), shape=compress_mat.shape).reshape(
            N * N * 3 * 3, -1
        )
