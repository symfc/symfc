"""O2 reps of space group ops with respect to atomic coordinate basis."""
from __future__ import annotations

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from scipy.sparse import coo_array

from .spg_reps_base import SpgRepsBase


class SpgRepsO2(SpgRepsBase):
    """Class of reps of space group operations for fc2."""

    def __init__(self, supercell: PhonopyAtoms):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.

        """
        self._r2_reps: list[coo_array]
        self._col: np.ndarray
        self._data: np.ndarray
        super().__init__(supercell)

    @property
    def r_reps(self) -> list[coo_array]:
        """Return 2nd rank tensor rotation matricies."""
        return self._r2_reps

    def get_sigma2_rep(self, i: int) -> coo_array:
        """Compute and return i-th atomic pair permutation matrix.

        Parameters
        ----------
        i : int
            Index of coset presentations of space group operations.

        """
        data, row, col, shape = self._get_sigma2_rep_data(i)
        return coo_array((data, (row, col)), shape=shape)

    def _prepare(self):
        super()._prepare()
        N = len(self._numbers)
        a = np.arange(N)
        self._atom_pairs = np.stack(np.meshgrid(a, a), axis=-1).reshape(-1, 2)
        self._coeff = np.array([1, N], dtype=int)
        self._col = self._atom_pairs @ self._coeff
        self._data = np.ones(N * N, dtype=int)
        self._compute_r2_reps()

    def _compute_r2_reps(self, tol: float = 1e-10):
        """Compute and return 2nd rank tensor rotation matricies."""
        r2_reps = []
        for r in self._unique_rotations:
            r_c = self._lattice.T @ r @ np.linalg.inv(self._lattice.T)
            r2_rep = np.kron(r_c, r_c)
            row, col = np.nonzero(np.abs(r2_rep) > tol)
            data = r2_rep[(row, col)]
            r2_reps.append(coo_array((data, (row, col)), shape=r2_rep.shape))
        self._r2_reps = r2_reps

    def _get_sigma2_rep_data(self, i: int) -> coo_array:
        uri = self._unique_rotation_indices
        permutation = self._permutations[uri[i]]
        NN = len(self._numbers) ** 2
        row = permutation[self._atom_pairs] @ self._coeff
        return self._data, row, self._col, (NN, NN)
