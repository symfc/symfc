"""O1 reps of space group ops with respect to atomic coordinate basis."""
from __future__ import annotations

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from scipy.sparse import coo_array

from .spg_reps_base import SpgRepsBase


class SpgRepsO1(SpgRepsBase):
    """Class of reps of space group operations for fc1."""

    def __init__(self, supercell: PhonopyAtoms):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.

        """
        self._r1_reps: list[coo_array]
        self._col: np.ndarray
        self._data: np.ndarray
        super().__init__(supercell)

    @property
    def r_reps(self) -> list[coo_array]:
        """Return 1st rank tensor rotation matricies."""
        return self._r1_reps

    def get_sigma1_rep(self, i: int) -> coo_array:
        """Compute and return i-th atomic pair permutation matrix.

        Parameters
        ----------
        i : int
            Index of coset presentations of space group operations.

        """
        data, row, col, shape = self._get_sigma1_rep_data(i)
        return coo_array((data, (row, col)), shape=shape)

    def _prepare(self):
        rotations = super()._prepare()
        N = len(self._numbers)
        self._col = np.arange(N, dtype=int)
        self._data = np.ones(N, dtype=int)
        self._compute_r1_reps(rotations)

    def _compute_r1_reps(self, rotations: np.ndarray, tol: float = 1e-10):
        """Compute and return 1st rank tensor rotation matricies."""
        uri = self._unique_rotation_indices
        r1_reps = []
        for r in rotations[uri]:
            r1_rep: np.ndarray = self._lattice.T @ r @ np.linalg.inv(self._lattice.T)
            row, col = np.nonzero(np.abs(r1_rep) > tol)
            data = r1_rep[(row, col)]
            r1_reps.append(coo_array((data, (row, col)), shape=r1_rep.shape))
        self._r1_reps = r1_reps

    def _get_sigma1_rep_data(self, i: int) -> coo_array:
        uri = self._unique_rotation_indices
        permutation = self._permutations[uri[i]]
        N = len(self._numbers)
        row = permutation
        return self._data, row, self._col, (N, N)
