"""O3 reps of space group ops with respect to atomic coordinate basis."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array

from symfc.spg_reps import SpgRepsBase
from symfc.utils.utils import SymfcAtoms


class SpgRepsO3(SpgRepsBase):
    """Class of reps of space group operations for fc3."""

    def __init__(self, supercell: SymfcAtoms):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.

        """
        self._r3_reps: list[csr_array]
        self._col: np.ndarray
        self._data: np.ndarray
        super().__init__(supercell)

    @property
    def r_reps(self) -> list[csr_array]:
        """Return 3rd rank tensor rotation matricies."""
        return self._r3_reps

    def get_sigma3_rep(self, i: int) -> csr_array:
        """Compute and return i-th atomic pair permutation matrix.

        Parameters
        ----------
        i : int
            Index of coset presentations of space group operations.

        """
        data, row, col, shape = self._get_sigma3_rep_data(i)
        return csr_array((data, (row, col)), shape=shape)

    def _prepare(self):
        super()._prepare()
        N = len(self._numbers)
        a = np.arange(N)
        self._atom_triplets = np.stack(np.meshgrid(a, a, a), axis=-1).reshape(-1, 3)
        self._coeff = np.array([1, N, N**2], dtype=int)
        self._col = self._atom_triplets @ self._coeff
        self._data = np.ones(N * N * N, dtype=int)
        self._compute_r3_reps()

    def _compute_r3_reps(self, tol: float = 1e-10):
        """Compute and return 3rd rank tensor rotation matricies."""
        r3_reps = []
        for r in self._unique_rotations:
            r_c = self._lattice.T @ r @ np.linalg.inv(self._lattice.T)
            r3_rep = np.kron(r_c, np.kron(r_c, r_c))
            row, col = np.nonzero(np.abs(r3_rep) > tol)
            data = r3_rep[(row, col)]
            r3_reps.append(csr_array((data, (row, col)), shape=r3_rep.shape))
        self._r3_reps = r3_reps

    def _get_sigma3_rep_data(self, i: int) -> csr_array:
        uri = self._unique_rotation_indices
        permutation = self._permutations[uri[i]]
        NNN = len(self._numbers) ** 3
        row = permutation[self._atom_triplets] @ self._coeff
        return self._data, row, self._col, (NNN, NNN)
