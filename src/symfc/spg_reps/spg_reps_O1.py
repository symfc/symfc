"""O1 reps of space group ops with respect to atomic coordinate basis."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.utils import SymfcAtoms

from .spg_reps_base import SpgRepsBase


class SpgRepsO1(SpgRepsBase):
    """Class of reps of space group operations for fc1."""

    def __init__(
        self, supercell: SymfcAtoms, spacegroup_operations: Optional[dict] = None
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        spacegroup_operations : dict, optional
            Space group operations in supercell, by default None. When None,
            spglib is used. The following keys and values correspond to spglib
            symmetry dataset:
                rotations : array_like
                translations : array_like

        """
        self._r1_reps: list[csr_array]
        self._col: np.ndarray
        self._data: np.ndarray
        super().__init__(supercell, spacegroup_operations=spacegroup_operations)

    @property
    def r_reps(self) -> list[csr_array]:
        """Return 1st rank tensor rotation matricies."""
        return self._r1_reps

    def get_sigma1_rep(self, i: int) -> csr_array:
        """Compute and return i-th atomic permutation matrix.

        Parameters
        ----------
        i : int
            Index of coset presentations of space group operations.

        """
        data, row, col, shape = self._get_sigma1_rep_data(i)
        return csr_array((data, (row, col)), shape=shape)

    def _prepare(self, spacegroup_operations):
        super()._prepare(spacegroup_operations)
        N = len(self._numbers)
        self._col = np.arange(N, dtype=int)
        self._data = np.ones(N, dtype=int)
        self._compute_r1_reps()

    def _compute_r1_reps(self, tol: float = 1e-10):
        """Compute and return 1st rank tensor rotation matricies.

        This is equivalent to rotation matrix of atomic position in Cartesian
        coordinates.

        """
        r1_reps = []
        for r in self._unique_rotations:
            r1_rep: np.ndarray = self._lattice.T @ r @ np.linalg.inv(self._lattice.T)
            row, col = np.nonzero(np.abs(r1_rep) > tol)
            data = r1_rep[(row, col)]
            r1_reps.append(csr_array((data, (row, col)), shape=r1_rep.shape))
        self._r1_reps = r1_reps

    def _get_sigma1_rep_data(self, i: int) -> csr_array:
        uri = self._unique_rotation_indices
        permutation = self._permutations[uri[i]]
        N = len(self._numbers)
        row = permutation
        return self._data, row, self._col, (N, N)
