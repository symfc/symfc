"""O2 reps of space group ops with respect to atomic coordinate basis."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.utils import SymfcAtoms

from .spg_reps_base import SpgRepsBase


class SpgRepsO2(SpgRepsBase):
    """Class of reps of space group operations for fc2."""

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
        self._r2_reps: list[csr_array]
        super().__init__(supercell, spacegroup_operations=spacegroup_operations)

    @property
    def r_reps(self) -> list[csr_array]:
        """Return 2nd rank tensor rotation matricies."""
        return self._r2_reps

    def get_sigma2_rep(self, i: int, nonzero: np.ndarray = None) -> csr_array:
        """Compute vector representation of i-th atomic pair permutation matrix.

        Parameters
        ----------
        i : int
            Index of coset presentations of space group operations.

        """
        return self._get_sigma2_rep_data(i, nonzero=nonzero)

    def _prepare(self, spacegroup_operations):
        super()._prepare(spacegroup_operations)
        N = len(self._numbers)
        self._atom_pairs = (np.mgrid[0:N, 0:N].reshape((2, -1)).T).astype(
            "uint16", copy=False
        )
        self._coeff = np.array([N, 1], dtype=int)
        self._compute_r2_reps()

    def _compute_r2_reps(self, tol: float = 1e-10):
        """Compute and return 2nd rank tensor rotation matricies."""
        r2_reps = []
        for r in self._unique_rotations:
            r_c = self._lattice.T @ r @ np.linalg.inv(self._lattice.T)
            r2_rep = np.kron(r_c, r_c)
            row, col = np.nonzero(np.abs(r2_rep) > tol)
            data = r2_rep[(row, col)]
            r2_reps.append(csr_array((data, (row, col)), shape=r2_rep.shape))
        self._r2_reps = r2_reps

    def _get_sigma2_rep_data(self, i: int, nonzero: np.ndarray = None) -> csr_array:
        """Compute vector representation of i-th atomic pair permutation matrix.

        Operation permutation[self._atom_pairs @ self._coeff is divided
        to reduce memory allocation.

        permutation_pairs represents row indices of nonzero elements
        in permutation matrix. Column indices of nonzero elements in permutation
        matrix are array indices of permutation_pairs.
        """
        uri = self._unique_rotation_indices
        permutation = self._permutations[uri[i]]
        if nonzero is not None:
            pairs = self._atom_pairs[nonzero]
        else:
            pairs = self._atom_pairs
        permutation_pairs = permutation[pairs[:, 0]] * self._coeff[0]
        permutation_pairs += permutation[pairs[:, 1]]
        return permutation_pairs


class SpgRepsO2MatrixReps(SpgRepsBase):
    """Class of reps of space group operations for fc2."""

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
        self._r2_reps: list[csr_array]
        self._col: np.ndarray
        self._data: np.ndarray
        super().__init__(supercell, spacegroup_operations=spacegroup_operations)

    @property
    def r_reps(self) -> list[csr_array]:
        """Return 2nd rank tensor rotation matricies."""
        return self._r2_reps

    def get_sigma2_rep(self, i: int) -> csr_array:
        """Compute and return i-th atomic pair permutation matrix.

        Parameters
        ----------
        i : int
            Index of coset presentations of space group operations.

        """
        data, row, col, shape = self._get_sigma2_rep_data(i)
        return csr_array((data, (row, col)), shape=shape)

    def _prepare(self, spacegroup_operations):
        super()._prepare(spacegroup_operations)
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
            r2_reps.append(csr_array((data, (row, col)), shape=r2_rep.shape))
        self._r2_reps = r2_reps

    def _get_sigma2_rep_data(self, i: int) -> csr_array:
        uri = self._unique_rotation_indices
        permutation = self._permutations[uri[i]]
        NN = len(self._numbers) ** 2
        row = permutation[self._atom_pairs] @ self._coeff
        return self._data, row, self._col, (NN, NN)
