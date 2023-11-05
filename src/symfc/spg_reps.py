"""Reps of space group operations with respect to atomic coordinate basis."""
from __future__ import annotations

from typing import Optional

import numpy as np
import spglib
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import compute_all_sg_permutations
from scipy.sparse import coo_array


class SpgReps:
    """Reps of space group operations with respect to atomic coordinate basis."""

    def __init__(self, supercell: PhonopyAtoms):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.

        """
        self._lattice = np.array(supercell.cell, dtype="double", order="C")
        self._positions = np.array(
            supercell.scaled_positions, dtype="double", order="C"
        )
        self._numbers = supercell.numbers
        self._r2_reps: Optional[list[coo_array]] = None
        self._unique_rotation_indices: Optional[np.ndarray] = None
        self._translation_permutations: Optional[np.ndarray] = None

        self._prepare()

    @property
    def translation_permutations(self) -> Optional[np.ndarray]:
        """Return permutations by lattice translation.

        Returns
        --------
        Atom indices after lattice translations.
        shape=(lattice_translations, supercell_atoms), dtype=int

        """
        return self._translation_permutations

    @property
    def unique_rotation_indices(self) -> Optional[np.ndarray]:
        """Return indices of coset representatives of space group operations."""
        return self._unique_rotation_indices

    @property
    def r2_reps(self) -> Optional[list[coo_array]]:
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
        self._rotations, translations = self._get_symops()
        self._permutations = compute_all_sg_permutations(
            self._positions, self._rotations, translations, self._lattice.T, 1e-5
        )
        self._translation_permutations = self._get_translation_permutations(
            self._permutations, self._rotations
        )
        self._unique_rotation_indices = self._get_unique_rotation_indices(
            self._rotations
        )
        N = len(self._numbers)
        a = np.arange(N)
        self._atom_pairs = np.stack(np.meshgrid(a, a), axis=-1).reshape(-1, 2)
        self._coeff = np.array([1, N], dtype=int)
        self._col = self._atom_pairs @ self._coeff
        self._data = np.ones(N * N, dtype=int)
        self._r2_reps = self._compute_r2_reps()

    def _compute_r2_reps(self, tol=1e-10) -> list:
        """Compute and return 2nd rank tensor rotation matricies."""
        uri = self._unique_rotation_indices
        r2_reps = []
        for r in self._rotations[uri]:
            r_c = self._lattice.T @ r @ np.linalg.inv(self._lattice.T)
            r2_rep = np.kron(r_c, r_c)
            row, col = np.nonzero(np.abs(r2_rep) > tol)
            data = r2_rep[(row, col)]
            r2_reps.append(coo_array((data, (row, col)), shape=r2_rep.shape))
        return r2_reps

    def _get_translation_permutations(self, permutations, rotations) -> np.ndarray:
        eye3 = np.eye(3, dtype=int)
        trans_perms = []
        for r, perm in zip(rotations, permutations):
            if np.array_equal(r, eye3):
                trans_perms.append(perm)
        return np.array(trans_perms, dtype="intc", order="C")

    def _get_unique_rotation_indices(self, rotations: np.ndarray) -> list[int]:
        unique_rotations: list[np.ndarray] = []
        indices = []
        for i, r in enumerate(rotations):
            is_found = False
            for ur in unique_rotations:
                if np.array_equal(r, ur):
                    is_found = True
                    break
            if not is_found:
                unique_rotations.append(r)
                indices.append(i)
        return indices

    def _get_symops(self) -> tuple[np.ndarray, np.ndarray]:
        """Return symmetry operations.

        The set of inverse operations is the same as the set of the operations.

        Returns
        -------
        rotations : array_like
            A set of rotation matrices of inverse space group operations.
            (n_symops, 3, 3), dtype='intc', order='C'
        translations : array_like
            A set of translation vectors. It is assumed that inverse matrices are
            included in this set.
            (n_symops, 3), dtype='double'.

        """
        symops = spglib.get_symmetry((self._lattice, self._positions, self._numbers))
        return symops["rotations"], symops["translations"]

    def _get_sigma2_rep_data(self, i: int) -> coo_array:
        uri = self._unique_rotation_indices
        permutation = self._permutations[uri[i]]
        NN = len(self._numbers) ** 2
        row = permutation[self._atom_pairs] @ self._coeff
        return self._data, row, self._col, (NN, NN)
