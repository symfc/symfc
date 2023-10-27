"""Reps of space group operations with respect to atomic coordinate basis."""
from __future__ import annotations

from typing import Optional

import numpy as np
import spglib
from phonopy.structure.cells import compute_all_sg_permutations
from phonopy.utils import similarity_transformation
from scipy.sparse import coo_array


class SpgReps:
    """Reps of space group operations with respect to atomic coordinate basis."""

    def __init__(
        self,
        lattice: np.ndarray,
        positions: np.ndarray,
        numbers: np.ndarray,
        only_coset_representatives: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        lattice : array_like
            Basis vectors as column vectors. shape=(3, 3), dtype='double'
        positions : array_like
            Position vectors given as column vectors. shape=(3, natom),
            dtype='double'
        numbers : array_like
            Atomic IDs idicated by integers larger or eqaul to 0.
        only_coset_representatives : bool
            Matrix reps are computed for only coset representatives. Default is
            True.

        """
        self._lattice = np.array(lattice, dtype="double", order="C")
        self._positions = np.array(positions, dtype="double", order="C")
        self._numbers = numbers
        self._reps: Optional[list[coo_array]] = None
        self._translation_permutations: Optional[np.ndarray] = None

        self._run(only_coset_representatives=only_coset_representatives)

    @property
    def representations(self) -> Optional[list[coo_array]]:
        """Return 3Nx3N matrix representations."""
        return self._reps

    @property
    def translation_permutations(self) -> Optional[np.ndarray]:
        """Return permutations by lattice translation.

        Returns
        --------
        Atom indices after lattice translations.
        shape=(lattice_translations, supercell_atoms), dtype=int

        """
        return self._translation_permutations

    def _run(self, only_coset_representatives=True):
        rotations, translations = self._get_symops()
        permutations = compute_all_sg_permutations(
            self._positions.T, rotations, translations, self._lattice, 1e-5
        )
        self._translation_permutations, _ = self._get_translation_permutations(
            permutations, rotations
        )
        unique_rotation_indices = self._get_unique_rotation_indices(rotations)
        if only_coset_representatives:
            self._reps = self._compute_reps(
                permutations, rotations, unique_rotation_indices
            )
        else:
            self._reps = self._compute_reps(
                permutations, rotations, list(range(len(rotations)))
            )

    def _get_translation_permutations(
        self, permutations, rotations
    ) -> tuple[np.ndarray, np.ndarray]:
        eye3 = np.eye(3, dtype=int)
        trans_perms = []
        trans_indices = []
        for i, (r, perm) in enumerate(zip(rotations, permutations)):
            if np.array_equal(r, eye3):
                trans_perms.append(perm)
                trans_indices.append(i)
        return np.array(trans_perms, dtype=int), np.array(trans_indices, dtype=int)

    def _get_unique_rotation_indices(self, rotations: np.ndarray) -> list[int]:
        unique_rotations = []
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
        symops = spglib.get_symmetry(
            (self._lattice.T, self._positions.T, self._numbers)
        )
        return symops["rotations"], symops["translations"]

    def _compute_reps(
        self,
        permutations: np.ndarray,
        rotations: np.ndarray,
        rotation_indices: list[int],
        tol=1e-10,
    ) -> list[coo_array]:
        """Construct representation matrices of rotations.

        Permutation of atoms by r, perm(r) = [0 1], means the permutation matrix:
            [1 0]
            [0 1]
        Rotation matrix in Cartesian coordinates:
            [0 1 0]
        r = [1 0 0]
            [0 0 1]

        Its representation matrix of perm(r) and r becomes

        [0 1 0 0 0 0]
        [1 0 0 0 0 0]
        [0 0 1 0 0 0]
        [0 0 0 0 1 0]
        [0 0 0 1 0 0]
        [0 0 0 0 0 1]

        Note
        ----
        np.add.outer(a, b).ravel() = [i + j for i in a for j in b].

        """
        size = 3 * len(self._numbers)
        atom_indices = np.arange(len(self._numbers))
        reps = []
        for perm, r in zip(permutations[rotation_indices], rotations[rotation_indices]):
            rot_cart = similarity_transformation(self._lattice, r)
            nonzero_r_row, nonzero_r_col = np.nonzero(np.abs(rot_cart) > tol)
            row = np.add.outer(perm * 3, nonzero_r_row).ravel()
            col = np.add.outer(atom_indices * 3, nonzero_r_col).ravel()
            nonzero_r_elems = [
                rot_cart[i, j] for i, j in zip(nonzero_r_row, nonzero_r_col)
            ]
            data = np.tile(nonzero_r_elems, len(self._numbers))
            reps.append(coo_array((data, (row, col)), shape=(size, size)))
        return reps
