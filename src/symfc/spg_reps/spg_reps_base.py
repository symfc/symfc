"""Reps base of space group ops with respect to atomic coordinate basis."""

from __future__ import annotations

from typing import Optional

import numpy as np

from symfc.utils.utils import (
    SymfcAtoms,
    compute_sg_permutations,
    get_indep_atoms_by_lat_trans,
)


class SpgRepsBase:
    """Base class of reps of space group operations."""

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
        self._lattice = np.array(supercell.cell, dtype="double", order="C")
        self._positions = np.array(
            supercell.scaled_positions, dtype="double", order="C"
        )
        self._numbers = supercell.numbers
        self._unique_rotations: Optional[np.ndarray] = None
        self._unique_rotation_indices: Optional[np.ndarray] = None
        self._translation_permutations: Optional[np.ndarray] = None
        self._p2s_map: Optional[np.ndarray] = None
        self._prepare(spacegroup_operations)

    @property
    def translation_permutations(self) -> np.ndarray:
        """Return permutations by lattice translation.

        Returns
        -------
        Atom indices after lattice translations.
        shape=(lattice_translations, supercell_atoms), dtype=int

        """
        return self._translation_permutations

    @property
    def unique_rotation_indices(self) -> np.ndarray:
        """Return indices of coset representatives of space group operations."""
        return self._unique_rotation_indices

    @property
    def p2s_map(self) -> np.ndarray:
        """Return indices of translationally independent atoms."""
        return self._p2s_map

    def _prepare(self, spacegroup_operations) -> np.ndarray:
        rotations, translations = self._get_symops(spacegroup_operations)
        (
            self._unique_rotation_indices,
            self._unique_rotations,
        ) = self._get_unique_rotation_indices(rotations)
        self._permutations = compute_sg_permutations(
            self._positions, rotations, translations, self._lattice.T, 1e-5
        )
        self._translation_permutations = self._get_translation_permutations(rotations)
        self._p2s_map = get_indep_atoms_by_lat_trans(self._translation_permutations)

    def _get_translation_permutations(self, rotations) -> np.ndarray:
        eye3 = np.eye(3, dtype=int)
        trans_perms = []
        for r, perm in zip(rotations, self._permutations):
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
        return indices, unique_rotations

    def _get_symops(
        self, spacegroup_operations: Optional[dict] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return symmetry operations.

        The set of inverse operations is the same as the set of the operations.

        Parameters
        ----------
        spacegroup_operations : dict, optional
            Space group operations in supercell, by default None.
            When None, spglib is used to get the operations.

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
        if spacegroup_operations is None:
            try:
                import spglib
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "Spglib python module was not found."
                ) from exc

            symops = spglib.get_symmetry(
                (self._lattice, self._positions, self._numbers)
            )
        else:
            symops = spacegroup_operations
        return symops["rotations"], symops["translations"]
