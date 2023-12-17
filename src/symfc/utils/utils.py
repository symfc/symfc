"""Utility functions."""
from __future__ import annotations

import numpy as np


def get_indep_atoms_by_lat_trans(trans_perms: np.ndarray) -> np.ndarray:
    """Return independent atoms by lattice translation symmetry.

    Parameters
    ----------
    trans_perms : np.ndarray
        Atom indices after lattice translations.
        shape=(lattice_translations, supercell_atoms)

    Returns
    -------
    np.ndarray
        Independent atoms.
        shape=(n_indep_atoms_by_lattice_translation,), dtype=int

    """
    unique_atoms: list[int] = []
    assert np.array_equal(trans_perms[0, :], range(trans_perms.shape[1]))
    for i, perms in enumerate(trans_perms.T):
        is_found = False
        for j in unique_atoms:
            if j in perms:
                is_found = True
                break
        if not is_found:
            unique_atoms.append(i)
    return np.array(unique_atoms, dtype=int)


def compute_sg_permutations(
    positions: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    lattice: np.ndarray,
    symprec: float = 1e-5,
) -> np.ndarray:
    """Compute permutations of atoms by space group operations in supercell.

    This function was originally obtained from the implemention in phonopy. Not
    to use the C-implementation, finally it almost looks different
    implementation by dropping the feature in the original implementation that
    sorts atoms by distances from the origin for the performance optimization.

    Parameters
    ----------
    positions : ndarray
        Scaled positions (like SymfcAtoms.scaled_positions) before applying the
        space group operation
    rotations : ndarray
        Matrix (rotation) parts of space group operations
        shape=(len(operations), 3, 3), dtype='intc'
    translations : ndarray
        Vector (translation) parts of space group operations
        shape=(len(operations), 3), dtype='double'
    lattice : ndarray
        Basis vectors in column vectors (like SymfcAtoms.cell.T)
    symprec : float
        Symmetry tolerance of the distance unit

    Returns
    -------
    perms : ndarray
        shape=(len(translations), len(positions)), dtype='intc', order='C'

    """
    out = []
    for sym, t in zip(rotations, translations):
        rotated_positions = positions @ sym.T + t
        diffs = positions[None, :, :] - rotated_positions[:, None, :]
        diffs -= np.rint(diffs)
        dists = np.linalg.norm(diffs @ lattice.T, axis=2)
        rows, cols = np.where(dists < symprec)
        assert len(positions) == len(np.unique(rows)) == len(np.unique(cols))
        out.append(cols[np.argsort(rows)])
    return np.array(out, dtype="intc", order="C")


class SymfcAtoms:
    """Class to represent crystal structure mimicing PhonopyAtoms."""

    def __init__(
        self,
        numbers=None,
        scaled_positions=None,
        cell=None,
    ):
        """Init method."""
        self._cell = np.array(cell, dtype="double")
        self._scaled_positions = np.array(scaled_positions, dtype="double")
        self._numbers = np.array(numbers, dtype="intc")

    @property
    def cell(self):
        """Setter and getter of basis vectors. For getter, copy is returned."""
        return self._cell.copy()

    @property
    def scaled_positions(self):
        """Setter and getter of scaled positions. For getter, copy is returned."""
        return self._scaled_positions.copy()

    @property
    def numbers(self):
        """Setter and getter of atomic numbers. For getter, copy is returned."""
        return self._numbers.copy()

    def totuple(self):
        """Return (cell, scaled_position, numbers)."""
        return (self._cell, self._scaled_positions, self._numbers)
