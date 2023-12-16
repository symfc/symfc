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
    positions: np.ndarray,  # scaled positions
    rotations: np.ndarray,  # scaled
    translations: np.ndarray,  # scaled
    lattice: np.ndarray,  # column vectors
    symprec: float,
):
    """Compute permutations of atoms by space group operations in supercell.

    This code is obtained from the implemention in phonopy, and modified to
    avoid using the C implementation.

    Parameters
    ----------
    positions : ndarray
        Scaled positions (like SymfcAtoms.scaled_positions) before applying
        the space group operation
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
        out.append(
            _compute_permutation_for_translation(
                positions, rotated_positions, lattice, symprec
            )
        )
    return np.array(out, dtype="intc", order="C")


def _compute_permutation_for_translation(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    lattice: np.ndarray,
    symprec: float,  # scaled positions  # column vectors
):
    """Get the overall permutation such that.

        positions_a[perm[i]] == positions_b[i]   (modulo the lattice)

    or in numpy speak,

        positions_a[perm] == positions_b   (modulo the lattice)

    This version is optimized for the case where positions_a and positions_b
    are related by a rotation.

    Parameters
    ----------
    positions_a : ndarray
        Scaled positions (like SymfcAtoms.scaled_positions) before applying
        the space group operation
    positions_b : ndarray
        Scaled positions (like SymfcAtoms.scaled_positions) after applying
        the space group operation
    lattice : ndarray
        Basis vectors in column vectors (like SymfcAtoms.cell.T)
    symprec : float
        Symmetry tolerance of the distance unit

    Returns
    -------
    perm : ndarray
        A list of atomic indices that maps atoms before the space group
        operation to those after as explained above.
        shape=(len(positions), ), dtype=int

    """

    def sort_by_lattice_distance(fracs):
        """Sort atoms by distance.

        Sort both sides by some measure which is likely to produce a small
        maximum value of (sorted_rotated_index - sorted_original_index).
        The C code is optimized for this case, reducing an O(n^2)
        search down to ~O(n). (for O(n log n) work overall, including the sort)

        We choose distance from the nearest bravais lattice point as our measure.

        """
        carts = np.dot(fracs - np.rint(fracs), lattice.T)
        perm = np.argsort(np.sum(carts**2, axis=1))
        sorted_fracs = np.array(fracs[perm], dtype="double", order="C")
        return perm, sorted_fracs

    (perm_a, sorted_a) = sort_by_lattice_distance(positions_a)
    (perm_b, sorted_b) = sort_by_lattice_distance(positions_b)

    perm_between = _compute_atom_mapping(sorted_a, sorted_b, lattice, symprec)

    # Compose all of the permutations for the full permutation.
    #
    # Note the following properties of permutation arrays:
    #
    # 1. Inverse:         if  x[perm] == y  then  x == y[argsort(perm)]
    # 2. Associativity:   x[p][q] == x[p[q]]
    return perm_a[perm_between][np.argsort(perm_b)]


def _compute_atom_mapping(
    positions_a, positions_b, lattice, symprec  # scaled positions  # column vectors
):
    """Return mapping defined by positions_a[perm[i]] == positions_b[i].

    Version of `_compute_permutation_for_rotation` which just directly
    calls the C function, without any conditioning of the data.
    Skipping the conditioning step makes this EXTREMELY slow on large
    structures.

    """
    atom_mapping = np.zeros(shape=(len(positions_a),), dtype="intc")

    for i, pos_b in enumerate(positions_b):
        diffs = positions_a - pos_b
        diffs -= np.rint(diffs)
        dists = np.linalg.norm(np.dot(diffs, lattice.T), axis=1)
        possible_j = np.where(dists < symprec)[0]
        if len(possible_j) != 1:
            raise ValueError("Mapping atoms failed. Maybe some atoms are too close.")
        atom_mapping[i] = possible_j[0]

    if -1 in atom_mapping:
        raise ValueError("Mapping atoms failed. Some atoms could not be mapped.")

    return atom_mapping


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
