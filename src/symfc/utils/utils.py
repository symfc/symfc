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


def round_positions(positions: np.ndarray, tol: float = 1e-13, decimals: int = 5):
    """Round fractional coordinates of positions (-0.5 <= p < 0.5)."""
    positions_rint = positions - np.rint(positions)
    positions_rint[np.abs(positions_rint - 0.5) < tol] = -0.5
    positions_rint = np.round(positions_rint, decimals)
    return positions_rint


def argsort_positions(positions: np.ndarray, tol: float = 1e-13, decimals: int = 5):
    """Round and sort fractional coordinates of positions (-0.5 <= p < 0.5)."""
    positions_round = round_positions(positions, tol=tol, decimals=decimals)
    # Not needed part?
    positions_rint = 10**decimals * positions_round
    positions_rint = [tuple(p) for p in positions_rint.astype(int)]
    # Not needed part? (end)
    sorted_ids = sorted(range(positions.shape[0]), key=positions_rint.__getitem__)
    sorted_positions = positions_round[sorted_ids]
    return sorted_ids, sorted_positions


def _find_optimal_decimals(positions: np.ndarray, tol: float = 1e-13):
    """Find optimal value of decimals used for sorting atoms."""
    n_atom = positions.shape[0]
    for decimals in range(1, 15):
        positions_round = round_positions(positions, tol=tol, decimals=decimals)
        n_atom_uniq = len(set([tuple(pos) for pos in positions_round]))
        if n_atom_uniq == n_atom:
            return decimals
    raise RuntimeError("Optimal decimals not found.")


def compute_sg_permutations(
    positions: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    lattice: np.ndarray,
    symprec: float = 1e-5,
) -> np.ndarray:
    """Compute permutations of atoms by space group operations in supercell.

    Permutations of atoms of pure translations and coset representatives are
    first computed. Then permutations of atoms of all the given space group
    operations are made as the combitation of these two permutations.

    Parameters
    ----------
    positions : ndarray
        Fractional positions (like SymfcAtoms.scaled_positions) before applying
        the space group operation.
    rotations : ndarray
        Matrix (rotation) parts of space group operations.
        shape=(len(operations), 3, 3), dtype='intc'
    translations : ndarray
        Vector (translation) parts of space group operations.
        shape=(len(operations), 3), dtype='double'
    lattice : ndarray
        Basis vectors in column vectors (like SymfcAtoms.cell.T).
    symprec : float
        Symmetry tolerance of the distance unit.

    Returns
    -------
    perms : ndarray
        shape=(len(translations), len(positions)), dtype='intc', order='C'

    """
    trans_perms = []
    pure_trans = []
    n_atom = positions.shape[0]
    decimals = _find_optimal_decimals(positions)
    sorted_ids, sorted_positions = argsort_positions(positions, decimals=decimals)
    for r, t in zip(rotations, translations):
        if (r != np.eye(3, dtype=int)).any():
            continue
        trans_positions = positions + t
        sorted_trans_ids, sorted_trans_positions = argsort_positions(
            trans_positions,
            decimals=decimals,
        )
        if np.allclose(sorted_trans_positions - sorted_positions, 0.0):
            tp = np.zeros(n_atom, dtype=int)
            tp[sorted_ids] = sorted_trans_ids
        else:
            diffs = positions[None, :, :] - trans_positions[:, None, :]
            diffs -= np.rint(diffs)
            dists = np.linalg.norm(diffs @ lattice.T, axis=2)
            rows, cols = np.where(dists < symprec)
            assert len(positions) == len(np.unique(rows)) == len(np.unique(cols))
            tp = cols[np.argsort(rows)]
        trans_perms.append(tp)
        pure_trans.append(t)
    trans_perms = np.array(trans_perms, dtype=int)
    pure_trans = np.array(pure_trans)

    unique_r = []
    unique_t = []
    r2ur = []
    unique_rotated_positions = []
    for r, t in zip(rotations, translations):
        is_found = False
        for j, ur in enumerate(unique_r):
            if (r == ur).all():
                is_found = True
                r2ur.append(j)
                break
        if not is_found:
            r2ur.append(len(unique_r))
            unique_r.append(r)
            unique_t.append(t)
            unique_rotated_positions.append(positions @ r.T + t)

    unique_rotation_perms = []
    for rotated_positions in unique_rotated_positions:
        diffs = positions[None, :, :] - rotated_positions[:, None, :]
        diffs -= np.rint(diffs)
        dists = np.linalg.norm(diffs @ lattice.T, axis=2)
        rows, cols = np.where(dists < symprec)
        assert len(positions) == len(np.unique(rows)) == len(np.unique(cols))
        unique_rotation_perms.append(cols[np.argsort(rows)])
    unique_rotation_perms = np.array(unique_rotation_perms, dtype=int)

    out = []
    for i, t in enumerate(translations):
        perms = unique_rotation_perms[r2ur[i]]
        lattice_trans = t - unique_t[r2ur[i]]
        diffs = pure_trans - lattice_trans
        diffs -= np.rint(diffs)
        dists = np.linalg.norm(diffs @ lattice.T, axis=1)
        lat_trans_idx = np.where(dists < symprec)
        assert len(lat_trans_idx) == 1
        out.append(trans_perms[lat_trans_idx[0], perms])
    out = np.array(out, dtype="intc", order="C")
    return out


def compute_sg_permutations_stable(
    positions: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    lattice: np.ndarray,
    symprec: float = 1e-5,
) -> np.ndarray:
    """Compute permutations of atoms by space group operations in supercell.

    Permutations of atoms of pure translations and coset representatives are
    first computed. Then permutations of atoms of all the given space group
    operations are made as the combitation of these two permutations.

    Parameters
    ----------
    positions : ndarray
        Fractional positions (like SymfcAtoms.scaled_positions) before applying
        the space group operation.
    rotations : ndarray
        Matrix (rotation) parts of space group operations.
        shape=(len(operations), 3, 3), dtype='intc'
    translations : ndarray
        Vector (translation) parts of space group operations.
        shape=(len(operations), 3), dtype='double'
    lattice : ndarray
        Basis vectors in column vectors (like SymfcAtoms.cell.T).
    symprec : float
        Symmetry tolerance of the distance unit.

    Returns
    -------
    perms : ndarray
        shape=(len(translations), len(positions)), dtype='intc', order='C'

    """
    trans_perms = []
    pure_trans = []
    """Bottleneck part"""
    for r, t in zip(rotations, translations):
        if (r != np.eye(3, dtype=int)).any():
            continue
        trans_positions = positions + t
        diffs = positions[None, :, :] - trans_positions[:, None, :]
        diffs -= np.rint(diffs)
        dists = np.linalg.norm(diffs @ lattice.T, axis=2)
        rows, cols = np.where(dists < symprec)
        assert len(positions) == len(np.unique(rows)) == len(np.unique(cols))
        trans_perms.append(cols[np.argsort(rows)])
        pure_trans.append(t)
    trans_perms = np.array(trans_perms, dtype=int)
    pure_trans = np.array(pure_trans)

    unique_r = []
    unique_t = []
    r2ur = []
    unique_rotated_positions = []
    for r, t in zip(rotations, translations):
        is_found = False
        for j, ur in enumerate(unique_r):
            if (r == ur).all():
                is_found = True
                r2ur.append(j)
                break
        if not is_found:
            r2ur.append(len(unique_r))
            unique_r.append(r)
            unique_t.append(t)
            unique_rotated_positions.append(positions @ r.T + t)

    unique_rotation_perms = []
    for rotated_positions in unique_rotated_positions:
        diffs = positions[None, :, :] - rotated_positions[:, None, :]
        diffs -= np.rint(diffs)
        dists = np.linalg.norm(diffs @ lattice.T, axis=2)
        rows, cols = np.where(dists < symprec)
        assert len(positions) == len(np.unique(rows)) == len(np.unique(cols))
        unique_rotation_perms.append(cols[np.argsort(rows)])
    unique_rotation_perms = np.array(unique_rotation_perms, dtype=int)

    out = []
    for i, t in enumerate(translations):
        perms = unique_rotation_perms[r2ur[i]]
        lattice_trans = t - unique_t[r2ur[i]]
        diffs = pure_trans - lattice_trans
        diffs -= np.rint(diffs)
        dists = np.linalg.norm(diffs @ lattice.T, axis=1)
        lat_trans_idx = np.where(dists < symprec)
        assert len(lat_trans_idx) == 1
        out.append(trans_perms[lat_trans_idx[0], perms])
    out = np.array(out, dtype="intc", order="C")
    return out


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

    def __len__(self):
        """Return number of atoms."""
        return len(self.numbers)

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
