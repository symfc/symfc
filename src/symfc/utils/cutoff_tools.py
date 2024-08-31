"""Utility functions for introducing zero force constants."""

import itertools
from typing import Union

import numpy as np

from symfc.utils.utils import SymfcAtoms


def apply_zeros(C, zero_ids):
    """Assign zero to matrix elements.

    Use this function, sparse C can become larger by assigning zeros. Zero
    elements should be applied to c_trans and c_perm in constructing them.
    Warning: Slow when zero_ids is large.

    Method 1
    --------
    C[zero_ids,:] = 0

    Method 2
    --------
    C = C.tolil() C[zero_ids, :] = 0 C = C.tocsr()

    """
    for i in zero_ids:
        nonzero_cols = C.getrow(i).nonzero()[1]
        for j in nonzero_cols:
            C[i, j] = 0
    return C


class FCCutoff:
    """Class for introducing cutoff radius."""

    def __init__(self, supercell: SymfcAtoms, cutoff: float = 7.0):
        """Find nonzero FC elements given by cutoff distance.

        For FC3, zero elements are determined by the distance between three
        pairs of atoms.

        Parameters
        ----------
        supercell: SymfcAtoms
            Supercell structure.
        cutoff: float
            Cutoff distance between two atoms.

        """
        self._supercell = supercell
        self._cutoff = cutoff
        self._n_atom = supercell.scaled_positions.shape[0]

        self._distances = self._calc_distances()
        self._neighbors = None
        self._nonzero_fc2 = None
        self._nonzero_fc3 = None
        self._nonzero_fc4 = None

    @property
    def neighbors(self) -> list[np.ndarray]:
        """Neighbor atoms: shape=(n_atom, n_neighbors)."""
        if self._neighbors is None:
            self._neighbors = [
                np.where(self._distances[i] < self._cutoff)[0]
                for i in range(self._n_atom)
            ]
        return self._neighbors

    @property
    def outsides(self) -> list[np.ndarray]:
        """Atoms outside cutoff radius: shape=(n_atom, n_neighbors)."""
        return [
            np.where(self._distances[i] >= self._cutoff)[0] for i in range(self._n_atom)
        ]

    @property
    def distances(self) -> np.ndarray:
        """Minimum distances between atoms: shape=(n_atom, n_atom)."""
        return self._distances

    def combinations1(self) -> np.ndarray:
        """Return combinations with single index ia."""
        return np.array([[i] for i in range(3 * self._n_atom)], dtype=int)

    def combinations2(self) -> np.ndarray:
        """Return combinations with two distinguished indices (ia,jb)."""
        combinations = []
        for jb in range(3 * self._n_atom):
            j = jb // 3
            combs = [
                [3 * i + a, jb]
                for i in self.neighbors[j]
                for a in range(3)
                if 3 * i + a < jb
            ]
            combinations.extend(combs)
        combinations_fc2 = np.array(combinations)
        return combinations_fc2

    def combinations3_all(self) -> np.ndarray:
        """Return combinations with three distinguished indices (ia,jb,kc)."""
        combinations = []
        for kc in range(3 * self._n_atom):
            combs = self.combinations3(kc)
            combinations.extend(combs)
        combinations_fc3 = np.array(combinations)
        return combinations_fc3

    def combinations3(self, kc) -> Union[list, np.ndarray]:
        """Return combinations with three distinguished indices (ia,jb,kc).

        Return only combinations with kc.
        """
        k = kc // 3
        neighbors_N3 = [
            3 * j + b for j in self.neighbors[k] for b in range(3) if 3 * j + b < kc
        ]
        combs = np.array(list(itertools.combinations(neighbors_N3, 2)))
        if len(combs) > 0:
            indices = np.where(
                self.distances[(combs[:, 0] // 3, combs[:, 1] // 3)] < self._cutoff
            )[0]
            combs = combs[indices]
            return np.hstack([combs, np.full((combs.shape[0], 1), kc)])
        return []

    def combinations4_all(self) -> np.ndarray:
        """Return combinations with three distinguished indices (ia,jb,kc,ld)."""
        combinations = []
        for ld in range(3 * self._n_atom):
            combs = self.combinations4(ld)
            combinations.extend(combs)
        combinations_fc4 = np.array(combinations)
        return combinations_fc4

    def combinations4(self, ld) -> Union[list, np.ndarray]:
        """Return combinations with three distinguished indices (ia,jb,kc,ld).

        Return only combinations with kc.

        """
        ll = ld // 3
        neighbors_N3 = [
            3 * j + b for j in self.neighbors[ll] for b in range(3) if 3 * j + b < ld
        ]
        combs = np.array(list(itertools.combinations(neighbors_N3, 3)))
        if len(combs) > 0:
            indices = np.where(
                (self.distances[(combs[:, 0] // 3, combs[:, 1] // 3)] < self._cutoff)
                & (self.distances[(combs[:, 0] // 3, combs[:, 2] // 3)] < self._cutoff)
                & (self.distances[(combs[:, 1] // 3, combs[:, 2] // 3)] < self._cutoff)
            )[0]
            combs = combs[indices]
            return np.hstack([combs, np.full((combs.shape[0], 1), ld)])
        return []

    def nonzero_atomic_indices_fc2(self) -> np.ndarray:
        """Return atomic indices of nonzero FC2.

        Returns
        -------
        nonzero : np.ndarray
            FC2 element is nonzero (True) or zero (False), shape=(NN).

        """
        if self._nonzero_fc2 is not None:
            return self._nonzero_fc2

        self._nonzero_fc2 = nonzero = np.zeros(self._n_atom**2, dtype=bool)
        for i in range(self._n_atom):
            ids = np.array(self.neighbors[i]) + i * self._n_atom
            nonzero[ids] = True
        return self._nonzero_fc2

    def nonzero_atomic_indices_fc3(self) -> np.ndarray:
        """Return atomic indices of nonzero FC3.

        Returns
        -------
        nonzero : np.ndarray
            FC3 element is nonzero (True) or zero (False), shape=(NNN).
        """
        if self._nonzero_fc3 is not None:
            return self._nonzero_fc3

        self._nonzero_fc3 = nonzero = np.zeros(self._n_atom**3, dtype=bool)
        for i in range(self._n_atom):
            jlist = self.neighbors[i]
            combs = np.array(list(itertools.product(jlist, jlist)))
            if len(combs) > 0:
                combs = combs[self.distances[(combs[:, 0], combs[:, 1])] < self._cutoff]
                ids = combs @ np.array([self._n_atom, 1]) + i * self._n_atom**2
                nonzero[ids] = True
        return self._nonzero_fc3

    def nonzero_atomic_indices_fc4(self) -> np.ndarray:
        """Return atomic indices of nonzero FC4.

        Returns
        -------
        nonzero : np.ndarray
            FC4 element is nonzero (True) or zero (False), shape=(NNNN).

        """
        if self._nonzero_fc4 is not None:
            return self._nonzero_fc4

        self._nonzero_fc4 = nonzero = np.zeros(self._n_atom**4, dtype=bool)
        for i in range(self._n_atom):
            jlist = self.neighbors[i]
            combs = np.array(list(itertools.product(*[jlist, jlist, jlist])))
            if len(combs) > 0:
                combs = combs[
                    (self.distances[(combs[:, 0], combs[:, 1])] < self._cutoff)
                    & (self.distances[(combs[:, 0], combs[:, 2])] < self._cutoff)
                    & (self.distances[(combs[:, 1], combs[:, 2])] < self._cutoff)
                ]
                ids = combs @ np.array([self._n_atom**2, self._n_atom, 1])
                ids += i * self._n_atom**3
                nonzero[ids] = True
        return self._nonzero_fc4

    def _calc_distances(self) -> np.ndarray:
        """Calculate minimum distances between atoms.

        This algorithm must be reconsidered.
        (Not applicable to structures with strange lattice shape)

        """
        try:
            import spglib
        except ImportError as exc:
            raise ModuleNotFoundError("Spglib python module was not found.") from exc

        reduced_bases = spglib.niggli_reduce(self._supercell.cell)
        trans_mat_float = self._supercell.cell @ np.linalg.inv(reduced_bases)
        trans_mat = np.rint(trans_mat_float).astype(int)
        assert (np.abs(trans_mat_float - trans_mat) < 1e-8).all()

        scaled_positions = self._supercell.scaled_positions @ trans_mat
        scaled_positions -= np.rint(scaled_positions)
        diff = scaled_positions[:, None, :] - scaled_positions[None, :, :]

        trans = np.array(list(itertools.product(*[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])))
        norms = np.ones((self._n_atom, self._n_atom)) * 1e10
        for t1 in trans:
            t1_tile = np.tile(t1, (self._n_atom, self._n_atom, 1))
            norms_trial = np.linalg.norm((diff - t1_tile) @ reduced_bases, axis=2)
            match = norms_trial < norms
            norms[match] = norms_trial[match]
        self._distances = norms
        return self._distances
