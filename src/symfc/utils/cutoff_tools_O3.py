"""Utility functions for introducing 3rd order zero force constants."""

import itertools

import numpy as np


def apply_zeros(C, zero_ids):
    """Assign zero to matrix elements.

    Use this function, sparse C can become larger by assigning zeros.
    Zero elements should be applied to c_trans and c_perm in constructing them.
    Warning: Slow when zero_ids is large.
    """
    """Method 1
    C[zero_ids,:] = 0
    """
    """Method 2
    C = C.tolil()
    C[zero_ids, :] = 0
    C = C.tocsr()
    """
    for i in zero_ids:
        nonzero_cols = C.getrow(i).nonzero()[1]
        for j in nonzero_cols:
            C[i, j] = 0
    return C


class FCCutoffO3:
    """Class for introducing cutoff radius to fc3."""

    def __init__(self, supercell, cutoff=7.0):
        """Find nonzero FC3 elements inside a sphere given by cutoff radius.

        Parameters
        ----------
        supercell: SymfcAtoms or PhonopyAtoms
        cutoff: Cutoff radius (in angstrom)
        """
        self.__supercell = supercell
        self.__cutoff = cutoff
        self.__n_atom = supercell.scaled_positions.shape[0]

        self.__distances = self.__calc_distances()
        self.__neighbors = None
        self.__nonzero = None

    def __calc_distances(self):
        """Calculate minimum distances between atoms.

        This algorithm must be reconsidered.
        (Not applicable to structures with strange lattice shape)
        """
        scaled_positions = self.__supercell.scaled_positions
        diff = scaled_positions[:, None, :] - scaled_positions[None, :, :]

        trans = np.array(list(itertools.product(*[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])))
        norms = np.ones((self.__n_atom, self.__n_atom)) * 1e10
        for t1 in trans:
            t1_tile = np.tile(t1, (self.__n_atom, self.__n_atom, 1))
            norms_trial = np.linalg.norm(
                (diff - t1_tile) @ self.__supercell.cell, axis=2
            )
            match = norms_trial < norms
            norms[match] = norms_trial[match]
        self.__distances = norms
        return self.__distances

    def combinations1(self):
        """Return combinations of FC3 with single index ia."""
        combinations = np.array(
            [[i, i, i] for i in range(3 * self.__n_atom)], dtype=int
        )
        return combinations

    def combinations2(self):
        """Return combinations of FC3 with two distinguished indices (ia,ia,jb)."""
        combinations = []
        for jb in range(3 * self.__n_atom):
            j = jb // 3
            combs = [
                [3 * i + a, jb]
                for i in self.neighbors[j]
                for a in range(3)
                if 3 * i + a < jb
            ]
            combinations.extend(combs)
        return np.array(combinations)

    def combinations3_all(self, dtype="uint16"):
        """Return combinations of FC3 with three distinguished indices (ia,jb,kc)."""
        combinations = []
        for kc in range(3 * self.__n_atom):
            combs = self.combinations3(kc)
            combinations.extend(combs)
        return np.array(combinations).astype(dtype, copy=False)

    def combinations3(self, kc, dtype="uint16"):
        """Return combinations of FC3 with three distinguished indices (ia,jb,kc).

        Return only combinations with kc.
        """
        k = kc // 3
        neighbors_N3 = [
            3 * j + b for j in self.neighbors[k] for b in range(3) if 3 * j + b < kc
        ]
        combs = np.array(list(itertools.combinations(neighbors_N3, 2)))
        if len(combs) > 0:
            """Algorithm to eliminate FC3 should be reconsidered."""
            indices = np.where(
                self.distances[(combs[:, 0] // 3, combs[:, 1] // 3)] < self.__cutoff
            )[0]
            combs = combs[indices]
            return np.hstack([combs, np.full((combs.shape[0], 1), kc)]).astype(
                dtype, copy=False
            )
        return []

    def nonzero_atomic_indices(self):
        """Return atomic indices of nonzero FC3.

        Return
        ------
        nonzero: FC3 element is nonzero (True) or zero (False), shape=(NNN).
        """
        if self.__nonzero is not None:
            return self.__nonzero

        self.__nonzero = nonzero = np.zeros(self.__n_atom**3, dtype=bool)
        for i in range(self.__n_atom):
            jlist = self.neighbors[i]
            combs = np.array(list(itertools.product(jlist, jlist)))
            if len(combs) > 0:
                combs = combs[
                    self.distances[(combs[:, 0], combs[:, 1])] < self.__cutoff
                ]
                ids = combs @ np.array([self.__n_atom, 1]) + i * self.__n_atom**2
                nonzero[ids] = True
        return self.__nonzero

    @property
    def neighbors(self):
        """Neighbor atoms: shape=(n_atom, n_neighbors)."""
        if self.__neighbors is None:
            self.__neighbors = [
                np.where(self.__distances[i] < self.__cutoff)[0]
                for i in range(self.__n_atom)
            ]
        return self.__neighbors

    @property
    def outsides(self):
        """Atoms outside cutoff radius: shape=(n_atom, n_neighbors)."""
        return [
            np.where(self.__distances[i] >= self.__cutoff)[0]
            for i in range(self.__n_atom)
        ]

    @property
    def distances(self):
        """Minimum distances between atoms: shape=(n_atom, n_atom)."""
        return self.__distances

    def find_zero_indices(self):
        """Find zero FC3 elements outside a sphere given by cutoff radius.

        Deprecated.
        The number of shells may be better than cutoff radius,
        because the minimal cutoff radius is strongly system-dependent.
        """
        NN27 = self.__n_atom**2 * 27
        N27 = self.__n_atom * 27

        """Algorithm to eliminate FC3 should be reconsidered."""
        zero_atom_indices = np.array(np.where(self.__distances >= self.__cutoff)).T
        zero_atom_indices = zero_atom_indices @ np.array([NN27, N27])

        zero_indices = zero_atom_indices[:, None] + np.arange(N27)[None, :]
        zero_indices = zero_indices.reshape(-1)
        return zero_indices
