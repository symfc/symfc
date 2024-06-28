"""Utility functions for introducing zero force constants."""

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


class FCCutoff:
    """Class for introducing cutoff radius."""

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
        self.__nonzero_fc2 = None
        self.__nonzero_fc3 = None
        self.__nonzero_fc4 = None

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
        """Return combinations with single index ia."""
        return np.array([[i] for i in range(3 * self.__n_atom)], dtype=int)

    def combinations2(self):
        """Return combinations with two distinguished indices (ia,jb)."""
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
        combinations_fc2 = np.array(combinations)
        return combinations_fc2

    def combinations3_all(self):
        """Return combinations with three distinguished indices (ia,jb,kc)."""
        combinations = []
        for kc in range(3 * self.__n_atom):
            combs = self.combinations3(kc)
            combinations.extend(combs)
        combinations_fc3 = np.array(combinations)
        return combinations_fc3

    def combinations3(self, kc):
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
                self.distances[(combs[:, 0] // 3, combs[:, 1] // 3)] < self.__cutoff
            )[0]
            combs = combs[indices]
            return np.hstack([combs, np.full((combs.shape[0], 1), kc)])
        return []

    def combinations4_all(self):
        """Return combinations with three distinguished indices (ia,jb,kc,ld)."""
        combinations = []
        for ld in range(3 * self.__n_atom):
            combs = self.combinations4(ld)
            combinations.extend(combs)
        combinations_fc4 = np.array(combinations)
        return combinations_fc4

    def combinations4(self, ld):
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
                (self.distances[(combs[:, 0] // 3, combs[:, 1] // 3)] < self.__cutoff)
                & (self.distances[(combs[:, 0] // 3, combs[:, 2] // 3)] < self.__cutoff)
                & (self.distances[(combs[:, 1] // 3, combs[:, 2] // 3)] < self.__cutoff)
            )[0]
            combs = combs[indices]
            return np.hstack([combs, np.full((combs.shape[0], 1), ld)])
        return []

    def nonzero_atomic_indices_fc2(self):
        """Return atomic indices of nonzero FC2.

        Return
        ------
        nonzero: FC2 element is nonzero (True) or zero (False), shape=(NN).
        """
        if self.__nonzero_fc2 is not None:
            return self.__nonzero_fc2

        self.__nonzero_fc2 = nonzero = np.zeros(self.__n_atom**2, dtype=bool)
        for i in range(self.__n_atom):
            ids = np.array(self.neighbors[i]) + i * self.__n_atom
            nonzero[ids] = True
        return self.__nonzero_fc2

    def nonzero_atomic_indices_fc3(self):
        """Return atomic indices of nonzero FC3.

        Return
        ------
        nonzero: FC3 element is nonzero (True) or zero (False), shape=(NNN).
        """
        if self.__nonzero_fc3 is not None:
            return self.__nonzero_fc3

        self.__nonzero_fc3 = nonzero = np.zeros(self.__n_atom**3, dtype=bool)
        for i in range(self.__n_atom):
            jlist = self.neighbors[i]
            combs = np.array(list(itertools.product(jlist, jlist)))
            if len(combs) > 0:
                combs = combs[
                    self.distances[(combs[:, 0], combs[:, 1])] < self.__cutoff
                ]
                ids = combs @ np.array([self.__n_atom, 1]) + i * self.__n_atom**2
                nonzero[ids] = True
        return self.__nonzero_fc3

    def nonzero_atomic_indices_fc4(self):
        """Return atomic indices of nonzero FC4.

        Return
        ------
        nonzero: FC4 element is nonzero (True) or zero (False), shape=(NNNN).
        """
        if self.__nonzero_fc4 is not None:
            return self.__nonzero_fc4

        self.__nonzero_fc4 = nonzero = np.zeros(self.__n_atom**4, dtype=bool)
        for i in range(self.__n_atom):
            jlist = self.neighbors[i]
            combs = np.array(list(itertools.product(*[jlist, jlist, jlist])))
            if len(combs) > 0:
                combs = combs[
                    (self.distances[(combs[:, 0], combs[:, 1])] < self.__cutoff)
                    & (self.distances[(combs[:, 0], combs[:, 2])] < self.__cutoff)
                    & (self.distances[(combs[:, 1], combs[:, 2])] < self.__cutoff)
                ]
                ids = combs @ np.array([self.__n_atom**2, self.__n_atom, 1])
                ids += i * self.__n_atom**3
                nonzero[ids] = True
        return self.__nonzero_fc4

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
