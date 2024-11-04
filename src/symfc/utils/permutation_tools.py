"""Permutation utility functions."""

import numpy as np
import scipy
from scipy.sparse import csr_array


def construct_basis_from_orbits(orbits: np.ndarray):
    """Transform orbits into basis matrix."""
    size_full = len(orbits)
    nonzero = orbits != -1
    if not np.all(nonzero):
        orbits = orbits[nonzero]
        nonzero_map = np.ones(size_full, dtype="int") * -1
        nonzero_map[nonzero] = np.arange(len(orbits))
        orbits = nonzero_map[orbits]

    size1 = len(orbits)
    orbits = csr_array(
        (np.ones(size1, dtype=bool), (np.arange(size1), orbits)),
        shape=(size1, size1),
        dtype=bool,
    )

    n_col, cols = scipy.sparse.csgraph.connected_components(orbits)
    key, cnt = np.unique(cols, return_counts=True)
    values = np.reciprocal(np.sqrt(cnt))

    rows = np.where(nonzero)[0]
    c_pt = csr_array(
        (values[cols], (rows, cols)),
        shape=(size_full, n_col),
        dtype="double",
    )
    return c_pt
