"""Tests of graph."""

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.graph import connected_components


def test_connected_components():
    """Test connected_component using DFS."""
    mat = np.array(
        [
            [5, 4, 0, 2, 0, 0, 0, 0],
            [4, 3, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 3, 0, 1],
            [0, 0, 0, 0, 3, 4, 1, 2],
            [0, 0, 0, 0, 7, 8, 1, 0],
            [0, 0, 0, 0, 7, 2, 0, 1],
        ]
    )
    mat = csr_array(mat + mat.T)

    group = connected_components(mat)
    np.testing.assert_array_equal(group[0], [0, 1, 3])
    np.testing.assert_array_equal(group[1], [2])
    np.testing.assert_array_equal(group[2], [4, 5, 6, 7])
