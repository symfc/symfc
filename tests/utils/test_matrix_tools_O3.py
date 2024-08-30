"""Tests of functions in matrix_tools_O3."""

import numpy as np

from symfc.utils.matrix_tools_O3 import (
    N3N3N3_to_NNNand333,
)


def test_N3N3N3_to_NNNand333():
    """Test N3N33_to_NNNand333."""
    N = 3
    combs = np.array([[0, 1, 2], [2, 4, 6], [3, 5, 8]])
    vecNNN, vec333 = N3N3N3_to_NNNand333(combs, N)
    np.testing.assert_allclose(vecNNN, [0, 5, 14])
    np.testing.assert_allclose(vec333, [5, 21, 8])


test_N3N3N3_to_NNNand333()
