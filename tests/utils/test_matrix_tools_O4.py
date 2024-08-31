"""Tests of functions in matrix_tools_O4."""

import numpy as np

from symfc.utils.matrix_tools_O4 import (
    N3N3N3N3_to_NNNNand3333,
)


def test_N3N3N3N3_to_NNNNand3333():
    """Test N3N33_to_NNNand333."""
    N = 3
    combs = np.array([[0, 1, 2, 5], [2, 4, 6, 8], [3, 4, 5, 8]])
    vecNNNN, vec3333 = N3N3N3N3_to_NNNNand3333(combs, N)
    np.testing.assert_allclose(vecNNNN, [1, 17, 41])
    np.testing.assert_allclose(vec3333, [17, 65, 17])
