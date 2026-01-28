"""Tests of core functions in eigenvalue problem solvers."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from symfc.utils.eig_tools_core import (
    EigenvectorResult,
    _divide_eigenvectors,
    eigh_projector,
)
from symfc.utils.matrix import root_block_matrix


def _set_projector():
    """Set projector."""
    # Atomic permutations of four atoms
    row = np.repeat(np.arange(12), 4)
    col = np.array([i + j for j in range(3) for i in range(0, 12, 3)])
    col = np.tile(col, 4)
    data = np.full(len(col), 0.25)
    proj = csr_array((data, (row, col)), shape=(12, 12), dtype=float)
    return proj


def _assert_eigvecs(eigvecs):
    """Assert eigenvectors."""
    assert eigvecs.shape[1] == 3
    assert np.linalg.norm(eigvecs[:, 0]) == pytest.approx(1.0)
    assert np.linalg.norm(eigvecs[:, 1]) == pytest.approx(1.0)
    assert np.linalg.norm(eigvecs[:, 2]) == pytest.approx(1.0)
    assert eigvecs[:, 0] @ eigvecs[:, 1] == pytest.approx(0.0)
    assert eigvecs[:, 1] @ eigvecs[:, 2] == pytest.approx(0.0)
    assert eigvecs[:, 2] @ eigvecs[:, 0] == pytest.approx(0.0)

    for i in range(3):
        nonzero = np.where(np.abs(eigvecs[:, i]) > 1e-12)[0]
        assert np.all(np.isclose(eigvecs[:, i][nonzero], eigvecs[nonzero[0], i]))


def test_eigh_projector():
    """Test eigh_projector."""
    proj = _set_projector()
    eigvecs = eigh_projector(proj, verbose=False).eigvecs
    _assert_eigvecs(eigvecs)


def test_eigenvector_result():
    """Test EigenvectorResult."""
    eigvecs = np.random.random((5, 3))

    res = EigenvectorResult(eigvecs)
    assert res.n_eigvecs == 3

    block = res.block_eigvecs
    block_root = root_block_matrix(shape=(5, 3), first_child=block)
    mat = np.random.random((3, 3))
    np.testing.assert_allclose(block_root.dot(mat), eigvecs @ mat)

    res = EigenvectorResult(block_root)
    assert res.n_eigvecs == 3


def test_eigenvector_result_with_compress():
    """Test EigenvectorResult with compression matrix."""
    eigvecs = np.random.random((3, 2))
    compress = np.random.random((5, 3))
    res = EigenvectorResult(eigvecs, compress=compress)
    assert res.n_eigvecs == 2

    block = res.block_eigvecs
    block_root = root_block_matrix(shape=(5, 2), first_child=block)
    np.testing.assert_allclose(block_root.recover(), compress @ eigvecs)


def test_divide_eigenvectors():
    """Test _divide_eigenvectors."""
    eigvecs = np.random.random((10, 5))
    eigvals = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
    res = _divide_eigenvectors(eigvals, eigvecs)
    assert res.eigvecs.shape == (10, 3)
    assert res.cmplt_eigvecs.shape == (10, 2)
    np.testing.assert_allclose(res.cmplt_eigvals, [0.5, 0.5])
