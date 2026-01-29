"""Tests of functions for eigsh."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from symfc.utils.eig_tools_core import find_projector_blocks
from symfc.utils.eig_tools_sparse import (
    CompressionProjector,
    _extract_sparse_projector_data,
    _recover_eigvecs_from_uniq_eigvecs,
    _solve_eigsh,
    eigsh_projector,
)


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


def test_eigsh_projector():
    """Test eigsh_projector."""
    proj = _set_projector()
    eigvecs = eigsh_projector(proj, verbose=False)
    _assert_eigvecs(eigvecs.toarray())


def test_compression_projector1():
    """Test CompressionProjector."""
    row = np.repeat(np.arange(4, 12, 4), 6)
    col = np.repeat(np.arange(4, 12, 4), 6)
    data = np.full(len(col), 1 / 6)
    proj = csr_array((data, (row, col)), shape=(12, 12), dtype=float)

    cp = CompressionProjector(proj)
    assert cp._compr.shape == (12, 2)
    assert cp._compressed_proj.shape == (2, 2)

    eigvecs = eigsh_projector(cp._compressed_proj, verbose=False)
    eigvecs = cp.recover(eigvecs)
    np.testing.assert_allclose((eigvecs @ eigvecs.T - proj).toarray(), 0.0, atol=1e-8)


def test_compression_projector2():
    """Test CompressionProjector."""
    row = col = [0, 2, 5]
    data = [1, 1, 1]
    proj = csr_array((data, (row, col)), shape=(6, 6), dtype=int)

    cp = CompressionProjector(proj)
    assert cp._compressed_proj.shape == (3, 3)
    np.testing.assert_allclose(cp._compressed_proj.toarray(), np.eye(3))
    assert cp._compr.shape == (6, 3)
    compr_ref = np.zeros(cp._compr.shape, dtype=int)
    for icol, irow in enumerate(row):
        compr_ref[irow, icol] = 1
    np.testing.assert_allclose(cp._compr.toarray(), compr_ref)


def test_DataCSR():
    """Test _extract_sparse_projector_data and DataCSR."""
    proj = _set_projector()
    group = find_projector_blocks(proj)
    data = _extract_sparse_projector_data(proj, group)

    for i, (d, label, size) in enumerate(data):
        np.testing.assert_allclose(d, 0.25)
        assert label == i
        assert size == 4


def test_solve_eigsh():
    """Test _solve_eigsh."""
    proj = _set_projector()
    group = find_projector_blocks(proj)
    eigvecs = _solve_eigsh(proj, group)
    assert len(eigvecs.data) == 12
    np.testing.assert_allclose(eigvecs.data, 0.5)


def test_recover_eigvecs_from_uniq_eigvecs():
    """Test _recover_eigvecs_from_uniq_eigvecs."""
    a = np.ones((2, 2))
    b = np.ones((2, 2)) * 3
    size_projector = 6
    group = {
        0: [0, 3],
        1: [1, 4],
        2: [2, 5],
    }
    uniq_eigvecs = {
        "a": [a, [0, 1]],
        "b": [b, [2]],
    }
    eigvecs = _recover_eigvecs_from_uniq_eigvecs(uniq_eigvecs, group, size_projector)
    assert eigvecs[0, 0] == 1.0
    assert eigvecs[0, 1] == 1.0
    assert eigvecs[1, 2] == 1.0
    assert eigvecs[1, 3] == 1.0
    assert eigvecs[2, 4] == 3.0
    assert eigvecs[2, 5] == 3.0
    assert eigvecs[3, 0] == 1.0
    assert eigvecs[3, 1] == 1.0
    assert eigvecs[4, 2] == 1.0
    assert eigvecs[4, 3] == 1.0
    assert eigvecs[5, 4] == 3.0
    assert eigvecs[5, 5] == 3.0
