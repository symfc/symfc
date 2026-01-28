"""Tests of eigsh functions."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from symfc.utils.eig_tools_sparse import (
    CompressionProjector,
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


def test_compression_projector():
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


# def test_compr_projector():
#     """Test compr_projector."""
#     row = col = [0, 2, 5]
#     data = [1, 1, 1]
#     proj = csr_array((data, (row, col)), shape=(6, 6), dtype=int)
#
#     proj_rev, compr = _compr_projector(proj)
#     assert proj_rev.shape == (3, 3)
#     np.testing.assert_allclose(proj_rev.toarray(), np.eye(3))
#     assert compr.shape == (6, 3)
#     compr_ref = np.zeros(compr.shape, dtype=int)
#     for icol, irow in enumerate(row):
#         compr_ref[irow, icol] = 1
#     np.testing.assert_allclose(compr.toarray(), compr_ref)
