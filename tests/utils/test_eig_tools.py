"""Tests of eigsh functions."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from symfc.utils.eig_tools import (
    eigh_projector,
    eigsh_projector,
    eigsh_projector_sumrule,
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
    eigvecs = eigvecs.toarray()
    _assert_eigvecs(eigvecs)


def test_eigsh_projector_sumrule():
    """Test eigsh_projector_sumrule."""
    proj = _set_projector()
    eigvecs = eigsh_projector_sumrule(proj, verbose=False)
    eigvecs = eigvecs.recover()
    _assert_eigvecs(eigvecs)


def test_eigh_projector():
    """Test eigh_projector."""
    proj = _set_projector()
    eigvecs = eigh_projector(proj, verbose=False)
    _assert_eigvecs(eigvecs)
