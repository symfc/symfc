"""Tests of API for eigh and eigsh."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from symfc import eigh, eigsh


def _assert_four_atoms_eigvecs(eigvecs: np.ndarray):
    """Test eigenvectors of four-atom permutation projector."""
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


def test_eigsh_and_eigh():
    """Test eigsh."""
    # Atomic permutations of four atoms
    row = np.repeat(np.arange(12), 4)
    col = np.array([i + j for j in range(3) for i in range(0, 12, 3)])
    col = np.tile(col, 4)
    data = np.full(len(col), 0.25)
    proj = csr_array((data, (row, col)), shape=(12, 12), dtype=float)

    eigvecs = eigsh(proj, log_level=0)
    eigvecs = eigvecs.toarray()
    _assert_four_atoms_eigvecs(eigvecs)

    eigvecs = eigsh(proj, is_large_block=True, log_level=0)
    eigvecs = eigvecs.recover()
    _assert_four_atoms_eigvecs(eigvecs)

    eigvecs = eigh(proj.toarray(), log_level=0)
    _assert_four_atoms_eigvecs(eigvecs)
