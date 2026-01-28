"""Tests of eigsh functions using division techniques."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from symfc.utils.eig_tools_division import (
    _calculate_batch_size_division,
    _find_complement_eigenvectors,
    _find_submatrix_eigenvectors,
    _get_descending_order,
    _run_division,
    _should_repeat_division,
    eigsh_projector_division,
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


def test_eigsh_projector_sumrule():
    """Test eigsh_projector_sumrule."""
    proj = _set_projector()
    block = eigsh_projector_sumrule(proj, verbose=False)
    eigvecs = block.recover()
    _assert_eigvecs(eigvecs)


def test_get_descending_order():
    """Test _get_descending_order."""
    group = {
        0: [0, 1],
        1: [2, 3, 4],
        2: [5],
    }
    order = _get_descending_order(group)
    np.testing.assert_equal(order, [1, 0, 2])


def test_eigsh_functions():
    """Test eigsh_projector_division and _run_division."""
    proj = _set_projector()
    res = eigsh_projector_division(proj, verbose=False)
    assert res.block_eigvecs.shape == (12, 3)

    res = _run_division(proj)
    assert res.block_eigvecs.shape == (12, 3)


def test_should_repeat_division():
    """Test _should_repeat_division."""
    assert not _should_repeat_division(100, depth=1)
    assert _should_repeat_division(10000, depth=1)
    assert not _should_repeat_division(10000, depth=5)
    assert not _should_repeat_division(30000, depth=5)
    assert _should_repeat_division(50000, depth=5)


def test_calculate_batch_size_division():
    """Test _calculate_batch_size_division."""
    assert _calculate_batch_size_division(100, depth=1) == 500
    assert _calculate_batch_size_division(10000, depth=1) == 1000
    assert _calculate_batch_size_division(10000, depth=5) == 7692
    assert _calculate_batch_size_division(30000, depth=5) == 20000
    assert _calculate_batch_size_division(50000, depth=5) == 20000


def test_find_submatrix_eigenvectors():
    """Test _find_submatrix_eigenvectors."""
    proj = _set_projector()
    res, cmplt = _find_submatrix_eigenvectors(proj, batch_size=100)
    assert res.block_eigvecs.shape == (12, 3)
    assert cmplt is None

    res, cmplt = _find_submatrix_eigenvectors(proj, batch_size=3)
    assert res.eigvecs is None
    assert cmplt.shape == (12, 12)
    for i, bl in enumerate(cmplt.traverse_data_nodes()):
        np.testing.assert_equal(bl.rows, [3 * i, 3 * i + 1, 3 * i + 2])
        np.testing.assert_allclose(bl.data, np.eye(3))

    res, cmplt = _find_submatrix_eigenvectors(proj, batch_size=1)
    assert res.eigvecs is None
    assert cmplt.shape == (12, 12)
    for i, bl in enumerate(cmplt.traverse_data_nodes()):
        np.testing.assert_equal(bl.rows, [i])
        np.testing.assert_allclose(bl.data, np.eye(1))


def test_find_complement_eigenvectors():
    """Test _find_complement_eigenvectors."""
    proj = _set_projector()
    _, cmplt = _find_submatrix_eigenvectors(proj, batch_size=3)
    res = _find_complement_eigenvectors(proj, cmplt)
    assert res.block_eigvecs.shape == (12, 3)
    assert res.cmplt_eigvals is None
    assert res.cmplt_eigvecs is None
