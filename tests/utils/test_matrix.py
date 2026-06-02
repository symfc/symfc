"""Tests of matrix utils."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.matrix import (
    blocked_product,
    blocked_triple_product,
    dot_product_sparse,
)


def test_dot_product_sparse():
    """Test dot_product_sparse."""
    mat1 = np.random.random((2, 3))
    mat2 = np.random.random((3, 2))
    true = mat1 @ mat2

    prod = dot_product_sparse(mat1, mat2)
    np.testing.assert_allclose(prod, true)

    mat1 = csr_array(mat1)
    prod = dot_product_sparse(mat1, mat2)
    np.testing.assert_allclose(prod, true)

    mat2 = csr_array(mat2)
    np.testing.assert_allclose(prod, true)

    prod = dot_product_sparse(mat1, mat2).toarray()
    np.testing.assert_allclose(prod, true)

    prod = dot_product_sparse(mat1, mat2, use_mkl=True).toarray()
    np.testing.assert_allclose(prod, true)


def test_blocked_triple_product():
    """Test blocked_triple_product."""
    mat = np.random.random((5, 5))
    cp1 = np.random.random((5, 6))
    cp2 = np.random.random((5, 3))
    cp = np.hstack((cp1, cp2))
    true = cp.T @ mat @ cp

    mat = csr_array(mat)
    cp1 = csr_array(cp1)
    cp2 = csr_array(cp2)
    cpt_array = [cp1, cp2]
    mat = blocked_triple_product(cpt_array, mat)

    np.testing.assert_allclose(mat.toarray(), true)


def test_blocked_product():
    """Test blocked_product."""
    mat = np.random.random((5, 3))
    cp1 = np.random.random((6, 3))
    cp2 = np.random.random((6, 2))
    cp = np.hstack((cp1, cp2))
    true = cp @ mat

    mat = csr_array(mat)
    cp1 = csr_array(cp1)
    cp2 = csr_array(cp2)
    cpt_array = [cp1, cp2]
    mat = blocked_product(cpt_array, mat)

    np.testing.assert_allclose(mat.toarray(), true)
