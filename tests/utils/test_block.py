"""Tests of block matrix functions."""

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.matrix import BlockMatrix, BlockMatrixComponent


def test_block_matrix():
    """Test BlockMatrix."""
    A = np.array([[2, 1], [4, 3]])
    B = np.array([[5, 4], [6, 2]])
    C = np.array([[3, 5], [7, 11]])
    mat = np.block([[A, B], [np.zeros((2, 2)), C]])

    block_A = BlockMatrixComponent(
        data=A,
        rows=[0, 1],
        col_begin=0,
        col_end=2,
    )
    block_B = BlockMatrixComponent(
        data=B,
        rows=[0, 1],
        col_begin=2,
        col_end=4,
    )
    block_C = BlockMatrixComponent(
        data=C,
        rows=[2, 3],
        col_begin=2,
        col_end=4,
    )
    blocks = [block_A, block_B, block_C]
    bm = BlockMatrix(blocks=blocks, shape=(4, 4))

    mat2 = np.array([[3, 1], [5, 7], [2, 8], [5, 9]])
    vec2 = np.array([3, 1, 5, 7])

    np.testing.assert_array_equal(bm.dot(vec2), mat @ vec2)
    np.testing.assert_array_equal(bm.transpose_dot(vec2), mat.T @ vec2)
    np.testing.assert_array_equal(bm.dot(vec2, left=True), vec2 @ mat)
    np.testing.assert_array_equal(bm.transpose_dot(vec2, left=True), vec2 @ mat.T)

    np.testing.assert_array_equal(bm.dot(mat2), mat @ mat2)
    np.testing.assert_array_equal(bm.transpose_dot(mat2), mat.T @ mat2)
    np.testing.assert_array_equal(bm.dot(mat2.T, left=True), mat2.T @ mat)
    np.testing.assert_array_equal(
        bm.transpose_dot(mat2.T, left=True),
        mat2.T @ mat.T,
    )

    mat3 = np.array([[3, 1, 4, 3], [5, 7, 3, 1], [2, 8, 7, 4], [5, 9, 0, 2]])
    np.testing.assert_array_equal(bm.compress_matrix(mat3), mat.T @ mat3 @ mat)
    np.testing.assert_array_equal(
        bm.compress_csr_matrix(csr_array(mat3)),
        mat.T @ mat3 @ mat,
    )


def test_compressed_block_matrix1():
    """Test compressed matrix in BlockMatrix."""
    A = mat = np.array([[2, 1], [4, 3], [3, 1], [5, 7]])
    comprA = np.array(
        [
            [4, 3, 2, 1],
            [6, 9, 8, 4],
            [2, 1, 6, 8],
            [9, 2, 3, 7],
            [7, 3, 0, 9],
            [0, 8, 1, 3],
        ]
    )
    compr1 = BlockMatrixComponent(
        data=comprA,
        rows=np.arange(6),
        col_begin=0,
        col_end=4,
    )
    compr = BlockMatrix(blocks=[compr1], shape=(6, 4))
    full_mat = compr.dot(mat)

    block_A = BlockMatrixComponent(
        data=A,
        rows=np.arange(6),
        col_begin=0,
        col_end=2,
        compress=compr,
    )
    blocks = [block_A]
    bm = BlockMatrix(blocks=blocks, shape=(6, 2))

    mat2 = np.array([[2, 8], [5, 9]])
    np.testing.assert_array_equal(bm.dot(mat2), full_mat @ mat2)
    np.testing.assert_array_equal(
        bm.transpose_dot(mat2.T, left=True), mat2.T @ full_mat.T
    )

    vec2 = np.array([3, 1])
    np.testing.assert_array_equal(bm.dot(vec2), full_mat @ vec2)
    np.testing.assert_array_equal(bm.transpose_dot(vec2, left=True), vec2 @ full_mat.T)


def test_compressed_block_matrix2():
    """Test complex compressed matrix in BlockMatrix."""
    A = np.array([[2, 1], [4, 3], [3, 1], [5, 7]])
    comprA = np.array(
        [
            [4, 3, 2, 1],
            [6, 9, 8, 4],
            [2, 1, 6, 8],
            [9, 2, 3, 7],
            [7, 3, 0, 9],
            [0, 8, 1, 3],
        ]
    )
    compr1 = BlockMatrixComponent(
        data=comprA,
        rows=np.arange(6),
        col_begin=0,
        col_end=4,
    )
    full_matA = comprA @ A
    comprA = BlockMatrix(blocks=[compr1], shape=(6, 4))

    B = np.array([[2, 1], [4, 3]])
    comprB = np.array(
        [
            [4, 3],
            [6, 9],
            [2, 1],
            [9, 2],
            [7, 3],
            [0, 8],
        ]
    )
    compr2 = BlockMatrixComponent(
        data=comprB,
        rows=np.arange(6),
        col_begin=0,
        col_end=2,
    )
    full_matB = comprB @ B
    comprB = BlockMatrix(blocks=[compr2], shape=(6, 2))

    full_mat = np.block([full_matA, full_matB])

    block_A = BlockMatrixComponent(
        data=A,
        rows=np.arange(6),
        col_begin=0,
        col_end=2,
        compress=comprA,
    )
    block_B = BlockMatrixComponent(
        data=B,
        rows=np.arange(6),
        col_begin=2,
        col_end=4,
        compress=comprB,
    )
    blocks = [block_A, block_B]
    bm = BlockMatrix(blocks=blocks, shape=(6, 4))

    mat2 = np.array([[2, 8], [5, 9], [3, 2], [7, 5]])
    np.testing.assert_array_equal(bm.dot(mat2), full_mat @ mat2)
    np.testing.assert_array_equal(
        bm.transpose_dot(mat2.T, left=True),
        mat2.T @ full_mat.T,
    )
    mat2 = np.array([[2, 8], [5, 9], [3, 2], [7, 5], [8, 4], [6, 3]])
    np.testing.assert_array_equal(bm.transpose_dot(mat2), full_mat.T @ mat2)
    np.testing.assert_array_equal(bm.dot(mat2.T, left=True), mat2.T @ full_mat)

    vec2 = np.array([3, 1, 4, 3])
    np.testing.assert_array_equal(bm.dot(vec2), full_mat @ vec2)
    np.testing.assert_array_equal(bm.transpose_dot(vec2, left=True), vec2 @ full_mat.T)

    vec2 = np.array([3, 1, 4, 3, 2, 3])
    np.testing.assert_array_equal(bm.transpose_dot(vec2), full_mat.T @ vec2)
    np.testing.assert_array_equal(bm.dot(vec2, left=True), vec2 @ full_mat)
