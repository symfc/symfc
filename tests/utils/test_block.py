"""Tests of block matrix functions."""

import numpy as np

from symfc.utils.eig_tools import BlockedMatrix, BlockedMatrixComponent


def test_block_matrix():
    """Test BlockedMatrix."""
    A = np.array([[2, 1], [4, 3]])
    B = np.array([[5, 4], [6, 2]])
    C = np.array([[3, 5], [7, 11]])
    mat = np.block([[A, B], [np.zeros((2, 2)), C]])

    block_A = BlockedMatrixComponent(
        data=A,
        rows=[0, 1],
        col_begin=0,
        col_end=2,
    )
    block_B = BlockedMatrixComponent(
        data=B,
        rows=[0, 1],
        col_begin=2,
        col_end=4,
    )
    block_C = BlockedMatrixComponent(
        data=C,
        rows=[2, 3],
        col_begin=2,
        col_end=4,
    )
    blocks = [block_A, block_B, block_C]
    bm = BlockedMatrix(blocks=blocks, shape=(4, 4))

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
