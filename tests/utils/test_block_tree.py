"""Tests of block matrix functions."""

import numpy as np

from symfc.utils.matrix import BlockMatrixNode


def test_block_matrix():
    """Test BlockMatrix."""
    mat = np.array(
        [
            [5, 4, 3, 2, 7, 4, 0, 0],
            [4, 3, 2, 1, 8, 3, 0, 0],
            [0, 0, 6, 2, 8, 7, 7, 2],
            [0, 0, 1, 4, 3, 3, 1, 8],
            [0, 0, 0, 0, 5, 3, 9, 1],
            [0, 0, 0, 0, 3, 4, 1, 2],
            [0, 0, 0, 0, 7, 8, 0, 0],
            [0, 0, 0, 0, 7, 2, 0, 0],
        ]
    )

    block1_1 = BlockMatrixNode(
        rows=[0, 1],
        col_begin=0,
        col_end=2,
        data=mat[0:2, 0:2],
        index=0,
    )
    block1_2 = BlockMatrixNode(
        rows=[0, 1],
        col_begin=2,
        col_end=4,
        data=mat[0:2, 2:4],
        next_sibling=block1_1,
        index=1,
    )
    block1_3 = BlockMatrixNode(
        rows=[2, 3],
        col_begin=2,
        col_end=4,
        data=mat[2:4, 2:4],
        next_sibling=block1_2,
        index=2,
    )
    block1 = BlockMatrixNode(
        rows=[0, 1, 2, 3],
        col_begin=0,
        col_end=4,
        first_child=block1_3,
        index=3,
    )

    block2_1 = BlockMatrixNode(
        rows=[0, 1],
        col_begin=0,
        col_end=2,
        data=mat[0:2, 4:6],
        index=4,
    )
    block2_2 = BlockMatrixNode(
        rows=[2, 3],
        col_begin=0,
        col_end=2,
        data=mat[2:4, 4:6],
        next_sibling=block2_1,
        index=5,
    )
    block2_3 = BlockMatrixNode(
        rows=[2, 3],
        col_begin=2,
        col_end=4,
        data=mat[2:4, 6:8],
        next_sibling=block2_2,
        index=6,
    )
    block2 = BlockMatrixNode(
        rows=[0, 1, 2, 3],
        col_begin=4,
        col_end=8,
        first_child=block2_3,
        next_sibling=block1,
        index=7,
    )

    block3_1 = BlockMatrixNode(
        rows=[0, 1],
        col_begin=0,
        col_end=2,
        data=mat[4:6, 4:6],
        index=8,
    )
    block3_2 = BlockMatrixNode(
        rows=[0, 1],
        col_begin=2,
        col_end=4,
        data=mat[4:6, 6:8],
        next_sibling=block3_1,
        index=9,
    )
    block3_3 = BlockMatrixNode(
        rows=[2, 3],
        col_begin=0,
        col_end=2,
        data=mat[6:8, 4:6],
        next_sibling=block3_2,
        index=10,
    )
    block3 = BlockMatrixNode(
        rows=[4, 5, 6, 7],
        col_begin=4,
        col_end=8,
        first_child=block3_3,
        next_sibling=block2,
        index=11,
    )

    cblock1 = BlockMatrixNode(
        rows=[0, 1],
        col_begin=0,
        col_end=1,
        data=np.array([[4], [5]]),
        index=101,
    )
    cblock2 = BlockMatrixNode(
        rows=[2, 3],
        col_begin=0,
        col_end=1,
        data=np.array([[1], [3]]),
        next_sibling=cblock1,
        index=102,
    )
    cblock3 = BlockMatrixNode(
        rows=[2, 3],
        col_begin=1,
        col_end=2,
        data=np.array([[2], [4]]),
        next_sibling=cblock2,
        index=103,
    )
    cmplt = BlockMatrixNode(
        rows=[0, 1, 2, 3],
        col_begin=0,
        col_end=2,
        first_child=cblock3,
        root=True,
        index=104,
    )
    eigvecs = np.array([[1, 0], [2, 1]])
    mat[4:8, 0:2] = cmplt.dot(eigvecs)

    block4 = BlockMatrixNode(
        rows=[4, 5, 6, 7],
        col_begin=0,
        col_end=2,
        data=eigvecs,
        next_sibling=block3,
        compress=cmplt,
        index=12,
    )

    bm = BlockMatrixNode(
        rows=[0, 1, 2, 3, 4, 5, 6, 7],
        col_begin=0,
        col_end=8,
        first_child=block4,
        root=True,
        index=13,
    )

    mat2 = np.array([[3, 1], [5, 7], [2, 8], [5, 9], [3, 2], [4, 5], [7, 2], [2, 1]])
    vec2 = np.array([3, 1, 5, 7, 2, 3, 3, 1])

    np.testing.assert_array_equal(bm.recover(), mat)

    np.testing.assert_array_equal(bm.dot(mat2), mat @ mat2)
    np.testing.assert_array_equal(bm.dot(vec2), mat @ vec2)

    np.testing.assert_array_equal(bm.transpose_dot(mat2), mat.T @ mat2)
    np.testing.assert_array_equal(bm.transpose_dot(vec2), mat.T @ vec2)

    # mat3 = np.array(
    #     [
    #         [3, 1, 4, 3, 0, 0, 2, 0],
    #         [5, 7, 3, 1, 2, 1, 0, 6],
    #         [2, 8, 7, 4, 3, 9, 9, 0],
    #         [5, 9, 0, 2, 0, 0, 0, 2],
    #         [2, 3, 4, 9, 1, 0, 1, 2],
    #         [0, 2, 0, 2, 0, 8, 0, 3],
    #         [5, 9, 0, 2, 0, 0, 2, 2],
    #         [0, 0, 0, 2, 2, 0, 9, 1],
    #     ]
    # )
    # np.testing.assert_array_equal(
    #     bm.compress_matrix(mat3),
    #     mat.T @ mat3 @ mat,
    # )
    # np.testing.assert_array_equal(
    #     bm.compress_matrix(csr_array(mat3)),
    #     mat.T @ mat3 @ mat,
    # )

    perm = np.array([2, 1, 0, 4, 3, 6, 5, 7])
    bm.rows = perm
    bm.set_root_indices()
    mat = mat[perm]
    np.testing.assert_array_equal(bm.dot(mat2), mat @ mat2)
