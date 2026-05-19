"""Tests of block matrix functions."""

import numpy as np
from scipy.sparse import csr_array

from symfc.eig_solvers.matrix import (
    BlockMatrixNode,
    block_matrix_sandwich,
    block_matrix_sandwich_sym,
    link_block_matrix_nodes,
    root_block_matrix,
)


def test_root_indices_in_block_matrix():
    """Test root indices."""
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

    sliced_data = mat[0:2, 0:2]
    block1_1 = BlockMatrixNode(
        rows=[0, 1],
        col_begin=0,
        col_end=2,
        data=sliced_data,
        index=0,
    )
    mat_prod = block1_1.dot(sliced_data)
    np.testing.assert_allclose(mat_prod, sliced_data @ sliced_data)

    sliced_data = mat[0:2, 2:4]
    block1_2 = BlockMatrixNode(
        rows=[0, 1],
        col_begin=2,
        col_end=4,
        data=sliced_data,
        next_sibling=block1_1,
        index=1,
    )

    sliced_data = mat[2:4, 2:4]
    block1_3 = BlockMatrixNode(
        rows=[2, 3],
        col_begin=2,
        col_end=4,
        data=sliced_data,
        next_sibling=block1_2,
        index=2,
    )

    assert block1_1.root
    assert block1_2.root
    assert block1_3.root

    block1 = BlockMatrixNode(
        rows=[4, 5, 6, 7],
        col_begin=3,
        col_end=7,
        first_child=block1_3,
        index=3,
    )
    block1.print_nodes()

    # shape and root
    assert block1.root
    assert not block1_1.root
    assert not block1_2.root
    assert not block1_3.root
    assert block1.data_shape == (4, 4)
    assert block1.shape == (8, 7)
    assert block1.n_eigvecs == 4

    # recover
    np_matrix = block1.recover()
    assert np_matrix.shape == (8, 7)
    np.testing.assert_allclose(np_matrix[4:, 3:], mat[:4, :4])

    # dot
    mat_ones = np.ones((8, 4))
    mat_prod = block1.dot(mat_ones)
    true = mat[:4, :4] @ np.ones((4, 4))
    np.testing.assert_allclose(mat_prod[4:], true)

    # @ operator
    mat_ones = np.ones((8, 4))
    mat_prod = block1 @ mat_ones
    true = mat[:4, :4] @ np.ones((4, 4))
    np.testing.assert_allclose(mat_prod[4:], true)

    # block.T @ mat
    mat_ones = np.ones((8, 4))
    mat_prod = block1.T @ mat_ones
    true = mat[:4, :4].T @ np.ones((4, 4))
    np.testing.assert_allclose(mat_prod[3:], true)


def test_compressed_block_matrix():
    """Test compressed block matrix."""
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
    assert cblock1.root
    assert cblock2.root
    assert cblock3.root

    cmplt = BlockMatrixNode(
        rows=[0, 1, 2, 3],
        col_begin=0,
        col_end=2,
        first_child=cblock3,
        index=104,
    )

    # shape and root
    assert cmplt.root
    assert not cblock1.root
    assert not cblock2.root
    assert not cblock3.root
    assert cmplt.shape == (4, 2)
    assert cmplt.data_shape == (4, 2)

    eigvecs = np.array([[1, 0], [2, 1]])
    block = BlockMatrixNode(
        rows=[0, 1, 2, 3],
        col_begin=0,
        col_end=2,
        data=eigvecs,
        compress=cmplt,
        index=12,
    )

    # recover
    true = cmplt @ eigvecs
    np_matrix = block.recover()
    assert np_matrix.shape == (4, 2)
    np.testing.assert_allclose(np_matrix, true)

    # block @ mat and block.T @ mat
    mat_one = np.ones((2, 3))
    np.testing.assert_allclose(block @ mat_one, true @ mat_one)
    mat_one = np.ones((4, 3))
    np.testing.assert_allclose(block.T @ mat_one, true.T @ mat_one)

    block = BlockMatrixNode(
        rows=[0, 3, 1, 2],
        col_begin=0,
        col_end=2,
        data=eigvecs,
        compress=cmplt,
        eigvals=[625, 36],
        index=12,
    )

    # recover
    true = cmplt @ eigvecs
    np_matrix = block.recover()
    assert np_matrix.shape == (4, 2)
    np.testing.assert_allclose(np_matrix[[0, 3, 1, 2]], true)


def test_compress_method():
    """Test compress_matrix method."""
    eigvecs = np.array(
        [[1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0], [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)]]
    ).T
    block = BlockMatrixNode(
        rows=[0, 1, 2, 3],
        col_begin=0,
        col_end=2,
        data=eigvecs,
        eigvals=[1, 1],
        index=12,
    )

    mat1 = eigvecs @ eigvecs.T
    compr_block1 = block.compress_matrix(mat1)
    assert compr_block1.shape == (2, 2)

    compr_block2 = block.compress_matrix(mat1, disable_simple_products=True)
    assert compr_block2.shape == (2, 2)
    np.testing.assert_allclose(compr_block1, compr_block2)

    # for sparse matrix input
    mat1 = csr_array(eigvecs @ eigvecs.T)
    compr_block1 = block.compress_matrix(mat1)
    assert compr_block1.shape == (2, 2)
    np.testing.assert_allclose(compr_block1, np.eye(2))


def test_large_block_matrix():
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

    bm = BlockMatrixNode(shape=(8, 8), first_child=block4, index=13)
    bm = root_block_matrix(shape=(8, 8), first_child=block4)

    mat2 = np.array([[3, 1], [5, 7], [2, 8], [5, 9], [3, 2], [4, 5], [7, 2], [2, 1]])
    vec2 = np.array([3, 1, 5, 7, 2, 3, 3, 1])

    np.testing.assert_array_equal(bm.recover(), mat)

    np.testing.assert_array_equal(bm @ mat2, mat @ mat2)
    np.testing.assert_array_equal(bm @ vec2, mat @ vec2)

    np.testing.assert_array_equal(bm.T @ mat2, mat.T @ mat2)
    np.testing.assert_array_equal(bm.T @ vec2, mat.T @ vec2)

    # Row index permutation
    perm = np.array([2, 1, 0, 4, 3, 6, 5, 7])
    bm.change_row_indices(mapping=perm)
    mat = mat[perm]
    np.testing.assert_array_equal(bm @ mat2, mat @ mat2)

    # Test attributes
    bm.print_nodes()
    bm.traverse_data_nodes()

    np.testing.assert_equal(bm.rows, perm)
    assert bm.col_begin == 0
    assert bm.col_end == 8
    assert bm.data_shape == (8, 8)
    assert bm.shape == (8, 8)
    assert bm.n_eigvecs == 8
    assert bm.next_sibling is None
    assert isinstance(bm.first_child, BlockMatrixNode)
    assert bm.compress is None
    assert bm.data is None
    assert bm.root
    assert bm.eigvals is None

    # Rest indices
    np.testing.assert_allclose(
        bm @ np.ones(8), [32.0, 21.0, 25.0, 22.0, 20.0, 22.0, 15.0, 24.0]
    )
    bm.reset_indices()
    np.testing.assert_allclose(
        bm @ np.ones(8), [25.0, 21.0, 32.0, 20.0, 22.0, 15.0, 22.0, 24.0]
    )


def test_link_block_matrices():
    """Test link_block_matrix_nodes and block_matrix_sandwich."""
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

    block1_1 = BlockMatrixNode(shape=(2, 2), data=mat[0:2, 0:2])
    block1_2 = BlockMatrixNode(shape=(2, 2), data=mat[0:2, 2:4])
    block1_2 = link_block_matrix_nodes(block1_2, block1_1, rows=[0, 1], col_begin=2)
    block1_3 = BlockMatrixNode(shape=(2, 2), data=mat[2:4, 2:4])
    block1_3 = link_block_matrix_nodes(block1_3, block1_2, rows=[2, 3], col_begin=2)
    block1 = root_block_matrix(shape=(4, 4), first_child=block1_3)

    block2_1 = BlockMatrixNode(shape=(2, 2), data=mat[0:2, 4:6])
    block2_2 = BlockMatrixNode(shape=(2, 2), data=mat[2:4, 4:6])
    block2_2 = link_block_matrix_nodes(block2_2, block2_1, rows=[2, 3], col_begin=0)
    block2_3 = BlockMatrixNode(shape=(2, 2), data=mat[2:4, 6:8])
    block2_3 = link_block_matrix_nodes(block2_3, block2_2, rows=[2, 3], col_begin=2)
    block2 = root_block_matrix(shape=(4, 4), first_child=block2_3)
    block2 = link_block_matrix_nodes(block2, block1, rows=[0, 1, 2, 3], col_begin=4)
    bm = root_block_matrix(shape=(8, 8), first_child=block2)

    mat1 = np.zeros((8, 8))
    mat1[:4] = mat[:4]
    np.testing.assert_allclose(bm.recover(), mat1, atol=1e-7)

    sand = block_matrix_sandwich(bm, bm, mat)
    true = mat1.T @ mat @ mat1
    np.testing.assert_allclose(sand, true)
    sand = block_matrix_sandwich(bm, bm, mat, disable_simple_products=True)
    np.testing.assert_allclose(sand, true)

    mat_t = mat + mat.T
    true = mat1.T @ mat_t @ mat1

    sand = block_matrix_sandwich_sym(bm, mat_t, disable_simple_products=False)
    np.testing.assert_allclose(sand, true)

    sand = block_matrix_sandwich_sym(bm, mat_t, disable_simple_products=True)
    np.testing.assert_allclose(sand, true)
