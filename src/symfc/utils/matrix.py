"""Utility functions for matrices."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from scipy.sparse import csr_array

try:
    from sparse_dot_mkl import dot_product_mkl  # type: ignore
except ImportError:
    pass


def dot_product_sparse(
    A: Union[np.ndarray, csr_array],
    B: Union[np.ndarray, csr_array],
    use_mkl: bool = False,
    dense: bool = False,
) -> csr_array:
    """Compute dot-product of sparse matrices."""
    if use_mkl:
        return dot_product_mkl(A, B, dense=dense)
    return A @ B


@dataclass
class BlockMatrixComponent:
    """Dataclass for each component of block matrix."""

    data: np.ndarray
    rows: np.ndarray
    col_begin: int
    col_end: int
    compress: Optional[Any] = None

    def change_indices(self, rows: np.ndarray, col_shift: int):
        """Change indices."""
        self.rows = rows[self.rows]
        self.col_begin += col_shift
        self.col_end += col_shift

    def recover(self):
        """Recover block from compression."""
        if self.compress is not None:
            return self.compress.dot(self.data)
        return self.data


@dataclass
class BlockMatrix:
    """Dataclass for block matrix."""

    blocks: list[BlockMatrixComponent]
    shape: tuple[int, int]
    data_full: Optional[np.ndarray] = None

    def dot(self, mat: np.ndarray, left: bool = False):
        """Dot product block_mat @ mat or mat @ block_mat."""
        if left:
            return self.dot_from_left(mat)
        return self.dot_from_right(mat)

    def transpose_dot(self, mat: np.ndarray, left: bool = False):
        """Dot product block_mat.T @ mat or mat @ block_mat.T."""
        if left:
            return self.transpose_dot_from_left(mat)
        return self.transpose_dot_from_right(mat)

    def dot_from_right(self, mat: np.ndarray):
        """Dot product block_mat @ mat."""
        if len(mat.shape) == 1:
            dot_matrix = np.zeros(self.shape[0])
        elif len(mat.shape) == 2:
            dot_matrix = np.zeros((self.shape[0], mat.shape[1]))
        else:
            raise RuntimeError("Dimension of input numpy array must be one or two.")

        for b in self.blocks:
            prod = b.data @ mat[b.col_begin : b.col_end]
            if b.compress is not None:
                prod = b.compress.dot_from_right(prod)
            dot_matrix[b.rows] += prod
        return dot_matrix

    def dot_from_left(self, mat: np.ndarray):
        """Dot product mat @ block_mat."""
        if len(mat.shape) == 1:
            dot_matrix = np.zeros(self.shape[1])
            for b in self.blocks:
                if b.compress is not None:
                    prod = b.compress.dot_from_left(mat[b.rows]) @ b.data
                else:
                    prod = mat[b.rows] @ b.data
                dot_matrix[b.col_begin : b.col_end] += prod
        elif len(mat.shape) == 2:
            dot_matrix = np.zeros((mat.shape[0], self.shape[1]))
            for b in self.blocks:
                if b.compress is not None:
                    prod = b.compress.dot_from_left(mat[:, b.rows]) @ b.data
                else:
                    prod = mat[:, b.rows] @ b.data
                dot_matrix[:, b.col_begin : b.col_end] += prod
        else:
            raise RuntimeError("Dimension of input numpy array must be one or two.")

        return dot_matrix

    def transpose_dot_from_right(self, mat: np.ndarray):
        """Dot product block_mat.T @ mat."""
        if len(mat.shape) == 1:
            dot_matrix = np.zeros(self.shape[1])
        elif len(mat.shape) == 2:
            dot_matrix = np.zeros((self.shape[1], mat.shape[1]))
        else:
            raise RuntimeError("Dimension of input numpy array must be one or two.")

        for b in self.blocks:
            if b.compress is not None:
                prod = b.data.T @ b.compress.transpose_dot_from_right(mat[b.rows])
            else:
                prod = b.data.T @ mat[b.rows]
            dot_matrix[b.col_begin : b.col_end] += prod
        return dot_matrix

    def transpose_dot_from_left(self, mat: np.ndarray):
        """Dot product mat @ block_mat.T."""
        if len(mat.shape) == 1:
            dot_matrix = np.zeros(self.shape[0])
            for b in self.blocks:
                prod = mat[b.col_begin : b.col_end] @ b.data.T
                if b.compress is not None:
                    prod = b.compress.transpose_dot_from_left(prod)
                dot_matrix[b.rows] += prod
        elif len(mat.shape) == 2:
            dot_matrix = np.zeros((mat.shape[0], self.shape[0]))
            for b in self.blocks:
                prod = mat[:, b.col_begin : b.col_end] @ b.data.T
                if b.compress is not None:
                    prod = b.compress.transpose_dot_from_left(prod)
                dot_matrix[:, b.rows] += prod
        else:
            raise RuntimeError("Dimension of input numpy array must be one or two.")

        return dot_matrix

    def compress_matrix(self, mat: np.ndarray):
        """Calculate block_mat.T @ mat @ block_mat."""
        # TODO: add compress attr
        for b1 in self.blocks:
            if b1.compress is not None:
                raise RuntimeError("Compression matrix does not work.")
        res = np.zeros((self.shape[1], self.shape[1]))
        for b1 in self.blocks:
            for b2 in self.blocks:
                prod = b1.data.T @ mat[np.ix_(b1.rows, b2.rows)] @ b2.data
                res[b1.col_begin : b1.col_end, b2.col_begin : b2.col_end] += prod
        return res

    def compress_csr_matrix(self, mat: csr_array, use_mkl: bool = False):
        """Calculate block_mat.T @ mat(csr) @ block_mat for csr_array."""
        # TODO: add compress attr
        for b1 in self.blocks:
            if b1.compress is not None:
                raise RuntimeError("Compression matrix does not work.")

        if mat.shape[0] < 10000:
            use_mkl = False
        res = np.zeros((self.shape[1], self.shape[1]))
        for b1 in self.blocks:
            for b2 in self.blocks:
                prod = b1.data.T @ dot_product_sparse(
                    mat[np.ix_(b1.rows, b2.rows)], b2.data, use_mkl=use_mkl, dense=True
                )
                res[b1.col_begin : b1.col_end, b2.col_begin : b2.col_end] += prod
        return res

    def recover_full_matrix(self):
        """Recover full block matrix."""
        if self.data_full is None:
            self.data_full = np.zeros(self.shape, dtype="double")  # type: ignore
            for b in self.blocks:
                if b.compress is not None:
                    mat = b.compress.dot_from_right(b.data)
                else:
                    mat = b.data
                self.data_full[b.rows, b.col_begin : b.col_end] = mat
        return self.data_full


def append_block(
    blocks_list: list,
    eigvecs: np.ndarray,
    rows: Optional[np.ndarray] = None,
    col_begin: Optional[int] = None,
    compress: Optional[BlockMatrix] = None,
):
    """Add eigenvectors to block matrix list."""
    if eigvecs is not None and eigvecs.shape[1] > 0:
        col_end = col_begin + eigvecs.shape[1]  # type: ignore
        block = BlockMatrixComponent(
            data=eigvecs,
            rows=rows,
            col_begin=col_begin,
            col_end=col_end,
            compress=compress,
        )
        blocks_list.append(block)
    return blocks_list


def block_matrix_sandwich(
    block_matrix1: BlockMatrix,
    block_matrix2: BlockMatrix,
    mat: np.ndarray,
):
    """Calculate block1.T @ mat @ block2."""
    res = np.zeros((block_matrix1.shape[1], block_matrix2.shape[1]))
    for b1 in block_matrix1.blocks:
        b1_full = b1.recover()
        for b2 in block_matrix2.blocks:
            b2_full = b2.recover()
            prod = b1_full.T @ mat[np.ix_(b1.rows, b2.rows)] @ b2_full
            res[b1.col_begin : b1.col_end, b2.col_begin : b2.col_end] += prod
    return res
