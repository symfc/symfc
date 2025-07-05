"""Utility functions for matrices."""

from dataclasses import dataclass
from typing import Optional, Union

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

    def change_indices(self, rows: np.ndarray, col_shift: int):
        """Change indices."""
        self.rows = rows[self.rows]
        self.col_begin += col_shift
        self.col_end += col_shift


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
            dot_matrix[b.rows] += b.data @ mat[b.col_begin : b.col_end]
        return dot_matrix

    def dot_from_left(self, mat: np.ndarray):
        """Dot product mat @ block_mat."""
        if len(mat.shape) == 1:
            dot_matrix = np.zeros(self.shape[1])
            for b in self.blocks:
                dot_matrix[b.col_begin : b.col_end] += mat[b.rows] @ b.data
        elif len(mat.shape) == 2:
            dot_matrix = np.zeros((mat.shape[0], self.shape[1]))
            for b in self.blocks:
                dot_matrix[:, b.col_begin : b.col_end] += mat[:, b.rows] @ b.data
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
            dot_matrix[b.col_begin : b.col_end] += b.data.T @ mat[b.rows]
        return dot_matrix

    def transpose_dot_from_left(self, mat: np.ndarray):
        """Dot product mat @ block_mat.T."""
        if len(mat.shape) == 1:
            dot_matrix = np.zeros(self.shape[0])
            for b in self.blocks:
                dot_matrix[b.rows] += mat[b.col_begin : b.col_end] @ b.data.T
        elif len(mat.shape) == 2:
            dot_matrix = np.zeros((mat.shape[0], self.shape[0]))
            for b in self.blocks:
                dot_matrix[:, b.rows] += mat[:, b.col_begin : b.col_end] @ b.data.T
        else:
            raise RuntimeError("Dimension of input numpy array must be one or two.")

        return dot_matrix

    def compress_matrix(self, mat: np.ndarray):
        """Calculate block_mat.T @ mat @ block_mat."""
        # TODO: Consider more efficient algorithm
        res = np.zeros((self.shape[1], self.shape[1]))
        for b in self.blocks:
            res[b.col_begin : b.col_end] += self.dot_from_left(b.data.T @ mat[b.rows])
        return res

    def compress_csr_matrix(self, mat: csr_array, use_mkl: bool = False):
        """Calculate block_mat.T @ mat @ block_mat."""
        if mat.shape[0] < 10000:
            use_mkl = False
        res = np.zeros((self.shape[1], self.shape[1]))
        for b in self.blocks:
            res[b.col_begin : b.col_end] += self.dot_from_left(
                dot_product_sparse(b.data.T, mat[b.rows], use_mkl=use_mkl, dense=True)
            )
        return res

    def recover_full_matrix(self):
        """Recover full block matrix."""
        if self.data_full is None:
            self.data_full = np.zeros(self.shape, dtype="double")  # type: ignore
            for b in self.blocks:
                self.data_full[b.rows, b.col_begin : b.col_end] = b.data
        return self.data_full


def append_block(
    blocks_list: list,
    eigvecs: np.ndarray,
    rows: Optional[np.ndarray] = None,
    col_begin: Optional[int] = None,
):
    """Add eigenvectors to block matrix list."""
    if eigvecs is not None and eigvecs.shape[1] > 0:
        col_end = col_begin + eigvecs.shape[1]  # type: ignore
        block = BlockMatrixComponent(
            data=eigvecs,
            rows=rows,
            col_begin=col_begin,
            col_end=col_end,
        )
        blocks_list.append(block)
    return blocks_list
