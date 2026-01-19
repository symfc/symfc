"""Utility functions for matrices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

try:
    from sparse_dot_mkl import dot_product_mkl  # type: ignore
except ImportError:
    pass


import sys

sys.setrecursionlimit(100000)


def dot_product_sparse(
    A: NDArray | csr_array,
    B: NDArray | csr_array,
    use_mkl: bool = False,
    dense: bool = False,
) -> csr_array:
    """Compute dot-product of sparse matrices."""
    if use_mkl:
        return dot_product_mkl(A, B, dense=dense)
    return A @ B


def is_sparse(p: NDArray | csr_array) -> bool:
    """Check whether matrix is sparse matrix or not."""
    if isinstance(p, np.ndarray):
        return False
    elif isinstance(p, list):
        return False
    return True


def return_numpy_array(p: NDArray | csr_array) -> NDArray:
    """Return numpy array."""
    if isinstance(p, np.ndarray):
        return p
    return p.toarray()


def matrix_rank(p: NDArray | csr_array) -> int:
    """Calculate projector rank."""
    if is_sparse(p):
        return int(round(p.trace()))
    assert isinstance(p, np.ndarray)
    return int(round(np.trace(p)))


class BlockMatrixNode:
    """Data structure class for node in block matrix tree.

    Note
    ----
    Avoided to use dataclass for complexity of handling type hints.

    Attributes
    ----------
    rows : NDArray
    col_begin : int
    col_end : int
    shape : tuple (readonly)
    rows_root : NDArray (readonly)
    col_begin_root : int (readonly)
    col_end_root : int (readonly)
    next_sibling : BlockMatrixNode | None
    first_child : BlockMatrixNode | None (readonly)
    compress : BlockMatrixNode | None (readonly)
    root : bool
    eigvals : NDArray | None
    index : int (readonly)

    """

    def __init__(
        self,
        rows: NDArray | Sequence[int],
        col_begin: int,
        col_end: int,
        data: NDArray | None = None,
        next_sibling: BlockMatrixNode | None = None,
        first_child: BlockMatrixNode | None = None,
        compress: BlockMatrixNode | None = None,
        root: bool = False,
        eigvals: NDArray | None = None,
        index: int | None = None,  # Used in test only
    ):
        self._rows = np.array(rows, dtype=int)
        self._col_begin = col_begin
        self._col_end = col_end
        self._data = data
        self._next_sibling = next_sibling
        self._first_child = first_child
        self._compress = compress
        self._root = root
        self._eigvals = eigvals
        self._index = index  # Used in test only

        self._shape = (len(self._rows), self._col_end - self._col_begin)

        self._rows_root: NDArray
        self._col_begin_root: int
        self._col_end_root: int

        self._check_errors()

        if self._root:
            self.set_root_indices()

    @property
    def rows(self) -> NDArray:
        """Return row indices."""
        return self._rows

    @rows.setter
    def rows(self, value: NDArray | Sequence[int]):
        """Set row indices."""
        self._rows = np.array(value)

    @property
    def col_begin(self) -> int:
        """Return column begin index."""
        return self._col_begin

    @col_begin.setter
    def col_begin(self, value: int):
        """Set column begin index."""
        self._col_begin = value

    @property
    def col_end(self) -> int:
        """Return column end index."""
        return self._col_end

    @col_end.setter
    def col_end(self, value: int):
        """Set column end index."""
        self._col_end = value

    @property
    def shape(self) -> tuple:
        """Return shape of block matrix."""
        return self._shape

    @property
    def rows_root(self) -> NDArray:
        """Return row indices compatible with root node and full matrix."""
        return self._rows_root

    @property
    def col_begin_root(self) -> int:
        """Return column begin index compatible with root node and full matrix."""
        return self._col_begin_root

    @property
    def col_end_root(self) -> int:
        """Return column end index compatible with root node and full matrix."""
        return self._col_end_root

    @property
    def next_sibling(self) -> BlockMatrixNode | None:
        """Return next sibling node."""
        return self._next_sibling

    @next_sibling.setter
    def next_sibling(self, value: BlockMatrixNode | None):
        """Set next sibling node."""
        self._next_sibling = value

    @property
    def first_child(self) -> BlockMatrixNode | None:
        """Return first child node."""
        return self._first_child

    @property
    def compress(self) -> BlockMatrixNode | None:
        """Return compression matrix node."""
        return self._compress

    @property
    def root(self) -> bool:
        """Return whether node is root or not."""
        return self._root

    @root.setter
    def root(self, value: bool):
        """Set whether node is root or not."""
        self._root = value

    @property
    def eigvals(self) -> NDArray | None:
        """Return eigenvalues."""
        return self._eigvals

    @eigvals.setter
    def eigvals(self, value: NDArray | None):
        """Set eigenvalues."""
        self._eigvals = value

    @property
    def index(self) -> int:
        """Return index. Only in test."""
        if self._index is None:
            raise RuntimeError("Index is not set.")
        return self._index

    def _check_errors(self):
        """Check errors in node."""
        if self._data is None:
            if self._first_child is None:
                raise RuntimeError("No data in this node and its children.")
            if self._compress is not None:
                raise RuntimeError("Data is required with compress matrix.")
        else:
            if self._compress is None:
                if self._data.shape != self.shape:
                    raise RuntimeError(
                        "Data shape is inconsistent with rows and columns."
                    )
            else:
                assert self._compress.shape is not None
                if self._compress.shape[0] != self.shape[0]:
                    raise RuntimeError("Data shape is inconsistent with rows.")
                if self._data.shape[1] != self.shape[1]:
                    raise RuntimeError("Data shape is inconsistent with columns.")

    def traverse_data_nodes(self) -> Iterator[BlockMatrixNode]:
        """Traverse all nodes with data."""
        if self._first_child is not None:
            yield from self._first_child.traverse_data_nodes()
        if self._next_sibling is not None:
            yield from self._next_sibling.traverse_data_nodes()
        if self._data is not None:
            yield self

    def print_nodes(self, depth: int = 0):
        """Print all nodes."""
        if depth == 0:
            header = "-"
        else:
            header = "  " * (depth - 1) + "|--"
        print(header, self.shape, end=", ", flush=True)
        if self._data is not None:
            if self.compress is not None:
                print("data:", self.compress.shape, "@", self._data.shape, flush=True)
            else:
                print("data: True", flush=True)
        else:
            print("data: False", flush=True)

        if self._first_child is not None:
            self._first_child.print_nodes(depth=depth + 1)
        if self._next_sibling is not None:
            self._next_sibling.print_nodes(depth=depth)
        return self

    def set_root_indices(
        self,
        parent_rows: NDArray | None = None,
        parent_col_begin: int | None = None,
    ):
        """Set row and columns indices compatible with root node and full matrix."""
        if parent_rows is not None and parent_col_begin is None:
            raise ValueError("parent_col_begin is required when parent_rows is set.")

        if parent_rows is not None:
            assert parent_col_begin is not None
            self._rows_root = parent_rows[self.rows]
            self._col_begin_root = self.col_begin + parent_col_begin
            self._col_end_root = self.col_end + parent_col_begin
            if self._root:
                self._root = False

        if self._first_child is not None:
            if parent_rows is None:
                self._first_child.set_root_indices(self.rows, self.col_begin)
            else:
                assert parent_col_begin is not None
                self._first_child.set_root_indices(
                    parent_rows[self.rows],
                    parent_col_begin + self.col_begin,
                )

        if self._next_sibling is not None:
            self._next_sibling.set_root_indices(parent_rows, parent_col_begin)

        return self

    def decompress(self) -> NDArray:
        """Decompress compressed data matrix."""
        if self.compress is not None:
            return self.compress.dot(self._data)
        return self._data

    def recover(self) -> NDArray:
        """Recover full block matrix."""
        if not self._root:
            raise RuntimeError("Node must be root of tree.")

        full = np.zeros(self.shape, dtype="double")  # type: ignore
        for b in self.traverse_data_nodes():
            full[b.rows_root, b.col_begin_root : b.col_end_root] = b.decompress()
        return full

    def dot(self, mat: NDArray) -> NDArray:
        """Calculate dot product block_mat @ mat."""
        if not self._root:
            raise RuntimeError("Node must be root of tree.")

        if len(mat.shape) == 1:
            prod = np.zeros(self.shape[0])
        elif len(mat.shape) == 2:
            prod = np.zeros((self.shape[0], mat.shape[1]))
        else:
            raise RuntimeError("Dimension of input numpy array must be one or two.")

        for b in self.traverse_data_nodes():
            res = b._data @ mat[b.col_begin_root : b.col_end_root]
            if b.compress is not None:
                res = b.compress.dot(res)
            prod[b.rows_root] += res

        return prod

    def transpose_dot(self, mat: NDArray) -> NDArray:
        """Calculate dot product block_mat.T @ mat."""
        if not self._root:
            raise RuntimeError("Node must be root of tree.")

        if len(mat.shape) == 1:
            prod = np.zeros(self.shape[1])
        elif len(mat.shape) == 2:
            prod = np.zeros((self.shape[1], mat.shape[1]))
        else:
            raise RuntimeError("Dimension of input numpy array must be one or two.")

        for b in self.traverse_data_nodes():
            assert b._data is not None
            if b.compress is not None:
                res = b._data.T @ b.compress.transpose_dot(mat[b.rows_root])
            else:
                res = b._data.T @ mat[b.rows_root]
            prod[b.col_begin_root : b.col_end_root] += res

        return prod

    def compress_matrix(
        self, mat: NDArray | csr_array, use_mkl: bool = False
    ) -> NDArray:
        """Calculate block_mat.T @ mat @ block_mat.

        Block matrix must be eigenvectors and include their eigenvalues.
        """
        if not self._root:
            raise RuntimeError("Node must be root of tree.")

        if is_sparse(mat):
            return self.compress_sparse_matrix(mat, use_mkl=use_mkl)  # type: ignore
        return self.compress_dense_matrix(mat)  # type: ignore

    def compress_dense_matrix(self, mat: NDArray) -> NDArray:
        """Calculate block_mat.T @ mat @ block_mat for numpy array.

        Block matrix must be eigenvectors and include their eigenvalues.
        """
        if not self._root:
            raise RuntimeError("Node must be root of tree.")

        if self.shape[1] < 10000:
            return self.recover().T @ mat @ self.recover()

        res = np.zeros((self.shape[1], self.shape[1]))
        for i, b1 in enumerate(self.traverse_data_nodes()):
            col_begin1, col_end1 = b1.col_begin_root, b1.col_end_root
            data1 = b1.decompress()
            for c, val in zip(range(col_begin1, col_end1), b1.eigvals, strict=True):
                res[c, c] = val
            for j, b2 in enumerate(self.traverse_data_nodes()):
                if i != j:
                    col_begin2, col_end2 = b2.col_begin_root, b2.col_end_root
                    data2 = b2.decompress()
                    prod = data1.T @ mat[np.ix_(b1.rows_root, b2.rows_root)] @ data2
                    res[col_begin1:col_end1, col_begin2:col_end2] = prod
        return res

    def compress_sparse_matrix(self, mat: csr_array, use_mkl: bool = False) -> NDArray:
        """Calculate block_mat.T @ mat(csr) @ block_mat for csr_array.

        Block matrix must be eigenvectors and include their eigenvalues.
        """
        if not self._root:
            raise RuntimeError("Node must be root of tree.")

        if mat.shape[0] < 30000:  # type: ignore
            use_mkl = False

        res = np.zeros((self.shape[1], self.shape[1]))
        for i, b1 in enumerate(self.traverse_data_nodes()):
            col_begin1, col_end1 = b1.col_begin_root, b1.col_end_root
            data1 = b1.decompress()
            mat1 = mat[b1.rows_root]
            for c, val in zip(range(col_begin1, col_end1), b1.eigvals, strict=True):
                res[c, c] = val
            for j, b2 in enumerate(self.traverse_data_nodes()):
                if i > j:
                    col_begin2, col_end2 = b2.col_begin_root, b2.col_end_root
                    data2 = b2.decompress()
                    mat_slice = mat1[:, b2.rows_root]
                    mat_slice_t = mat[np.ix_(b2.rows_root, b1.rows_root)].T
                    mat_slice = 0.5 * (mat_slice + mat_slice_t)
                    prod = dot_product_sparse(mat_slice, data2, use_mkl=use_mkl)
                    prod = dot_product_sparse(
                        data1.T, prod, use_mkl=use_mkl, dense=True
                    )
                    res[col_begin1:col_end1, col_begin2:col_end2] = prod
                    res[col_begin2:col_end2, col_begin1:col_end1] = prod.T

        return res


def append_node(
    eigvecs: NDArray | BlockMatrixNode | None,
    next_sibling: BlockMatrixNode | None,
    rows: NDArray,
    col_begin: int,
    compress: BlockMatrixNode | None = None,
    eigvals: NDArray | None = None,
) -> BlockMatrixNode | None:
    """Add eigenvectors to block matrix node."""
    if isinstance(eigvecs, BlockMatrixNode):
        block = eigvecs
        assert block.shape is not None
        if block.shape[1] > 0:
            block.rows = rows
            block.col_begin = col_begin
            block.col_end = col_begin + block.shape[1]  # type: ignore
            block.next_sibling = next_sibling
            block.eigvals = eigvals
            block.root = False
            next_sibling = block
    else:
        if eigvecs is not None and eigvecs.shape[1] > 0:
            col_end = col_begin + eigvecs.shape[1]  # type: ignore
            block = BlockMatrixNode(
                rows=rows,
                col_begin=col_begin,
                col_end=col_end,
                data=eigvecs,
                next_sibling=next_sibling,
                compress=compress,
                eigvals=eigvals,
            )
            next_sibling = block
    return next_sibling


def root_block_matrix(
    shape: tuple | None = None,
    data: NDArray | None = None,
    first_child: BlockMatrixNode | None = None,
) -> BlockMatrixNode | None:
    """Return root block matrix."""
    if shape is None and data is None:
        raise RuntimeError("Shape or data is required.")

    if data is not None:
        shape = data.shape

    assert shape is not None
    if shape[1] == 0:
        return None

    return BlockMatrixNode(
        rows=np.arange(shape[0]),
        col_begin=0,
        col_end=shape[1],
        first_child=first_child,
        data=data,
        root=True,
    )


def block_matrix_sandwich(
    bm1: BlockMatrixNode, bm2: BlockMatrixNode, mat: NDArray
) -> NDArray:
    """Calculate block1.T @ mat @ block2."""
    if not bm1.root or not bm2.root:
        raise RuntimeError("Nodes must be root of tree.")

    if bm1.shape[1] < 20000 and bm2.shape[1] < 20000:
        return bm1.recover().T @ mat @ bm2.recover()

    res = np.zeros((bm1.shape[1], bm2.shape[1]))
    for b1 in bm1.traverse_data_nodes():
        col_begin1, col_end1 = b1.col_begin_root, b1.col_end_root
        data1 = b1.decompress()
        for b2 in bm2.traverse_data_nodes():
            col_begin2, col_end2 = b2.col_begin_root, b2.col_end_root
            data2 = b2.decompress()
            prod = data1.T @ mat[np.ix_(b1.rows_root, b2.rows_root)] @ data2
            res[col_begin1:col_end1, col_begin2:col_end2] += prod
    return res
