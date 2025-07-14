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


def is_sparse(p: Union[np.ndarray, csr_array]) -> bool:
    """Check whether matrix is sparse matrix or not."""
    if isinstance(p, np.ndarray):
        return False
    elif isinstance(p, list):
        return False
    return True


def return_numpy_array(p: Union[np.ndarray, csr_array]) -> np.ndarray:
    """Return numpy array."""
    if isinstance(p, np.ndarray):
        return p
    return p.toarray()


def matrix_rank(p: Union[np.ndarray, csr_array]) -> int:
    """Calculate projector rank."""
    if is_sparse(p):
        return int(round(p.trace()))
    return int(round(np.trace(p)))


@dataclass
class BlockMatrixNode:
    """Dataclass for node in block matrix tree."""

    rows: np.ndarray
    col_begin: int
    col_end: int
    root: bool = False
    shape: Optional[tuple] = None

    data: Optional[np.ndarray] = None
    first_child: Optional[Any] = None
    next_sibling: Optional[Any] = None
    compress: Optional[Any] = None

    rows_root: Optional[np.ndarray] = None
    col_begin_root: Optional[int] = None
    col_end_root: Optional[int] = None
    index: Optional[int] = None
    eigvals: Optional[np.ndarray] = None

    def __post_init__(self):
        """Post init method."""
        self.shape = (len(self.rows), self.col_end - self.col_begin)
        self.rows = np.array(self.rows)
        self._check_errors()

        if self.root:
            self.set_root_indices()

    def _check_errors(self):
        """Check errors in node."""
        if self.data is None:
            if self.first_child is None:
                raise RuntimeError("No data in this node and its children.")
            if self.compress is not None:
                raise RuntimeError("Data is required with compress matrix.")
        else:
            if self.compress is None:
                if self.data.shape != self.shape:
                    raise RuntimeError(
                        "Data shape is inconsistent with rows and columns."
                    )
            else:
                if self.compress.shape[0] != self.shape[0]:
                    raise RuntimeError("Data shape is inconsistent with rows.")
                if self.data.shape[1] != self.shape[1]:
                    raise RuntimeError("Data shape is inconsistent with columns.")

    def traverse_data_nodes(self):
        """Traverse all nodes with data."""
        if self.first_child is not None:
            yield from self.first_child.traverse_data_nodes()
        if self.next_sibling is not None:
            yield from self.next_sibling.traverse_data_nodes()
        if self.data is not None:
            yield self

    def print_nodes(self, depth: int = 0):
        """Print all nodes."""
        if depth == 0:
            header = "-"
        else:
            header = "  " * (depth - 1) + "|--"
        print(header, self.shape, end=", ", flush=True)
        if self.data is not None:
            if self.compress is not None:
                print("data:", self.compress.shape, "@", self.data.shape, flush=True)
            else:
                print("data: True", flush=True)
        else:
            print("data: False", flush=True)

        if self.first_child is not None:
            self.first_child.print_nodes(depth=depth + 1)
        if self.next_sibling is not None:
            self.next_sibling.print_nodes(depth=depth)
        return self

    def set_root_indices(
        self,
        parent_rows: Optional[np.ndarray] = None,
        parent_col_begin: Optional[int] = None,
    ):
        """Set row and columns indices compatible with root node and full matrix."""
        if parent_rows is not None:
            self.rows_root = parent_rows[self.rows]
            self.col_begin_root = self.col_begin + parent_col_begin
            self.col_end_root = self.col_end + parent_col_begin
            if self.root:
                self.root = False

        if self.first_child is not None:
            if parent_rows is None:
                self.first_child.set_root_indices(self.rows, self.col_begin)
            else:
                self.first_child.set_root_indices(
                    parent_rows[self.rows],
                    parent_col_begin + self.col_begin,
                )

        if self.next_sibling is not None:
            self.next_sibling.set_root_indices(parent_rows, parent_col_begin)

        return self

    def decompress(self):
        """Decompress compressed data matrix."""
        if self.compress is not None:
            return self.compress.dot(self.data)
        return self.data

    def recover(self):
        """Recover full block matrix."""
        if not self.root:
            raise RuntimeError("Node must be root of tree.")

        full = np.zeros(self.shape, dtype="double")  # type: ignore
        for b in self.traverse_data_nodes():
            full[b.rows_root, b.col_begin_root : b.col_end_root] = b.decompress()
        return full

    def dot(self, mat: np.ndarray):
        """Calculate dot product block_mat @ mat."""
        if not self.root:
            raise RuntimeError("Node must be root of tree.")

        if len(mat.shape) == 1:
            prod = np.zeros(self.shape[0])
        elif len(mat.shape) == 2:
            prod = np.zeros((self.shape[0], mat.shape[1]))
        else:
            raise RuntimeError("Dimension of input numpy array must be one or two.")

        for b in self.traverse_data_nodes():
            res = b.data @ mat[b.col_begin_root : b.col_end_root]
            if b.compress is not None:
                res = b.compress.dot(res)
            prod[b.rows_root] += res

        return prod

    def transpose_dot(self, mat: np.ndarray):
        """Calculate dot product block_mat.T @ mat."""
        if not self.root:
            raise RuntimeError("Node must be root of tree.")

        if len(mat.shape) == 1:
            prod = np.zeros(self.shape[1])
        elif len(mat.shape) == 2:
            prod = np.zeros((self.shape[1], mat.shape[1]))
        else:
            raise RuntimeError("Dimension of input numpy array must be one or two.")

        for b in self.traverse_data_nodes():
            if b.compress is not None:
                res = b.data.T @ b.compress.transpose_dot(mat[b.rows_root])
            else:
                res = b.data.T @ mat[b.rows_root]
            prod[b.col_begin_root : b.col_end_root] += res

        return prod

    def compress_matrix(self, mat: Union[csr_array, np.ndarray], use_mkl: bool = False):
        """Calculate block_mat.T @ mat @ block_mat.

        Block matrix must be eigenvectors and include their eigenvalues.
        """
        if not self.root:
            raise RuntimeError("Node must be root of tree.")

        if is_sparse(mat):
            return self.compress_sparse_matrix(mat, use_mkl=use_mkl)
        return self.compress_dense_matrix(mat)

    def compress_dense_matrix(self, mat: np.ndarray):
        """Calculate block_mat.T @ mat @ block_mat for numpy array.

        Block matrix must be eigenvectors and include their eigenvalues.
        """
        if not self.root:
            raise RuntimeError("Node must be root of tree.")

        if self.shape[1] < 10000:
            return self.recover().T @ mat @ self.recover()

        res = np.zeros((self.shape[1], self.shape[1]))
        for i, b1 in enumerate(self.traverse_data_nodes()):
            col_begin1, col_end1 = b1.col_begin_root, b1.col_end_root
            data1 = b1.decompress()
            for c, val in zip(range(col_begin1, col_end1), b1.eigvals):
                res[c, c] = val
            for j, b2 in enumerate(self.traverse_data_nodes()):
                if i != j:
                    col_begin2, col_end2 = b2.col_begin_root, b2.col_end_root
                    data2 = b2.decompress()
                    prod = data1.T @ mat[np.ix_(b1.rows_root, b2.rows_root)] @ data2
                    res[col_begin1:col_end1, col_begin2:col_end2] = prod
        return res

    def compress_sparse_matrix(self, mat: csr_array, use_mkl: bool = False):
        """Calculate block_mat.T @ mat(csr) @ block_mat for csr_array.

        Block matrix must be eigenvectors and include their eigenvalues.
        """
        if not self.root:
            raise RuntimeError("Node must be root of tree.")

        if mat.shape[0] < 30000:
            use_mkl = False

        res = np.zeros((self.shape[1], self.shape[1]))
        for i, b1 in enumerate(self.traverse_data_nodes()):
            col_begin1, col_end1 = b1.col_begin_root, b1.col_end_root
            data1 = b1.decompress()
            mat1 = mat[b1.rows_root]
            for c, val in zip(range(col_begin1, col_end1), b1.eigvals):
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
    eigvecs: np.ndarray,
    next_sibling: BlockMatrixNode,
    rows: np.ndarray,
    col_begin,
    compress: Optional[BlockMatrixNode] = None,
    eigvals: Optional[np.ndarray] = None,
):
    """Add eigenvectors to block matrix node."""
    if isinstance(eigvecs, BlockMatrixNode):
        block = eigvecs
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
    shape: Optional[tuple] = None,
    data: Optional[np.ndarray] = None,
    first_child: Optional[BlockMatrixNode] = None,
):
    """Return root block matrix."""
    if shape is None and data is None:
        raise RuntimeError("Shape or data is required.")

    if data is not None:
        shape = data.shape

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


def block_matrix_sandwich(bm1: BlockMatrixNode, bm2: BlockMatrixNode, mat: np.ndarray):
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
