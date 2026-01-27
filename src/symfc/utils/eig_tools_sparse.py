"""Utility functions for sparse eigenvalue solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from symfc.utils.eig_tools_core import (
    eigh_projector,
    find_projector_blocks,
)
from symfc.utils.matrix import matrix_rank

# Threshold constants for eigenvalue solvers
SPARSE_DATA_LIMIT = 2147483647

# Tolerance constants
DEFAULT_EIGVAL_TOL = 1e-8


def eigsh_projector(
    p: csr_array,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    verbose: bool = True,
) -> csr_array:
    """Solve eigenvalue problem for matrix p.

    Return sparse matrix for eigenvectors of matrix p.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved. When p = diag(A,B), Av =
    v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w] are solutions.

    This function avoids solving numpy.eigh for duplicate block matrices. This
    function is efficient for matrix p composed of many duplicate block
    matrices.

    """
    compr = CompressionProjector(p)
    group = find_projector_blocks(compr.compressed_projector, verbose=verbose)
    if verbose:
        rank = matrix_rank(compr.compressed_projector)
        print("Rank of projector:", rank, flush=True)
        print("Number of blocks in projector:", len(group), flush=True)

    eigvecs = _solve_eigsh(
        compr.compressed_projector, group, atol=atol, rtol=rtol, verbose=verbose
    )
    return compr.recover(eigvecs)


class CompressionProjector:
    """Compress projection matrix p with many zero rows and columns."""

    def __init__(self, p: csr_array):
        """Init method."""
        self._compr = None
        self._compressed_proj = self._compress(p)

    def _compress(self, p: csr_array):
        """Compress projector."""
        _, col_p = p.nonzero()  # type: ignore
        col_p = np.unique(col_p)
        size = len(col_p)

        if p.shape[1] == size:
            self._compressed_proj = p
            return self._compressed_proj

        self._compr = csr_array(
            (np.ones(size), (col_p, np.arange(size))),
            shape=(p.shape[1], size),
            dtype="int",
        )
        # Matrix compression using p = compr.T @ p @ compr
        self._compressed_proj = p[col_p].T
        self._compressed_proj = self._compressed_proj[col_p].T
        return self._compressed_proj

    def recover(self, eigvecs: csr_array):
        """Recover eigenvectors of compressed projector."""
        if self._compr is None:
            return eigvecs
        return self._compr @ eigvecs

    @property
    def compressed_projector(self) -> csr_array:
        """Return compressed projector."""
        return self._compressed_proj


def _recover_eigvecs_from_uniq_eigvecs(
    uniq_eigvecs: dict,
    group: dict,
    size_projector: int,
) -> csr_array:
    """Recover all eigenvectors from unique eigenvectors.

    Parameters
    ----------
    uniq_eigvecs: Unique eigenvectors and submatrixblock indices
                  are included in its values.
    group: Row indices comprising submatrix blocks.

    """
    total_length = sum(
        len(labels) * v.shape[0] * v.shape[1]
        for v, labels in uniq_eigvecs.values()
        if v is not None
    )
    row = np.zeros(total_length, dtype=int)
    col = np.zeros(total_length, dtype=int)
    data = np.zeros(total_length, dtype="double")

    current_id, col_id = 0, 0
    for eigvecs, labels in uniq_eigvecs.values():
        if eigvecs is not None:
            n_row, n_col = eigvecs.shape
            num_labels = len(labels)
            end_id = current_id + n_row * n_col * num_labels

            row[current_id:end_id] = np.repeat(
                [i for ll in labels for i in group[ll]], n_col
            )
            col[current_id:end_id] = [
                j
                for seq, _ in enumerate(labels)
                for i in range(n_row)
                for j in range(col_id + seq * n_col, col_id + (seq + 1) * n_col)
            ]
            data[current_id:end_id] = np.tile(eigvecs.flatten(), num_labels)

            col_id += n_col * num_labels
            current_id = end_id

    n_col = col_id
    eigvecs = csr_array((data, (row, col)), shape=(size_projector, n_col))
    return eigvecs


@dataclass
class DataCSR:
    """Dataclass for extracting data in projector."""

    data: NDArray
    block_labels: NDArray
    block_sizes: NDArray
    slice_begin: NDArray | None = None
    slice_end: NDArray | None = None

    def __post_init__(self):
        """Init method."""
        self.slice_end = np.cumsum(self.block_sizes**2)
        self.slice_begin = np.zeros_like(self.slice_end)
        self.slice_begin[1:] = self.slice_end[:-1]

        self._index: int = 0
        self._length = len(self.block_sizes)

    def __iter__(self):
        """Define iter method."""
        return self

    def __next__(self):
        """Define next method."""
        if self._index >= self._length:
            raise StopIteration

        value = (
            self.get_data(self._index),
            self.block_labels[self._index],
            self.block_sizes[self._index],
        )
        self._index += 1
        return value

    def get_data(self, idx: int):
        """Get data in projector for i-th block."""
        if self.slice_begin is None or self.slice_end is None:
            raise ValueError("Slice begin and end are not initialized.")
        s1 = self.slice_begin[idx]
        s2 = self.slice_end[idx]
        return self.data[s1:s2]


def _extract_sparse_projector_data(p: csr_array, group: dict) -> DataCSR:
    """Extract data in projector in csr_format efficiently.

    Parameters
    ----------
    p: Projection matrix in CSR format.
    group: Row indices comprising submatrix blocks.

    """
    # r = np.array([i for ids in group.values() for i in ids for j in ids])
    group_ravel = [i for ids in group.values() for i in ids]
    lengths = [len(ids) for ids in group.values() for i in ids]
    r = np.repeat(group_ravel, lengths)
    c = np.array([j for ids in group.values() for _ in ids for j in ids])
    sizes = np.array([len(ids) for ids in group.values()], dtype=int)

    p_data = DataCSR(
        data=np.ravel(p[r, c]),
        block_labels=np.array(list(group.keys())),
        block_sizes=sizes,
    )
    return p_data


def _solve_eigsh(
    p: csr_array,
    group: dict,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    verbose: bool = True,
) -> csr_array:
    """Solve eigenvalue problem for matrix p."""
    p_data = _extract_sparse_projector_data(p, group)
    uniq_eigvecs = {"one": [np.array([[1.0]]), []]}
    for p_block, block_label, block_size in p_data:
        if block_size == 1 and not np.isclose(p_block[0], 0.0):
            uniq_eigvecs["one"][1].append(block_label)
        elif block_size > 1:
            # key = tuple(p_block)
            key = (p_block.tobytes(), p_block.shape, p_block.dtype.str)
            try:
                uniq_eigvecs[key][1].append(block_label)
            except KeyError:
                p_numpy = p_block.reshape((block_size, block_size))
                res = eigh_projector(p_numpy, atol=atol, rtol=rtol, verbose=verbose)
                uniq_eigvecs[key] = [res.eigvecs, [block_label]]

    eigvecs = _recover_eigvecs_from_uniq_eigvecs(uniq_eigvecs, group, p.shape[0])  # type: ignore
    return eigvecs
