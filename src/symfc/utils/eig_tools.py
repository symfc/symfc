"""Utility functions for eigenvalue solutions."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy
from scipy.linalg import get_lapack_funcs
from scipy.sparse import csr_array

from symfc.utils.solver_funcs import get_batch_slice

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


def _compr_projector(p: csr_array) -> tuple[csr_array, Optional[csr_array]]:
    """Compress projection matrix p with many zero rows and columns."""
    _, col_p = p.nonzero()  # type: ignore
    col_p = np.unique(col_p)
    size = len(col_p)

    if p.shape[1] > size:
        compr = csr_array(
            (np.ones(size), (col_p, np.arange(size))),
            shape=(p.shape[1], size),
            dtype="int",
        )
        """p = compr.T @ p @ compr"""
        p = p[col_p].T
        p = p[col_p].T
        return p, compr
    return p, None


def _find_projector_blocks(p: csr_array):
    """Find block structures in projection matrix."""
    n_components, labels = scipy.sparse.csgraph.connected_components(p)
    group = defaultdict(list)
    for i, ll in enumerate(labels):
        group[ll].append(i)
    return group


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

    data: np.ndarray
    block_labels: np.ndarray
    block_sizes: np.ndarray
    slice_begin: Optional[np.ndarray] = None
    slice_end: Optional[np.ndarray] = None

    def __post_init__(self):
        """Init method."""
        self.slice_end = np.cumsum(self.block_sizes**2)
        self.slice_begin = np.zeros_like(self.slice_end)
        self.slice_begin[1:] = self.slice_end[:-1]

    def get_data(self, idx: int):
        """Get data in projector for i-th block."""
        if self.slice_begin is None or self.slice_end is None:
            raise ValueError("Slice begin and end are not initialized.")
        s1 = self.slice_begin[idx]
        s2 = self.slice_end[idx]
        return self.data[s1:s2]

    def get_block_label(self, idx: int):
        """Get block label for i-th block."""
        return self.block_labels[idx]

    def get_block_size(self, idx: int):
        """Get block size for i-th block."""
        return self.block_sizes[idx]

    @property
    def n_blocks(self):
        """Return number of blocks in projector."""
        return len(self.block_labels)


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


def eigh_projector(
    p: np.ndarray,
    return_complement: bool = False,
    verbose: bool = True,
) -> Union[
    np.ndarray,
    tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
    None,
    tuple[None, None],
]:
    """Solve eigenvalue problem using numpy and eliminate eigenvectors with e < 1.0."""
    rank = int(round(np.trace(p)))
    if rank == 0:
        if return_complement:
            return None, None
        return None

    if rank < 32768:
        eigvals, eigvecs = np.linalg.eigh(p)
    else:
        if verbose:
            print("Eigsh_solver: lapack dsyevr is used.", flush=True)
        (syevr,) = get_lapack_funcs(("syevr",), ilp64=False)
        eigvals, eigvecs, _, _, _ = syevr(p, compute_v=True)

    tol = 1e-8
    if np.count_nonzero((eigvals > 1.0 + tol) | (eigvals < -tol)):
        raise ValueError("Eigenvalue error: e > 1 or e < 0.")

    nonzero = np.isclose(eigvals, 1.0)
    if return_complement:
        compr_bool = np.logical_not(nonzero)
        return (
            eigvecs[:, nonzero],
            (eigvals[compr_bool], eigvecs[:, compr_bool]),
        )
    return eigvecs[:, nonzero]


def eigsh_projector(p: csr_array, verbose: bool = True) -> csr_array:
    """Solve eigenvalue problem for matrix p.

    Return sparse matrix for eigenvectors of matrix p.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved. When p = diag(A,B), Av =
    v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w] are solutions.

    This function avoids solving numpy.eigh for duplicate block matrices. This
    function is efficient for matrix p composed of many duplicate block
    matrices.

    """
    p, compr_p = _compr_projector(p)
    group = _find_projector_blocks(p)
    if verbose:
        rank = int(round(sum(p.diagonal())))
        print("Rank of projector:", rank, flush=True)
        print("Number of blocks in projector:", len(group), flush=True)

    p_data = _extract_sparse_projector_data(p, group)
    uniq_eigvecs = dict()
    for i in range(p_data.n_blocks):
        p_block = p_data.get_data(i)
        block_label = p_data.get_block_label(i)
        block_size = p_data.get_block_size(i)
        if block_size > 1:
            key = tuple(p_block)
            try:
                uniq_eigvecs[key][1].append(block_label)
            except KeyError:
                p_np = p_block.reshape((block_size, block_size))
                eigvecs = eigh_projector(p_np, verbose=verbose)
                uniq_eigvecs[key] = [eigvecs, [block_label]]
        else:
            if not np.isclose(p_block[0], 0.0):
                if "one" in uniq_eigvecs:
                    uniq_eigvecs["one"][1].append(block_label)
                else:
                    uniq_eigvecs["one"] = [np.array([[1.0]]), [block_label]]

    c_p = _recover_eigvecs_from_uniq_eigvecs(uniq_eigvecs, group, p.shape[0])  # type: ignore
    if compr_p is not None:
        return compr_p @ c_p
    return c_p


def _block_eigh_projector(p_block: np.ndarray, verbose: bool = False):
    """Solve eigenvalue problem using block divisions."""
    eigvecs_block = np.zeros(p_block.shape, dtype="double")
    cmplt = np.zeros((p_block.shape[0], p_block.shape[0]), dtype="double")
    # TODO: memory allocation of cmlpt should be more efficient
    # cmplt = np.zeros((p_block.shape[0], p_block.shape[0] // 2), dtype="double")

    p_size = p_block.shape[0]
    target_size = min(max(p_size // 10, 1000), 3000)

    col_id, col_id_cmplt = 0, 0
    for begin, end in zip(*get_batch_slice(p_size, target_size)):
        if verbose:
            print("Block:", end, "/", p_size, flush=True)
        p_small = p_block[begin:end, begin:end]
        rank = int(round(np.trace(p_small)))
        if rank > 0:
            result = eigh_projector(
                p_small,
                return_complement=True,
                verbose=verbose,
            )
            assert result is not None
            assert result[0] is not None
            eigvecs, (cmplt_eigvals, cmplt_small) = result
            col_end = col_id + eigvecs.shape[1]
            col_end_cmplt = col_id_cmplt + cmplt_small.shape[1]
            eigvecs_block[begin:end, col_id:col_end] = eigvecs
            cmplt[begin:end, col_id_cmplt:col_end_cmplt] = cmplt_small
            p_block[begin:end, begin:end] -= eigvecs @ eigvecs.T
            col_id = col_end
            col_id_cmplt = col_end_cmplt
            if verbose:
                print(eigvecs.shape[1], "eigenvectors are found.", flush=True)

    rank = int(round(np.trace(p_block)))
    if rank > 0:
        if verbose:
            print("Solving complementary projector.", flush=True)
        cmplt = cmplt[:, :col_end_cmplt]
        p_block_rem = cmplt.T @ p_block @ cmplt
        eigvecs = eigh_projector(p_block_rem, verbose=verbose)
        eigvecs_shape1 = eigvecs.shape[1]  # type: ignore
        if verbose:
            print(eigvecs_shape1, "eigenvectors are found.", flush=True)
        if eigvecs_shape1 > 0:
            col_end = col_id + eigvecs_shape1
            eigvecs_block[:, col_id:col_end] = cmplt @ eigvecs
            col_id = col_end

    if col_id == 0:
        return None
    return eigvecs_block[:, :col_id]


def eigsh_projector_sumrule(
    p: csr_array,
    size_threshold: int = 1000,
    verbose: bool = True,
) -> np.ndarray:
    """Solve eigenvalue problem for matrix p.

    Return dense matrix for eigenvectors of matrix p.
    """
    if p.shape[0] > size_threshold:  # type: ignore
        return eigsh_projector_sumrule_large(p, verbose=verbose)
    return eigsh_projector_sumrule_stable(p, verbose=verbose)


def eigsh_projector_sumrule_stable(p: csr_array, verbose: bool = True) -> np.ndarray:
    """Solve eigenvalue problem for matrix p.

    Return dense matrix for eigenvectors of matrix p.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved.
    When p = diag(A,B), Av = v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w]
    are solutions.

    This function solves numpy.eigh for all block matrices.
    This function is efficient for matrix p composed of nonequivalent
    block matrices.

    """
    group = _find_projector_blocks(p)
    if verbose:
        print("Use standard normal eigsh solver.", flush=True)
        print("Number of blocks in projector (Sum rule):", len(group), flush=True)

    eigvecs_full = np.zeros(p.shape, dtype="double")  # type: ignore
    col_id = 0
    for i, ids in enumerate(group.values()):
        if verbose:
            print("Eigsh_solver_block:", i + 1, "/", len(group), flush=True)
            print(" - Block_size:", len(ids), flush=True)
        p_block = p[np.ix_(ids, ids)].toarray()
        rank = int(round(np.trace(p_block)))
        if rank > 0:
            eigvecs = eigh_projector(p_block, verbose=verbose)
            col_end = col_id + eigvecs.shape[1]  # type: ignore
            eigvecs_full[ids, col_id:col_end] = eigvecs
            col_id = col_end
    return eigvecs_full[:, :col_id]


def eigsh_projector_sumrule_large(p: csr_array, verbose: bool = True) -> np.ndarray:
    """Solve eigenvalue problem for matrix p.

    Return dense matrix for eigenvectors of matrix p.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved.
    When p = diag(A,B), Av = v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w]
    are solutions.

    This algorithm is optimized to solve an eigenvalue problem for
    projection matrix p with large block submatrices.
    The algorithm for solving each block submatrix is as follows.
    1. Divide each block submatrix into reasonable sizes of submatrices,
       not projectors.
    2. Solve eigenvalue problems for these submatrices and eigenvectors
       with eigenvalues of one are extracted. The eigenvectors with
       eigenvalues e < 1 are used for compressing the complementary matrix
       in the next step.
    3. Calculate the complementary matrix, which corresponds to the complement
       of the vector space spanned by the eigenvectors. The complementary matrix
       is compressed using the eigenvectors with e < 1.
    4. Solve eigenvalue problem for the complementary matrix and eigenvectors
       with e = 1 are calculated. The eigenvalue problems are efficiently solved
       using the compressed complementary matrix and the eigenvectors are recovered by
       the compression matrix.
    5. Collect all eigenvectors with e = 1.
    """
    group = _find_projector_blocks(p)
    if verbose:
        print("Use eigsh solver for large matrices.", flush=True)
        print("Number of blocks in projector (Sum rule):", len(group), flush=True)

    eigvecs_full = np.zeros(p.shape, dtype="double")  # type: ignore
    col_id = 0
    for i, ids in enumerate(group.values()):
        if verbose and len(ids) > 2:
            print("Eigsh_solver_block:", i + 1, "/", len(group), flush=True)
            print(" - Block_size:", len(ids), flush=True)
        ids = np.array(ids)
        p_block = p[np.ix_(ids, ids)].toarray()
        rank = int(round(np.trace(p_block)))
        if rank > 0:
            eigvecs = _block_eigh_projector(p_block, verbose=verbose)
            if eigvecs is not None:
                col_end = col_id + eigvecs.shape[1]
                eigvecs_full[ids, col_id:col_end] = eigvecs
                col_id = col_end
    return eigvecs_full[:, :col_id]
