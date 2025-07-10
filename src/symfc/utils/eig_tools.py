"""Utility functions for eigenvalue solutions."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy
from scipy.sparse import csr_array

from symfc.utils.matrix import BlockMatrix, BlockMatrixNode, append_node
from symfc.utils.solver_funcs import get_batch_slice


def eigh_projector(
    p: np.ndarray,
    return_cmplt: bool = False,
    atol: float = 1e-8,
    rtol: float = 0.0,
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
        if return_cmplt:
            return None, None
        return None

    if rank < 32768:
        eigvals, eigvecs = np.linalg.eigh(p)
    else:
        raise RuntimeError("Projector rank is too large in eigh.")

    tol = 1e-8
    if np.count_nonzero((eigvals > 1.0 + tol) | (eigvals < -tol)):
        raise ValueError("Eigenvalue error: e > 1 or e < 0.")

    nonzero = np.isclose(eigvals, 1.0, atol=atol, rtol=rtol)
    if return_cmplt:
        compr_bool = np.logical_and(np.logical_not(nonzero), eigvals > 1e-12)
        return (
            eigvecs[:, nonzero],
            (eigvals[compr_bool], eigvecs[:, compr_bool]),
        )
    return eigvecs[:, nonzero]


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
    # from symfc.utils.graph import connected_components
    # n_components, labels = connected_components(p, verbose=True)
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


def eigsh_projector(
    p: csr_array,
    atol: float = 1e-8,
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
                eigvecs = eigh_projector(p_np, atol=atol, rtol=rtol, verbose=verbose)
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


def eigh_projector_use_submatrix(
    p: np.ndarray,
    return_cmplt: bool = False,
    atol: float = 1e-8,
    rtol: float = 0.0,
    target_size: Optional[int] = None,
    repeat: bool = True,
    verbose: bool = False,
):
    """Solve eigenvalue problem for numpy array."""
    p_size = p.shape[0]
    if p_size < 2000:
        return eigh_projector(
            p,
            return_cmplt=return_cmplt,
            atol=atol,
            rtol=rtol,
            verbose=verbose,
        )

    sibling, sibling_c = None, None
    col_id, col_id_cmplt = 0, 0
    if target_size is None:
        target_size = min(p_size // 5, 10000)

    for begin, end in zip(*get_batch_slice(p_size, target_size)):
        if verbose:
            print("- Block:", end, "/", p_size, flush=True)
        p_small = p[begin:end, begin:end]
        rank = int(round(np.trace(p_small)))
        if rank > 0:
            result = eigh_projector(
                p_small,
                return_cmplt=True,
                atol=1e-12,
                rtol=0.0,
                verbose=verbose,
            )
            eigvecs, (cmplt_eigvals, cmplt_small) = result
            if eigvecs is not None:
                if verbose:
                    print("  eigenvectors:", eigvecs.shape[1], flush=True)

                sibling = append_node(
                    eigvecs, sibling, rows=np.arange(begin, end), col_begin=col_id
                )
                col_id += eigvecs.shape[1]
            if cmplt_small is not None:
                sibling_c = append_node(
                    cmplt_small,
                    sibling_c,
                    rows=np.arange(begin, end),
                    col_begin=col_id_cmplt,
                )
                col_id_cmplt += cmplt_small.shape[1]

    if col_id_cmplt > 0:
        if verbose:
            print("- Block_complement_size:", col_id_cmplt, flush=True)

        cmplt = BlockMatrixNode(
            rows=np.arange(p_size),
            col_begin=0,
            col_end=col_id_cmplt,
            first_child=sibling_c,
            root=True,
        )
        if not repeat or cmplt.shape[1] < 20000:
            result = eigh_projector(
                cmplt.compress_matrix(p),
                atol=atol,
                rtol=rtol,
                return_cmplt=True,
                verbose=verbose,
            )
        else:
            target_size = min(cmplt.shape[1] // 2, 20000)
            result = eigh_projector_use_submatrix(
                cmplt.compress_matrix(p),
                atol=atol,
                rtol=rtol,
                target_size=target_size,
                repeat=False,
                return_cmplt=True,
                verbose=verbose,
            )

        assert result is not None
        assert result[0] is not None
        eigvecs, (cmplt_eigvals, cmplt_small) = result

        if eigvecs is not None:
            if verbose:
                print("  eigenvectors:", eigvecs.shape[1], flush=True)

            sibling = append_node(
                eigvecs,
                sibling,
                rows=np.arange(p_size),
                col_begin=col_id,
                compress=cmplt,
            )
            col_id += eigvecs.shape[1]

        if return_cmplt:
            if cmplt_small is not None and cmplt_small.shape[1] > 0:
                cmplt_eigvecs = cmplt.dot(cmplt_small)
            else:
                cmplt_eigvals, cmplt_eigvecs = None, None
    else:
        if return_cmplt:
            cmplt_eigvals, cmplt_eigvecs = None, None

    block = BlockMatrixNode(
        rows=np.arange(p_size),
        col_begin=0,
        col_end=col_id,
        first_child=sibling,
        root=True,
    )
    eigvecs = block.recover()

    if return_cmplt:
        return eigvecs, (cmplt_eigvals, cmplt_eigvecs)
    return eigvecs


def _solve_use_submatrix(p_block: csr_array, verbose: bool = False):
    """Solve eigenvalue problem for submatrix parts."""
    sibling, sibling_c = None, None
    col_id, col_id_cmplt = 0, 0
    p_size = p_block.shape[0]
    target_size = max(p_size // 10, 500)

    for begin, end in zip(*get_batch_slice(p_size, target_size)):
        if verbose:
            print("Block:", end, "/", p_size, flush=True)

        p_small = p_block[begin:end, begin:end].toarray()
        rank = int(round(np.trace(p_small)))
        if rank > 0:
            # Numerical noise may increase in eigh_projector_use_submatrix.
            result = eigh_projector_use_submatrix(
                p_small,
                return_cmplt=True,
                atol=1e-12,
                rtol=0.0,
                verbose=verbose,
            )
            assert result is not None
            assert result[0] is not None
            eigvecs, (cmplt_eigvals, cmplt_small) = result
            if eigvecs is not None:
                if verbose:
                    print(eigvecs.shape[1], "eigenvectors are found.", flush=True)
                sibling = append_node(
                    eigvecs, sibling, rows=np.arange(begin, end), col_begin=col_id
                )
                col_id += eigvecs.shape[1]
            if cmplt_small is not None:
                sibling_c = append_node(
                    cmplt_small,
                    sibling_c,
                    rows=np.arange(begin, end),
                    col_begin=col_id_cmplt,
                )
                col_id_cmplt += cmplt_small.shape[1]

    cmplt = BlockMatrixNode(
        rows=np.arange(p_size),
        col_begin=0,
        col_end=col_id_cmplt,
        first_child=sibling_c,
        root=True,
    )
    return sibling, col_id, cmplt


def eigsh_projector_use_submatrix(
    p_block: csr_array,
    atol: float = 1e-8,
    rtol: float = 0.0,
    size: Optional[int] = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> BlockMatrixNode:
    r"""Solve eigenvalue problem using submatrix division algorithm.

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
    p_size = p_block.shape[0]
    sibling, col_id, cmplt = _solve_use_submatrix(p_block, verbose=verbose)
    if cmplt.shape[1] > 0:
        if verbose:
            print("Solve complementary projector.", flush=True)
            print("Complementary block size:", cmplt.shape[1], flush=True)

        p_block_cmr = cmplt.compress_matrix(p_block, use_mkl=use_mkl)
        size_cmplt = min(max(p_block_cmr.shape[0] // 3, p_size // 15), 20000)
        if verbose:
            print("Submatrix size for complementary block:", size_cmplt, flush=True)
        eigvecs = eigh_projector_use_submatrix(
            p_block_cmr,
            atol=atol,
            rtol=rtol,
            target_size=size_cmplt,
            verbose=verbose,
        )
        if eigvecs is not None:
            if verbose:
                print(eigvecs.shape[1], "eigenvectors are found.", flush=True)
            sibling = append_node(
                eigvecs,
                sibling,
                rows=np.arange(p_size),
                col_begin=col_id,
                compress=cmplt,
            )
            col_id += eigvecs.shape[1]

    if col_id == 0:
        return None

    return BlockMatrixNode(
        rows=np.arange(p_size),
        col_begin=0,
        col_end=col_id,
        first_child=sibling,
        root=True,
    )


def eigsh_projector_sumrule(
    p: csr_array,
    atol: float = 1e-8,
    rtol: float = 0.0,
    size_threshold: int = 500,
    use_mkl: bool = False,
    verbose: bool = True,
) -> BlockMatrix:
    """Solve eigenvalue problem for matrix p.

    Return dense matrix for eigenvectors of matrix p.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved.
    When p = diag(A,B), Av = v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w]
    are solutions.

    If p.shape[0] < size_threshold, this function solves numpy.eigh for all block
    matrices. Otherwise, this function use a submatrix division algorithm
    to solve the eigenvalue problem of each block matrix.
    """
    group = _find_projector_blocks(p)
    lengths = [-len(ids) for ids in group.values()]
    order = np.array(list(group.keys()))[np.argsort(lengths)]
    if verbose:
        print("Number of blocks in projector (Sum rule):", len(group), flush=True)

    sibling, col_id = None, 0
    for i, key in enumerate(order):
        ids = np.array(group[key])
        if verbose and len(ids) > 0:
            print("Eigsh_solver_block:", i + 1, "/", len(group), flush=True)
            print("- Block_size:", len(ids), flush=True)

        p_block = p[np.ix_(ids, ids)]
        rank = int(round(p_block.trace()))
        if rank > 0 and p_block.shape[0] < size_threshold:
            if verbose:
                print("Use standard eigh solver.", flush=True)
            p_block = p_block.toarray()
            eigvecs = eigh_projector(p_block, atol=atol, rtol=rtol, verbose=verbose)
            if eigvecs is not None:
                sibling = append_node(eigvecs, sibling, rows=ids, col_begin=col_id)
                col_id += eigvecs.shape[1]

        elif rank > 0 and p_block.shape[0] >= size_threshold:
            if verbose:
                print("Use submatrix version of eigsh solver.", flush=True)
            block = eigsh_projector_use_submatrix(
                p_block, atol=atol, rtol=rtol, use_mkl=use_mkl, verbose=verbose
            )
            if block.shape[1] > 0:
                block.next_sibling = sibling
                block.rows = ids
                block.col_begin = col_id
                block.col_end = col_id + block.shape[0]
                block.root = False

                sibling = block
                col_id += block.shape[1]

        del p_block

    return BlockMatrixNode(
        rows=np.arange(p.shape[0]),
        col_begin=0,
        col_end=col_id,
        first_child=sibling,
        root=True,
    )
