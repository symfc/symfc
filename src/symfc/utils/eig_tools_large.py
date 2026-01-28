"""Utility functions for large sparse eigenvalue solutions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from symfc.utils.eig_tools_core import (
    EigenvectorResult,
    eigh_projector,
    find_projector_blocks,
)
from symfc.utils.matrix import (
    BlockMatrixNode,
    link_block_matrix_nodes,
    matrix_rank,
    root_block_matrix,
)
from symfc.utils.solver_funcs import get_batch_slice

# Threshold constants for eigenvalue solvers
MIN_BLOCK_SIZE = 500
LARGE_BLOCK_SIZE = 5000
VERY_LARGE_BLOCK_SIZE = 30000
MAX_BATCH_SIZE = 20000

# Tolerance constants
DEFAULT_EIGVAL_TOL = 1e-8
# MIN_EIGVAL_THRESHOLD = 1e-12


def _should_repeat_division(
    p_size: int, depth: int, threshold_small: int = MIN_BLOCK_SIZE
) -> bool:
    """Determine if division should be repeated based on size and depth."""
    if p_size < threshold_small:
        return False
    elif p_size > VERY_LARGE_BLOCK_SIZE:
        return depth < 8
    else:
        return depth < 3


def _calculate_batch_size_division(
    p_size: int, depth: int, batch_size: int | None = None
) -> int:
    """Calculate batch size for submatrix division."""
    if batch_size is not None:
        return batch_size

    if depth == 1:
        return max(p_size // 10, MIN_BLOCK_SIZE)
    elif depth == 2:
        return max(p_size // 5, MIN_BLOCK_SIZE // 2)
    elif depth == 3:
        return p_size // 2
    else:
        return min(int(round(p_size / 1.3)), MAX_BATCH_SIZE)


def _find_submatrix_eigenvectors(
    p: NDArray | csr_array,
    batch_size: int | None = None,
    depth: int = 0,
    use_mkl: bool = False,
    verbose: bool = False,
):
    """Find eigenvectors in division part of submatrix division algorithm."""
    p_size = p.shape[0]
    repeat = _should_repeat_division(p_size, depth)
    batch_size = _calculate_batch_size_division(p_size, depth, batch_size)

    sibling, sibling_c = None, None
    col_id, col_id_c = 0, 0
    header = "  " * (depth - 1) + "(" + str(depth) + ")"
    for begin, end in zip(*get_batch_slice(p_size, batch_size), strict=True):
        if verbose:
            print(header, "Block:", end, "/", p_size, flush=True)

        p_small = p[begin:end, begin:end]
        rank = matrix_rank(p_small)
        if rank == 0:
            continue

        if repeat:
            res = eigh_projector_division(
                p_small,
                atol=1e-12,
                rtol=0.0,
                depth=depth,
                use_mkl=use_mkl,
                verbose=verbose,
            )
        else:
            res = eigh_projector(
                p_small,
                atol=1e-12,
                rtol=0.0,
                verbose=verbose,
            )
        block = res.eigvecs
        cmplt_eigvals = res.cmplt_eigvals
        cmplt_vecs = res.cmplt_eigvecs

        if verbose:
            print(header, " ", res.n_eigvecs, "eigenvectors found.", flush=True)

        rows = np.arange(begin, end)
        sibling = link_block_matrix_nodes(block, sibling, rows=rows, col_begin=col_id)
        col_id += res.n_eigvecs

        if cmplt_vecs is not None:
            sibling_c = link_block_matrix_nodes(
                cmplt_vecs,
                sibling_c,
                rows=rows,
                col_begin=col_id_c,
                eigvals=cmplt_eigvals,
            )
            col_id_c += cmplt_vecs.shape[1]

    if col_id_c > 0:
        cmplt = root_block_matrix((p_size, col_id_c), first_child=sibling_c)
    else:
        cmplt = None

    # block = root_block_matrix((p_size, col_id), first_child=sibling)
    # return EigenvectorResult(eigvecs=block), cmplt
    # return EigenvectorResult(eigvecs=block), col_id, cmplt
    # return sibling, col_id, cmplt
    return sibling, col_id, cmplt


def _calculate_batch_size_complement(cmplt_size: int, p_size: int, depth: int) -> int:
    """Calculate batch size for complement eigenvector computation."""
    if depth == 1:
        return max(cmplt_size // 3, p_size // 15)
    elif depth == 2:
        return int(round(cmplt_size / 1.5))
    elif depth == 3:
        return int(round(cmplt_size / 1.3))
    else:
        return min(int(round(cmplt_size / 1.2)), MAX_BATCH_SIZE)


def _find_complement_eigenvectors(
    p: NDArray | csr_array,
    sibling: BlockMatrixNode | None,
    col_id: int,
    cmplt: BlockMatrixNode,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    depth: int = 0,
    return_cmplt: bool = True,
    use_mkl: bool = False,
    verbose: bool = False,
) -> EigenvectorResult:
    """Find eigenvectors in complementary part of submatrix division algorithm."""
    p_size = p.shape[0]
    repeat = _should_repeat_division(p_size, depth, threshold_small=LARGE_BLOCK_SIZE)
    if p_size >= LARGE_BLOCK_SIZE and p_size <= VERY_LARGE_BLOCK_SIZE:
        repeat = depth < 4  # slightly different threshold for complement

    batch_size_cmplt = _calculate_batch_size_complement(cmplt.shape[1], p_size, depth)

    header = "  " * (depth - 1) + "(" + str(depth) + ")"
    if verbose:
        print(header, "Complementary block size:", cmplt.shape[1], flush=True)
        print(header, "Compute compressed projector.", flush=True)

    p_cmr = cmplt.compress_matrix(p, use_mkl=use_mkl)
    if not repeat:
        if verbose:
            print(header, "Use standard solver.", flush=True)
        res = eigh_projector(
            p_cmr,
            atol=atol,
            rtol=rtol,
            verbose=verbose,
        )
    else:
        if verbose:
            print(header, "Use submatrix size of", batch_size_cmplt, flush=True)
        res = eigh_projector_division(
            p_cmr,
            atol=atol,
            rtol=rtol,
            depth=depth,
            batch_size=batch_size_cmplt,
            return_cmplt=return_cmplt,
            use_mkl=use_mkl,
            verbose=verbose,
        )

    if verbose:
        print(header, " ", res.n_eigvecs, "eigenvectors found.", flush=True)

    sibling = link_block_matrix_nodes(
        res.eigvecs,
        sibling,
        rows=np.arange(p_size),
        col_begin=col_id,
        compress=cmplt,
    )
    col_id += res.n_eigvecs

    cmplt_eigvals, cmplt_eigvecs = None, None
    if return_cmplt:
        cmplt_small = res.cmplt_eigvecs
        if cmplt_small is not None and cmplt_small.shape[1] > 0:
            cmplt_eigvals = res.cmplt_eigvals
            cmplt_eigvecs = cmplt.dot(cmplt_small)

    return EigenvectorResult(
        eigvecs=sibling,
        col_id=col_id,
        cmplt_eigvals=cmplt_eigvals,
        cmplt_eigvecs=cmplt_eigvecs,
    )


def _run_division(
    p: NDArray | csr_array,
    batch_size: int | None = None,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    depth: int = 0,
    return_cmplt: bool = True,
    use_mkl: bool = False,
    verbose: bool = False,
) -> EigenvectorResult:
    """Find eigenvectors in division and complementary parts."""
    depth += 1
    # res, cmplt = _find_submatrix_eigenvectors(
    sibling, col_id, cmplt = _find_submatrix_eigenvectors(
        p,
        batch_size=batch_size,
        depth=depth,
        use_mkl=use_mkl,
        verbose=verbose,
    )
    # col_id = res.n_eigvecs
    # sibling = res.block_eigvecs

    cmplt_eigvals, cmplt_eigvecs = None, None
    if cmplt is not None:
        result = _find_complement_eigenvectors(
            p,
            sibling,
            col_id,
            cmplt,
            atol=atol,
            rtol=rtol,
            depth=depth,
            return_cmplt=return_cmplt,
            use_mkl=use_mkl,
            verbose=verbose,
        )
        sibling = result.eigvecs
        col_id = result.col_id
        cmplt_eigvals = result.cmplt_eigvals
        cmplt_eigvecs = result.cmplt_eigvecs
        # col_id += result.n_eigvecs

        # sibling = link_block_matrix_nodes(
        #     res.eigvecs,
        #     sibling,
        #     rows=np.arange(p_size),
        #     col_begin=col_id,
        #     compress=cmplt,
        # )

    block = root_block_matrix((p.shape[0], col_id), first_child=sibling)
    return EigenvectorResult(
        eigvecs=block,
        cmplt_eigvals=cmplt_eigvals,
        cmplt_eigvecs=cmplt_eigvecs,
    )


def eigh_projector_division(
    p: NDArray | csr_array,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    depth: int = 0,
    return_cmplt: bool = True,
    batch_size: int | None = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> EigenvectorResult:
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
    p_size = p.shape[0]
    if p_size < MIN_BLOCK_SIZE:
        return eigh_projector(p, atol=atol, rtol=rtol, verbose=verbose)

    result = _run_division(
        p,
        batch_size=batch_size,
        atol=atol,
        rtol=rtol,
        depth=depth,
        return_cmplt=return_cmplt,
        use_mkl=use_mkl,
        verbose=verbose,
    )
    return result


def _get_descending_order(group: dict):
    """Return descending order of eigenvector calculations."""
    lengths = [-len(ids) for ids in group.values()]
    order = np.array(list(group.keys()))[np.argsort(lengths)]
    return order


def eigsh_projector_sumrule(
    p: csr_array,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    size_threshold: int = MIN_BLOCK_SIZE,
    use_mkl: bool = False,
    verbose: bool = True,
) -> BlockMatrixNode:
    """Solve eigenvalue problem for matrix p.

    Return block matrix for eigenvectors of matrix p.
    This function should be used for sparse matrix but its solution
    is composed of dense block matrices.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved.
    When p = diag(A,B), Av = v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w]
    are solutions.

    If p.shape[0] < size_threshold, this function solves numpy.eigh for all block
    matrices. Otherwise, this function use a submatrix division algorithm
    to solve the eigenvalue problem of each block matrix.
    """
    group = find_projector_blocks(p, verbose=verbose)
    order = _get_descending_order(group)
    if verbose:
        print("Number of blocks in projector (Sum rule):", len(group), flush=True)

    sibling, col_id = None, 0
    for i, key in enumerate(order):
        ids = np.array(group[key])
        if verbose and len(ids) > 0:
            prefix = "------------ Eigsh_solver_block:"
            print(prefix, i + 1, "/", len(group), "------------", flush=True)
            print("Block_size:", len(ids), flush=True)

        p_block = p[np.ix_(ids, ids)]
        rank = matrix_rank(p_block)
        if rank == 0:
            if verbose:
                print("No eigenvectors.", flush=True)
            continue

        if p_block.shape[0] < size_threshold:
            if verbose:
                print("Use standard eigh solver.", flush=True)
            p_block = p_block.toarray()
            res = eigh_projector(p_block, atol=atol, rtol=rtol, verbose=verbose)
            sibling = link_block_matrix_nodes(
                res.eigvecs, sibling, rows=ids, col_begin=col_id
            )
            col_id += res.n_eigvecs

        else:
            if verbose:
                print("Use submatrix version of eigh solver.", flush=True)
            res = eigh_projector_division(
                p_block,
                atol=atol,
                rtol=rtol,
                return_cmplt=False,
                use_mkl=use_mkl,
                verbose=verbose,
            )
            sibling = link_block_matrix_nodes(
                res.eigvecs, sibling, rows=ids, col_begin=col_id
            )
            col_id += res.n_eigvecs

        del p_block

    block = root_block_matrix((p.shape[0], col_id), first_child=sibling)
    if verbose:
        print("---------------------------------------------------", flush=True)
        print("Tree of FC basis block matrices:", flush=True)
        block.print_nodes()
    return block
