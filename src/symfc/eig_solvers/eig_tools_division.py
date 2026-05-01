"""Functions for eigenvalue solutions using matrix divisions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from symfc.eig_solvers.eig_tools_core import (
    EigenvectorResult,
    eigh_projector,
    find_projector_blocks,
)
from symfc.eig_solvers.matrix import (
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


def _find_submatrix_eigenvectors(
    p: NDArray | csr_array,
    batch_size: int | None = None,
    use_mkl: bool = False,
    verbose: bool = False,
):
    """Find eigenvectors in division part of submatrix division algorithm."""
    p_size = p.shape[0]  # type: ignore
    # repeat = _should_repeat_division(p_size, depth)
    # batch_size = _calculate_batch_size_division(p_size, depth, batch_size)

    sibling, sibling_c = None, None
    col_id, col_id_c = 0, 0
    for begin, end in zip(*get_batch_slice(p_size, batch_size), strict=True):
        if verbose:
            print("- Block:", end, "/", p_size, flush=True)

        rows = np.arange(begin, end)
        p_small = p[begin:end, begin:end]
        res = eigh_projector(p_small, atol=1e-12, rtol=0.0, verbose=verbose)

        block_eigvecs = res.block_eigvecs
        if block_eigvecs is not None:
            sibling = link_block_matrix_nodes(
                block_eigvecs,
                sibling,
                rows=rows,
                col_begin=col_id,
            )
            col_id += res.n_eigvecs

        if verbose:
            print(res.n_eigvecs, "eigenvectors found.", flush=True)

        cmplt_eigvecs = res.cmplt_eigvecs
        if cmplt_eigvecs is not None:
            sibling_c = link_block_matrix_nodes(
                cmplt_eigvecs,
                sibling_c,
                rows=rows,
                col_begin=col_id_c,
                eigvals=res.cmplt_eigvals,
            )
            col_id_c += cmplt_eigvecs.shape[1]

    block_eigvecs, blocked_cmplt_eigvecs = None, None
    if col_id > 0:
        block_eigvecs = root_block_matrix(
            (p_size, col_id),
            first_child=sibling,
        )
    if col_id_c > 0:
        blocked_cmplt_eigvecs = root_block_matrix(
            (p_size, col_id_c),
            first_child=sibling_c,
        )
    return (
        EigenvectorResult(eigvecs=block_eigvecs),
        EigenvectorResult(eigvecs=blocked_cmplt_eigvecs),
    )


def eigsh_projector_division(
    p: NDArray | csr_array,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
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

    sibling = None
    col_id = 0
    compress_mat1, compress_mat2 = None, None
    for iter1 in range(300):
        if p.shape[0] < 1000:
            res = eigh_projector(p, atol=atol, rtol=rtol, verbose=verbose)
            print(res.block_eigvecs.shape)

            if iter1 < 2:
                block_eigvecs = res.block_eigvecs
            else:
                block_eigvecs = compress_mat2 @ res.block_eigvecs.recover()
            print(res.block_eigvecs.shape)

            sibling = link_block_matrix_nodes(
                block_eigvecs,
                sibling,
                rows=np.arange(p_size),
                col_begin=col_id,
                compress=compress_mat1,
            )
            col_id += res.n_eigvecs
            break

        batch_size = (iter1 + 1) * 300
        res, res_cmplt = _find_submatrix_eigenvectors(
            p,
            batch_size=batch_size,
            use_mkl=use_mkl,
            verbose=verbose,
        )

        if iter1 < 2:
            block_eigvecs = res.block_eigvecs
        else:
            block_eigvecs = compress_mat2 @ res.block_eigvecs.recover()

        sibling = link_block_matrix_nodes(
            block_eigvecs,
            sibling,
            rows=np.arange(p_size),
            col_begin=col_id,
            compress=compress_mat1,
        )
        col_id += res.n_eigvecs

        block_cmplt = res_cmplt.block_eigvecs
        p = block_cmplt.compress_matrix(p, use_mkl=use_mkl)
        if iter1 == 0:
            compress_mat1 = block_cmplt
        elif iter1 == 1:
            compress_mat2 = block_cmplt
        else:
            compress_mat2 = compress_mat2 @ block_cmplt.recover()

    block_eigvecs = root_block_matrix((p_size, col_id), first_child=sibling)
    return EigenvectorResult(eigvecs=block_eigvecs)


def solve_blocked_projector(
    p_block: csr_array,
    ids: NDArray,
    col_id: int,
    sibling: BlockMatrixNode | None,
    size_threshold: int = MIN_BLOCK_SIZE,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    use_mkl: bool = False,
    verbose: bool = False,
):
    """Solve."""
    if p_block.shape[0] < size_threshold:
        if verbose:
            print("Use standard eigh solver.", flush=True)
        p_block = p_block.toarray()
        res = eigh_projector(p_block, atol=atol, rtol=rtol, verbose=verbose)
    else:
        if verbose:
            print("Use submatrix version of eigh solver.", flush=True)
        res = eigsh_projector_division(
            p_block,
            atol=atol,
            rtol=rtol,
            use_mkl=use_mkl,
            verbose=verbose,
        )

    block_eigvecs = link_block_matrix_nodes(
        res.eigvecs,
        sibling,
        rows=ids,
        col_begin=col_id,
    )
    col_id += res.n_eigvecs
    return block_eigvecs, col_id


def _get_descending_order(group: dict):
    """Return descending order of eigenvector calculations."""
    lengths = [-len(ids) for ids in group.values()]
    order = np.array(list(group.keys()))[np.argsort(lengths)]
    return order


def eigsh_projector_sumrule(
    p: csr_array,
    size_threshold: int = MIN_BLOCK_SIZE,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    use_mkl: bool = False,
    verbose: bool = False,
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
        if matrix_rank(p_block) == 0:
            continue

        sibling, col_id = solve_blocked_projector(
            p_block,
            ids,
            col_id,
            sibling,
            atol=atol,
            rtol=rtol,
            size_threshold=size_threshold,
            use_mkl=use_mkl,
            verbose=verbose,
        )

    block = root_block_matrix((p.shape[0], col_id), first_child=sibling)  # type: ignore
    assert block is not None
    # if verbose:
    #     print("---------------------------------------------------", flush=True)
    #     print("Tree of FC basis block matrices:", flush=True)
    #     block.print_nodes()
    return block
