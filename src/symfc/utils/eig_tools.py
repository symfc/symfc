"""Utility functions for eigenvalue solutions."""

from __future__ import annotations

from numpy.typing import NDArray
from scipy.sparse import csr_array

from symfc.utils.eig_tools_core import eigh_projector as eigh_standard
from symfc.utils.eig_tools_sparse import eigsh_projector as eigsh_standard
from symfc.utils.matrix import BlockMatrixNode

# from symfc.utils.eig_tools_large import eigh_projector as eigsh_standard


# Threshold constants for eigenvalue solvers
MIN_BLOCK_SIZE = 500
LARGE_BLOCK_SIZE = 5000
VERY_LARGE_BLOCK_SIZE = 30000
MAX_BATCH_SIZE = 20000
MAX_PROJECTOR_RANK = 32767
SPARSE_DATA_LIMIT = 2147483647

# Tolerance constants
DEFAULT_EIGVAL_TOL = 1e-8


def eigh_projector(
    p: NDArray | csr_array,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    verbose: bool = True,
) -> NDArray:
    """Solve eigenvalue problem using numpy and eliminate eigenvectors with e < 1.0."""
    return eigh_standard(p, atol=atol, rtol=rtol, verbose=verbose).eigvecs


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
    return eigsh_standard(p, atol=atol, rtol=rtol, verbose=verbose)


def eigsh_projector_sumrule(
    p: csr_array,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    size_threshold: int = MIN_BLOCK_SIZE,
    use_mkl: bool = False,
    verbose: bool = True,
) -> BlockMatrixNode:
    r"""Solve eigenvalue problem for matrix p.

    Return dense matrix for eigenvectors of matrix p.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved.
    When p = diag(A,B), Av = v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w]
    are solutions.

    If p.shape[0] < size_threshold, this function solves numpy.eigh for all block
    matrices. Otherwise, this function use a submatrix division algorithm
    to solve the eigenvalue problem of each block matrix.

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
    pass
