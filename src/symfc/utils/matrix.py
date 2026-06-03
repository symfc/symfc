"""Utility functions for matrices."""

from __future__ import annotations

from numpy.typing import NDArray
from scipy.sparse import csr_array, hstack, vstack

try:
    from sparse_dot_mkl import dot_product_mkl  # type: ignore
except ImportError:
    pass


def dot_product_sparse(
    A: NDArray | csr_array,
    B: NDArray | csr_array,
    use_mkl: bool = False,
    dense: bool = False,
) -> csr_array:
    """Compute dot-product of sparse matrices.

    dense option is enabled if use_mkl = True and dot_product_mkl is found.
    """
    if use_mkl:
        try:
            return dot_product_mkl(A, B, dense=dense)
        except NameError:
            pass
    return A @ B


def blocked_triple_product(cpt_array: list, mat: csr_array, use_mkl: bool = False):
    """Calculate c_pt.T @ mat @ c_pt.

    Input matrix `mat` is overwritten.

    Input
    -----
    cpt_array: Column-stacked list of matrix, cpt_array = [c_pt1, c_pt2, ...].
    mat: Central matrix for triple product.
    """
    if len(cpt_array) == 1:
        basis = cpt_array[0]
        temp = dot_product_sparse(mat, basis, use_mkl=use_mkl)
        mat = dot_product_sparse(basis.T, temp, use_mkl=use_mkl)
        return mat

    rows = []
    for c_pt1 in cpt_array:
        blk_i = dot_product_sparse(c_pt1.T, mat, use_mkl=use_mkl)
        row_blocks = [
            dot_product_sparse(blk_i, c_pt2, use_mkl=use_mkl) for c_pt2 in cpt_array
        ]
        rows.append(hstack(row_blocks))

    blk_mat = vstack(rows).tocsr()
    return blk_mat


def blocked_product(cpt_array: list, mat: csr_array, use_mkl: bool = False):
    """Calculate c_pt @ mat.

    Input
    -----
    cpt_array: Column-stacked list of matrix, cpt_array = [c_pt1, c_pt2, ...].
    mat: Central matrix for double product.
    """
    if len(cpt_array) == 1:
        return dot_product_sparse(cpt_array[0], mat, use_mkl=use_mkl)

    shape = (cpt_array[0].shape[0], mat.shape[1])
    res = csr_array(shape, dtype="double")
    start = 0
    for c_pt1 in cpt_array:
        end = start + c_pt1.shape[1]
        res += dot_product_sparse(c_pt1, mat[start:end], use_mkl=use_mkl)
        start = end
    return res
