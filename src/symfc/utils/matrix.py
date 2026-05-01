"""Utility functions for matrices."""

from __future__ import annotations

from numpy.typing import NDArray
from scipy.sparse import csr_array

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
    """Compute dot-product of sparse matrices."""
    if use_mkl:
        try:
            return dot_product_mkl(A, B, dense=dense)
        except NameError:
            pass
    return A @ B
