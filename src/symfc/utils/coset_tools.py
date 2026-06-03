"""Utility functions for coset calculation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array


def kron_spg_reps(
    rows: NDArray, cols: NDArray, r_rep_csr: csr_array, factor: float, size1: int
):
    """Compute Kronecker product of atomic permutationand spg representation."""
    r_shape0, r_shape1 = r_rep_csr.shape
    r_coo = r_rep_csr.tocoo()
    r_rows, r_cols, r_data = r_coo.row, r_coo.col, r_coo.data * factor

    new_rows = np.add.outer(rows * r_shape0, r_rows).ravel()
    new_cols = np.add.outer(cols * r_shape1, r_cols).ravel()
    new_data = np.tile(r_data, len(rows))

    mat = csr_array(
        (new_data, (new_rows, new_cols)),
        shape=(size1 * r_shape0, size1 * r_shape1),
    )
    return mat
