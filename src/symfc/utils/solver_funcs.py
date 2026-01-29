"""Solver functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg.lapack import get_lapack_funcs
from scipy.sparse import csr_array


def solve_linear_equation(A: NDArray, b: NDArray) -> NDArray:
    """Solve linear equation using lapack in scipy.

    numpy and scipy implementations
    x = np.linalg.solve(A, b)
    x = scipy.linalg.solve(A, b, check_finite=False, assume_a='pos')

    """
    (posv,) = get_lapack_funcs(("posv",), (A, b))
    _, x, _ = posv(A, b, lower=False, overwrite_a=False, overwrite_b=False)
    return x


def fit(X: NDArray, y: NDArray) -> NDArray:
    """Solve a normal equation in least-squares.

    (X.T @ X) @ coefs = X.T @ y

    n_samples, n_features = X.shape

    """
    coefs = solve_linear_equation(X.T @ X, X.T @ y)
    return coefs


def get_batch_slice(n_data: int, batch_size: int) -> tuple[list[int], list[int]]:
    """Calculate slice indices for a given batch size."""
    begin_batch = list(range(0, n_data, batch_size))
    if len(begin_batch) > 1:
        end_batch = list(begin_batch[1:]) + [n_data]
        if (end_batch[-1] - end_batch[-2]) < batch_size // 5:
            end_batch[-2] = end_batch[-1]
            begin_batch = begin_batch[:-1]
            end_batch = end_batch[:-1]
    else:
        end_batch = [n_data]
    return begin_batch, end_batch


def get_displacement_sparse_matrix(
    atoms: NDArray,
    displacements: NDArray,
    n_atom: int,
    tol: float = 1e-15,
) -> csr_array:
    """Return sparse matrix with displacements.

    Parameter
    ---------
    atoms: Indices of atoms displaced, shape = (n_snapshot).
    displacements: Displacement vectors, shape = (n_snapshot, 3).
    n_atom: Number of atoms in structure.
    tol: Tolerance value for defining nonzero elements.
    """
    if atoms.shape[0] != displacements.shape[0]:
        raise RuntimeError("Sizes of atoms and displacements are inconsistent.")

    N3 = n_atom * 3
    nonzero = np.abs(displacements) > tol
    rows, cols = np.where(nonzero)
    cols += atoms[rows] * 3

    mat = csr_array(
        (displacements[nonzero], (rows, cols)),
        shape=(displacements.shape[0], N3),
        dtype="double",
    )
    return mat
