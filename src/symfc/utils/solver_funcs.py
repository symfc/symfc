"""Solver functions."""

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs


def solve_linear_equation(A: np.ndarray, b: np.ndarray):
    """Solve linear equation using lapack in scipy.

    numpy and scipy implementations
    x = np.linalg.solve(A, b)
    x = scipy.linalg.solve(A, b, check_finite=False, assume_a='pos')

    """
    (posv,) = get_lapack_funcs(("posv",), (A, b))
    _, x, _ = posv(A, b, lower=False, overwrite_a=False, overwrite_b=False)
    return x


def fit(X: np.ndarray, y: np.ndarray):
    """Solve a normal equation in least-squares.

    (X.T @ X) @ coefs = X.T @ y

    n_samples, n_features = X.shape

    """
    A = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    coefs = solve_linear_equation(A, Xy)
    return coefs


def get_batch_slice(n_data: int, batch_size: int):
    """Calculate slice indices for a given batch size."""
    begin_batch = list(range(0, n_data, batch_size))
    if len(begin_batch) > 1:
        end_batch = list(begin_batch[1:]) + [n_data]
    else:
        end_batch = [n_data]
    return begin_batch, end_batch
