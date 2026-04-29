"""Solver utility functions."""

from __future__ import annotations

from numpy.typing import NDArray

from symfc.utils.solver_funcs import get_batch_slice


def calc_sum_xtx(
    xtx: NDArray | None,
    x: NDArray,
    nbytes_threshold: float = 5.0,
    verbose: bool = False,
):
    """Add X_i.T @ X_i to large X.T @ X using batch calculation."""
    if xtx is None:
        return x.T @ x

    if nbytes_threshold < 0.0:
        nbytes_threshold = 0.0

    mem_size = (xtx.nbytes + x.nbytes) * 1e-9
    if mem_size < nbytes_threshold:
        xtx += x.T @ x
        return xtx

    try:
        n_batch = min(int(mem_size / nbytes_threshold), 10) + 1
    except ZeroDivisionError:
        n_batch = 2

    size = x.shape[1]
    batch_size = size // n_batch
    begin_ids, end_ids = get_batch_slice(size, batch_size)
    for i, (begin_row, end_row) in enumerate(zip(begin_ids, end_ids, strict=True)):
        if verbose:
            print("- Batch:", end_row, "/", size, flush=True)

        x1 = x[:, begin_row:end_row]
        for j in range(i, len(begin_ids)):
            begin_col, end_col = begin_ids[j], end_ids[j]
            prod = x1.T @ x[:, begin_col:end_col]
            xtx[begin_row:end_row, begin_col:end_col] += prod

            if i < j:
                xtx[begin_col:end_col, begin_row:end_row] += prod.T
    return xtx
