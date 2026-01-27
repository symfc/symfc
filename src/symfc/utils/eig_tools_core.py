"""Utility functions for dense eigenvalue solutions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.sparse import csr_array

from symfc.utils.graph import connected_components
from symfc.utils.matrix import (
    BlockMatrixNode,
    matrix_rank,
    return_numpy_array,
    root_block_matrix,
)

# Threshold constants for eigenvalue solvers
MAX_PROJECTOR_RANK = 32767
SPARSE_DATA_LIMIT = 2147483647

# Tolerance constants
DEFAULT_EIGVAL_TOL = 1e-8
SYMMETRY_TOL_STRICT = 1e-15
SYMMETRY_TOL_LOOSE = 1e-3
# MIN_EIGVAL_THRESHOLD = 1e-12


@dataclass
class EigenvectorResult:
    """Result of eigenvector computation."""

    eigvecs: NDArray | BlockMatrixNode | None
    cmplt_eigvals: NDArray | None = None
    cmplt_eigvecs: NDArray | None = None
    col_id: int | None = None

    @property
    def block_eigvecs(self):
        """Return eigenvectors in BlockMatrixNode."""
        if self.eigvecs is None:
            return None
        if isinstance(self.eigvecs, BlockMatrixNode):
            return self.eigvecs
        return root_block_matrix(data=self.eigvecs)

    @property
    def numpy_eigvecs(self):
        """Return eigenvectors in BlockMatrixNode."""
        if self.eigvecs is None:
            return None
        if isinstance(self.eigvecs, BlockMatrixNode):
            return self.eigvecs.recover()
        return self.eigvecs


def eigh_projector(
    p: NDArray | csr_array,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
    verbose: bool = True,
) -> EigenvectorResult:
    """Solve eigenvalue problem using numpy and eliminate eigenvectors with e < 1.0."""
    p = return_numpy_array(p)
    rank = matrix_rank(p)
    if rank == 0:
        return EigenvectorResult(eigvecs=None)

    if rank > MAX_PROJECTOR_RANK:
        raise RuntimeError("Projector rank is too large in eigh.")

    eigvals, eigvecs = _solve_eigh(p, tol=atol, verbose=verbose)
    return _divide_eigenvectors(eigvals, eigvecs, atol=atol, rtol=rtol)


def _solve_eigh(
    proj: NDArray,
    tol: float = DEFAULT_EIGVAL_TOL,
    verbose: bool = True,
) -> tuple[NDArray, NDArray]:
    """Solve eigenvalue problem with fallback to LAPACK dsyev."""
    try:
        eigvals, eigvecs = np.linalg.eigh(proj)
    except np.linalg.LinAlgError as e:
        if verbose:
            print(f"np.linalg.eigh failed: {str(e)}")
            print("Try scipy.linalg.lapack.dsyev")
        eigvals, eigvecs, info = scipy.linalg.lapack.dsyev(proj.T)
        if info != 0:
            raise scipy.linalg.LinAlgError(
                "scipy.linalg.lapack.dsyev failed: Eigenvalues did not converge"
            ) from e

    eigvals, eigvecs = _validate_eigenvalues(eigvals, eigvecs, proj, tol=tol)
    return eigvals, eigvecs


def _validate_eigenvalues(
    eigvals: NDArray,
    eigvecs: NDArray,
    proj: NDArray,
    tol: float = DEFAULT_EIGVAL_TOL,
) -> tuple[NDArray, NDArray]:
    """Validate eigenvalues and symmetrize matrix if needed."""
    if np.count_nonzero((eigvals > 1.0 + tol) | (eigvals < -tol)):
        diff = np.abs(proj - proj.T)
        if np.any(diff > SYMMETRY_TOL_LOOSE):
            raise RuntimeError("Transpose equality not satisfied")
        elif np.any(diff > SYMMETRY_TOL_STRICT):
            eigvals, eigvecs = np.linalg.eigh(0.5 * (proj + proj.T))

    if np.count_nonzero((eigvals > 1.0 + tol) | (eigvals < -tol)):
        raise ValueError("Eigenvalue error: e > 1 or e < 0.")

    return eigvals, eigvecs


def _divide_eigenvectors(
    eigvals: NDArray,
    eigvecs: NDArray,
    atol: float = DEFAULT_EIGVAL_TOL,
    rtol: float = 0.0,
) -> EigenvectorResult:
    """Divide eigenvectors into those with eigenvalues one and their complements."""
    is_one = np.isclose(eigvals, 1.0, atol=atol, rtol=rtol)
    is_complement = (~is_one) & (eigvals > atol)

    one_eigvecs = eigvecs[:, is_one] if np.any(is_one) else None

    if np.any(is_complement):
        cmplt_eigvals = eigvals[is_complement]
        cmplt_eigvecs = eigvecs[:, is_complement]
    else:
        cmplt_eigvals = None
        cmplt_eigvecs = None

    return EigenvectorResult(
        eigvecs=one_eigvecs,
        cmplt_eigvals=cmplt_eigvals,
        cmplt_eigvecs=cmplt_eigvecs,
    )


def find_projector_blocks(p: csr_array, verbose: bool = False) -> dict:
    """Find block structures in projection matrix."""
    if verbose:
        print("Finding block diagonal structure in projector.", flush=True)

    if len(p.data) < SPARSE_DATA_LIMIT:
        if verbose:
            print("Using scipy connected_components.", flush=True)
        n_components, labels = scipy.sparse.csgraph.connected_components(p)
        group = defaultdict(list)
        for i, ll in enumerate(labels):
            group[ll].append(i)
    else:
        if verbose:
            print("Using symfc connected_components with DFS.", flush=True)
        group = connected_components(p)

    return group
