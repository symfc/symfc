"""Base class of force constants solvers."""
from abc import ABC
from typing import Optional, Union

import numpy as np
from scipy.sparse import csc_array, csr_array


class FCSolverBase(ABC):
    """Abstract base class of force constants solvers."""

    def __init__(
        self,
        basis_set: Union[np.ndarray, csr_array],
        translation_permutations: np.ndarray,
        compression_matrix: Optional[Union[csr_array, csc_array]] = None,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._basis_set = basis_set
        self._translation_permutations = translation_permutations
        self._compression_matrix = compression_matrix
        self._use_mkl = use_mkl
        self._log_level = log_level
        _, self._natom = self._translation_permutations.shape
