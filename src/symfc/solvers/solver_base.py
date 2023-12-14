"""Base class of force constants solvers."""
from abc import ABC
from typing import Union

import numpy as np
from scipy.sparse import csr_array


class FCSolverBase(ABC):
    """Abstract base class of force constants solvers."""

    def __init__(
        self,
        basis_set: Union[np.ndarray, csr_array],
        translation_permutations: np.ndarray,
        log_level: int = 0,
    ):
        """Init method."""
        self._basis_set = basis_set
        self._translation_permutations = translation_permutations
        self._log_level = log_level

        _, self._natom = self._translation_permutations.shape
