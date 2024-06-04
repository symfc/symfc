"""Base class of force constants solvers."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from symfc.basis_sets import FCBasisSetBase


class FCSolverBase(ABC):
    """Abstract base class of force constants solvers."""

    def __init__(
        self,
        basis_set: FCBasisSetBase,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._basis_set: FCBasisSetBase = basis_set
        self._use_mkl: bool = use_mkl
        self._log_level: int = log_level
        self._natom: int = self._basis_set.translation_permutations.shape[1]
        self._coefs: Optional[np.ndarray] = None

    @property
    def coefs(self) -> Optional[np.ndarray]:
        """Return coefficients of force constants with respect to basis set."""
        return self._coefs

    @property
    @abstractmethod
    def full_fc(self) -> Optional[np.ndarray]:
        """Return full force constants."""
        pass

    @property
    @abstractmethod
    def compact_fc(self) -> Optional[np.ndarray]:
        """Return compact force constants."""
        pass

    @abstractmethod
    def solve(self):
        """Solve coefficients of basis set from displacements and forces."""
        pass
