"""Base class of force constants solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from symfc.basis_sets import FCBasisSetBase


class FCSolverBase(ABC):
    """Abstract base class of force constants solvers."""

    def __init__(
        self,
        basis_set: Union[FCBasisSetBase, Sequence[FCBasisSetBase]],
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._basis_set: FCBasisSetBase = basis_set
        self._use_mkl: bool = use_mkl
        self._log_level: int = log_level
        if isinstance(self._basis_set, Sequence):
            _basis_set: FCBasisSetBase = self._basis_set[0]
        else:
            _basis_set = basis_set
        self._natom: int = _basis_set.translation_permutations.shape[1]
        self._coefs: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None

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
