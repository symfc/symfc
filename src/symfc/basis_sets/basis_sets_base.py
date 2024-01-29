"""Symmetry adapted basis sets of force constants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from symfc.spg_reps import SpgRepsBase
from symfc.utils.utils import SymfcAtoms


class FCBasisSetBase(ABC):
    """Abstract base class of symmetry adapted basis set for force constants."""

    def __init__(
        self,
        supercell: SymfcAtoms,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        log_level : int, optional
            Log level. Default is 0.

        """
        self._natom = len(supercell)
        self._use_mkl = use_mkl
        self._log_level = log_level
        self._basis_set: Optional[np.ndarray] = None
        self._spg_reps: Optional[SpgRepsBase] = None

    @abstractmethod
    def basis_set(self):
        """Return (compressed) basis set."""
        pass

    @abstractmethod
    def full_basis_set(self):
        """Return full (decompressed) basis set."""
        pass

    @property
    def translation_permutations(self) -> np.ndarray:
        """Return permutations by lattice translation."""
        return self._spg_reps.translation_permutations

    @abstractmethod
    def run(self):
        """Run basis set calculation."""
        pass
