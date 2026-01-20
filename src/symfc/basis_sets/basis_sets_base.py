"""Symmetry adapted basis sets of force constants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from symfc.spg_reps import SpgRepsBase
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.matrix import BlockMatrixNode
from symfc.utils.utils import SymfcAtoms


class FCBasisSetBase(ABC):
    """Abstract base class of symmetry adapted basis set for force constants."""

    def __init__(
        self,
        supercell: SymfcAtoms,
        cutoff: Optional[float] = None,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        cutoff: float
            Cutoff distance in angstroms. Default is None.
        use_mkl : bool
            Use MKL or not. Default is False.
        log_level : int, optional
            Log level. Default is 0.

        """
        self._supercell = supercell
        self._natom = len(supercell)
        self._use_mkl = use_mkl
        self._log_level = log_level
        self._spg_reps: SpgRepsBase
        self._atomic_decompr_idx: np.ndarray
        self._basis_set: np.ndarray
        self._blocked_basis_set: BlockMatrixNode

        if cutoff is None:
            self._fc_cutoff = None
        else:
            self._fc_cutoff = FCCutoff(supercell, cutoff=cutoff)

    @property
    @abstractmethod
    def compact_compression_matrix(self) -> Optional[np.ndarray]:
        """Return compression matrix for compact basis set."""
        pass

    @property
    @abstractmethod
    def compression_matrix(self) -> Optional[np.ndarray]:
        """Return compression matrix."""
        pass

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return compressed basis set.

        shape=(n_c, n_bases), dtype='double'.

        """
        return self._blocked_basis_set.recover()

    @property
    def blocked_basis_set(self) -> Optional[BlockMatrixNode]:
        """Return compressed basis set in blocked format."""
        return self._blocked_basis_set

    @property
    def atomic_decompr_idx(self) -> np.ndarray:
        """Return atomic permutations by lattice translations."""
        return self._atomic_decompr_idx

    @property
    def translation_permutations(self) -> np.ndarray:
        """Return permutations by lattice translation."""
        if self._spg_reps is None:
            raise ValueError("SpgRepsBase is not set.")
        return self._spg_reps.translation_permutations

    @property
    def p2s_map(self) -> np.ndarray:
        """Return indices of translationally independent atoms."""
        if self._spg_reps is None:
            raise ValueError("SpgRepsBase is not set.")
        if self._spg_reps.p2s_map is None:
            raise ValueError("p2s_map is not set.")
        return self._spg_reps.p2s_map

    @property
    def fc_cutoff(self) -> Optional[FCCutoff]:
        """Return force constants cutoff."""
        return self._fc_cutoff

    @abstractmethod
    def run(self):
        """Run basis set calculation."""
        pass
