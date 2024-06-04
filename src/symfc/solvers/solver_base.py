"""Base class of force constants solvers."""

from abc import ABC

from symfc.basis_sets import FCBasisSetO2


class FCSolverBase(ABC):
    """Abstract base class of force constants solvers."""

    def __init__(
        self,
        fc_basis_set: FCBasisSetO2,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._fc_basis_set = fc_basis_set
        self._use_mkl = use_mkl
        self._log_level = log_level
        _, self._natom = fc_basis_set.translation_permutations.shape
