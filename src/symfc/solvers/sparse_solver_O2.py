"""2nd order force constants solver using finite displacements."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from symfc.basis_sets import FCBasisSetO2
from symfc.utils.solver_funcs import get_displacement_sparse_matrix

from .solver_base import FCSolverBase
from .solver_O2 import run_solver_O2


class FCSparseSolverO2(FCSolverBase):
    """Third order force constants solver."""

    def __init__(
        self,
        basis_set: FCBasisSetO2,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._basis_set: FCBasisSetO2
        super().__init__(basis_set, use_mkl=use_mkl, log_level=log_level)

    def solve(
        self,
        atoms: np.ndarray,
        displacements: np.ndarray,
        forces: np.ndarray,
        batch_size: int = 10000,
    ) -> FCSparseSolverO2:
        """Solve coefficients of basis set from displacements and forces.

        Parameters
        ----------
        atoms : ndarray
            Indices of atoms displaced.
            shape=(n_snapshot), dtype='int'
        displacements : ndarray
            Displacements of atoms in Cartesian coordinates.
            shape=(n_snapshot, 3), dtype='double'
        forces : ndarray
            Forces of atoms in Cartesian coordinates.
            shape=(n_snapshot, N, 3), dtype='double'

        Returns
        -------
        self : FCSparseSolverO2

        """
        n_data, n_atom, _ = forces.shape
        f = forces.reshape(n_data, -1)
        d = get_displacement_sparse_matrix(atoms, displacements, n_atom)

        fc2_basis = self._basis_set
        compress_mat_fc2 = fc2_basis.compact_compression_matrix
        basis_set_fc2 = fc2_basis.blocked_basis_set

        atomic_decompr_idx_fc2 = fc2_basis.atomic_decompr_idx

        self._coefs = run_solver_O2(
            d,
            f,
            compress_mat_fc2,
            basis_set_fc2,
            atomic_decompr_idx_fc2,
            batch_size=batch_size,
            use_sparse_disps=True,
            use_mkl=self._use_mkl,
            verbose=self._log_level > 0,
        )
        return self

    @property
    def full_fc(self) -> Optional[np.ndarray]:
        """Return full force constants.

        Returns
        -------
        np.ndarray
            shape=(N, N, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("full")

    @property
    def compact_fc(self) -> Optional[np.ndarray]:
        """Return full force constants.

        Returns
        -------
        np.ndarray
            shape=(n_a, N, 3, 3), dtype='double', order='C'

        """
        return self._recover_fcs("compact")

    def _recover_fcs(
        self, comp_mat_type: Literal["full", "compact"]
    ) -> Optional[np.ndarray]:
        fc2_basis = self._basis_set

        if self._coefs is None or fc2_basis.basis_set is None:
            return None

        if comp_mat_type == "full":
            comp_mat_fc2 = fc2_basis.compression_matrix
        elif comp_mat_type == "compact":
            comp_mat_fc2 = fc2_basis.compact_compression_matrix
        else:
            raise ValueError("Invalid comp_mat_type.")

        N = self._natom
        fc2 = fc2_basis.blocked_basis_set.dot(self._coefs)
        fc2 = np.array(
            (comp_mat_fc2 @ fc2).reshape((-1, N, 3, 3)), dtype="double", order="C"
        )
        return fc2
