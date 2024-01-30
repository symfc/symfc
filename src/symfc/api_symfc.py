"""Symfc API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from symfc.basis_sets import FCBasisSetBase, FCBasisSetO2Slow
from symfc.solvers import FCSolverO2
from symfc.utils.utils import SymfcAtoms


class Symfc:
    """Symfc API."""

    def __init__(
        self,
        supercell: SymfcAtoms,
        displacements: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        orders: Optional[Sequence[int]] = None,
        log_level: int = 0,
    ):
        """Init method."""
        self._supercell: SymfcAtoms = supercell
        self._displacements: Optional[np.ndarray] = displacements
        self._forces: Optional[np.ndarray] = forces
        self._log_level = log_level

        self._basis_set: dict[FCBasisSetBase] = {}
        self._force_constants: dict[np.ndarray] = {}

        if orders:
            self.run(orders)

    @property
    def basis_set(self) -> dict[FCBasisSetBase]:
        """Return basis set instance.

        Returns
        -------
        dict[FCBasisSet]
            The key is the order of basis set in int.

        """
        return self._basis_set

    @property
    def force_constants(self) -> dict[np.ndarray]:
        """Return force constants.

        Returns
        -------
        dict[np.ndarray]
            The key is the order of force_constants in int.

        """
        return self._force_constants

    @property
    def displacements(self) -> np.ndarray:
        """Setter and getter of supercell displacements.

        ndarray
            shape=(n_snapshot, natom, 3), dtype='double', order='C'

        """
        return self._displacements

    @displacements.setter
    def displacements(self, displacements: Union[np.ndarray, list, tuple]):
        self._displacements = np.array(displacements, dtype="double", order="C")

    @property
    def forces(self) -> np.ndarray:
        """Setter and getter of supercell forces.

        ndarray
            shape=(n_snapshot, natom, 3), dtype='double', order='C'

        """
        return self._forces

    @forces.setter
    def forces(self, forces: Union[np.ndarray, list, tuple]):
        self._forces = np.array(forces, dtype="double", order="C")

    def run(self, orders: Sequence[int]):
        """Run basis set and force constants calculation."""
        if (
            orders is not None
            and self._displacements is not None
            and self._forces is not None
        ):
            for order in orders:
                self.compute_basis_set(order)
            self.solve(orders)

    def compute_basis_set(self, order: int):
        """Set order of force constants."""
        if order == 2:
            basis_set_o2 = FCBasisSetO2Slow(
                self._supercell, log_level=self._log_level
            ).run()
            self._basis_set[2] = basis_set_o2
        else:
            raise NotImplementedError("Only fc2 basis set is implemented.")

    def solve(self, orders: Sequence[int], is_compact_fc=True) -> Symfc:
        """Calculate force constants.

        orders : Sequence[int]
            Sequence of fc orders.

        """
        self._check_dataset()
        for order in orders:
            if order == 2:
                basis_set: FCBasisSetO2Slow = self._basis_set[2]
                solver = FCSolverO2(
                    basis_set,
                    log_level=self._log_level,
                )
                fc2 = solver.solve(
                    self._displacements, self._forces, is_compact_fc=is_compact_fc
                )
                self._force_constants[2] = fc2
            else:
                raise NotImplementedError("Only order-2 is implemented.")
        return self

    def _check_dataset(self):
        if self._displacements is None:
            raise RuntimeError("Dispalcements not found.")
        if self._forces is None:
            raise RuntimeError("Forces not found.")
        if self._displacements.shape != self._forces.shape:
            raise RuntimeError("Shape mismatch between dispalcements and forces.")
        if self._displacements.shape != self._forces.shape:
            raise RuntimeError("Shape mismatch between dispalcements and forces.")
        if self._displacements.ndim != 3 or self._displacements.shape[1:] != (
            len(self._supercell),
            3,
        ):
            raise RuntimeError(
                "Inconsistent array shape of displacements "
                f"{self._displacements.shape} with respect to supercell "
                f"{len(self._supercell)}."
            )
        if self._forces.ndim != 3 or self._forces.shape[1:] != (
            len(self._supercell),
            3,
        ):
            raise RuntimeError(
                "Inconsistent array shape of forces "
                f"{self._forces.shape} with respect to supercell "
                f"{len(self._supercell)}."
            )
