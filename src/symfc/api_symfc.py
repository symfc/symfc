"""Symfc API."""

from typing import Optional, Union

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms

from symfc.basis_set import FCBasisSetO2


class Symfc:
    """Symfc API."""

    def __init__(
        self,
        supercell: PhonopyAtoms,
        displacements: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        log_level: int = 0,
    ):
        """Init method."""
        self._supercell: PhonopyAtoms = supercell
        self._displacements: Optional[np.ndarray] = displacements
        self._forces: Optional[np.ndarray] = forces
        self._basis_set = FCBasisSetO2(supercell, log_level=log_level)
        self._force_constants: Optional[np.ndarray] = None
        if self._displacements is not None and self._forces is not None:
            self._check_dataset()
            self._basis_set.run()
            self.solve()

    @property
    def force_constants(self) -> Optional[np.ndarray]:
        """Return force constants."""
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

    def solve(self):
        """Calculate force constants."""
        self._force_constants = self._basis_set.solve(self._displacements, self._forces)

    def calculate_basis_set(self):
        """Calculate force constants basis set."""
        self._basis_set.run()

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