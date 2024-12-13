"""Symfc API."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from symfc.basis_sets import FCBasisSetBase, FCBasisSetO2, FCBasisSetO3, FCBasisSetO4
from symfc.solvers import (
    FCSolverO2,
    FCSolverO2O3,
    FCSolverO2O3O4,
    FCSolverO3,
    FCSolverO3O4,
    FCSolverO4,
)
from symfc.utils.utils import SymfcAtoms


class Symfc:
    """Symfc API."""

    def __init__(
        self,
        supercell: SymfcAtoms,
        displacements: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        spacegroup_operations: Optional[dict] = None,
        cutoff: Optional[dict] = None,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        displacements : ndarray, optional
            Displacements of supercell atoms. shape=(n_snapshot, natom, 3),
            dtype='double', order='C'
        forces : ndarray, optional
            Forces of supercell atoms. shape=(n_snapshot, natom, 3),
            dtype='double', order='C'
        spacegroup_operations : dict, optional
            Space group operations in supercell, by default None. When None,
            spglib is used. The following keys and values correspond to spglib
            symmetry dataset:
                rotations : array_like translations : array_like
        cutoff : dict, optional
            Cutoff radii in angstrom for FC3 and FC4, by default None.
        use_mkl : bool, optional
            Use MKL library, by default False.
        log_level : int, optional
            Log level, by default 0.

        """
        self._supercell = supercell
        self._displacements = displacements
        self._forces = forces
        self._spacegroup_operations = spacegroup_operations
        self._use_mkl = use_mkl
        self._log_level = log_level

        self._basis_set: dict[FCBasisSetBase] = {}
        self._force_constants: dict[np.ndarray] = {}
        self._prepare_cutoff(cutoff)

    @property
    def p2s_map(self) -> Optional[np.ndarray]:
        """Return indices of translationally independent atoms."""
        if self._basis_set:
            return next(iter(self._basis_set.values())).p2s_map
        else:
            raise ValueError("No FCBasisSet set is not set.")

    @property
    def basis_set(self) -> dict[FCBasisSetBase]:
        """Setter and getter of basis set.

        dict[FCBasisSet]
            The key is the order of basis set in int.

        """
        return self._basis_set

    @basis_set.setter
    def basis_set(self, basis_set):
        self._basis_set = basis_set

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

    def run(
        self,
        max_order: Optional[int] = None,
        orders: Optional[list] = None,
        is_compact_fc: bool = True,
        batch_size: int = 100,
    ) -> Symfc:
        """Run basis set and force constants calculation.

        Parameters
        ----------
        max_order : int
            Maximum fc order.
        orders: list
            Orders of force constants.
        is_compact_fc: bool
            Return compact force constants.
        batch_size : int, optional
            Batch size in solvers, by default 100.
        """
        if self._displacements is not None and self._forces is not None:
            self.compute_basis_set(max_order=max_order, orders=orders)
            self.solve(
                max_order=max_order,
                orders=orders,
                is_compact_fc=is_compact_fc,
                batch_size=batch_size,
            )
        return self

    def solve(
        self,
        max_order: Optional[int] = None,
        orders: Optional[list] = None,
        is_compact_fc: bool = True,
        batch_size: int = 100,
    ) -> Symfc:
        """Calculate force constants.

        Parameters
        ----------
        max_order : int
            Maximum fc order.
        orders: list
            Orders of force constants.
        is_compact_fc: bool
            Return compact force constants.
        batch_size : int, optional
            Batch size in solvers, by default 100.
        """
        self._check_dataset()
        orders = self._check_orders(max_order, orders)

        if orders == (2,):
            basis_set: FCBasisSetO2 = self._basis_set[2]
            solver_o2 = FCSolverO2(
                basis_set,
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces)
            if is_compact_fc:
                self._force_constants[2] = solver_o2.compact_fc
            else:
                self._force_constants[2] = solver_o2.full_fc
        elif orders == (3,):
            basis_set: FCBasisSetO3 = self._basis_set[3]
            solver_o3 = FCSolverO3(
                basis_set,
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces)
            if is_compact_fc:
                self._force_constants[3] = solver_o3.compact_fc
            else:
                self._force_constants[3] = solver_o3.full_fc
        elif orders == (4,):
            basis_set: FCBasisSetO4 = self._basis_set[4]
            solver_o4 = FCSolverO4(
                basis_set,
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces)
            if is_compact_fc:
                self._force_constants[4] = solver_o4.compact_fc
            else:
                self._force_constants[4] = solver_o4.full_fc
        elif orders == (2, 3):
            basis_set_o2: FCBasisSetO2 = self._basis_set[2]
            basis_set_o3: FCBasisSetO3 = self._basis_set[3]
            solver_o2o3 = FCSolverO2O3(
                [basis_set_o2, basis_set_o3],
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces, batch_size=batch_size)
            if is_compact_fc:
                fc2, fc3 = solver_o2o3.compact_fc
            else:
                fc2, fc3 = solver_o2o3.full_fc
            self._force_constants[2] = fc2
            self._force_constants[3] = fc3
        elif orders == (3, 4):
            basis_set_o3: FCBasisSetO3 = self._basis_set[3]
            basis_set_o4: FCBasisSetO4 = self._basis_set[4]
            solver_o3o4 = FCSolverO3O4(
                [basis_set_o3, basis_set_o4],
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces, batch_size=batch_size)
            if is_compact_fc:
                fc3, fc4 = solver_o3o4.compact_fc
            else:
                fc3, fc4 = solver_o3o4.full_fc
            self._force_constants[3] = fc3
            self._force_constants[4] = fc4
        elif orders == (2, 3, 4):
            basis_set_o2: FCBasisSetO2 = self._basis_set[2]
            basis_set_o3: FCBasisSetO3 = self._basis_set[3]
            basis_set_o4: FCBasisSetO4 = self._basis_set[4]
            solver_o2o3o4 = FCSolverO2O3O4(
                [basis_set_o2, basis_set_o3, basis_set_o4],
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces, batch_size=batch_size)
            if is_compact_fc:
                fc2, fc3, fc4 = solver_o2o3o4.compact_fc
            else:
                fc2, fc3, fc4 = solver_o2o3o4.full_fc
            self._force_constants[2] = fc2
            self._force_constants[3] = fc3
            self._force_constants[4] = fc4

        return self

    def compute_basis_set(
        self,
        max_order: Optional[int] = None,
        orders: Optional[list] = None,
    ) -> Symfc:
        """Run basis set calculations.

        Parameters
        ----------
        max_order : int
            Maximum fc order.
        orders: list
            Orders of force constants.
        """
        orders = self._check_orders(max_order, orders)
        for order in orders:
            if order == 2:
                basis_set_o2 = FCBasisSetO2(
                    self._supercell,
                    spacegroup_operations=self._spacegroup_operations,
                    use_mkl=self._use_mkl,
                    log_level=self._log_level,
                ).run()
                self._basis_set[2] = basis_set_o2
            elif order == 3:
                basis_set_o3 = FCBasisSetO3(
                    self._supercell,
                    spacegroup_operations=self._spacegroup_operations,
                    cutoff=self._cutoff[3],
                    use_mkl=self._use_mkl,
                    log_level=self._log_level,
                ).run()
                self._basis_set[3] = basis_set_o3
            elif order == 4:
                basis_set_o4 = FCBasisSetO4(
                    self._supercell,
                    spacegroup_operations=self._spacegroup_operations,
                    cutoff=self._cutoff[4],
                    use_mkl=self._use_mkl,
                    log_level=self._log_level,
                ).run()
                self._basis_set[4] = basis_set_o4
        return self

    def _check_orders(self, max_order: int, orders: list) -> tuple:
        if max_order is None and orders is None:
            raise RuntimeError("Maximum order and orders not found.")

        if max_order is not None:
            if max_order not in (2, 3, 4):
                raise NotImplementedError(
                    "Only fc2, fc3 and fc4 basis sets are implemented."
                )
            orders = tuple(list(range(2, max_order + 1)))
        else:
            orders = tuple(sorted(orders))
            if orders not in [(2,), (3,), (4,), (2, 3), (3, 4), (2, 3, 4)]:
                raise RuntimeError("Invalid FC orders.")
        return orders

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

    def _prepare_cutoff(self, cutoff):
        if cutoff is None:
            self._cutoff = {3: None, 4: None}
        else:
            self._cutoff = cutoff
            for order in (3, 4):
                if order not in self._cutoff:
                    self._cutoff[order] = None
