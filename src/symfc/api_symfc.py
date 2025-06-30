"""Symfc API."""

from __future__ import annotations

from typing import Optional, Union, cast

import numpy as np
from scipy.sparse import csr_array

from symfc.basis_sets import FCBasisSetBase, FCBasisSetO2, FCBasisSetO3, FCBasisSetO4
from symfc.solvers import (
    FCSolverO2,
    FCSolverO2O3,
    FCSolverO2O3O4,
    FCSolverO3,
    FCSolverO3O4,
    FCSolverO4,
)
from symfc.utils.eig_tools import (
    eigh_projector,
    eigsh_projector,
    eigsh_projector_sumrule,
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
        cutoff: Optional[dict[int, float]] = None,
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
        cutoff : dict[int, float], optional
            Cutoff radii in angstrom for FC3 and FC4, by default None.
            For example, {3: 4.0, 4: 4.0}
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

        self._basis_set: dict[int, FCBasisSetBase] = {}
        self._force_constants: dict[int, np.ndarray] = {}
        self._prepare_cutoff(cutoff)

    @property
    def supercell(self) -> SymfcAtoms:
        """Return supercell."""
        return self._supercell

    @property
    def p2s_map(self) -> Optional[np.ndarray]:
        """Return indices of translationally independent atoms."""
        if self._basis_set:
            return next(iter(self._basis_set.values())).p2s_map
        else:
            raise ValueError("No FCBasisSet set is not set.")

    @property
    def basis_set(self) -> dict[int, FCBasisSetBase]:
        """Setter and getter of basis set.

        dict[FCBasisSet]
            The key is the order of basis set in int.

        """
        return self._basis_set

    @basis_set.setter
    def basis_set(self, basis_set):
        self._basis_set = basis_set

    @property
    def force_constants(self) -> dict[int, np.ndarray]:
        """Return force constants.

        Returns
        -------
        dict[np.ndarray]
            The key is the order of force_constants in int.

        """
        return self._force_constants

    @property
    def displacements(self) -> Optional[np.ndarray]:
        """Setter and getter of supercell displacements.

        ndarray
            shape=(n_snapshot, natom, 3), dtype='double', order='C'

        """
        return self._displacements

    @displacements.setter
    def displacements(self, displacements: Union[np.ndarray, list, tuple]):
        self._displacements = np.array(displacements, dtype="double", order="C")

    @property
    def forces(self) -> Optional[np.ndarray]:
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
        _orders = self._check_orders(max_order, orders)

        if self._displacements is None:
            raise RuntimeError("Displacements not found.")
        if self._forces is None:
            raise RuntimeError("Forces not found.")

        if _orders == (2,):
            basis_set_o2: FCBasisSetO2 = cast(FCBasisSetO2, self._basis_set[2])
            solver_o2 = FCSolverO2(
                basis_set_o2,
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces)
            if is_compact_fc:
                if solver_o2.compact_fc is not None:
                    self._force_constants[2] = solver_o2.compact_fc
                else:
                    raise RuntimeError("Failed to obtain compact force constants")
            else:
                if solver_o2.full_fc is not None:
                    self._force_constants[2] = solver_o2.full_fc
                else:
                    raise RuntimeError("Failed to obtain full force constants")
        elif _orders == (3,):
            basis_set_o3: FCBasisSetO3 = cast(FCBasisSetO3, self._basis_set[3])
            solver_o3 = FCSolverO3(
                basis_set_o3,
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces)
            if is_compact_fc:
                if solver_o3.compact_fc is not None:
                    self._force_constants[3] = solver_o3.compact_fc
                else:
                    raise RuntimeError("Failed to obtain compact force constants")
            else:
                if solver_o3.full_fc is not None:
                    self._force_constants[3] = solver_o3.full_fc
                else:
                    raise RuntimeError("Failed to obtain full force constants")
        elif _orders == (4,):
            basis_set_o4: FCBasisSetO4 = cast(FCBasisSetO4, self._basis_set[4])
            solver_o4 = FCSolverO4(
                basis_set_o4,
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces)
            if is_compact_fc:
                if solver_o4.compact_fc is not None:
                    self._force_constants[4] = solver_o4.compact_fc
                else:
                    raise RuntimeError("Failed to obtain compact force constants")
            else:
                if solver_o4.full_fc is not None:
                    self._force_constants[4] = solver_o4.full_fc
                else:
                    raise RuntimeError("Failed to obtain full force constants")
        elif _orders == (2, 3):
            basis_set_o2: FCBasisSetO2 = cast(FCBasisSetO2, self._basis_set[2])
            basis_set_o3: FCBasisSetO3 = cast(FCBasisSetO3, self._basis_set[3])
            solver_o2o3 = FCSolverO2O3(
                [basis_set_o2, basis_set_o3],
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces, batch_size=batch_size)
            if is_compact_fc and solver_o2o3.compact_fc is not None:
                fc2, fc3 = solver_o2o3.compact_fc
            elif solver_o2o3.full_fc is not None:
                fc2, fc3 = solver_o2o3.full_fc
            else:
                raise RuntimeError("Failed to obtain force constants")
            self._force_constants[2] = fc2
            self._force_constants[3] = fc3
        elif _orders == (3, 4):
            basis_set_o3: FCBasisSetO3 = cast(FCBasisSetO3, self._basis_set[3])
            basis_set_o4: FCBasisSetO4 = cast(FCBasisSetO4, self._basis_set[4])
            solver_o3o4 = FCSolverO3O4(
                [basis_set_o3, basis_set_o4],
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces, batch_size=batch_size)
            if is_compact_fc and solver_o3o4.compact_fc is not None:
                fc3, fc4 = solver_o3o4.compact_fc
            elif solver_o3o4.full_fc is not None:
                fc3, fc4 = solver_o3o4.full_fc
            else:
                raise RuntimeError("Failed to obtain force constants")
            self._force_constants[3] = fc3
            self._force_constants[4] = fc4
        elif _orders == (2, 3, 4):
            basis_set_o2: FCBasisSetO2 = cast(FCBasisSetO2, self._basis_set[2])
            basis_set_o3: FCBasisSetO3 = cast(FCBasisSetO3, self._basis_set[3])
            basis_set_o4: FCBasisSetO4 = cast(FCBasisSetO4, self._basis_set[4])
            solver_o2o3o4 = FCSolverO2O3O4(
                [basis_set_o2, basis_set_o3, basis_set_o4],
                use_mkl=self._use_mkl,
                log_level=self._log_level,
            ).solve(self._displacements, self._forces, batch_size=batch_size)
            if is_compact_fc and solver_o2o3o4.compact_fc is not None:
                fc2, fc3, fc4 = solver_o2o3o4.compact_fc
            elif solver_o2o3o4.full_fc is not None:
                fc2, fc3, fc4 = solver_o2o3o4.full_fc
            else:
                raise RuntimeError("Failed to obtain force constants")
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
        for order in self._check_orders(max_order, orders):
            if order == 2:
                basis_set_o2 = FCBasisSetO2(
                    self._supercell,
                    spacegroup_operations=self._spacegroup_operations,
                    cutoff=self._cutoff[2],
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

    def estimate_basis_size(
        self,
        max_order: Optional[int] = None,
        orders: Optional[list] = None,
    ) -> dict:
        """Estimate the size of basis set.

        Parameters
        ----------
        max_order : int
            Maximum fc order.
        orders: list
            Orders of force constants.

        Returns
        -------
        dict :
            Estimates of basis set sizes for each order. The key of dict is the
            order.
        """
        basis_size_estimates = {}
        for order in self._check_orders(max_order, orders):
            if order < 2 or order > 4:
                raise NotImplementedError(
                    "Only fc2, fc3 and fc4 basis sets are implemented."
                )

            if order == 2:
                basis_size_estimates[order] = FCBasisSetO2(
                    self._supercell,
                    spacegroup_operations=self._spacegroup_operations,
                    cutoff=self._cutoff[2],
                    use_mkl=self._use_mkl,
                    log_level=self._log_level,
                ).estimate_basis_size()
            elif order == 3:
                basis_size_estimates[order] = FCBasisSetO3(
                    self._supercell,
                    spacegroup_operations=self._spacegroup_operations,
                    cutoff=self._cutoff[3],
                    use_mkl=self._use_mkl,
                    log_level=self._log_level,
                ).estimate_basis_size()
            elif order == 4:
                basis_size_estimates[order] = FCBasisSetO4(
                    self._supercell,
                    spacegroup_operations=self._spacegroup_operations,
                    cutoff=self._cutoff[4],
                    use_mkl=self._use_mkl,
                    log_level=self._log_level,
                ).estimate_basis_size()

        return basis_size_estimates

    def _check_orders(self, max_order: Optional[int], orders: Optional[list]) -> tuple:
        if max_order is None and orders is None:
            raise RuntimeError("Maximum order and orders not found.")

        if max_order is not None:
            if max_order not in (2, 3, 4):
                raise NotImplementedError(
                    "Only fc2, fc3 and fc4 basis sets are implemented."
                )
            _orders = tuple(list(range(2, max_order + 1)))
        elif orders is not None:
            _orders = tuple(sorted(orders))
            if _orders not in [(2,), (3,), (4,), (2, 3), (3, 4), (2, 3, 4)]:
                raise RuntimeError("Invalid FC orders.")
        return _orders

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
            self._cutoff = {2: None, 3: None, 4: None}
        else:
            self._cutoff = {}
            for order in (2, 3, 4):
                if order in cutoff:
                    self._cutoff[order] = cutoff[order]
                else:
                    self._cutoff[order] = None


def eigh(
    p: np.ndarray,
    atol: float = 1e-8,
    rtol: float = 0.0,
    log_level: int = 0,
) -> np.ndarray:
    """Solve eigenvalue problem for projector in numpy ndarray.

    Parameters
    ----------
    p: np.ndarray
        Projection matrix to be solved.
    atol : float, optional
        atol used in np.isclose.
    rtol : float, optional
        rtol used in np.isclose.
    log_level : int, optional
        Log level, by default 0.

    Return
    ------
    Eigenvectors with eigenvalues = 1.0 in np.ndarray format.
    Eigenvectors with eigenvalues < 1.0 are eliminated.
    """
    return eigh_projector(p, atol=atol, rtol=rtol, verbose=log_level > 0)


def eigsh(
    p: csr_array,
    atol: float = 1e-8,
    rtol: float = 0.0,
    is_large_block: bool = False,
    log_level: int = 0,
) -> Union[csr_array, np.ndarray]:
    """Solve eigenvalue problem for projector in scipy sparse csr_array.

    Parameters
    ----------
    p: csr_array
        Projection matrix to be solved.
    atol : float, optional
        atol used in np.isclose.
    rtol : float, optional
        rtol used in np.isclose.
    is_large_block: bool, optional
        Use an algorithm for solving projector with large block matrices.
    log_level : int, optional
        Log level, by default 0.

    Return
    ------
    Eigenvectors with eigenvalues = 1.0.
    If is_large_block is True, eigenvectors in np.ndarray are returned.
    Otherwise, eigenvectors in csr_array are returned.
    Eigenvectors with eigenvalues < 1.0 are eliminated.
    """
    if is_large_block:
        return eigsh_projector_sumrule(p, atol=atol, rtol=rtol, verbose=log_level > 0)
    return eigsh_projector(p, atol=atol, rtol=rtol, verbose=log_level > 0)
