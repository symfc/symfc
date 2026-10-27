"""Tests of Symfc."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from symfc import Symfc
from symfc.solvers.solver_O2 import FCSolverO2
from symfc.utils.utils import SymfcAtoms

cwd = Path(__file__).parent


def test_api_NaCl_222_with_dataset_fd(
    ph_nacl_222: tuple[SymfcAtoms, np.ndarray, np.ndarray],
):
    """Test solver with sparse displacements and forces as input."""
    supercell, displacements, forces = ph_nacl_222
    symfc = Symfc(
        supercell,
    )
    symfc.displacements = displacements
    symfc.forces = forces
    symfc.compute_basis_set(orders=(2,))

    n_data, n_atom, _ = forces.shape
    f = forces.reshape(n_data, -1)
    d = displacements.reshape(n_data, -1)

    fc2_basis = symfc.basis_set[2]

    solver = FCSolverO2(fc2_basis)
    solver.solve(d, f, use_large_basis_solver=True)
    fc = solver.compact_fc

    fc_ref = np.loadtxt(cwd / "compact_fc_NaCl_222.xz").reshape(fc.shape)
    np.testing.assert_allclose(fc, fc_ref)
