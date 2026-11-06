"""Tests of Symfc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from symfc import Symfc
from symfc.solvers import solver_O2
from symfc.solvers.solver_O2 import FCSolverO2
from symfc.utils.utils import SymfcAtoms

cwd = Path(__file__).parent


def _record_calls(monkeypatch: pytest.MonkeyPatch, name: str) -> list[int]:
    """Replace a function in solver_O2 by a wrapper that records its calls.

    A test of a solver usually sees only the final force constants, so it cannot
    tell which internal function was called or how many times.  This function
    makes the number of calls visible.

    `monkeypatch` is a fixture of pytest that replaces an attribute of an object
    during one test.  Here the attribute is the function of the module solver_O2
    whose name is the string in `name`.  For example, when name is
    "reshape_compr_mat_O2_for_large_basis", the call of `monkeypatch.setattr` below
    works like this assignment:

        solver_O2.reshape_compr_mat_O2_for_large_basis = wrapper

    After this call, a call of reshape_compr_mat_O2_for_large_basis in solver_O2
    runs wrapper.  wrapper runs calls.append(1) and then calls the original function
    with the same arguments, so the calculation gives the same result as without
    wrapper.

    pytest restores the original function when the test ends, so other tests are
    not affected.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The `monkeypatch` fixture of the test that calls this function.
    name : str
        Name of the function in the module solver_O2.

    Returns
    -------
    list[int]
        List that has one element for each call of the function.  The elements
        have no meaning.  Use the length of the list as the number of calls.  The
        list is empty until the function is called.

    """
    calls: list[int] = []
    original = getattr(solver_O2, name)

    def wrapper(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_O2, name, wrapper)
    return calls


def test_api_NaCl_222_with_dataset_fd(
    ph_nacl_222: tuple[SymfcAtoms, np.ndarray, np.ndarray],
):
    """Test solver with sparse displacements and forces as input."""
    supercell, displacements, forces = ph_nacl_222
    symfc = Symfc(
        supercell,
    )
    symfc.compute_basis_set(orders=(2,))

    fc2_basis = symfc.basis_set[2]

    solver = FCSolverO2(fc2_basis)
    solver.solve(displacements, forces, use_large_basis_solver=True)
    fc = solver.compact_fc

    fc_ref = np.loadtxt(cwd / "compact_fc_NaCl_222.xz").reshape(fc.shape)
    np.testing.assert_allclose(fc, fc_ref)


def test_large_basis_solver_with_multiple_atom_batches(
    ph_gan_442: tuple[SymfcAtoms, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the sum over atom batches agrees with the default solver."""
    supercell, displacements, forces = ph_gan_442
    fc2_basis = Symfc(supercell).compute_basis_set(orders=(2,)).basis_set[2]
    fc_ref = FCSolverO2(fc2_basis).solve(displacements, forces).compact_fc

    reshape_calls = _record_calls(monkeypatch, "reshape_compr_mat_O2_for_large_basis")
    solver = FCSolverO2(fc2_basis)
    solver.solve(displacements, forces, use_large_basis_solver=True)

    # The function is called once for each atom batch.
    assert len(reshape_calls) > 1
    np.testing.assert_allclose(solver.compact_fc, fc_ref, rtol=1e-7, atol=1e-10)


def test_large_basis_solver_selected_automatically(
    ph_sio2_221: tuple[SymfcAtoms, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the large basis solver is selected for a large dataset.

    The solver is selected when n_compr * n_snapshot / N > 10000.  Repeating the
    snapshots increases n_snapshot above this limit, and the least-squares solution
    stays the same because every equation is only repeated.

    """
    supercell, displacements, forces = ph_sio2_221
    fc2_basis = Symfc(supercell).compute_basis_set(orders=(2,)).basis_set[2]
    large_basis_calls = _record_calls(
        monkeypatch, "prepare_normal_equation_O2_for_large_basis"
    )

    fc_ref = FCSolverO2(fc2_basis).solve(displacements, forces).compact_fc
    assert len(large_basis_calls) == 0

    n_repeat = 35
    solver = FCSolverO2(fc2_basis)
    solver.solve(
        np.tile(displacements, (n_repeat, 1, 1)),
        np.tile(forces, (n_repeat, 1, 1)),
    )

    assert len(large_basis_calls) == 1
    np.testing.assert_allclose(solver.compact_fc, fc_ref, rtol=1e-7, atol=1e-10)
