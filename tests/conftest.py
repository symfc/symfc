"""Pytest conftest.py."""
from pathlib import Path

import numpy as np
import phonopy
import pytest

from symfc.symfc import SymBasisSets, SymOpReps

cwd = Path(__file__).parent


def pytest_addoption(parser):
    """Add command option to pytest."""
    parser.addoption(
        "--runbig", action="store_true", default=False, help="run big tests"
    )


def pytest_configure(config):
    """Set up marker big."""
    config.addinivalue_line("markers", "big: mark test as big to run")


def pytest_collection_modifyitems(config, items):
    """Add mechanism to run with --runbig."""
    if config.getoption("--runbig"):
        # --runbig given in cli: do not skip slow tests
        return
    skip_big = pytest.mark.skip(reason="need --runbig option to run")
    for item in items:
        if "big" in item.keywords:
            item.add_marker(skip_big)


@pytest.fixture(scope="session")
def bs_nacl_222() -> np.ndarray:
    """Return basis sets of NaCl222."""
    ph = phonopy.load(cwd / "phonopy_NaCl_222_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSets(sym_op_reps.representations, log_level=1)
    return sbs.basis_sets


@pytest.fixture(scope="session")
def bs_sno2_223() -> np.ndarray:
    """Return basis sets of SnO2-223."""
    ph = phonopy.load(cwd / "phonopy_SnO2_223_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSets(sym_op_reps.representations, log_level=1)
    return sbs.basis_sets


@pytest.fixture(scope="session")
def bs_sio2_222() -> np.ndarray:
    """Return basis sets of SiO2-222."""
    ph = phonopy.load(cwd / "phonopy_SiO2_222_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSets(sym_op_reps.representations, log_level=1, lang="C")
    return sbs.basis_sets


@pytest.fixture(scope="session")
def bs_gan_442() -> np.ndarray:
    """Return basis sets of GaN-442."""
    ph = phonopy.load(cwd / "phonopy_GaN_442_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSets(sym_op_reps.representations, log_level=1, lang="C")
    return sbs.basis_sets
