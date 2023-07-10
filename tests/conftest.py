"""Pytest conftest.py."""
from pathlib import Path

import numpy as np
import phonopy
import pytest

from symfc.spg_reps import SymOpReps
from symfc.symfc import SymBasisSets
from symfc.symfc_compact import SymBasisSetsCompact

cwd = Path(__file__).parent
scope = "function"


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


@pytest.fixture(scope=scope)
def ph_nacl_222() -> SymOpReps:
    """Return basis sets of NaCl222."""
    ph = phonopy.load(cwd / "phonopy_NaCl_222_rd.yaml.xz", produce_fc=False)
    return ph


@pytest.fixture(scope=scope)
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


@pytest.fixture(scope=scope)
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


@pytest.fixture(scope=scope)
def bs_sno2_222() -> np.ndarray:
    """Return basis sets of SnO2-222."""
    ph = phonopy.load(cwd / "phonopy_SnO2_222_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSets(sym_op_reps.representations, log_level=1)
    return sbs.basis_sets


@pytest.fixture(scope=scope)
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


@pytest.fixture(scope=scope)
def bs_sio2_221() -> np.ndarray:
    """Return basis sets of SiO2-221."""
    ph = phonopy.load(cwd / "phonopy_SiO2_221_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSets(sym_op_reps.representations, log_level=1, lang="C")
    return sbs.basis_sets


@pytest.fixture(scope=scope)
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


@pytest.fixture(scope=scope)
def bs_gan_222() -> np.ndarray:
    """Return basis sets of GaN-222."""
    ph = phonopy.load(cwd / "phonopy_GaN_222_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSets(sym_op_reps.representations, log_level=1, lang="C")
    return sbs.basis_sets


#
# Compact form
#
@pytest.fixture(scope=scope)
def bs_nacl_222_compact() -> np.ndarray:
    """Return basis sets of NaCl222."""
    ph = phonopy.load(cwd / "phonopy_NaCl_222_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSetsCompact(sym_op_reps.representations, log_level=1)
    return sbs.basis_sets


@pytest.fixture(scope=scope)
def bs_sno2_223_compact() -> np.ndarray:
    """Return basis sets of SnO2-223."""
    ph = phonopy.load(cwd / "phonopy_SnO2_223_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSetsCompact(sym_op_reps.representations, log_level=1)
    return sbs.basis_sets


@pytest.fixture(scope=scope)
def bs_sno2_222_compact() -> np.ndarray:
    """Return basis sets of SnO2-222."""
    ph = phonopy.load(cwd / "phonopy_SnO2_222_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSetsCompact(sym_op_reps.representations, log_level=1)
    return sbs.basis_sets


@pytest.fixture(scope=scope)
def bs_sio2_222_compact() -> np.ndarray:
    """Return basis sets of SiO2-222."""
    ph = phonopy.load(cwd / "phonopy_SiO2_222_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSetsCompact(sym_op_reps.representations, log_level=1)
    return sbs.basis_sets


@pytest.fixture(scope=scope)
def bs_sio2_221_compact() -> np.ndarray:
    """Return basis sets of SiO2-221."""
    ph = phonopy.load(cwd / "phonopy_SiO2_221_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSetsCompact(sym_op_reps.representations, log_level=1)
    return sbs.basis_sets


@pytest.fixture(scope=scope)
def bs_gan_442_compact() -> np.ndarray:
    """Return basis sets of GaN-442."""
    ph = phonopy.load(cwd / "phonopy_GaN_442_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSetsCompact(sym_op_reps.representations, log_level=1)
    return sbs.basis_sets


@pytest.fixture(scope=scope)
def bs_gan_222_compact() -> np.ndarray:
    """Return basis sets of GaN-222."""
    ph = phonopy.load(cwd / "phonopy_GaN_222_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSetsCompact(sym_op_reps.representations, log_level=1)
    return sbs.basis_sets
