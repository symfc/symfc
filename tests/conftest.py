"""Pytest conftest.py."""
from pathlib import Path
from typing import Final

import phonopy
import pytest
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms

from symfc.basis_set import FCBasisSet

cwd = Path(__file__).parent
scope: Final = "function"


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
def cell_nacl_111() -> PhonopyAtoms:
    """Return unitcell of NaCl."""
    cell = read_cell_yaml(cwd / "NaCl-unitcell.yaml")
    return cell


@pytest.fixture(scope="session")
def ph_nacl_222() -> Phonopy:
    """Return phonopy instance of NaCl222."""
    ph = phonopy.load(cwd / "phonopy_NaCl_222_rd.yaml.xz", produce_fc=False)
    return ph


@pytest.fixture(scope="session")
def ph_gan_222() -> Phonopy:
    """Return phonopy instance of GaN222."""
    ph = phonopy.load(cwd / "phonopy_GaN_222_rd.yaml.xz", produce_fc=False)
    return ph


@pytest.fixture(scope=scope)
def bs_nacl_222() -> FCBasisSet:
    """Return basis set of NaCl222."""
    ph = phonopy.load(cwd / "phonopy_NaCl_222_rd.yaml.xz", produce_fc=False)
    sbs = FCBasisSet(ph.supercell, log_level=1)
    return sbs


@pytest.fixture(scope=scope)
def bs_sno2_223() -> FCBasisSet:
    """Return basis set of SnO2-223."""
    ph = phonopy.load(cwd / "phonopy_SnO2_223_rd.yaml.xz", produce_fc=False)
    sbs = FCBasisSet(ph.supercell, log_level=1)
    return sbs


@pytest.fixture(scope=scope)
def bs_sno2_222() -> FCBasisSet:
    """Return basis set of SnO2-222."""
    ph = phonopy.load(cwd / "phonopy_SnO2_222_rd.yaml.xz", produce_fc=False)
    sbs = FCBasisSet(ph.supercell, log_level=1)
    return sbs


@pytest.fixture(scope=scope)
def bs_sio2_222() -> FCBasisSet:
    """Return basis set of SiO2-222."""
    ph = phonopy.load(cwd / "phonopy_SiO2_222_rd.yaml.xz", produce_fc=False)
    sbs = FCBasisSet(ph.supercell, log_level=1)
    return sbs


@pytest.fixture(scope=scope)
def bs_sio2_221() -> FCBasisSet:
    """Return basis set of SiO2-221."""
    ph = phonopy.load(cwd / "phonopy_SiO2_221_rd.yaml.xz", produce_fc=False)
    sbs = FCBasisSet(ph.supercell, log_level=1)
    return sbs


@pytest.fixture(scope=scope)
def bs_gan_442() -> FCBasisSet:
    """Return basis set of GaN-442."""
    ph = phonopy.load(cwd / "phonopy_GaN_442_rd.yaml.xz", produce_fc=False)
    sbs = FCBasisSet(ph.supercell, log_level=1)
    return sbs


@pytest.fixture(scope=scope)
def bs_gan_222() -> FCBasisSet:
    """Return basis set of GaN-222."""
    ph = phonopy.load(cwd / "phonopy_GaN_222_rd.yaml.xz", produce_fc=False)
    sbs = FCBasisSet(ph.supercell, log_level=1)
    return sbs
