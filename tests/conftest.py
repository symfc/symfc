"""Pytest conftest.py."""
from pathlib import Path

import phonopy
import pytest

from symfc.symfc import SymBasisSets, SymOpReps

cwd = Path(__file__).parent


@pytest.fixture(scope="session")
def sbs_nacl222() -> SymBasisSets:
    """Return SymBasisSets instance of NaCl222."""
    ph = phonopy.load(cwd / "phonopy_NaCl222_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSets(sym_op_reps.representations, log_level=1)
    return sbs


@pytest.fixture(scope="session")
def sbs_sno2_223() -> SymBasisSets:
    """Return SymBasisSets instance of SnO2-223."""
    ph = phonopy.load(cwd / "phonopy_SnO2_223_rd.yaml.xz", produce_fc=False)
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSets(sym_op_reps.representations, log_level=1)
    return sbs
