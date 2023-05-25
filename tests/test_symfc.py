from pathlib import Path

import numpy as np
import phonopy
import pytest

from symfc.symfc import SymBasisSets, SymOpReps

cwd = Path(__file__).parent


@pytest.mark.parametrize("lang", ["Py", "Py_for_C", "C"])
def test_fc_basis_sets(lang):
    """Test symmetry adapted basis sets of FC."""
    basis_ref = [
        [-0.28867513, 0, 0, 0.28867513, 0, 0],
        [0, -0.28867513, 0, 0, 0.28867513, 0],
        [0, 0, -0.28867513, 0, 0, 0.28867513],
        [0.28867513, 0, 0, -0.28867513, 0, 0],
        [0, 0.28867513, 0, 0, -0.28867513, 0],
        [0, 0, 0.28867513, 0, 0, -0.28867513],
    ]

    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]).T
    types = [0, 0]

    sym_op_reps = SymOpReps(lattice, positions, types, log_level=1)
    rep = sym_op_reps.representations
    sbs = SymBasisSets(rep, log_level=1, lang=lang)
    basis = sbs.basis_sets_matrix_form
    np.testing.assert_allclose(basis[0], basis_ref, atol=1e-6)
    assert np.linalg.norm(basis[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("lang", ["C"])
def test_fc_NaCl222(lang):
    """Test force constants by NaCl 64 atoms supercell."""
    ph = phonopy.load(cwd / "phonopy_NaCl222_rd.yaml.xz", produce_fc=False)
    f = ph.dataset["forces"]
    d = ph.dataset["displacements"]
    sym_op_reps = SymOpReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        log_level=1,
    )
    sbs = SymBasisSets(sym_op_reps.representations, log_level=1, lang=lang)
    basis_sets = sbs.basis_sets

    # To save and load basis_sets.
    # np.savetxt(
    #     "basis_sets_NaCl.dat",
    #     basis_sets.reshape((basis_sets.shape[0], -1)),
    # )
    # Data size of basis_sets_NaCl in bz2 is ~6.8M.
    # This is too large to store in git for distribution.
    # So below is just an example and for convenience in writing test.
    # basis_sets = np.loadtxt(cwd / "basis_sets_NaCl.dat.bz2", dtype="double").reshape(
    #     (31, 64, 64, 3, 3)
    # )

    Bf = np.einsum("ijklm,nkm->injl", basis_sets, d).reshape(basis_sets.shape[0], -1)
    c = -f.ravel() @ np.linalg.pinv(Bf)
    fc = np.einsum("i,ijklm->jklm", c, basis_sets)
    fc_compact = fc[ph.primitive.p2s_map]

    # To save force constants in phonopy-yaml.
    # save_settings = {
    #     "force_constants": True,
    #     "displacements": False,
    #     "force_sets": False,
    # }
    # ph.save(
    #     "phonopy_NaCl222_fc.yaml",
    #     settings=save_settings
    # )

    ph_ref = phonopy.load(cwd / "phonopy_NaCl222_fc.yaml.xz", produce_fc=False)
    np.testing.assert_allclose(ph_ref.force_constants, fc_compact, atol=1e-6)
