"""Tests of Symfc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_array

from symfc import Symfc
from symfc.solvers.solver_O2 import run_solver_O2
from symfc.utils.utils import SymfcAtoms

cwd = Path(__file__).parent


def test_api_NaCl_222_with_dataset_fd(
    ph_nacl_222: tuple[SymfcAtoms, np.ndarray, np.ndarray],
):
    """Test solver with sparse displacements and forces as input."""
    supercell, displacements, forces = ph_nacl_222
    symfc = Symfc(
        supercell,
        displacements=displacements,
        forces=forces,
    )
    symfc.compute_basis_set(orders=(2,))

    n_data, n_atom, _ = forces.shape
    f = forces.reshape(n_data, -1)
    d = csr_array(displacements.reshape(n_data, -1))

    fc2_basis = symfc.basis_set[2]
    compress_mat_fc2 = fc2_basis.compact_compression_matrix
    basis_set_fc2 = fc2_basis.blocked_basis_set

    atomic_decompr_idx_fc2 = fc2_basis.atomic_decompr_idx

    coefs = run_solver_O2(
        d,
        f,
        compress_mat_fc2,
        basis_set_fc2,
        atomic_decompr_idx_fc2,
        use_sparse_disps=True,
        use_mkl=False,
    )

    fc = fc2_basis.blocked_basis_set.dot(coefs)
    fc = np.array(
        (compress_mat_fc2 @ fc).reshape((-1, n_atom, 3, 3)), dtype="double", order="C"
    )

    fc_ref = np.loadtxt(cwd / "compact_fc_NaCl_222.xz").reshape(fc.shape)
    np.testing.assert_allclose(fc, fc_ref)
