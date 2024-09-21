"""Matrix utility functions for setting rotational invariants."""

# from typing import Optional
#
import itertools

import numpy as np
from scipy.sparse import csr_array

from symfc.utils.utils import SymfcAtoms, get_indep_atoms_by_lat_trans
from symfc.utils.utils_O2 import (
    get_lat_trans_decompr_indices,
)

# try:
#     from symfc.utils.eig_tools import dot_product_sparse
# except ImportError:
#     pass


def _orthogonalize_constraints(positions_cartesian: np.ndarray):
    """Orthogonalize constraints derived from rotational invariance."""
    natom = positions_cartesian.shape[0]
    N3 = 3 * natom

    C = np.zeros((N3, 3), dtype="double")
    C[1::3, 0] = -positions_cartesian[:, 0]
    C[2::3, 2] = positions_cartesian[:, 0]
    C[2::3, 1] = -positions_cartesian[:, 1]
    C[0::3, 0] = positions_cartesian[:, 1]
    C[0::3, 2] = -positions_cartesian[:, 2]
    C[1::3, 1] = positions_cartesian[:, 2]

    proj_C = C @ np.linalg.inv(C.T @ C) @ C.T
    eigvals, eigvecs = np.linalg.eigh(proj_C)
    nonzero = np.isclose(eigvals, 1.0)
    C = eigvecs[:, nonzero]
    return C


def complementary_compr_projector_rot_sum_rules_O2(
    supercell: SymfcAtoms,
    trans_perms: np.ndarray,
    n_a_compress_mat: np.ndarray,
    use_mkl: bool = False,
) -> csr_array:
    """Test function for setting rotational invariants."""
    #     atomic_decompr_idx: Optional[np.ndarray] = None,
    #     fc_cutoff: Optional[FCCutoff] = None,
    n_lp, natom = trans_perms.shape

    # TODO: decompr_idx -> atomic_decompr_idx
    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    indep_atoms = [indep_atoms[0]]
    # indep_atoms = list(range(natom))

    positions_cartesian = (supercell.scaled_positions) @ supercell.cell
    positions_cartesian[:, 0] -= np.average(positions_cartesian[:, 0])
    positions_cartesian[:, 1] -= np.average(positions_cartesian[:, 1])
    positions_cartesian[:, 2] -= np.average(positions_cartesian[:, 2])
    C2 = _orthogonalize_constraints(positions_cartesian)

    N3 = natom * 3
    N33 = N3 * 3
    NN33 = N3 * N3

    n_expand = len(indep_atoms) * 3
    data = np.tile(C2.T.reshape(-1), n_expand)
    col = np.repeat(np.arange(n_expand * 3), N3)
    row = []
    ids_ialpha = (np.arange(natom) * N33)[:, None] + (np.arange(3) * 3)[None, :]
    ids_ialpha = ids_ialpha.reshape(-1)
    for j, beta in itertools.product(indep_atoms, range(3)):
        ids = ids_ialpha + (j * 9 + beta)
        row.extend(np.tile(ids, 3))

    data /= np.sqrt(n_lp)
    c_rot = csr_array(
        (data, (decompr_idx[row], col)),
        shape=(NN33 // n_lp, n_expand * 3),
        dtype="double",
    )

    c_cmplt = n_a_compress_mat.T @ c_rot
    proj = c_cmplt @ c_cmplt.T
    return proj
