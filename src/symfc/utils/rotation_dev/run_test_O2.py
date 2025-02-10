"""Matrix utility functions for setting rotational invariants."""

import itertools

import numpy as np
import scipy
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal
from scipy.sparse import csr_array

from symfc import Symfc
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.spg_reps import SpgRepsBase
from symfc.utils.eig_tools import eigsh_projector
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import (
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
)


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


def complement_sum_rule(natom, decompr_idx=None, n_lp=None):
    """Construct sum rule."""
    N33 = natom * 9
    NN33 = natom**2 * 9
    row = []
    for j, alpha, beta in itertools.product(range(natom), range(3), range(3)):
        i = np.arange(natom) * N33
        const_add = j * 9 + alpha * 3 + beta
        row.extend(i + const_add)
    col = np.repeat(np.arange(N33), natom)
    data = np.ones(len(col)) / np.sqrt(natom)

    if decompr_idx is not None:
        data /= np.sqrt(n_lp)
        c_sum = csr_array(
            (data, (decompr_idx[row], col)),
            shape=(NN33 // n_lp, N33),
            dtype="double",
        )
    else:
        c_sum = csr_array(
            (data, (row, col)),
            shape=(NN33, N33),
            dtype="double",
        )

    return c_sum


def complement_rotational_sum_rule(supercell, indep_atoms, decompr_idx=None, n_lp=None):
    """Get complement_rotational_sum_rule."""
    positions_cartesian = (supercell.scaled_positions) @ supercell.cell
    positions_cartesian[:, 0] -= np.average(positions_cartesian[:, 0])
    positions_cartesian[:, 1] -= np.average(positions_cartesian[:, 1])
    positions_cartesian[:, 2] -= np.average(positions_cartesian[:, 2])
    C2 = _orthogonalize_constraints(positions_cartesian)

    natom = positions_cartesian.shape[0]
    N3 = natom * 3
    N33 = N3 * 3
    NN33 = N3 * N3

    n_expand = len(indep_atoms) * 3
    n_col = n_expand * 3

    data = np.tile(C2.T.reshape(-1), n_expand)
    col = np.repeat(np.arange(n_col), N3)
    row = []
    ids_ialpha = (np.arange(natom) * N33)[:, None] + (np.arange(3) * 3)[None, :]
    ids_ialpha = ids_ialpha.reshape(-1)
    for j, beta in itertools.product(indep_atoms, range(3)):
        ids = ids_ialpha + (j * 9 + beta)
        row.extend(np.tile(ids, 3))

    if decompr_idx is None:
        c_rot = csr_array((data, (row, col)), shape=(NN33, n_col), dtype="double")
    else:
        data /= np.sqrt(n_lp)
        c_rot = csr_array(
            (data, (decompr_idx[row], col)),
            shape=(NN33 // n_lp, n_col),
            dtype="double",
        )
    return c_rot


def complementary_compr_projector_rot_O2_test(
    supercell: SymfcAtoms,
    trans_perms: np.ndarray,
    basis_set_fc2: FCBasisSetO2,
    use_mkl: bool = False,
) -> csr_array:
    """Test function for setting rotational invariants."""
    n_lp, natom = trans_perms.shape
    N3 = natom * 3
    N33 = N3 * 3
    NN33 = N3 * N3

    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    c_trans = get_lat_trans_compr_matrix(decompr_idx, natom, n_lp)
    # indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    # indep_atoms = [indep_atoms[0]]
    # if indep_atoms is None:
    indep_atoms = list(range(natom))

    """
    c_rot = complement_rotational_sum_rule(supercell, indep_atoms)
    c_sum = complement_sum_rule(natom)
    proj_rot = scipy.sparse.identity(NN33) - c_rot @ c_rot.T
    proj_sum = scipy.sparse.identity(NN33) - c_sum @ c_sum.T
    """
    c_sum = complement_sum_rule(natom, decompr_idx=decompr_idx, n_lp=n_lp)
    # proj = scipy.sparse.identity(NN33 // n_lp) - c_sum @ c_sum.T
    # c = eigsh_projector(proj, verbose=True)

    print("-------------- Only rotational sum rules ---------------")
    c_rot = complement_rotational_sum_rule(
        supercell,
        indep_atoms,
        decompr_idx=decompr_idx,
        n_lp=n_lp,
    )
    proj = scipy.sparse.identity(NN33 // n_lp) - c_rot @ c_rot.T - c_sum @ c_sum.T
    basis = eigsh_projector(proj, verbose=True)
    print(basis.shape)
    # vals, vecs = np.linalg.eigh(proj.toarray())
    # print(vals)
    print(c_trans.shape)

    basis = c_trans @ basis
    print(basis.shape)
    proj_basis = basis @ basis.T

    positions_cartesian = (supercell.scaled_positions) @ supercell.cell
    C = np.zeros((N3, 3), dtype="double")
    C[1::3, 0] = -positions_cartesian[:, 0]
    C[2::3, 2] = positions_cartesian[:, 0]
    C[2::3, 1] = -positions_cartesian[:, 1]
    C[0::3, 0] = positions_cartesian[:, 1]
    C[0::3, 2] = -positions_cartesian[:, 2]
    C[1::3, 1] = positions_cartesian[:, 2]

    trial_vec = np.zeros(NN33, dtype="double")
    ids_ialpha = (np.arange(natom) * N33)[:, None] + (np.arange(3) * 3)[None, :]
    ids_ialpha = ids_ialpha.reshape(-1)

    for gamma in range(3):
        for j in range(natom):
            for beta in range(3):
                ids = ids_ialpha + (j * 9 + beta)
                trial_vec[ids] = C[:, gamma]
                projected_vec = proj_basis @ trial_vec
                print(projected_vec[np.abs(projected_vec) > 1e-14])

    return proj


if __name__ == "__main__":
    import argparse
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--poscar", type=str, default=None, help="poscar")
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).structure
    supercell = supercell_diagonal(unitcell, args.supercell)
    supercell = SymfcAtoms(
        numbers=supercell.types,
        cell=supercell.axis.T,
        scaled_positions=supercell.positions.T,
    )

    spg_reps = SpgRepsBase(supercell)
    trans_perms = spg_reps.translation_permutations
    #    n_lp, natom = trans_perms.shape

    symfc = Symfc(supercell, use_mkl=True, log_level=1).compute_basis_set(2)
    basis_set_fc2 = symfc.basis_set[2]
    #    n_a_compress_mat = np.sqrt(n_lp) * basis_set_fc2.compact_compression_matrix

    proj1 = complementary_compr_projector_rot_O2_test(
        supercell,
        trans_perms,
        basis_set_fc2,
    )
#    proj2 = complementary_compr_projector_rot_O2_test(
#        supercell,
#        trans_perms,
#        basis_set_fc2,
#        indep_atoms = [1, 6],
#    )
# print(proj2 - proj1)
