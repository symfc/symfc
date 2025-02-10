"""Matrix utility functions for setting rotational invariants."""

# from typing import Optional
#
import itertools

import numpy as np
import scipy
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal
from scipy.sparse import csr_array

from symfc import Symfc
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.spg_reps import SpgRepsBase
from symfc.utils.eig_tools import eigsh_projector, eigsh_projector_sumrule
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import (
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
)

try:
    from symfc.utils.eig_tools import dot_product_sparse
except ImportError:
    pass

import copy


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

    #    # U, S, V = np.linalg.svd(C, full_matrices=False)
    #    # nonzero = np.abs(S) > 1e-15
    #    # C = U[:,nonzero]
    #    # C[np.abs(C) < 1e-15] = 0.0
    #    # print(S)
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


def complementary_compr_projector_rot_sum_rules_O2(
    supercell: SymfcAtoms,
    trans_perms: np.ndarray,
    n_a_compress_mat: np.ndarray,
    use_mkl: bool = False,
) -> csr_array:
    """Test function for setting rotational invariants."""
    n_lp, natom = trans_perms.shape

    # TODO: decompr_idx -> atomic_decompr_idx
    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    indep_atoms = list(range(natom))

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
    c_rot_cmplt = csr_array(
        (data, (decompr_idx[row], col)),
        shape=(NN33 // n_lp, n_expand * 3),
        dtype="double",
    )

    proj = scipy.sparse.identity(NN33 // n_lp) - c_rot_cmplt @ c_rot_cmplt.T
    c_rot = eigsh_projector(proj, verbose=True)

    c_cmplt = n_a_compress_mat.T @ c_rot
    proj = c_cmplt @ c_cmplt.T
    return proj


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
    data = np.tile(C2.T.reshape(-1), n_expand)
    col = np.repeat(np.arange(n_expand * 3), N3)
    row = []
    ids_ialpha = (np.arange(natom) * N33)[:, None] + (np.arange(3) * 3)[None, :]
    ids_ialpha = ids_ialpha.reshape(-1)
    for j, beta in itertools.product(indep_atoms, range(3)):
        ids = ids_ialpha + (j * 9 + beta)
        row.extend(np.tile(ids, 3))

    if decompr_idx is None:
        c_rot = csr_array(
            (data, (row, col)),
            shape=(NN33, n_expand * 3),
            dtype="double",
        )
    else:
        data /= np.sqrt(n_lp)
        c_rot = csr_array(
            (data, (decompr_idx[row], col)),
            shape=(NN33 // n_lp, n_expand * 3),
            dtype="double",
        )
    return c_rot


def complementary_compr_projector_rot_O2_test(
    supercell: SymfcAtoms,
    trans_perms: np.ndarray,
    basis_set_fc2: FCBasisSetO2,
    indep_atoms=None,
    use_mkl: bool = False,
) -> csr_array:
    """Test function for setting rotational invariants."""
    n_lp, natom = trans_perms.shape
    N3 = natom * 3
    NN33 = N3 * N3

    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    c_trans = get_lat_trans_compr_matrix(decompr_idx, natom, n_lp)
    # indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    # indep_atoms = [indep_atoms[0]]
    if indep_atoms is None:
        indep_atoms = list(range(natom))
    # indep_atoms = [2]
    # indep_atoms = [0,1,2,3,4,5,6,7]

    c_rot = complement_rotational_sum_rule(supercell, indep_atoms)
    proj_rot = c_rot @ c_rot.T

    c_sum = complement_sum_rule(natom)
    proj_sum = c_sum @ c_sum.T

    print("--------------")
    print("Only lattice translation")
    proj = dot_product_sparse(c_trans, c_trans.T, use_mkl=use_mkl)
    print(proj.shape)
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)

    print("--------------")
    print("Only translational sum rule")
    proj = scipy.sparse.identity(NN33) - proj_sum
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)
    c = c_trans.T @ c
    print(c.shape)
    proj = dot_product_sparse(c, c.T, use_mkl=use_mkl)
    print(proj.shape)
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)

    c_sum = complement_sum_rule(natom, decompr_idx=decompr_idx, n_lp=n_lp)
    proj = scipy.sparse.identity(NN33 // n_lp) - c_sum @ c_sum.T
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)

    #    print("--------------")
    #    print("Only rotational sum rules (indep_atoms)")
    #
    #    c_rot1 = complement_rotational_sum_rule(
    #        supercell, [0], decompr_idx=decompr_idx, n_lp=n_lp
    #    )
    #
    #    c_rot2 = complement_rotational_sum_rule(
    #        supercell, [2], decompr_idx=decompr_idx, n_lp=n_lp
    #    )
    #    #print(c_rot1 - c_rot2)

    print("--------------")
    print("Only rotational sum rules")

    c_rot = complement_rotational_sum_rule(
        supercell, indep_atoms, decompr_idx=decompr_idx, n_lp=n_lp
    )
    proj = scipy.sparse.identity(NN33) - proj_rot
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)
    c = c_trans.T @ c
    proj = dot_product_sparse(c, c.T, use_mkl=use_mkl)
    print(proj.shape)
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)

    proj = scipy.sparse.identity(NN33 // n_lp) - c_rot @ c_rot.T
    proj_return = copy.copy(proj)
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)
    vals, vecs = np.linalg.eigh(proj.toarray())
    print(vals)

    print("--------------")
    print("Only rotational sum rules (cmplt)")

    c = eigsh_projector(proj_rot, verbose=True)
    print(c.shape)

    print("--------------")
    print("Only translational and rotational sum rules")
    proj = scipy.sparse.identity(NN33) - proj_sum - proj_rot
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)
    c = c_trans.T @ c
    proj = dot_product_sparse(c, c.T, use_mkl=use_mkl)
    print(proj.shape)
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)

    proj = scipy.sparse.identity(NN33 // n_lp) - c_rot @ c_rot.T - c_sum @ c_sum.T
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)

    print("--------------")
    """
    Compressed projector I - P^(c)
    P^(c) = n_a_compress_mat.T @ C_trans.T @ C_sum^(c)
            @ C_sum^(c).T @ C_trans @ n_a_compress_mat
    """

    proj = scipy.sparse.identity(NN33 // n_lp) - c_rot @ c_rot.T
    c = eigsh_projector(proj, verbose=True)
    print(c.shape)

    n_a_compress_mat = np.sqrt(n_lp) * basis_set_fc2.compact_compression_matrix
    print(n_a_compress_mat.shape)
    # compress_mat = n_a_compress_mat @ basis_set_fc2.basis_set
    compress_mat = n_a_compress_mat
    print(compress_mat.shape)

    c = compress_mat.T @ c
    print(c.shape)
    proj = csr_array(c @ c.T)
    eigvecs = eigsh_projector(proj)
    print(eigvecs.shape)

    vals, vecs = np.linalg.eigh(proj.toarray())
    print(vals)
    return proj_return


if __name__ == "__main__":
    import argparse
    import signal
    import time

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
    n_lp, natom = trans_perms.shape

    symfc = Symfc(supercell, use_mkl=True, log_level=1).compute_basis_set(2)
    basis_set_fc2 = symfc.basis_set[2]
    n_a_compress_mat = np.sqrt(n_lp) * basis_set_fc2.compact_compression_matrix

    t1 = time.time()
    proj = complementary_compr_projector_rot_sum_rules_O2(
        supercell,
        trans_perms,
        n_a_compress_mat,
        use_mkl=True,
    )
    # proj = scipy.sparse.identity(proj.shape[0]) - proj
    eigvecs = eigsh_projector_sumrule(proj, verbose=True)
    t2 = time.time()
    print("Elapsed time:", t2 - t1)
    print("Number of eigenvectors:", eigvecs.shape[1])

    proj1 = complementary_compr_projector_rot_O2_test(
        supercell,
        trans_perms,
        basis_set_fc2,
        indep_atoms=[2],
    )
    proj2 = complementary_compr_projector_rot_O2_test(
        supercell,
        trans_perms,
        basis_set_fc2,
        indep_atoms=[1, 6],
    )
    # print(proj2 - proj1)
