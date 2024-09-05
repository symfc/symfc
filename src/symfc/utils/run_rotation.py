"""Matrix utility functions for setting rotational invariants."""

# from typing import Optional
#
import itertools

import numpy as np
import scipy

#
# from symfc.utils.cutoff_tools import FCCutoff
# from symfc.utils.matrix_tools import get_combinations, permutation_dot_lat_trans
# from symfc.utils.solver_funcs import get_batch_slice
# from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal
from scipy.sparse import csr_array, vstack

from symfc.spg_reps import SpgRepsBase
from symfc.utils.eig_tools import eigsh_projector
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import get_lat_trans_compr_matrix_O2
from symfc.utils.utils_O3 import get_lat_trans_compr_matrix_O3

try:
    from symfc.utils.eig_tools import dot_product_sparse
except ImportError:
    pass


def delta(a: int, b: int):
    """Delta function."""
    if a == b:
        return 1
    return 0


def orthogonalize_constraints(positions_cartesian: np.ndarray):
    """Orthogonalize constraints derived from rotational invariance."""
    natom = positions_cartesian.shape[0]
    N3 = 3 * natom

    C = np.zeros((N3, 3))
    C[1::3, 0] = -positions_cartesian[:, 0]
    C[2::3, 2] = positions_cartesian[:, 0]
    C[2::3, 1] = -positions_cartesian[:, 1]
    C[0::3, 0] = positions_cartesian[:, 1]
    C[0::3, 2] = -positions_cartesian[:, 2]
    C[1::3, 1] = positions_cartesian[:, 2]

    C23 = np.zeros((9 + N3 * 9, 27))
    col_begin = 0
    for alpha, beta in itertools.product(range(3), range(3)):
        col_end = col_begin + 3
        row = beta
        add = np.array([delta(alpha, 1), 0, -delta(alpha, 2)])
        C23[row, col_begin:col_end] += add
        row = alpha * 3
        add = np.array([delta(beta, 1), 0, -delta(beta, 2)])
        C23[row, col_begin:col_end] += add
        row = 3 + beta
        add = np.array([-delta(alpha, 0), delta(alpha, 2), 0])
        C23[row, col_begin:col_end] += add
        row = alpha * 3 + 1
        add = np.array([-delta(beta, 0), delta(beta, 2), 0])
        C23[row, col_begin:col_end] += add
        row = 6 + beta
        add = np.array([0, -delta(alpha, 1), delta(alpha, 0)])
        C23[row, col_begin:col_end] += add
        row = alpha * 3 + 2
        add = np.array([0, -delta(beta, 1), delta(beta, 0)])
        C23[row, col_begin:col_end] += add
        col_begin = col_end

    k_ids = np.repeat(np.arange(natom), 3) * 27
    gamma_ids = np.tile(np.arange(3), natom)
    row_const = k_ids + gamma_ids + 9

    col_begin = 0
    for alpha, beta in itertools.product(range(3), range(3)):
        row = row_const + (alpha * 9 + beta * 3)
        col_end = col_begin + 3
        C23[row, col_begin:col_end] = C
        col_begin = col_end

    C23, _ = np.linalg.qr(C23)

    assert np.linalg.matrix_rank(C23) == 27
    assert np.allclose(C23.T @ C23, np.eye(27))
    return C23


def get_complement_matrix(mat, natom, order):
    """Get complementary matrix."""
    NN = natom**2
    NN33 = 9 * NN
    N333 = 27 * natom
    NN333 = N333 * natom
    NNN333 = NN333 * natom
    if order == 2:
        n_row = 9
        size1 = NN33
    elif order == 3:
        n_row = N333
        size1 = NNN333

    data = np.tile(mat.T.reshape(-1), NN)
    col = np.repeat(np.arange(NN333), n_row)
    row = []
    for i, j in itertools.product(range(natom), range(natom)):
        begin = i * n_row * natom + j * n_row
        end = begin + n_row
        row.extend(np.tile(np.arange(begin, end), 27))

    nonzero = np.abs(data) > 1e-15
    row = np.array(row)[nonzero]
    col = np.array(col)[nonzero]
    data = data[nonzero]
    return csr_array((data, (row, col)), shape=(size1, NN333), dtype="double")


def set_rotational_invariants(supercell: SymfcAtoms):
    """Test function for setting rotational invariants."""
    natom = len(supercell.numbers)
    positions_cartesian = (supercell.scaled_positions) @ supercell.cell
    C23 = orthogonalize_constraints(positions_cartesian)
    print(C23.shape)

    c_rot_fc2 = get_complement_matrix(C23[:9], natom, order=2)
    c_rot_fc3 = get_complement_matrix(C23[9:], natom, order=3)
    c_rot = vstack([c_rot_fc2, c_rot_fc3])

    correlation = c_rot.T @ c_rot
    diff = correlation - scipy.sparse.identity(c_rot.shape[1])
    assert np.allclose(diff.data, 0.0)
    print("Finished setting csr_array")

    compress = True
    if compress:
        spg_reps = SpgRepsBase(supercell)
        trans_perms = spg_reps.translation_permutations

        c_trans_fc2 = get_lat_trans_compr_matrix_O2(trans_perms)
        c_trans_fc3 = get_lat_trans_compr_matrix_O3(trans_perms)
        print(c_trans_fc2.shape)
        print(c_trans_fc3.shape)

        c_rot = vstack([c_trans_fc2.T @ c_rot_fc2, c_trans_fc3.T @ c_rot_fc3])
        print("C_rot:", c_rot.shape)

    proj = dot_product_sparse(c_rot, c_rot.T, use_mkl=True)
    print(proj.shape)
    eigvecs = eigsh_projector(proj, verbose=True)
    print(eigvecs.shape)
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
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="FC order.",
    )
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).structure
    supercell = supercell_diagonal(unitcell, args.supercell)
    supercell = SymfcAtoms(
        numbers=supercell.types,
        cell=supercell.axis.T,
        scaled_positions=supercell.positions.T,
    )

    set_rotational_invariants(supercell)


# def constraint_orthogonalization(positions_cartesian: np.ndarray):
#
#    natom = positions_cartesian.shape[0]
#    N3 = 3 * natom
#
#    C = np.zeros((N3, 3))
#    C[1::3, 0] = - positions_cartesian[:, 0]
#    C[2::3, 2] = positions_cartesian[:, 0]
#    C[2::3, 1] = - positions_cartesian[:, 1]
#    C[0::3, 0] = positions_cartesian[:, 1]
#    C[0::3, 2] = - positions_cartesian[:, 2]
#    C[1::3, 1] = positions_cartesian[:, 2]
#
#    id_fc2_dict = dict()
#    C23_dict = dict()
#    for alpha, beta in itertools.product(range(3), range(3)):
#        id_fc2 = np.array([
#            beta, alpha * 3, 3 + beta, alpha * 3 + 1, 6 + beta, alpha * 3 + 2,
#        ])
#        id_fc2_dict[(alpha, beta)] = id_fc2
#        C23 = np.zeros((N3 + 6, 3), dtype="double")
#        if alpha == 0:
#            C23[2, 0] = -1
#            C23[4, 2] = 1
#        elif alpha == 1:
#            C23[0, 0] = 1
#            C23[4, 1] = -1
#        elif alpha == 2:
#            C23[0, 2] = -1
#            C23[2, 1] = 1
#        if beta == 0:
#            C23[3, 0] = -1
#            C23[5, 2] = 1
#        elif beta == 1:
#            C23[1, 0] = 1
#            C23[5, 1] = -1
#        elif beta == 2:
#            C23[1, 2] = -1
#            C23[3, 1] = 1
#        C23[6:] = C
#        C23, _ = np.linalg.qr(C23)
#        C23_dict[(alpha, beta)] = C23
#    return C23_dict, id_fc2_dict


#    data = np.tile(C23[:9].reshape(-1), NN)
#    row, col = [], []
#    for i, j in itertools.product(range(natom), range(natom)):
#        begin = i * N33 + j * 9
#        end = begin + 9
#        row.extend(np.repeat(np.arange(begin, end), 27))
#        begin = i * NN333 + j * N333 + NN33
#        end = begin + N333
#        row.extend(np.repeat(np.arange(begin, end), 27))
#
#        begin = i * N333 + j * 27
#        end = begin + 27
#        col.extend(np.tile(np.arange(begin, end), N333 + 9))
#
#    nonzero = (np.abs(data) > 1e-15)
#    row = np.array(row)[nonzero]
#    col = np.array(col)[nonzero]
#    data = data[nonzero]
#
#
