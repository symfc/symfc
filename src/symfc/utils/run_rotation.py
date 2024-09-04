"""Matrix utility functions for setting rotational invariants."""

# from typing import Optional
#
import itertools

import numpy as np

#
# from symfc.utils.cutoff_tools import FCCutoff
# from symfc.utils.matrix_tools import get_combinations, permutation_dot_lat_trans
# from symfc.utils.solver_funcs import get_batch_slice
# from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal

# import scipy
from scipy.sparse import csr_array

from symfc.utils.eig_tools import eigsh_projector
from symfc.utils.utils import SymfcAtoms

try:
    from symfc.utils.eig_tools import dot_product_sparse
except ImportError:
    pass


def delta(a: int, b: int):
    """Delta function."""
    if a == b:
        return 1
    return 0


def constraint_orthogonalization(positions_cartesian: np.ndarray):
    """Constraint."""
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

    print(np.linalg.matrix_rank(C23))

    row_begin = 9
    col_begin = 0
    for _ in range(9):
        row_end = row_begin + N3
        col_end = col_begin + 3
        C23[row_begin:row_end, col_begin:col_end] = C
    C23, _ = np.linalg.qr(C23)
    print(C23.shape)

    for _ii in range(9):
        print(C23[:9, 3 * _ii : 3 * _ii + 3])


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


def set_rotational_invariants(supercell: SymfcAtoms):
    """Test function for setting rotational invariants."""
    natom = len(supercell.numbers)
    N3 = 3 * natom
    NN = natom**2
    NN33 = 9 * natom**2
    NNN333 = 27 * natom**3

    positions_cartesian = (supercell.scaled_positions) @ supercell.cell
    C23_dict, id_fc2_dict = constraint_orthogonalization(positions_cartesian)

    id_fc3 = np.array([k * 27 + gamma for k in range(natom) for gamma in range(3)])

    col = np.repeat(np.arange(NN33 * 3), N3 + 6)
    row, data = [], []
    for alpha, beta in itertools.product(range(3), range(3)):
        id_fc2 = id_fc2_dict[(alpha, beta)]
        C23 = C23_dict[(alpha, beta)]
        data.extend(np.tile(C23.T.reshape(-1), NN))
        seq_id_fc3_prefix = 9 * alpha + 3 * beta + NN33
        for i, j in itertools.product(range(natom), range(natom)):
            seq_id_fc2 = i * 9 * natom + j * 9
            seq_id_fc3 = i * 27 * natom**2 + j * 27 * natom + seq_id_fc3_prefix

            id2 = id_fc2 + seq_id_fc2
            id3 = id_fc3 + seq_id_fc3
            row.extend(np.hstack([id2, id3, id2, id3, id2, id3]))

    c_rot = csr_array(
        (data, (row, col)),
        shape=(NN33 + NNN333, NN33 * 3),
        dtype="double",
    )
    print(c_rot.T @ c_rot)
    print(c_rot.shape)
    proj = dot_product_sparse(c_rot, c_rot.T, use_mkl=True)
    print(proj.shape)
    eigvecs = eigsh_projector(proj)
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
