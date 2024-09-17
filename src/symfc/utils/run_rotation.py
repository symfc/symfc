"""Matrix utility functions for setting rotational invariants."""

# from typing import Optional
#
import itertools

import numpy as np
import scipy
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal
from scipy.sparse import csr_array, vstack

from symfc import Symfc
from symfc.basis_sets import FCBasisSetO2, FCBasisSetO3
from symfc.spg_reps import SpgRepsBase
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import (
    get_lat_trans_compr_matrix_O2,
    get_lat_trans_decompr_indices,
)
from symfc.utils.utils_O3 import (
    get_lat_trans_compr_matrix_O3,
    get_lat_trans_decompr_indices_O3,
)

# from symfc.utils.solver_funcs import get_batch_slice
# from symfc.utils.cutoff_tools import FCCutoff
# from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3

try:
    from symfc.utils.eig_tools import dot_product_sparse
except ImportError:
    pass


def delta(a: int, b: int):
    """Delta function."""
    if a == b:
        return 1
    return 0


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

    C23 = np.zeros((9 + N3 * 9, 27), dtype="double")
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

    U, S, V = np.linalg.svd(C23, full_matrices=False)
    nonzero = np.abs(S) > 1e-15
    C23 = U[:, nonzero]
    print(S)
    #    assert np.linalg.matrix_rank(C23) == 27
    #    assert np.allclose(C23.T @ C23, np.eye(27))
    return C23


def _get_complement_matrix(mat, natom, order=2, n_lp=None, decompr_idx=None):
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
    n_col = mat.shape[1]

    data = np.tile(mat.T.reshape(-1), NN)
    col = np.repeat(np.arange(NN * n_col), n_row)
    row = []
    for i, j in itertools.product(range(natom), range(natom)):
        # id must be examined.
        begin = i * n_row * natom + j * n_row
        end = begin + n_row
        row.extend(np.tile(np.arange(begin, end), n_col))

    nonzero = np.abs(data) > 1e-15
    row = np.array(row)[nonzero]
    col = np.array(col)[nonzero]
    data = data[nonzero]
    if decompr_idx is None:
        return csr_array((data, (row, col)), shape=(size1, NN * n_col), dtype="double")

    size_row = decompr_idx.shape[0] // n_lp
    return csr_array(
        (data / np.sqrt(n_lp), (decompr_idx[row], col)),
        shape=(size_row, NN * n_col),
        dtype="double",
    )


def complementary_compr_projector_rot(
    supercell: SymfcAtoms,
    trans_perms: np.ndarray,
    basis_set_fc2: FCBasisSetO2,
    basis_set_fc3: FCBasisSetO3,
    use_mkl: bool = False,
) -> csr_array:
    """Test function for setting rotational invariants."""
    n_lp, natom = trans_perms.shape
    positions_cartesian = (supercell.scaled_positions) @ supercell.cell
    C23 = _orthogonalize_constraints(positions_cartesian)

    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    c_rot_fc2 = _get_complement_matrix(
        C23[:9],
        natom,
        order=2,
        decompr_idx=decompr_idx,
        n_lp=n_lp,
    )

    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    c_rot_fc3 = _get_complement_matrix(
        C23[9:],
        natom,
        order=3,
        decompr_idx=decompr_idx,
        n_lp=n_lp,
    )
    c_rot_fc2 = dot_product_sparse(
        basis_set_fc2.compact_compression_matrix.T,
        c_rot_fc2,
        use_mkl=True,
    )
    print(c_rot_fc2.shape)
    c_rot_fc2 = dot_product_sparse(
        csr_array(basis_set_fc2.basis_set.T),
        c_rot_fc2,
        use_mkl=True,
    )
    print(c_rot_fc2.shape)
    c_rot_fc3 = dot_product_sparse(
        basis_set_fc3.compact_compression_matrix.T,
        c_rot_fc3,
        use_mkl=True,
    )
    print(c_rot_fc3.shape)
    c_rot_fc3 = dot_product_sparse(
        csr_array(basis_set_fc3.basis_set.T),
        c_rot_fc3,
        use_mkl=True,
    )
    print(c_rot_fc3.shape)

    c_rot = vstack([c_rot_fc2, c_rot_fc3])
    proj = dot_product_sparse(c_rot, c_rot.T, use_mkl=use_mkl)
    return proj


def complementary_compr_projector_rot_reference(
    supercell: SymfcAtoms,
    trans_perms: np.ndarray,
    basis_set_fc2: FCBasisSetO2,
    basis_set_fc3: FCBasisSetO3,
) -> csr_array:
    """Test function for setting rotational invariants."""
    natom = len(supercell.numbers)
    positions_cartesian = (supercell.scaled_positions) @ supercell.cell
    C23 = _orthogonalize_constraints(positions_cartesian)

    c_rot_fc2 = _get_complement_matrix(C23[:9], natom, order=2)
    c_rot_fc3 = _get_complement_matrix(C23[9:], natom, order=3)
    c_rot = vstack([c_rot_fc2, c_rot_fc3])

    correlation = c_rot.T @ c_rot
    diff = correlation - scipy.sparse.identity(c_rot.shape[1])
    assert np.allclose(diff.data, 0.0)

    c_trans_fc2 = get_lat_trans_compr_matrix_O2(trans_perms)
    c_trans_fc3 = get_lat_trans_compr_matrix_O3(trans_perms)

    c_rot_fc2 = c_trans_fc2.T @ c_rot_fc2
    c_rot_fc2 = dot_product_sparse(
        basis_set_fc2.compact_compression_matrix.T,
        c_rot_fc2,
        use_mkl=True,
    )
    c_rot_fc2 = dot_product_sparse(
        csr_array(basis_set_fc2.basis_set.T),
        c_rot_fc2,
        use_mkl=True,
    )

    c_rot_fc3 = c_trans_fc3.T @ c_rot_fc3
    c_rot_fc3 = dot_product_sparse(
        basis_set_fc3.compact_compression_matrix.T,
        c_rot_fc3,
        use_mkl=True,
    )
    c_rot_fc3 = dot_product_sparse(
        csr_array(basis_set_fc3.basis_set.T),
        c_rot_fc3,
        use_mkl=True,
    )

    c_rot = vstack([c_rot_fc2, c_rot_fc3])
    proj = dot_product_sparse(c_rot, c_rot.T, use_mkl=True)
    return proj


def complementary_projector_rot_reference(supercell: SymfcAtoms) -> np.ndarray:
    """Test function for setting rotational invariants."""
    natom = len(supercell.numbers)
    positions_cartesian = (supercell.scaled_positions) @ supercell.cell
    C23 = _orthogonalize_constraints(positions_cartesian)

    c_rot_fc2 = _get_complement_matrix(C23[:9], natom, order=2)
    c_rot_fc3 = _get_complement_matrix(C23[9:], natom, order=3)
    c_rot = vstack([c_rot_fc2, c_rot_fc3])
    c_rot = c_rot.toarray()
    proj = c_rot @ c_rot.T
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

    spg_reps = SpgRepsBase(supercell)
    trans_perms = spg_reps.translation_permutations

    symfc = Symfc(supercell, use_mkl=True, log_level=1).compute_basis_set(3)
    basis_set_fc2 = symfc.basis_set[2]
    basis_set_fc3 = symfc.basis_set[3]

    import time

    t1 = time.time()
    proj = complementary_compr_projector_rot(
        supercell,
        trans_perms,
        basis_set_fc2,
        basis_set_fc3,
        use_mkl=True,
    )
    t2 = time.time()
    if len(supercell.numbers) < 17:
        proj_ref = complementary_compr_projector_rot_reference(
            supercell,
            trans_perms,
            basis_set_fc2,
            basis_set_fc3,
        )
        assert np.allclose((proj_ref - proj).data, 0.0)
    t3 = time.time()
    print(t2 - t1, t3 - t2)

    proj = proj.toarray()
    print(proj.shape)
    eigvals, eigvecs = np.linalg.eigh(proj)
    # eigvecs = eigsh_projector(proj, verbose = True)

    np.set_printoptions(threshold=np.inf)
    print(eigvals)
    print(sum(eigvals))
    print(eigvecs.shape)

    nonzero = np.isclose(eigvals, 1.0)
    print("Number of eigenvectors:", np.count_nonzero(nonzero))
