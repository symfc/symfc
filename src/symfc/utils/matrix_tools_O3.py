"""Matrix utility functions for 3rd order force constants."""
import itertools
from math import sqrt

import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix

from symfc.utils.eig_tools import dot_sp_mats


def N3N3N3_to_NNN333(combinations_perm, N):
    """Transform index order."""
    vec = combinations_perm[:, 0] // 3 * 27 * N**2
    vec += combinations_perm[:, 1] // 3 * 27 * N
    vec += combinations_perm[:, 2] // 3 * 27
    vec += combinations_perm[:, 0] % 3 * 9
    vec += combinations_perm[:, 1] % 3 * 3
    vec += combinations_perm[:, 2] % 3
    return vec


def permutation_symmetry_basis(N):
    """Return compression matrix by permutation symmetry.

    Matrix shape is (NNN333, (N*3)(N*3+1)(N*3+2)/6).

    """
    NNN333 = 27 * N**3
    combinations3 = np.array(list(itertools.combinations(range(3 * N), 3)))
    combinations2 = np.array(list(itertools.combinations(range(3 * N), 2)))
    combinations1 = np.array([[i, i, i] for i in range(3 * N)])

    n_col3 = combinations3.shape[0]
    n_col2 = combinations2.shape[0] * 2
    n_col1 = combinations1.shape[0]
    n_col = n_col3 + n_col2 + n_col1
    n_data3 = combinations3.shape[0] * 6
    n_data2 = combinations2.shape[0] * 6
    n_data1 = combinations1.shape[0]
    n_data = n_data3 + n_data2 + n_data1

    row = np.zeros(n_data, dtype="int_")
    col = np.zeros(n_data, dtype="int_")
    data = np.zeros(n_data, dtype=float)

    # (3) for FC3 with three distinguished indices (ia,jb,kc)
    begin_id, end_id = 0, n_data3
    perms = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
    combinations_perm = combinations3[:, perms].reshape((-1, 3))
    row[begin_id:end_id] = N3N3N3_to_NNN333(combinations_perm, N)
    col[begin_id:end_id] = np.repeat(range(n_col3), 6)
    data[begin_id:end_id] = 1 / sqrt(6)

    # (2) for FC3 with two distinguished indices (ia,ia,jb)
    begin_id = end_id
    end_id = begin_id + n_data2
    perms = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    combinations_perm = combinations2[:, perms].reshape((-1, 3))
    row[begin_id:end_id] = N3N3N3_to_NNN333(combinations_perm, N)
    col[begin_id:end_id] = np.repeat(range(n_col3, n_col3 + n_col2), 3)
    data[begin_id:end_id] = 1 / sqrt(3)

    # (1) for FC3 with single index ia
    begin_id = end_id
    row[begin_id:] = N3N3N3_to_NNN333(combinations1, N)
    col[begin_id:] = np.array(range(n_col3 + n_col2, n_col))
    data[begin_id:] = 1.0

    return coo_matrix((data, (row, col)), shape=(NNN333, n_col))


def compressed_complement_projector_sum_rules_algo1(C, N, mkl=False):
    r"""Return complementary projection matrix for sum rule compressed by C.

    proj_sum_cplmt = [C.T @ Csum(c)] @ [Csum(c).T @ C]
                   = c_sum_cplmt_compr.T @ c_sum_cplmt_compr
    Matrix shape of proj_sum_cplmt is (C.shape[1], C.shape[1]).
    C.shape[0] must be equal to NNN333.

    Sum rules are given as sums over i: \sum_i \phi_{ia,jb,kc} = 0

    """
    NNN333 = 27 * N**3
    NN333 = 27 * N**2

    row = np.arange(NNN333)
    col = np.tile(range(NN333), N)
    data = np.zeros(NNN333)
    data[:] = 1 / sqrt(N)
    c_sum_cplmt = coo_matrix((data, (row, col)), shape=(NNN333, NN333))

    if mkl:
        C = C.tocsr()
        c_sum_cplmt = c_sum_cplmt.tocsr()

    # bottleneck part
    c_sum_cplmt_compr = dot_sp_mats(c_sum_cplmt.transpose(), C, mkl=mkl)
    proj_sum_cplmt = dot_sp_mats(
        c_sum_cplmt_compr.transpose(), c_sum_cplmt_compr, mkl=mkl
    )
    # bottleneck part: end
    return proj_sum_cplmt


def compressed_complement_projector_sum_rules_algo2(C, N, mkl=False):
    """Return complementary projection matrix for sum rule compressed by C.

    proj_sum_cplmt = [C.T @ Csum(c)] @ [Csum(c).T @ C]
    This version does not make Csum but is slow if N is large.

    """
    NN333 = 27 * N**2
    c_sum_cplmt_compr = csr_matrix(([], ([], [])), shape=(NN333, C.shape[1]))
    for i in range(N):
        c_sum_cplmt_compr += C[i * NN333 : (i + 1) * NN333]

    proj_sum_cplmt = dot_sp_mats(
        c_sum_cplmt_compr.transpose(), c_sum_cplmt_compr, mkl=mkl
    )
    proj_sum_cplmt /= N
    return proj_sum_cplmt


def compressed_complement_projector_sum_rules_algo3(C, N, mkl=False):
    """Return complementary projection matrix for sum rule compressed by C.

    proj_sum_cplmt = [C.T @ Csum(c)] @ [Csum(c).T @ C]
    This version does not make Csum but is too slow.

    """
    NN333 = 27 * N**2
    proj_sum_cplmt = csr_matrix(([], ([], [])), shape=(C.shape[1], C.shape[1]))
    for i, j in itertools.product(range(N), range(N)):
        proj_sum_cplmt += dot_sp_mats(
            C[i * NN333 : (i + 1) * NN333].transpose(),
            C[j * NN333 : (j + 1) * NN333],
            mkl=mkl,
        )
    proj_sum_cplmt /= N
    return proj_sum_cplmt


def compressed_complement_projector_sum_rules(C, N, mkl=False):
    """Return complementary projection matrix for sum rule compressed by C."""
    return compressed_complement_projector_sum_rules_algo1(C, N, mkl=mkl)


def compressed_projector_sum_rules(C, N, mkl=False):
    """Return projection matrix for sum rule compressed by C."""
    proj_cplmt = compressed_complement_projector_sum_rules(C, N, mkl=mkl)
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
