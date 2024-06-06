"""Matrix utility functions for 1st order force constants."""

import numpy as np
import scipy
from scipy.sparse import csr_array


def compressed_projector_sum_rules(
    compress_mat: csr_array, N: int, use_mkl: bool = False
):
    """Return projection matrix for sum rule compressed by C."""
    proj_cplmt = _compressed_complement_projector_sum_rules(
        compress_mat, N, use_mkl=use_mkl
    )
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


def _compressed_complement_projector_sum_rules_algo1(
    compress_mat: csr_array, N: int, use_mkl: bool = False
):
    r"""Return complementary projection matrix for sum rule compressed by C.

    proj_sum_cplmt = [C.T @ Csum(c)] @ [Csum(c).T @ C]
                   = c_sum_cplmt_compr.T @ c_sum_cplmt_compr
    Matrix shape of proj_sum_cplmt is (C.shape[1], C.shape[1]).
    C.shape[0] must be equal to NN33.

    Sum rules are given as sums over i: \sum_i \phi_{ia,jb} = 0

    """
    N3 = 3 * N

    row = np.arange(N3)
    col = np.tile(range(3), N)
    data = np.zeros(N3)
    data[:] = 1 / np.sqrt(N)
    c_sum_cplmt = csr_array((data, (row, col)), shape=(N3, 3))

    c_sum_cplmt_compr = c_sum_cplmt.T @ compress_mat
    proj_sum_cplmt = c_sum_cplmt_compr.T @ c_sum_cplmt_compr
    return proj_sum_cplmt


def _compressed_complement_projector_sum_rules(
    compress_mat: csr_array, N: int, use_mkl: bool = False
):
    """Return complementary projection matrix for sum rule compressed by C."""
    return _compressed_complement_projector_sum_rules_algo1(
        compress_mat, N, use_mkl=use_mkl
    )
