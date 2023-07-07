"""Generate symmetrized force constants using compact projection matrix."""
import itertools
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import coo_array

from symfc.matrix_funcs import convert_basis_sets_matrix_form, kron_c, to_serial
from symfc.symfc import get_projector_constraints_sum_rule_array


class SymBasisSetsCompact:
    """Compact symmetry adapted basis sets for force constants.

    The strategy is as follows:
    Construct compression matrix using permutation symmetry C.
    The matrix shape is (NN33, N(N+1)/2).
    This matrix expands elements of upper right triagle to
    full elements NN33 of matrix. (C @ C.T) is made to be identity matrix.
    The projection matrix of space group operations is multipiled by C
    from both side, and the resultant matrix is diagonalized.
    The eigenvectors thus obtained are tricky further applying constraints
    of translational symmetry.

    """

    def __init__(
        self,
        reps: list[coo_array],
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        reps : list[coo_array]
            Matrix representations of symmetry operations.
        log_level : int, optional
            Log level. Default is 0.

        """
        self._reps: list[coo_array] = reps
        self._log_level = log_level

        self._natom = int(round(self._reps[0].shape[0] / 3))

        self._basis_sets: Optional[np.ndarray] = None

        self._run()

    @property
    def basis_sets_matrix_form(self) -> Optional[list[np.ndarray]]:
        """Retrun a list of FC basis in 3Nx3N matrix."""
        if self._basis_sets is None:
            return None

        return convert_basis_sets_matrix_form(self._basis_sets)

    @property
    def basis_sets(self) -> Optional[np.ndarray]:
        """Return a list of FC basis in (N, N, 3, 3) dimentional arrays."""
        return self._basis_sets

    def _run(self, tol: float = 1e-8):
        row, col, data = kron_c(self._reps, self._natom)
        size_sq = (3 * self._natom) ** 2
        proj_spg = coo_array(
            (data, (row, col)), shape=(size_sq, size_sq), dtype="double"
        )
        perm_mat = _get_permutation_compression_matrix(self._natom)
        perm_spg_mat = (perm_mat.T @ proj_spg) @ perm_mat
        rank = int(round(perm_spg_mat.diagonal(k=0).sum()))
        print(f"Solving eigenvalue problem of projection matrix (rank={rank}).")
        vals, vecs = scipy.sparse.linalg.eigsh(perm_spg_mat, k=rank, which="LM")
        nonzero_elems = np.nonzero(np.abs(vals) > tol)[0]
        np.testing.assert_allclose(vals[nonzero_elems], 1.0, rtol=0, atol=tol)
        vecs = vecs[:, nonzero_elems]
        vals = vals[nonzero_elems]
        if self._log_level:
            print(f" eigenvalues of projector = {vals}")

        # pattern 1
        # C = get_projector_constraints_sum_rule_array(self._natom)
        # # For this C, C.T @ C = self._natom * np.eye(C.shape[1])
        # proj_trans = scipy.sparse.eye(size_sq) - (C @ C.T) / self._natom

        # # # checking commutativity of two projectors that are symmetric matrices.
        # # prod_mat = proj_spg @ proj_trans
        # # # When A, B are symmetric matrices, BA = B^T.A^T = (AB)^T.
        # # comm = prod_mat - prod_mat.T
        # # if np.all(np.abs(comm.data) < tol) is False:
        # #     raise ValueError("Two projectors do not satisfy commutation rule.")

        # U = proj_trans @ (perm_mat @ vecs)
        # U, s, _ = np.linalg.svd(U, full_matrices=False)
        # U = U[:, np.where(np.abs(s) > tol)[0]]

        # if self._log_level:
        #     print(f"  - svd eigenvalues = {np.abs(s)}")
        #     print(f"  - basis size = {U.shape}")

        # fc_basis = [b.reshape((self._natom, self._natom, 3, 3)) for b in U.T]

        # pattern 2
        # C = _get_projector_constraints_sum_rule_perm_array(self._natom)
        # Cinv = scipy.sparse.linalg.inv(C.T @ C)
        # proj_trans_perm = scipy.sparse.eye(C.shape[0]) - ((C @ Cinv) @ C.T)
        # U = proj_trans_perm @ vecs
        # U, s, _ = np.linalg.svd(U, full_matrices=False)
        # U = U[:, np.where(np.abs(s) > tol)[0]]

        # if self._log_level:
        #     print(f"  - svd eigenvalues = {np.abs(s)}")
        #     print(f"  - basis size = {U.shape}")

        # fc_basis = [
        #     b.reshape((self._natom, self._natom, 3, 3)) for b in (perm_mat @ U).T
        # ]

        # pattern 3
        C = get_projector_constraints_sum_rule_array(self._natom)
        proj_trans = scipy.sparse.eye(size_sq) - (C @ C.T) / self._natom
        U = perm_mat.T @ (proj_trans @ (perm_mat @ vecs))
        U, s, _ = np.linalg.svd(U, full_matrices=False)
        U = U[:, np.where(np.abs(s) > 0.9)[0]]

        if self._log_level:
            print(f"  - svd eigenvalues = {np.abs(s)}")
            print(f"  - basis size = {U.shape}")

        fc_basis = [
            b.reshape((self._natom, self._natom, 3, 3)) for b in (perm_mat @ U).T
        ]

        # pattern 4
        # C = get_projector_constraints_sum_rule_array(self._natom)
        # D = perm_mat.T @ C
        # Dinv = np.linalg.inv((D.T @ D).toarray())
        # proj_trans = scipy.sparse.eye(D.shape[0]) - (D @ (Dinv @ D.T))
        # U = proj_trans @ vecs
        # U, s, _ = np.linalg.svd(U, full_matrices=False)
        # U = U[:, np.where(np.abs(s) > 0.9)[0]]

        # if self._log_level:
        #     print(f"  - svd eigenvalues = {np.abs(s)}")
        #     print(f"  - basis size = {U.shape}")

        # fc_basis = [
        #     b.reshape((self._natom, self._natom, 3, 3)) for b in (perm_mat @ U).T
        # ]

        self._basis_sets = np.array(fc_basis, dtype="double", order="C")


def _get_permutation_compression_matrix(
    natom: int, val: float = np.sqrt(2) / 2
) -> coo_array:
    """Return compression matrix by permutation matrix.

    Matrix shape is (NN33,(N*3)((N*3)+1)/2).
    Non-zero only ijab and jiba column elements for ijab rows.
    Rows upper right NN33 matrix elements are selected for rows.

    """
    col, row, data = [], [], []
    size = natom**2 * 9

    n = 0
    for ia, jb in itertools.combinations_with_replacement(range(natom * 3), 2):
        i_i = ia // 3
        i_a = ia % 3
        i_j = jb // 3
        i_b = jb % 3
        col.append(n)
        row.append(to_serial(i_i, i_a, i_j, i_b, natom))
        if i_i == i_j and i_a == i_b:
            data.append(1)
        else:
            data.append(val)
            col.append(n)
            row.append(to_serial(i_j, i_b, i_i, i_a, natom))
            data.append(val)
        n += 1
    if (natom * 3) % 2 == 1:
        assert (natom * 3) * ((natom * 3 + 1) // 2) == n, f"{natom}, {n}"
    else:
        assert ((natom * 3) // 2) * (natom * 3 + 1) == n, f"{natom}, {n}"
    return coo_array((data, (row, col)), shape=(size, n), dtype="double")


# def _get_projector_constraints_sum_rule_perm_array(natom: int) -> coo_array:
#     perm_serial_table = _get_perm_serial_table(natom)
#     assert len(perm_serial_table) == (natom * 3) * ((natom * 3) + 1) // 2, len(
#         perm_serial_table
#     )
#     col, row, data = [], [], []
#     n = 0
#     for i in range(natom):
#         for alpha, beta in itertools.product(range(3), range(3)):
#             for j in range(natom):
#                 ia = i * 3 + alpha
#                 jb = j * 3 + beta
#                 if ia > jb:
#                     row.append(perm_serial_table[(jb, ia)])
#                 else:
#                     row.append(perm_serial_table[(ia, jb)])
#                 col.append(n)
#                 if ia == jb:
#                     data.append(1)
#                 else:
#                     data.append(2)
#             n += 1

#     dtype = "intc"
#     row = np.array(row, dtype=dtype)
#     col = np.array(col, dtype=dtype)
#     return coo_array(
#         (data, (row, col)), shape=(len(perm_serial_table), n), dtype="double"
#     )


def _get_perm_serial_table(natom: int) -> dict:
    """Return upper right triangle NN33-1D index mapping table."""
    combination = itertools.combinations_with_replacement(range(natom * 3), 2)
    return {pair: i for i, pair in enumerate(combination)}
