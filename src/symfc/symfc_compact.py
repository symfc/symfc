"""Generate symmetrized force constants using compact projection matrix."""
import itertools
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import coo_array

from symfc.matrix_funcs import convert_basis_sets_matrix_form, kron_c, to_serial
from symfc.symfc import get_projector_constraints_sum_rule_array


class SymBasisSetsCompact:
    """Compact symmetry adapted basis sets for force constants."""

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
        print("Solving eigenvalue problem of projection matrix.")
        vals, vecs = scipy.sparse.linalg.eigsh(perm_spg_mat, k=rank, which="LM")

        nonzero_cols = np.where(np.isclose(vals, 1.0, rtol=0, atol=tol))[0]
        vecs = vecs[:, nonzero_cols]
        if self._log_level:
            print(
                f" eigenvalues of projector = {vals}, len(nonzero-vals)={vecs.shape[1]}"
            )

        C = get_projector_constraints_sum_rule_array(self._natom)
        # For this C, C.T @ C = self._natom * np.eye(C.shape[1])
        proj_trans = scipy.sparse.eye(size_sq) - (C @ C.T) / self._natom

        # checking commutativity of two projectors that are symmetric matrices.
        prod_mat = proj_spg @ proj_trans
        # When A, B are symmetric matrices, BA = B^T.A^T = (AB)^T.
        comm = prod_mat - prod_mat.T
        if np.all(np.abs(comm.data) < tol) is False:
            raise ValueError("Two projectors do not satisfy commutation rule.")

        U = proj_trans @ (perm_mat @ vecs)
        U, s, _ = np.linalg.svd(U, full_matrices=False)
        U = U[:, np.where(np.abs(s) > tol)[0]]

        if self._log_level:
            print(f"  - svd eigenvalues = {np.abs(s)}")
            print(f"  - basis size = {U.shape}")

        fc_basis = [b.reshape((self._natom, self._natom, 3, 3)) for b in U.T]

        self._basis_sets = np.array(fc_basis, dtype="double", order="C")


def _get_permutation_compression_matrix(natom: int) -> coo_array:
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
            data.append(np.sqrt(2) / 2)
            col.append(n)
            row.append(to_serial(i_j, i_b, i_i, i_a, natom))
            data.append(np.sqrt(2) / 2)
        n += 1
    if (natom * 3) % 2 == 1:
        assert (natom * 3) * ((natom * 3 + 1) // 2) == n, f"{natom}, {n}"
    else:
        assert ((natom * 3) // 2) * (natom * 3 + 1) == n, f"{natom}, {n}"
    return coo_array((data, (row, col)), shape=(size, n), dtype="double")
