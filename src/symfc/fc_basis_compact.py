"""Generate symmetrized force constants using compact projection matrix."""
import itertools
import time
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import coo_array, csc_array, csr_array

from symfc.utils import (
    convert_basis_sets_matrix_form,
    get_permutation_spg_proj_c,
    get_projector_permutations,
    get_projector_sum_rule,
    to_serial,
)


def measure_time(func):
    """Measure time consumed by func."""

    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print(f"|--- {t1 - t0} ---")
        return result

    return wrapper


class FCBasisSetsCompact:
    """Compact symmetry adapted basis sets for force constants.

    Strategy-1
    ----------
    Construct compression matrix using permutation symmetry C. The matrix shape
    is (NN33, N(N+1)/2). This matrix expands elements of upper right triagle to
    full elements NN33 of matrix. (C @ C.T) is made to be identity matrix. The
    projection matrix of space group operations is multipiled by C from both
    side, and the resultant matrix is diagonalized.

    Strategy-2
    ----------
    Construct compression matrix using lattice translation symmetry C. The
    matrix shape is (NN33, n_aN33), where n_a is the number of atoms in
    primitive cell. This matrix expands elements of full elements NN33 of
    matrix. (C @ C.T) is made to be identity matrix. The projection matrix of
    space group operations is multipiled by C from both side, and the resultant
    matrix is diagonalized.

    """

    def __init__(
        self,
        reps: list[coo_array],
        translation_permutations: Optional[np.ndarray] = None,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        reps : list[coo_array]
            Matrix representations of symmetry operations.
        translation_permutations:
            Atom indices after lattice translations.
            shape=(lattice_translations, supercell_atoms)
        log_level : int, optional
            Log level. Default is 0.

        """
        self._reps: list[coo_array] = reps
        self._translation_permutations = translation_permutations
        self._log_level = log_level

        self._natom = self._reps[0].shape[0] // 3

        self._basis_sets: Optional[np.ndarray] = None

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

    def run(self, tol: float = 1e-8):
        """Compute force constants basis."""
        compression_mat = _get_permutation_compression_matrix(self._natom)
        vecs = self._step2(tol=tol)
        U = self._step3(vecs, compression_mat)
        self._step4(U, compression_mat, tol=tol)
        return self

    def _step2(self, tol: float = 1e-8) -> csr_array:
        row, col, data = get_permutation_spg_proj_c(self._reps, self._natom)
        if self._translation_permutations is not None:
            size = self._natom * 3 * (self._natom * 3 + 1)
            size = size // 2
        else:
            size = self._natom * 3 * (self._natom * 3 + 1)
            size = size // 2
        compression_spg_mat = csc_array(
            (data, (row, col)), shape=(size, size), dtype="double"
        )
        rank = int(round(compression_spg_mat.diagonal(k=0).sum()))
        if self._log_level:
            print(f"Solving eigenvalue problem of projection matrix (rank={rank}).")
        vals, vecs = scipy.sparse.linalg.eigsh(compression_spg_mat, k=rank, which="LM")
        nonzero_elems = np.nonzero(np.abs(vals) > tol)[0]
        vals = vals[nonzero_elems]
        # Check non-zero values are all ones. This is a weak check of commutativity.
        np.testing.assert_allclose(vals, 1.0, rtol=0, atol=tol)
        vecs = vecs[:, nonzero_elems]
        if self._log_level:
            print(f" eigenvalues of projector = {vals}")
        return vecs

    def _step3(self, vecs: csr_array, compression_mat: csr_array) -> csr_array:
        U = compression_mat @ vecs
        U = get_projector_sum_rule(self._natom) @ U
        if self._translation_permutations is not None:
            U = get_projector_permutations(self._natom) @ U
        U = compression_mat.T @ U
        return U

    def _step4(self, U: csr_array, perm_mat: csr_array, tol: float = 1e-8):
        # Note: proj_trans and (perm_mat @ perm_mat.T) are considered not commute.
        # for i in range(30):
        #     U = perm_mat.T @ (proj_trans @ (perm_mat @ U))
        U, s, _ = np.linalg.svd(U, full_matrices=False)
        # Instead of making singular value small by repeating, just removing
        # non one eigenvalues.
        U = U[:, np.where(np.abs(s) > 1 - tol)[0]]

        if self._log_level:
            print(f"  - svd eigenvalues = {np.abs(s)}")
            print(f"  - basis size = {U.shape}")

        fc_basis = [
            b.reshape((self._natom, self._natom, 3, 3)) for b in (perm_mat @ U).T
        ]
        self._basis_sets = np.array(fc_basis, dtype="double", order="C")


def _get_permutation_compression_matrix(natom: int) -> csr_array:
    """Return compression matrix by permutation matrix.

    Matrix shape is (NN33,(N*3)((N*3)+1)/2).
    Non-zero only ijab and jiba column elements for ijab rows.
    Rows upper right NN33 matrix elements are selected for rows.

    """
    col, row, data = [], [], []
    val = np.sqrt(2) / 2
    size_sq = natom**2 * 9

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
    return csr_array((data, (row, col)), shape=(size_sq, n), dtype="double")


def _get_lattice_translation_compression_matrix(natom: int) -> csr_array:
    """Return compression matrix by permutation matrix.

    Matrix shape is (NN33,(N*3)((N*3)+1)/2).
    Non-zero only ijab and jiba column elements for ijab rows.
    Rows upper right NN33 matrix elements are selected for rows.

    """
    col, row, data = [], [], []
    val = np.sqrt(2) / 2
    size_sq = natom**2 * 9

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
    return csr_array((data, (row, col)), shape=(size_sq, n), dtype="double")
