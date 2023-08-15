"""Generate symmetrized force constants using compact projection matrix."""
import itertools
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import coo_array

from symfc.spg_reps import SpgReps
from symfc.utils import (
    convert_basis_sets_matrix_form,
    get_compression_spg_proj,
    get_indep_atoms_by_lattice_translation,
    get_projector_permutations,
    to_serial,
)


class FCBasisSetsCompact:
    """Compact symmetry adapted basis sets for force constants.

    Strategy-1 : run(use_permutation=True)
    --------------------------------------
    Construct compression matrix using permutation symmetry C. The matrix shape
    is (NN33, N(N+1)/2). This matrix expands elements of upper right triagle to
    full elements NN33 of matrix. (C.T @ C) is made to be identity matrix. The
    projection matrix of space group operations is multipiled by C from both
    side, and the resultant matrix is diagonalized.

    Strategy-2 : run(use_permutation=False)
    ---------------------------------------
    Construct compression matrix using lattice translation symmetry C. The
    matrix shape is (NN33, n_aN33), where n_a is the number of atoms in
    primitive cell. This matrix expands elements of full elements NN33 of
    matrix. (C.T @ C) is made to be identity matrix. The projection matrix of
    space group operations is multipiled by C from both side, and the resultant
    matrix is diagonalized.

    """

    def __init__(self, spg_reps: SpgReps, log_level: int = 0):
        """Init method.

        Parameters
        ----------
        reps : list[coo_array]
            3Nx3N matrix representations of symmetry operations.
        translation_permutations:
            Atom indices after lattice translations.
            shape=(lattice_translations, supercell_atoms)
        log_level : int, optional
            Log level. Default is 0.

        """
        self._reps: list[coo_array] = spg_reps.representations
        self._translation_permutations = spg_reps.translation_permutations
        self._rotations = spg_reps.rotations
        self._translation_indices = spg_reps.translation_indices
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

    def run(self, use_permutation=False, tol: float = 1e-8):
        """Compute force constants basis."""
        if use_permutation:
            compression_mat = _get_permutation_compression_matrix(self._natom)
        else:
            compression_mat = _get_lattice_translation_compression_matrix(
                self._translation_permutations
            )
        vecs = self._step2(compression_mat, tol=tol)
        U = self._step3(vecs, compression_mat, use_permutation)
        U = compression_mat.T @ U
        self._step4(U, compression_mat, tol=tol)
        return self

    def _step2(self, compression_mat, tol: float = 1e-8) -> np.ndarray:
        compression_spg_mat = get_compression_spg_proj(
            self._reps,
            self._natom,
            compression_mat,
            rotations=self._rotations,
            translation_indices=self._translation_indices,
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

    def _step3(
        self, vecs: np.ndarray, compression_mat: coo_array, use_permutation: bool
    ) -> np.ndarray:
        if not use_permutation:
            print("Multiply index permutation projector")
            U = get_projector_permutations(self._natom) @ compression_mat
            U = U @ vecs
        else:
            U = compression_mat @ vecs
        print("Multiply sum rule projector")
        block = (
            np.tile(np.eye(9, dtype=float), (self._natom, self._natom)) / self._natom
        )
        for i in range(self._natom):
            U[i * self._natom * 9 : (i + 1) * self._natom * 9, :] -= (
                block @ U[i * self._natom * 9 : (i + 1) * self._natom * 9, :]
            )
        return U

    def _step4(self, U: np.ndarray, compression_mat: coo_array, tol: float = 1e-8):
        U, s, _ = np.linalg.svd(U, full_matrices=False)
        U = U[:, np.where(np.abs(s) > 1 - tol)[0]]

        if self._log_level:
            print(f"  - svd eigenvalues = {np.abs(s)}")
            print(f"  - basis size = {U.shape}")

        basis = (compression_mat @ U).T
        self._basis_sets = np.zeros(
            (basis.shape[0], self._natom, self._natom, 3, 3),
            dtype="double",
            order="C",
        )
        for i, b in enumerate(basis):
            self._basis_sets[i] = b.reshape((self._natom, self._natom, 3, 3))


def _get_permutation_compression_matrix(natom: int) -> coo_array:
    """Return compression matrix by permutation symmetry.

    Matrix shape is (NN33,(N*3)(N*3+1)/2).
    Non-zero only ijab and jiba column elements for ijab rows.
    Rows upper right NN33 matrix elements are selected for rows.

    """
    col, row, data = [], [], []
    val = np.sqrt(2) / 2
    size_row = natom**2 * 9

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
    return coo_array((data, (row, col)), shape=(size_row, n), dtype="double")


def _get_lattice_translation_compression_matrix(trans_perms: np.ndarray) -> coo_array:
    """Return compression matrix by lattice translation symmetry.

    Matrix shape is (NN33, n_a*N33), where n_a is the number of independent
    atoms by lattice translation symmetry.

    """
    col, row, data = [], [], []
    indep_atoms = get_indep_atoms_by_lattice_translation(trans_perms)
    n_a = len(indep_atoms)
    N = trans_perms.shape[1]
    n_lp = N // n_a
    val = 1.0 / np.sqrt(n_lp)
    size_row = (N * 3) ** 2

    n = 0
    for i_patom in indep_atoms:
        for j in range(N):
            for a, b in itertools.product(range(3), range(3)):
                for i_trans, j_trans in zip(trans_perms[:, i_patom], trans_perms[:, j]):
                    data.append(val)
                    col.append(n)
                    row.append(to_serial(i_trans, a, j_trans, b, N))
                n += 1

    assert n * n_lp == size_row
    return coo_array((data, (row, col)), shape=(size_row, n), dtype="double")
