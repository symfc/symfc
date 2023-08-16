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
    to_serial,
)


class FCBasisSetsCompact:
    """Compact symmetry adapted basis sets for force constants.

    Strategy
    --------
    Construct compression matrix using lattice translation symmetry C. The
    matrix shape is (NN33, n_aN33), where n_a is the number of atoms in
    primitive cell. This matrix expands elements of full elements NN33 of
    matrix. (C.T @ C) is made to be identity matrix. The projection matrix of
    space group operations is multipiled by C from both side, and the resultant
    matrix is diagonalized. In addition, (C_perm @ C_perm.T) that is the
    projector of index permutation symmetry is multiplied.

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

    def run(
        self,
        with_all_operations: bool = False,
        tol: float = 1e-8,
    ):
        """Compute force constants basis.

        Parameters
        ----------
        with_all_operations : bool, optional
            With True, space group operation projector is constructued using all
            given symmetry operations. With False, the projector is constructued
            as a product of sum of those operations with unique rotations (up to
            48 operations) and lattice translation projector. The former is
            considered as the coset representatives and the lattice is the
            translation group. Default is False.

        """
        if self._log_level:
            print("Construct compression matrix of lattice translation.")
        compression_mat = _get_lattice_translation_compression_matrix(
            self._translation_permutations
        )
        vecs = self._step2(
            compression_mat, with_all_operations=with_all_operations, tol=tol
        )
        U = self._step3(vecs, compression_mat)
        self._step4(U, compression_mat, tol=tol)
        return self

    def _step2(
        self,
        compression_mat: coo_array,
        with_all_operations: bool = False,
        tol: float = 1e-8,
    ) -> np.ndarray:
        if self._log_level:
            print(
                "Construct projector of product of space group and "
                "index permutation symmetry."
            )
        compression_spg_mat = get_compression_spg_proj(
            self._reps,
            self._natom,
            compression_mat,
            rotations=self._rotations,
            translation_indices=self._translation_indices,
            with_all_operations=with_all_operations,
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

    def _step3(self, vecs: np.ndarray, compression_mat: coo_array) -> np.ndarray:
        # print("Multiply index permutation projector")
        # U = get_projector_permutations(self._natom) @ compression_mat
        print("Multiply sum rule projector")
        U = compression_mat @ vecs
        block = np.tile(
            np.eye(9, dtype=float) / self._natom, (self._natom, self._natom)
        )
        for i in range(self._natom):
            U[i * self._natom * 9 : (i + 1) * self._natom * 9, :] -= (
                block @ U[i * self._natom * 9 : (i + 1) * self._natom * 9, :]
            )
        U = compression_mat.T @ U
        return U

    def _step4(self, U: np.ndarray, compression_mat: coo_array, tol: float = 1e-8):
        U, s, _ = np.linalg.svd(U, full_matrices=False)
        U = U[:, np.where(np.abs(s) > 1 - tol)[0]]

        if self._log_level:
            print(f"  - svd eigenvalues = {np.abs(s)}")
            print(f"  - basis size = {U.shape}")

        self._basis_sets = (compression_mat @ U).T.reshape(
            (U.shape[1], self._natom, self._natom, 3, 3)
        )


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
