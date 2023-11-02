"""Generate symmetrized force constants using compact projection matrix."""
from typing import Optional

import numpy as np
import scipy
from phonopy.structure.atoms import PhonopyAtoms

from symfc.spg_reps import SpgReps
from symfc.utils import (
    get_lat_trans_compr_indices,
    get_lat_trans_decompr_indices,
    get_spg_projector,
)


class FCBasisSet:
    """Symmetry adapted basis set for force constants.

    Attributes
    ----------
    basis_set : ndarray
        Force constants basis set.
        shape=(n_a * N * 9, n_basis), dtype='double'

    """

    def __init__(
        self,
        supercell: PhonopyAtoms,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.
        log_level : int, optional
            Log level. Default is 0.

        """
        self._spg_reps = SpgReps(supercell).run()
        self._natom = len(supercell)
        self._log_level = log_level
        self._basis_set: Optional[np.ndarray] = None
        self._compression_mat = None

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return basis set in (n_a * N * 9, n_basis) array."""
        return self._basis_set

    @property
    def decompression_indices(self):
        """Return decompression indices in (N,N,3,3) order.

        shape=(N^2*9,), dtype='int_'.

        """
        trans_perms = self._spg_reps.translation_permutations
        return get_lat_trans_decompr_indices(trans_perms)

    @property
    def compression_indices(self):
        """Return compression indices in (n_a,N,3,3) order.

        shape=(n_a*N*9, n_lp), dtype='int_'.

        """
        trans_perms = self._spg_reps.translation_permutations
        return get_lat_trans_compr_indices(trans_perms)

    @property
    def translation_permutations(self):
        """Return permutations by lattice translation.

        Returns
        --------
        Atom indices after lattice translations.
        shape=(lattice_translations, supercell_atoms), dtype=int

        """
        return self._spg_reps.translation_permutations

    def run(self, tol: float = 1e-8):
        """Compute force constants basis.

        Parameters
        ----------
        tol : float
            Tolerance to identify zero eigenvalues. Default=1e-8.

        """
        decompr_idx = get_lat_trans_decompr_indices(
            self._spg_reps.translation_permutations
        )
        vecs = self._step1(decompr_idx, tol=tol)
        U = self._step2(decompr_idx, vecs)
        self._step3(U, tol=tol)
        return self

    def solve(
        self, displacements: np.ndarray, forces: np.ndarray, is_compact_fc=True
    ) -> Optional[np.ndarray]:
        """Solve force constants.

        Parameters
        ----------
        displacements : ndarray
            Displacements of atoms in Cartesian coordinates. shape=(n_snapshot,
            N, 3), dtype='double'
        forces : ndarray
            Forces of atoms in Cartesian coordinates. shape=(n_snapshot, N, 3),
            dtype='double'
        is_compact_fc : bool
            Shape of force constants array is (n_a, N, 3, 3) if True or (N, N,
            3, 3) if False.

        Returns
        -------
        ndarray
            Force constants.
            shape=(n_a, N, 3, 3) or (N, N, 3, 3). See `is_compact_fc` parameter.
            dtype='double', order='C'

        """
        if self._basis_set is None:
            return None
        assert displacements.shape == forces.shape
        coeff = self._get_basis_coeff(displacements, forces)
        N = self._natom
        if is_compact_fc:
            fc = (self._basis_set @ coeff).reshape(-1, N, 3, 3)
        else:
            trans_perms = self._spg_reps.translation_permutations
            decompr_idx = get_lat_trans_decompr_indices(trans_perms)
            fc = (self._basis_set @ coeff)[decompr_idx].reshape(N, N, 3, 3)
        return fc

    def _get_basis_coeff(
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self._basis_set is None:
            return None
        trans_perms = self._spg_reps.translation_permutations
        N = trans_perms.shape[1]
        decompr_idx = np.transpose(
            get_lat_trans_decompr_indices(trans_perms).reshape(N, N, 3, 3), (0, 2, 1, 3)
        ).reshape(N * 3, N * 3)
        n_snapshot = displacements.shape[0]
        disps = displacements.reshape(n_snapshot, -1)
        N = self._natom
        d_basis = np.zeros(
            (n_snapshot * N * 3, self._basis_set.shape[1]), dtype="double", order="C"
        )
        for i, vec in enumerate(self._basis_set.T):
            d_basis[:, i] = (disps @ vec[decompr_idx]).ravel()
        coeff = -(np.linalg.pinv(d_basis) @ forces.ravel())
        return coeff

    def _step1(self, decompr_idx: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """Compute eigenvectors of projection matrix.

        Projection matrix is made of the product of the projection matrices of
        space group operations and index permutation symmetry in supercell.

        The eigenvalues are 1 or 0. Therefore eigenvectors corresponding to
        eigenvalue=1 are collected using sparce eigen solver. The collected
        eigenvectors are basis vectors of force constants.

        """
        if self._log_level:
            print(
                "Construct projector matrix of space group and "
                "index permutation symmetry..."
            )
        compression_spg_mat = get_spg_projector(self._spg_reps, decompr_idx)
        rank = int(round(compression_spg_mat.diagonal(k=0).sum()))
        if self._log_level:
            N = self._natom**2 * 9
            N_c = compression_spg_mat.shape[0]
            print(f"Projection matrix ({N}, {N}) was compressed to ({N_c}, {N_c}).")
            print(
                f"Solving eigenvalue problem of projection matrix with rank={rank}..."
            )
        vals, vecs = scipy.sparse.linalg.eigsh(compression_spg_mat, k=rank, which="LM")
        # Check non-zero values are all ones. This is a weak check of
        # commutativity.
        np.testing.assert_allclose(vals, 1.0, rtol=0, atol=tol)
        return vecs

    def _step2(self, decompr_idx: np.ndarray, vecs: np.ndarray) -> np.ndarray:
        if self._log_level:
            print("Multiply sum rule projector with current basis set...")
        trans_perms = self._spg_reps.translation_permutations
        n_lp, N = trans_perms.shape
        n_a = N // n_lp
        U = np.zeros(shape=(n_a * 9 * N, vecs.shape[1]), dtype="double")
        compr_idx = get_lat_trans_compr_indices(trans_perms)
        for i, vec in enumerate(vecs.T):
            basis = vec[decompr_idx].reshape(N, N, 9).sum(axis=1)
            basis = np.tile(basis, N).ravel()
            basis = basis[compr_idx].sum(axis=1) / (compr_idx.shape[1] * N)
            U[:, i] = vec - basis
        return U

    def _step3(self, U: np.ndarray, tol: float = 1e-8):
        """Extract basis vectors that satisfies sum rule.

        Eigenvectors corresponding to SVD eigenvalues that are not 1 are
        rejected.

        """
        if self._log_level:
            print("Accomodate sum rule by SVD...")

        U, s, _ = np.linalg.svd(U, full_matrices=False)
        U = U[:, np.where(np.abs(s) > 1 - tol)[0]]

        if self._log_level:
            print("Excluded SVD eigenvalues:")
            print(f"{s[np.abs(s) < 1 - tol]}")
            print(f"Final size of basis set: {U.shape}")
            print(
                "Non-zero elems: "
                f"{np.count_nonzero(np.abs(U) > tol)}/{np.prod(U.shape)}"
            )

        self._basis_set = U
