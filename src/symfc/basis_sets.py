"""Symmetry adapted basis sets of force constants."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import scipy
from phonopy.structure.atoms import PhonopyAtoms

from symfc.spg_reps import SpgReps, SpgRepsO2
from symfc.utils import (
    get_lat_trans_compr_indices,
    get_lat_trans_decompr_indices,
    get_spg_perm_projector,
)


class FCBasisSet(ABC):
    """Abstract base class of symmetry adapted basis set for force constants."""

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
        self._natom = len(supercell)
        self._log_level = log_level
        self._basis_set: Optional[np.ndarray] = None
        self._spg_reps: Optional[SpgReps] = None

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return compressed basis set."""
        return self._basis_set

    @abstractmethod
    def full_basis_set(self):
        """Return full (decompressed) basis set."""
        pass

    @abstractmethod
    def decompression_indices(self):
        """Return decompression indices."""
        pass

    @abstractmethod
    def compression_indices(self):
        """Return compression indices."""
        pass

    @property
    def translation_permutations(self) -> np.ndarray:
        """Return permutations by lattice translation."""
        return self._spg_reps.translation_permutations

    @abstractmethod
    def run(self):
        """Run basis set calculation."""
        pass


class FCBasisSetO2(FCBasisSet):
    """Symmetry adapted basis set for force constants.

    Attributes
    ----------
    basis_set : ndarray
        Compressed force constants basis set.
        shape=(n_a * N * 9, n_basis), dtype='double'
    full_basis_set : ndarray
        Full (decompressed) force constants basis set.
        shape=(N * N * 9, n_basis), dtype='double'
    decompression_indices : ndarray
        Decompression indices in (N,N,3,3) order.
        shape=(N^2*9,), dtype='int_'.
    compresssion_indices : ndarray
        Compression indices in (n_a,N,3,3) order.
        shape=(n_a*N*9, n_lp), dtype='int_'.
    translation_permutations : ndarray
        Atom indices after lattice translations.
        shape=(lattice_translations, supercell_atoms), dtype=int.

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
        super().__init__(supercell, log_level=log_level)
        self._spg_reps = SpgRepsO2(supercell)

    @property
    def full_basis_set(self) -> Optional[np.ndarray]:
        """Return full (decompressed) basis set."""
        if self._basis_set is None:
            return None
        return self._basis_set[self.decompression_indices, :]

    @property
    def decompression_indices(self) -> np.ndarray:
        """Return decompression indices in (N,N,3,3) order."""
        trans_perms = self.translation_permutations
        return get_lat_trans_decompr_indices(trans_perms)

    @property
    def compression_indices(self) -> np.ndarray:
        """Return compression indices in (n_a,N,3,3) order."""
        return get_lat_trans_compr_indices(self.translation_permutations)

    def run(self, tol: float = 1e-8) -> FCBasisSetO2:
        """Compute compressed force constants basis set.

        Parameters
        ----------
        tol : float
            Tolerance to identify zero eigenvalues. Default=1e-8.

        """
        decompr_idx = get_lat_trans_decompr_indices(self.translation_permutations)
        vecs = self._get_tilde_basis_set(decompr_idx, tol=tol)
        U = self._multiply_sum_rule_projector(decompr_idx, vecs)
        self._extract_basis_set(U, tol=tol)
        return self

    def _get_tilde_basis_set(
        self, decompr_idx: np.ndarray, tol: float = 1e-8
    ) -> np.ndarray:
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
        compression_spg_mat = get_spg_perm_projector(self._spg_reps, decompr_idx)
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

    def _multiply_sum_rule_projector(
        self, decompr_idx: np.ndarray, vecs: np.ndarray
    ) -> np.ndarray:
        if self._log_level:
            print("Multiply sum rule projector with current basis set...")
        n_lp, N = self.translation_permutations.shape
        n_a = N // n_lp
        U = np.zeros(shape=(n_a * 9 * N, vecs.shape[1]), dtype="double")
        compr_idx = self.compression_indices
        for i, vec in enumerate(vecs.T):
            basis = vec[decompr_idx].reshape(N, N, 9).sum(axis=1)
            basis = np.tile(basis, N).ravel()
            basis = basis[compr_idx].sum(axis=1) / (compr_idx.shape[1] * N)
            U[:, i] = vec - basis
        return U

    def _extract_basis_set(self, U: np.ndarray, tol: float = 1e-8):
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
