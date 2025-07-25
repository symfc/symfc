"""Symmetry adapted basis sets of 4th order force constants."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.spg_reps import SpgRepsO4
from symfc.utils.eig_tools import (
    eigsh_projector,
    eigsh_projector_sumrule,
)
from symfc.utils.matrix import BlockMatrixNode

try:
    from symfc.utils.matrix import dot_product_sparse
except ImportError:
    pass
from symfc.utils.permutation_tools_O4 import compr_permutation_lat_trans_O4
from symfc.utils.translation_tools_O4 import compressed_projector_sum_rules_O4
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O4 import (
    get_atomic_lat_trans_decompr_indices_O4,
    get_compr_coset_projector_O4,
    get_lat_trans_compr_matrix_O4,
)

from . import FCBasisSetBase


class FCBasisSetO4(FCBasisSetBase):
    r"""Symmetry adapted basis set for 4th order force constants.

    Attributes
    ----------
    basis_set : ndarray
        Compressed force constants basis set. The first dimension n_compr
        (<< 81 * N ** 4, \sim n_bases) is given as a result of compression,
        which depends on the system.  shape=(n_compr, n_bases), dtype='double'
    n_a_compression_matrix : csr_array
        Compression matrix compressed by lattice translation. The basis set
        compressed only by lattice translation is obtained by
        n_a_compression_matrix @ basis_set.
        shape=(n_a * N * N * N * 81, n_compr), dtype='double'
    translation_permutations : ndarray
        Atom indices after lattice translations. shape=(lattice_translations,
        supercell_atoms), dtype=int.

    """

    def __init__(
        self,
        supercell: SymfcAtoms,
        cutoff: Optional[float] = None,
        spacegroup_operations: Optional[dict] = None,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        cutoff: float
            Cutoff distance in angstroms. Default is None.
        spacegroup_operations : dict, optional
            Space group operations in supercell, by default None. When None,
            spglib is used. The following keys and values correspond to spglib
            symmetry dataset:
                rotations : array_like
                translations : array_like
        use_mkl : bool
            Use MKL or not. Default is False.
        log_level : int, optional
            Log level. Default is 0.

        """
        super().__init__(supercell, cutoff=cutoff, use_mkl=use_mkl, log_level=log_level)
        self._spg_reps = SpgRepsO4(
            supercell, spacegroup_operations=spacegroup_operations
        )

        trans_perms = self._spg_reps.translation_permutations
        self._atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)

        self._n_a_compression_matrix: Optional[csr_array] = None
        self._basis_set: Optional[np.ndarray] = None
        self._blocked_basis_set: Optional[BlockMatrixNode] = None

    @property
    def compression_matrix(self) -> Optional[csr_array]:
        """Return compression matrix.

        This expands fc basis_sets to (N*N*N*N*3*3*3*3, n_bases).

        """
        if self._n_a_compression_matrix is None:
            raise ValueError(
                "Compression matrix is not computed yet. "
                "Call run() method to compute it."
            )
        trans_perms = self._spg_reps.translation_permutations
        c_trans = get_lat_trans_compr_matrix_O4(trans_perms)
        return dot_product_sparse(
            c_trans, self._n_a_compression_matrix, use_mkl=self._use_mkl
        )

    @property
    def compact_compression_matrix(self) -> Optional[csr_array]:
        """Return compact compression matrix.

        This expands fc basis_sets to (n_a*N*N*N*3*3*3*3, n_bases).

        """
        if self._n_a_compression_matrix is None:
            raise ValueError(
                "Compression matrix is not computed yet. "
                "Call run() method to compute it."
            )
        n_lp = self.translation_permutations.shape[0]
        return self._n_a_compression_matrix / np.sqrt(n_lp)

    def run(self) -> FCBasisSetO4:
        """Compute compressed force constants basis set."""
        trans_perms = self._spg_reps.translation_permutations

        tt0 = time.time()
        c_pt = compr_permutation_lat_trans_O4(
            trans_perms,
            atomic_decompr_idx=self._atomic_decompr_idx,
            fc_cutoff=self._fc_cutoff,
            verbose=self._log_level > 0,
        )

        if self._log_level:
            print(" c_pt (size) :", c_pt.shape, flush=True)
        tt2 = time.time()

        proj_rpt = get_compr_coset_projector_O4(
            self._spg_reps,  # type: ignore
            fc_cutoff=self._fc_cutoff,
            atomic_decompr_idx=self._atomic_decompr_idx,
            c_pt=c_pt,
            use_mkl=self._use_mkl,
            verbose=self._log_level > 0,
        )
        tt3 = time.time()

        c_rpt = eigsh_projector(proj_rpt, verbose=self._log_level > 0)
        if self._log_level:
            print(" c_rpt (size) :", c_rpt.shape, flush=True)
        tt4 = time.time()

        n_a_compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=self._use_mkl)
        tt5 = time.time()

        proj = compressed_projector_sum_rules_O4(
            trans_perms,
            n_a_compress_mat,
            atomic_decompr_idx=self._atomic_decompr_idx,
            fc_cutoff=self._fc_cutoff,
            use_mkl=self._use_mkl,
            verbose=self._log_level > 0,
        )
        tt6 = time.time()
        eigvecs = eigsh_projector_sumrule(
            proj,
            use_mkl=self._use_mkl,
            verbose=self._log_level > 0,
        )

        if self._log_level:
            print("Final size of basis set:", eigvecs.shape, flush=True)
        tt7 = time.time()

        if self._log_level:
            print(
                "Time (perm @ ltrans)               :",
                "{:.3f}".format(tt2 - tt0),
                flush=True,
            )
            print(
                "Time (coset)                       :",
                "{:.3f}".format(tt3 - tt2),
                flush=True,
            )
            print(
                "Time (eigh(coset @ perm @ ltrans)) :",
                "{:.3f}".format(tt4 - tt3),
                flush=True,
            )
            print(
                "Time (c_pt @ c_rpt)                :",
                "{:.3f}".format(tt5 - tt4),
                flush=True,
            )
            print(
                "Time (proj(sum))                   :",
                "{:.3f}".format(tt6 - tt5),
                flush=True,
            )
            print(
                "Time (eigh(sum))                   :",
                "{:.3f}".format(tt7 - tt6),
                flush=True,
            )
            print("---")
            print(
                "Time (Basis FC4)                   :",
                "{:.3f}".format(tt7 - tt0),
                flush=True,
            )

        self._blocked_basis_set = eigvecs
        self._n_a_compression_matrix = n_a_compress_mat

        return self

    def estimate_basis_size(self) -> int:
        """Estimate basis set size."""
        if self._fc_cutoff is None:
            n_sym, N = self._spg_reps._permutations.shape
            basis_size_estimates = 81 * (N**4) / n_sym / 24
            return int(np.round(basis_size_estimates).astype(int))

        trans_perms = self._spg_reps.translation_permutations
        c_pt = compr_permutation_lat_trans_O4(
            trans_perms,
            atomic_decompr_idx=self._atomic_decompr_idx,
            fc_cutoff=self._fc_cutoff,
            verbose=False,
        )
        n_sym_prim = len(self._spg_reps._unique_rotations)
        basis_size_estimates = c_pt.shape[1] / n_sym_prim  # type: ignore
        return int(np.round(basis_size_estimates).astype(int))
