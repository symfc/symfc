"""Symmetry adapted basis sets of 2nd order force constants."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.spg_reps import SpgRepsO2
from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)
from symfc.utils.matrix_tools_O2 import compressed_projector_sum_rules
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import (
    _get_atomic_lat_trans_decompr_indices,
    get_compr_coset_reps_sum,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
    get_perm_compr_matrix,
)

from .basis_sets_base import FCBasisSetBase


class FCBasisSetO2(FCBasisSetBase):
    """Dense symmetry adapted basis set for 2nd order force constants.

    Attributes
    ----------
    basis_set : ndarray
        Compressed force constants basis set. The first dimension n_x (< n_a) is
        given as a result of compression, which is depends on the system.
        shape=(n_x * N * 9, n_bases), dtype='double'
    full_basis_set : ndarray
        Full (decompressed) force constants basis set. shape=(N * N * 9,
        n_bases), dtype='double'
    translation_permutations : ndarray
        Atom indices after lattice translations. shape=(lattice_translations,
        supercell_atoms), dtype=int.

    """

    def __init__(
        self,
        supercell: SymfcAtoms,
        spacegroup_operations: Optional[dict] = None,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
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
        super().__init__(supercell, use_mkl=use_mkl, log_level=log_level)
        self._spg_reps = SpgRepsO2(
            supercell, spacegroup_operations=spacegroup_operations
        )
        self._n_a_compression_matrix: Optional[csr_array] = None

        trans_perms = self._spg_reps.translation_permutations
        self._atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return compressed basis set.

        n_c = len(compressed_indices).

        shape=(n_c*N*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (n_c,N,3,3).

        """
        return self._basis_set

    @property
    def compression_matrix(self) -> Optional[csr_array]:
        """Return compression matrix.

        This expands fc basis_sets to (N*N*3*3, n_bases).

        """
        c_trans = self._get_c_trans()
        return dot_product_sparse(
            c_trans, self._n_a_compression_matrix, use_mkl=self._use_mkl
        )

    @property
    def compact_compression_matrix(self) -> Optional[csr_array]:
        """Return compact compression matrix.

        This expands basis_sets to (n_a*N*3*3, n_bases).

        """
        n_lp = self.translation_permutations.shape[0]
        return self._n_a_compression_matrix / np.sqrt(n_lp)

    @property
    def decompression_indices(self) -> np.ndarray:
        """Return decompression indices in (N,N,3,3) order."""
        trans_perms = self.translation_permutations
        return get_lat_trans_decompr_indices(trans_perms)

    @property
    def atomic_decompr_idx(self) -> np.ndarray:
        """Return atomic permutation."""
        return self._atomic_decompr_idx

    def run(self) -> FCBasisSetO2:
        """Compute compressed force constants basis set.

        compression using C(pt)
            = eigvecs of C(trans).T @ C(perm) @ C(perm).T @ C(trans)
        proj_rpt = c_pt.transpose() @ coset_reps_sum @ c_pt

        [C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt] @ C_rpt
         = C_rpt @ [C_rpt.T @ C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt @ C_rpt]
         = C_rpt @ compression_mat.T @ (I - P_sum^(c)) @ compression_mat
         = C_rpt @ compression_mat.T @ (I - Csum(c) @ Csum(c).T) @ compression_mat
         = C_rpt @ proj

        compress_mat = c_trans @ c_pt @ c_rpt

        """
        N = self._natom
        c_trans = self._get_c_trans()
        n_a_compress_mat = self._get_n_a_compress_mat(c_trans)
        compress_mat = dot_product_sparse(
            c_trans, n_a_compress_mat, use_mkl=self._use_mkl
        )
        proj = compressed_projector_sum_rules(compress_mat, N, use_mkl=self._use_mkl)
        eigvecs = eigsh_projector_sumrule(proj, verbose=self._log_level > 0)

        if self._log_level:
            print(f"Final size of basis set: {eigvecs.shape}", flush=True)

        self._basis_set = eigvecs
        self._n_a_compression_matrix = n_a_compress_mat
        return self

    def _get_n_a_compress_mat(self, c_trans: csr_array) -> csr_array:
        """Return compression matrix without c_trans mutiplied.

        This compression matrix is preserved as a class instance variable.
        The full compression matrix is obtained by

        c_trans @ n_a_compression_matrix.

        The compact compression matrix is obtained by

        n_a_compression_matrix / sqrt(n_lp).

        """
        N = self._natom
        c_perm = get_perm_compr_matrix(N)
        c_pt = dot_product_sparse(c_perm.T, c_trans, use_mkl=self._use_mkl)
        proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=self._use_mkl)
        coset_reps_sum = get_compr_coset_reps_sum(self._spg_reps)
        c_pt = eigsh_projector(proj_pt, verbose=self._log_level > 0)
        proj_rpt = dot_product_sparse(coset_reps_sum, c_pt, use_mkl=self._use_mkl)
        proj_rpt = dot_product_sparse(c_pt.T, proj_rpt)
        c_rpt = eigsh_projector(proj_rpt, verbose=self._log_level > 0)
        n_a_compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=self._use_mkl)
        return n_a_compress_mat

    def _get_c_trans(self) -> csr_array:
        trans_perms = self._spg_reps.translation_permutations
        n_lp, N = trans_perms.shape
        decompr_idx = get_lat_trans_decompr_indices(trans_perms)
        c_trans = get_lat_trans_compr_matrix(decompr_idx, N, n_lp)
        return c_trans
