"""Symmetry adapted basis sets of 2nd order force constants."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.spg_reps import SpgRepsO2
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)
from symfc.utils.matrix_tools_O2 import (
    compressed_projector_sum_rules_O2,
    projector_permutation_lat_trans_O2,
)
from symfc.utils.permutation_tools_O2 import compr_permutation_lat_trans_O2
from symfc.utils.rotation_tools_O2 import complementary_compr_projector_rot_sum_rules_O2
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import (
    _get_atomic_lat_trans_decompr_indices,
    get_compr_coset_projector_O2,
    get_lat_trans_compr_matrix_O2,
)

from .basis_sets_base import FCBasisSetBase


class FCBasisSetO2(FCBasisSetBase):
    r"""Symmetry adapted basis set for 2nd order force constants.

    Attributes
    ----------
    basis_set : ndarray
        Compressed force constants basis set. The first dimension n_compr
        (<< 9 * N ** 2, \sim n_bases) is given as a result of compression,
        which depends on the system.  shape=(n_compr, n_bases), dtype='double'
    n_a_compression_matrix : csr_array
        Compression matrix compressed by lattice translation. The basis set
        compressed only by lattice translation is obtained by
        n_a_compression_matrix @ basis_set.
        shape=(n_a * N * 9, n_compr), dtype='double'
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
        super().__init__(supercell, use_mkl=use_mkl, log_level=log_level)
        self._spg_reps = SpgRepsO2(
            supercell, spacegroup_operations=spacegroup_operations
        )
        if cutoff is None:
            self._fc_cutoff = None
        else:
            self._fc_cutoff = FCCutoff(supercell, cutoff=cutoff)

        trans_perms = self._spg_reps.translation_permutations
        self._atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

        self._n_a_compression_matrix: Optional[csr_array] = None
        self._basis_set: Optional[np.ndarray] = None

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return compressed basis set.

        shape=(n_c, n_bases), dtype='double'.

        """
        return self._basis_set

    @property
    def compression_matrix(self) -> Optional[csr_array]:
        """Return compression matrix.

        This expands fc basis_sets to (N*N*3*3, n_bases).

        """
        trans_perms = self._spg_reps.translation_permutations
        c_trans = get_lat_trans_compr_matrix_O2(trans_perms)
        return dot_product_sparse(
            c_trans, self._n_a_compression_matrix, use_mkl=self._use_mkl
        )

    @property
    def compact_compression_matrix(self) -> Optional[csr_array]:
        """Return compact compression matrix.

        This expands fc basis_sets to (n_a*N*3*3, n_bases).

        """
        n_lp = self.translation_permutations.shape[0]
        return self._n_a_compression_matrix / np.sqrt(n_lp)

    @property
    def atomic_decompr_idx(self) -> np.ndarray:
        """Return atomic permutations by lattice translations."""
        return self._atomic_decompr_idx

    def run(self, rotational_sum_rules: bool = False) -> FCBasisSetO2:
        """Compute compressed force constants basis set."""
        trans_perms = self._spg_reps.translation_permutations

        direct_permutation = True
        if direct_permutation:
            c_pt = compr_permutation_lat_trans_O2(
                trans_perms,
                atomic_decompr_idx=self._atomic_decompr_idx,
                fc_cutoff=self._fc_cutoff,
                verbose=self._log_level > 0,
            )
        else:
            proj_pt = projector_permutation_lat_trans_O2(
                trans_perms,
                atomic_decompr_idx=self._atomic_decompr_idx,
                fc_cutoff=self._fc_cutoff,
                use_mkl=self._use_mkl,
                verbose=self._log_level > 0,
            )
            c_pt = eigsh_projector(proj_pt, verbose=self._log_level > 0)

        proj_rpt = get_compr_coset_projector_O2(
            self._spg_reps,
            fc_cutoff=self._fc_cutoff,
            atomic_decompr_idx=self._atomic_decompr_idx,
            c_pt=c_pt,
            # verbose=self._log_level > 0,
        )
        c_rpt = eigsh_projector(proj_rpt, verbose=self._log_level > 0)
        n_a_compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=self._use_mkl)

        proj = compressed_projector_sum_rules_O2(
            trans_perms,
            n_a_compress_mat,
            atomic_decompr_idx=self._atomic_decompr_idx,
            fc_cutoff=self._fc_cutoff,
            use_mkl=self._use_mkl,
            # verbose=self._log_level > 0,
        )

        if rotational_sum_rules:
            proj_rot_cmplt = complementary_compr_projector_rot_sum_rules_O2(
                self._supercell,
                trans_perms,
                n_a_compress_mat,
                use_mkl=self._use_mkl,
            )
            proj -= proj_rot_cmplt

        eigvecs = eigsh_projector_sumrule(proj, verbose=self._log_level > 0)

        self._basis_set = eigvecs
        self._n_a_compression_matrix = n_a_compress_mat

        return self
