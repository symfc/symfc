"""Symmetry adapted basis sets of 1st order force constants."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_array

from symfc.spg_reps import SpgRepsO1
from symfc.utils.eig_tools import eigsh_projector
from symfc.utils.matrix_tools_O1 import compressed_projector_sum_rules
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O1 import (
    get_compr_coset_reps_sum,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
)

from .basis_sets_base import FCBasisSetBase


class FCBasisSetO1Base(FCBasisSetBase):
    """Base class of FCBasisSetO1."""

    def __init__(
        self,
        supercell: SymfcAtoms,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        use_mkl : bool
            Use MKL or not. Default is False.
        log_level : int, optional
            Log level. Default is 0.

        """
        super().__init__(supercell, use_mkl=use_mkl, log_level=log_level)
        self._spg_reps = SpgRepsO1(supercell)

    def _get_c_trans(self) -> csr_array:
        trans_perms = self._spg_reps.translation_permutations
        n_lp, N = trans_perms.shape
        decompr_idx = get_lat_trans_decompr_indices(trans_perms)
        c_trans = get_lat_trans_compr_matrix(decompr_idx, N, n_lp)
        return c_trans


class FCBasisSetO1(FCBasisSetO1Base):
    """Dense symmetry adapted basis set for 1st order force constants.

    Attributes
    ----------
    basis_set : ndarray
        Compressed force constants basis set. The first dimension n_x (< n_a) is
        given as a result of compression, which is depends on the system.
        shape=(n_x, n_bases), dtype='double'
    full_basis_set : ndarray
        Full (decompressed) force constants basis set. shape=(N * 3,
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
        log_level : int, optional
            Log level. Default is 0.

        """
        super().__init__(supercell, use_mkl=use_mkl, log_level=log_level)
        self._spg_reps = SpgRepsO1(
            supercell, spacegroup_operations=spacegroup_operations
        )
        self._n_a_compression_matrix: Optional[csr_array] = None

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return compressed basis set.

        n_c = len(compressed_indices).

        shape=(n_c, n_bases), dtype='double'.

        """
        return self._basis_set

    @property
    def full_basis_set(self) -> Optional[np.ndarray]:
        """Return full (decompressed) basis set.

        shape=(N*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (N,3).

        """
        return self._full_basis_set

    @property
    def compact_compression_matrix(self) -> Optional[np.ndarray]:
        """Return compression matrix for compact basis set."""
        pass

    @property
    def compression_matrix(self) -> Optional[np.ndarray]:
        """Return compression matrix."""
        pass

    def run(self) -> FCBasisSetO1:
        """Compute compressed force constants basis set."""
        c_trans = self._get_c_trans()
        coset_reps_sum = get_compr_coset_reps_sum(self._spg_reps)
        proj_rt = coset_reps_sum

        if len(proj_rt.data) == 0:
            raise ValueError("No basis vectors exist.")

        c_rt = eigsh_projector(proj_rt, verbose=self._log_level > 0)
        compress_mat = c_trans @ c_rt
        proj = compressed_projector_sum_rules(compress_mat, self._natom)
        self._basis_set = eigsh_projector(proj, verbose=self._log_level > 0)
        self._full_basis_set = compress_mat @ self._basis_set

        return self
