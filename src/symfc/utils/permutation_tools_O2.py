"""Permutation utility functions for 2nd order force constants."""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array, hstack

from symfc.utils.combination_tools import get_combinations
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.matrix import blocked_triple_product
from symfc.utils.permutation_tools import construct_basis_from_perm_decompr_indices
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices


def _N3N3_to_NNand33(combs: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Transform index order."""
    vecNN, vec33 = np.divmod(combs[:, 0], 3)
    vecNN *= N
    vec33 *= 3
    div, mod = np.divmod(combs[:, 1], 3)
    vecNN += div
    vec33 += mod
    return vecNN, vec33


def _update_perm_decompr_indices(
    combinations: np.ndarray,
    permutations: Union[np.ndarray, list],
    atomic_decompr_idx: np.ndarray,
    trans_perms: np.ndarray,
    perm_decompr_idx: np.ndarray,
    n_perms_group: int = 1,
    n_batch: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Apply permutations to lattice translation basis.

    Return
    ------
    perm_decompr_idx: Updated decompression indices of lattice translation basis
                      using permutations.
    """
    if len(combinations) == 0:
        return perm_decompr_idx
    n_lp, natom = trans_perms.shape
    n_comb = combinations.shape[0]
    n_perms = len(permutations)
    n_perms_sym = n_perms // n_perms_group
    for begin, end in zip(*get_batch_slice(n_comb, n_comb // n_batch), strict=True):
        if verbose:
            print("Permutation basis:", str(end) + "/" + str(n_comb), flush=True)
        combs_perm = combinations[begin:end][:, permutations].reshape((-1, 2))
        combs_perm, combs33 = _N3N3_to_NNand33(combs_perm, natom)
        decompr_idx_combs_perm = atomic_decompr_idx[combs_perm] * 9 + combs33
        decompr_idx_combs_perm = decompr_idx_combs_perm.reshape(-1, n_perms_sym)
        for orbit_components in decompr_idx_combs_perm.T:
            perm_decompr_idx[orbit_components] = decompr_idx_combs_perm[:, 0]
    return perm_decompr_idx


class PermutationO2:
    """Class for constructing permutation basis for 2nd order."""

    def __init__(
        self,
        trans_perms: np.ndarray,
        atomic_decompr_idx: Optional[NDArray] = None,
        fc_cutoff: Optional[FCCutoff] = None,
        verbose: bool = False,
    ):
        r"""Init method.

        Build a compression matrix for permutation rules compressed by C_trans.
        This calculates C_(trans,perm) without allocating C_trans and C_perm.
        Batch calculations are used to reduce memory allocation.

        Parameters
        ----------
        trans_perms : ndarray
            Permutation of atomic indices by lattice translational symmetry.
            dtype='intc'.
            shape=(n_l, N), where n_l and N are the numbers of lattce points and
            atoms in supercell.
        fc_cutoff : FCCutoff class object. Default is None.
        """
        self._trans_perms = trans_perms
        self._fc_cutoff = fc_cutoff
        self._verbose = verbose

        n_lp, natom = self._trans_perms.shape
        self._size_row = (natom**2 * 9) // n_lp

        self._indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
        if atomic_decompr_idx is None:
            atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
        self._atomic_decompr_idx = atomic_decompr_idx

        self._cpt_array = None

    def _run_indep1(self, perm_decompr_idx: NDArray):
        """Construct basis for N3-IDs (i, i)."""
        _, natom = self._trans_perms.shape
        combinations = np.array([[i] for i in range(3 * natom)], dtype=int)
        perms = [[0, 0]]
        perm_decompr_idx = _update_perm_decompr_indices(
            combinations,
            perms,
            self._atomic_decompr_idx,
            self._trans_perms,
            perm_decompr_idx,
            n_perms_group=1,
            n_batch=1,
            verbose=self._verbose,
        )
        return perm_decompr_idx

    def _run_indep2(self, perm_decompr_idx: NDArray):
        """Construct basis for N3-IDs (i, j)."""
        _, natom = self._trans_perms.shape
        combinations = get_combinations(
            natom, order=2, fc_cutoff=self._fc_cutoff, indep_atoms=self._indep_atoms
        )
        perms = [[0, 1], [1, 0]]
        perm_decompr_idx = _update_perm_decompr_indices(
            combinations,
            perms,
            self._atomic_decompr_idx,
            self._trans_perms,
            perm_decompr_idx,
            n_perms_group=1,
            n_batch=1,
            verbose=self._verbose,
        )
        return perm_decompr_idx

    def _initialize_perm_decompr_idx(self):
        """Initialize permutation IDs."""
        perm_decompr_idx = np.ones(self._size_row, dtype="int") * -1
        return perm_decompr_idx

    def _convert_to_matrix(self, perm_decompr_idx: NDArray):
        """Convert permutation decomposition indices into matrix."""
        c_pt = construct_basis_from_perm_decompr_indices(
            perm_decompr_idx,
            verbose=self._verbose,
        )
        self._cpt_array.append(c_pt)
        return self

    def run(self):
        """Construct basis for permutation rules compressed by C_trans."""
        self._cpt_array = []
        perm_decompr_idx = self._initialize_perm_decompr_idx()
        perm_decompr_idx = self._run_indep1(perm_decompr_idx)
        perm_decompr_idx = self._run_indep2(perm_decompr_idx)
        self._convert_to_matrix(perm_decompr_idx)
        return self

    @property
    def col_shape(self):
        """Return basis-set matrix shape."""
        return sum([c_pt.shape[1] for c_pt in self._cpt_array])

    @property
    def basis_set(self):
        """Return basis-set matrix for permutation compressed by lattice translation."""
        if len(self._cpt_array) == 1:
            return self._cpt_array[0]
        return hstack(self._cpt_array)

    @property
    def divided_basis_set(self):
        """Return basis-set matrices for permutation divided into reasonable sizes."""
        return self._cpt_array

    def blocked_triple_product(self, mat: csr_array, use_mkl: bool = False):
        """Calculate c_pt.T @ mat @ c_pt.

        Input matrix is overwritten.
        """
        return blocked_triple_product(self._cpt_array, mat, use_mkl=use_mkl)
