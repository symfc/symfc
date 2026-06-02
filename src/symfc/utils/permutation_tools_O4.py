"""Permutation utility functions for 4th order force constants."""

import itertools
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array, hstack

from symfc.utils.combination_tools import get_combinations, get_combinations_first_atom
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.matrix import blocked_product, blocked_triple_product
from symfc.utils.permutation_tools import (
    construct_basis_from_perm_decompr_indices,
    find_groups_perm_decompr_indices,
)
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O4 import get_atomic_lat_trans_decompr_indices_O4


def _N3N3N3N3_to_NNNNand3333(
    combs: np.ndarray, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Transform index order."""
    vecNNNN, vec3333 = np.divmod(combs[:, 0], 3)
    vecNNNN *= N**3
    vec3333 *= 27
    div, mod = np.divmod(combs[:, 1], 3)
    vecNNNN += div * N**2
    vec3333 += mod * 9
    div, mod = np.divmod(combs[:, 2], 3)
    vecNNNN += div * N
    vec3333 += mod * 3
    div, mod = np.divmod(combs[:, 3], 3)
    vecNNNN += div
    vec3333 += mod
    return vecNNNN, vec3333


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
        combs_perm = combinations[begin:end][:, permutations].reshape((-1, 4))
        combs_perm, combs3333 = _N3N3N3N3_to_NNNNand3333(combs_perm, natom)
        decompr_idx_combs_perm = atomic_decompr_idx[combs_perm] * 81 + combs3333
        decompr_idx_combs_perm = decompr_idx_combs_perm.reshape(-1, n_perms_sym)
        for orbit_components in decompr_idx_combs_perm.T:
            perm_decompr_idx[orbit_components] = decompr_idx_combs_perm[:, 0]
    return perm_decompr_idx


def _construct_basis_from_perm_decompr_indices(
    perm_decompr_idx: np.ndarray,
    cpt_array: list,
    n_div: int = 3,
    verbose: bool = False,
):
    """Transform perm_decompr_idx into partitioned basis matrices.

    Parameters
    ----------
    perm_decompr_idx: Decompression indices of lattice translation basis
                      using permutations.
    Return
    ------
    c_pt: Compressed basis matrix for permutations and lattice translations.
          c_pt = eigh(C_trans.T @ C_perm @ C_perm.T @ C_trans)
    """
    if verbose:
        print("Construct permutation basis matrix.", flush=True)

    size_full = len(perm_decompr_idx)
    rows, cols, values, n_col = find_groups_perm_decompr_indices(
        perm_decompr_idx, verbose=verbose
    )

    # Partitioning of C_pt matrix.
    chunks = [(i * n_col // n_div, (i + 1) * n_col // n_div) for i in range(n_div)]
    for start, end in chunks:
        n_col_batch = end - start
        match = (cols >= start) & (cols < end)
        cols_match = cols[match]
        c_pt = csr_array(
            (values[cols_match], (rows[match], cols_match - start)),
            shape=(size_full, n_col_batch),
            dtype="double",
        )
        cpt_array.append(c_pt)
    return cpt_array


class PermutationO4:
    """Class for constructing permutation basis for 4th order."""

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

        Return
        ------
        c_pt: Compressed basis matrix for permutations and lattice translations.
              c_pt = eigh(C_trans.T @ C_perm @ C_perm.T @ C_trans)
              shape: (NNNN3333//n_lp, n_basis_pt).
              n_basis_pt denotes the size of basis for permutations and
              lattice translations.

        Algorithm
        ---------
        1. Calculate combinations {(i, a), (j, b), (k, c)}.
        2. Apply permutations to these combinations and calculate the orbits of
           {(i, a), (j, b), (k, c)} related to each other by permutations.
        3. Calculate the lattice translation basis indices of the orbits,
           and represent the lattice translation indices related to each other
           by a representative in perm_decompr_idx.
        """
        self._trans_perms = trans_perms
        self._fc_cutoff = fc_cutoff
        self._verbose = verbose

        n_lp, natom = self._trans_perms.shape
        self._size_row = (natom**4 * 81) // n_lp

        self._indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
        if atomic_decompr_idx is None:
            atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
        self._atomic_decompr_idx = atomic_decompr_idx

        self._cpt_array = None

    def _run_indep1(self, perm_decompr_idx: NDArray):
        """Construct basis for N3-IDs (i, i, i, i)."""
        _, natom = self._trans_perms.shape
        combinations = np.array([[i] for i in range(3 * natom)], dtype=int)
        perms = [[0, 0, 0, 0]]
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
        """Construct basis for N3-IDs (i, i, i, j) and (i, j, i, j)."""
        _, natom = self._trans_perms.shape
        combinations = get_combinations(
            natom, order=2, fc_cutoff=self._fc_cutoff, indep_atoms=self._indep_atoms
        )
        perms = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ]
        perm_decompr_idx = _update_perm_decompr_indices(
            combinations,
            perms,
            self._atomic_decompr_idx,
            self._trans_perms,
            perm_decompr_idx,
            n_perms_group=2,
            n_batch=1,
            verbose=self._verbose,
        )
        perms = [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ]
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

    def _run_indep3(self, perm_decompr_idx: NDArray, n_batch: Optional[int] = None):
        """Construct basis for N3-IDs (i, i, j, k)."""
        _, natom = self._trans_perms.shape
        perms = [
            [0, 0, 1, 2],
            [0, 0, 2, 1],
            [0, 1, 0, 2],
            [0, 2, 0, 1],
            [0, 1, 2, 0],
            [0, 2, 1, 0],
            [1, 0, 0, 2],
            [2, 0, 0, 1],
            [1, 0, 2, 0],
            [2, 0, 1, 0],
            [1, 2, 0, 0],
            [2, 1, 0, 0],
            [1, 1, 0, 2],
            [1, 1, 2, 0],
            [1, 0, 1, 2],
            [1, 2, 1, 0],
            [1, 0, 2, 1],
            [1, 2, 0, 1],
            [0, 1, 1, 2],
            [2, 1, 1, 0],
            [0, 1, 2, 1],
            [2, 1, 0, 1],
            [0, 2, 1, 1],
            [2, 0, 1, 1],
            [2, 2, 1, 0],
            [2, 2, 0, 1],
            [2, 1, 2, 0],
            [2, 0, 2, 1],
            [2, 1, 0, 2],
            [2, 0, 1, 2],
            [1, 2, 2, 0],
            [0, 2, 2, 1],
            [1, 2, 0, 2],
            [0, 2, 1, 2],
            [1, 0, 2, 2],
            [0, 1, 2, 2],
        ]
        for first_atom in self._indep_atoms:
            if self._verbose:
                print("Permutation - atom:", first_atom, flush=True)

            combs_first_atom = get_combinations_first_atom(
                natom, order=3, first_atom=first_atom, fc_cutoff=self._fc_cutoff
            )
            for combinations in combs_first_atom:
                perm_decompr_idx = _update_perm_decompr_indices(
                    combinations,
                    perms,
                    self._atomic_decompr_idx,
                    self._trans_perms,
                    perm_decompr_idx,
                    n_perms_group=3,
                    n_batch=n_batch,
                    verbose=self._verbose,
                )
        return perm_decompr_idx

    def _run_indep4(self, perm_decompr_idx: NDArray, n_batch: Optional[int] = None):
        """Construct basis for N3-IDs (i, j, k, l)."""
        _, natom = self._trans_perms.shape
        perms = np.array(list(itertools.permutations(range(4))))
        for first_atom in self._indep_atoms:
            if self._verbose:
                print("Permutation - atom:", first_atom, flush=True)

            combs_first_atom = get_combinations_first_atom(
                natom, order=4, first_atom=first_atom, fc_cutoff=self._fc_cutoff
            )
            for combinations in combs_first_atom:
                perm_decompr_idx = _update_perm_decompr_indices(
                    combinations,
                    perms,
                    self._atomic_decompr_idx,
                    self._trans_perms,
                    perm_decompr_idx,
                    n_perms_group=1,
                    n_batch=n_batch,
                    verbose=self._verbose,
                )
        return perm_decompr_idx

    def _initialize_perm_decompr_idx(self):
        """Initialize permutation IDs."""
        perm_decompr_idx = np.ones(self._size_row, dtype="int_") * -1
        return perm_decompr_idx

    def _convert_to_matrix(self, perm_decompr_idx: NDArray):
        """Convert permutation decomposition indices into matrix."""
        c_pt = construct_basis_from_perm_decompr_indices(
            perm_decompr_idx,
            verbose=self._verbose,
        )
        self._cpt_array.append(c_pt)
        return self

    def _convert_to_matrix_partition(self, perm_decompr_idx: NDArray):
        """Convert permutation decomposition indices into matrix."""
        self._cpt_array = _construct_basis_from_perm_decompr_indices(
            perm_decompr_idx,
            self._cpt_array,
            verbose=self._verbose,
        )
        return self

    def run(self, n_batch: Optional[int] = None):
        """Construct basis for permutation rules compressed by C_trans."""
        self._cpt_array = []
        n_lp, natom = self._trans_perms.shape
        if n_batch is None:
            n_batch3 = 1 if natom <= 256 else int(round((natom / 256) ** 2))
        else:
            n_batch3 = n_batch
        if n_batch is None:
            n_batch4 = 1 if natom <= 64 else int(round((natom / 64) ** 2))
        else:
            n_batch4 = n_batch

        perm_decompr_idx = self._initialize_perm_decompr_idx()
        perm_decompr_idx = self._run_indep1(perm_decompr_idx)
        perm_decompr_idx = self._run_indep2(perm_decompr_idx)
        perm_decompr_idx = self._run_indep3(perm_decompr_idx, n_batch=n_batch3)
        perm_decompr_idx = self._run_indep4(perm_decompr_idx, n_batch=n_batch4)

        if natom <= 500:
            self._convert_to_matrix(perm_decompr_idx)
        else:
            self._convert_to_matrix_partition(perm_decompr_idx)
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

    def blocked_product(self, mat: csr_array, use_mkl: bool = False):
        """Calculate c_pt @ mat."""
        return blocked_product(self._cpt_array, mat, use_mkl=use_mkl)
