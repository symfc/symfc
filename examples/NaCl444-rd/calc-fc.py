"""Calculate force constants using symfc."""

import phonopy
from symfc import Symfc

ph = phonopy.load("phonopy_NaCl444_rd.yaml.xz", produce_fc=False)
symfc = Symfc(
    ph.supercell,
    displacements=ph.dataset["displacements"],
    forces=ph.dataset["forces"],
    orders=[
        2,
    ],
    log_level=1,
)
ph.force_constants = symfc.force_constants[2]
ph.auto_band_structure(plot=True).show()
