"""Calculate force constants using symfc."""
import phonopy
from symfc import Symfc

ph = phonopy.load("phonopy_NaCl444_rd.yaml.xz", produce_fc=False)
symfc = Symfc(
    ph.supercell,
    displacements=ph.dataset["displacements"],
    forces=ph.dataset["forces"],
    log_level=1,
)
ph.force_constants = symfc.force_constants
ph.auto_band_structure(plot=True).show()
