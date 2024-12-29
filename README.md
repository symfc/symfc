# symfc

## What does symfc do?

Atomic vibrations in crystals are often conveniently described using the phonon
model. In this model, the crystal potential is expanded into a Taylor series
with respect to atomic displacements from their equilibrium positions, and the
expansion coefficients are referred to as force constants.

Predicting phonon properties through computer simulations is becoming
increasingly popular, with the supercell approach being one of the techniques
employed for phonon calculations. In this method, force constants are derived
from datasets of atomic forces and displacements obtained from supercell
snapshots, which feature various configurations of atomic displacements.

While force constants possess specific symmetries, those computed from
displacement-force datasets often do not adhere to these symmetries due to
factors such as numerical noise or approximations used. Symfc is a software
designed to compute force constants from displacement-force datasets in the
supercell approach, ensuring they meet the required symmetry constraints.

## Citation of symfc

"Projector-based efficient estimation of force constants", A. Seko and A. Togo,
Phys. Rev. B, **110**, 214302 (2024)
[[doi](https://doi.org/10.1103/PhysRevB.110.214302)]
[[arxiv](https://arxiv.org/abs/2403.03588)].

```
@article{PhysRevB.110.214302,
  title = {Projector-based efficient estimation of force constants},
  author = {Seko, Atsuto and Togo, Atsushi},
  journal = {Phys. Rev. B},
  volume = {110},
  issue = {21},
  pages = {214302},
  numpages = {18},
  year = {2024},
  month = {Dec},
}
```
