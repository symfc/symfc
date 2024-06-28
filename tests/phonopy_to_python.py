"""A script to convert the phonopy.yaml to raw data in python."""

import numpy as np
import phono3py
import phonopy

ph3 = phono3py.load("phono3py_params_Si-111-222-rd.yaml.xz", produce_fc=False)
ph = phonopy.load("phonopy_GaN_222_rd.yaml.xz", produce_fc=False)

print("    lattice = [")
for v in ph.supercell.cell:
    print("[", ", ".join([f"{x:.15f}" for x in v]), "],")
print("    ]")

print("    points = [")
for v in ph.supercell.scaled_positions:
    print("[", ", ".join([f"{x:.15f}" for x in v]), "],")
print("    ]")

print("    numbers = [")
for v in ph.supercell.numbers:
    print(f"{v},")
print("    ]")

print("    cell = SymfcAtoms(cell=lattice, scaled_positions=points, numbers=numbers)")


print(f"    N = {ph.displacements.shape[0]}")
print('    dfset = np.loadtxt(cwd / "dfset_Si_111_fc3_rd.xz")')
print("    d = dfset[:, :3].reshape(N, -1, 3)")
print("    f = dfset[:, 3:].reshape(N, -1, 3)")
print("    return cell, d, f")

d = ph.displacements.reshape(-1, 3)
f = ph.forces.reshape(-1, 3)

with open("dfset_Si_111_fc3_rd", "w") as w:
    for v in np.c_[(d, f)]:
        print(" ".join([f"{x:18.15f}" for x in v]), file=w)
