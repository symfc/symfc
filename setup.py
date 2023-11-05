"""Setup script."""
from setuptools import setup

setup(
    name="symfc",
    version="0.1",
    setup_requires=["numpy", "setuptools"],
    description="This is the symfc module.",
    author="Atsuto Seko",
    author_email="seko@cms.mtl.kyoto-u.ac.jp",
    install_requires=["numpy", "scipy", "phonopy", "spglib"],
    provides=["symfc"],
    platforms=["all"],
)
