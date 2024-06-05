"""Setup script."""

from setuptools import setup

setup(
    name="symfc",
    version="0.1",
    description="This is the symfc module.",
    author="Atsuto Seko",
    author_email="seko@cms.mtl.kyoto-u.ac.jp",
    install_requires=["numpy", "scipy", "spglib"],
    provides=["symfc"],
    platforms=["all"],
)
