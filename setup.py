from distutils.core import setup

setup(
    name="sim2sim",
    version="1.0.0",
    packages=["sim2sim"],
    install_requires=[
        "numpy",
        "matplotlib",
        "pre-commit",
        "py",
        "trimesh",
        "manipulation",
        "pydrake",
        "pytorch3d",
        "open3d",
        "scipy",
        "pyrender",
        "prettytable",
        "wandb",
        "coacd",
    ],
)
