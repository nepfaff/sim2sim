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
        "y-py",
        "pyrr",
        "trimesh",
        "pydrake",
        "pytorch3d",
        "open3d",
        "scipy",
        "pyrender",
        "prettytable",
        "wandb",
        "coacd",
        "setuptools",
        "opencv-python",
        "transforms3d",
        "PyOpenGL-accelerate",
    ],
)
