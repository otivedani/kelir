try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import setup

version = "0.0.1"

install_requires = [
    "numpy>=1.20.1",
    "scipy",
]
extras_require = {"plotting": ["matplotlib", "jupyter"]}
setup_requires = ["flake8"]

setup(
    name="kelir",
    version=version,
    author="Dani Devito",
    author_email="otivedani@outlook.com",
    description="Image and Color Manipulation Tools",
    packages=find_packages(),
    python_requires="~=3.7",
    install_requires=install_requires,
    extras_require=extras_require,
    setup_requires=setup_requires,
    test_suite="pytest"
)
