import setuptools
from PyCO2SYS import __author__, __version__

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
setuptools.setup(
    name="PyCO2SYS",
    version=__version__,
    author=__author__,
    author_email="m.p.humphreys@icloud.com",
    description="PyCO2SYS: marine carbonate system calculations in Python",
    url="https://github.com/mvdh7/PyCO2SYS",
    packages=setuptools.find_packages(exclude=["manuscript", "tests", "validate"]),
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
