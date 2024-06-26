# PyCO2SYS

[![Tests](https://github.com/mvdh7/PyCO2SYS/workflows/Tests/badge.svg?branch=main)](https://github.com/mvdh7/PyCO2SYS/actions)
[![pypi badge](https://img.shields.io/pypi/v/PyCO2SYS.svg?style=popout)](https://pypi.org/project/PyCO2SYS/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyco2sys.svg)](https://anaconda.org/conda-forge/pyco2sys)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.3744275-informational)](https://doi.org/10.5281/zenodo.3744275)
[![Docs](https://readthedocs.org/projects/pyco2sys/badge/?version=latest&style=flat)](https://mvdh.xyz/PyCO2SYS/)
[![Coverage](https://github.com/mvdh7/PyCO2SYS/blob/main/.misc/coverage.svg)](https://github.com/mvdh7/PyCO2SYS/blob/main/.misc/coverage.txt)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- TOC -->

- [PyCO2SYS](#pyco2sys)
    - [Introduction](#introduction)
    - [Citation](#citation)
    - [Installation](#installation)
        - [With pip](#with-pip)
        - [With conda/mamba](#with-condamamba)
    - [Documentation](#documentation)
    - [Basic use](#basic-use)
    - [About](#about)
    - [License](#license)

<!-- /TOC -->

## Introduction

**PyCO2SYS** is a Python implementation of CO2SYS, based on the [MATLAB v2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB) but also including the updates made for [MATLAB CO2SYS v3](https://github.com/jonathansharp/CO2-System-Extd) as well as some additional related calculations.  PyCO2SYS solves the full marine carbonate system from the values of any two of its parameters.

Every combination of input parameters has been tested, with differences in the results small enough to be attributable to floating point errors and iterative solver endpoint differences (i.e. negligible).  See the scripts in [validate](https://github.com/mvdh7/PyCO2SYS/tree//validate) to see how and check this for yourself, and their [discussion](https://pyco2sys.readthedocs.io/en/latest/validate/) in the online docs.  **Please [let us know](https://github.com/mvdh7/PyCO2SYS/issues) ASAP if you discover a discrepancy that we have not spotted!**

Documentation is available online at [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/).

There are also some usage examples that you can either download or run live in your web browser (with no Python installation required) at [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples#pyco2sys-examples).

## Citation

A paper describing PyCO2SYS is freely available:

> Humphreys, M. P., Lewis, E. R., Sharp, J. D., and Pierrot, D. (2022).  PyCO2SYS v1.8: marine carbonate system calculations in Python.  *Geoscientific Model Development* 15, 15-43.  [doi:10.5194/gmd-15-15-2022](https://doi.org/10.5194/gmd-15-15-2022).

Please also specify the full version number that you used.  You can find this in Python with:

```python
import PyCO2SYS as pyco2
pyco2.hello()
```

As per the instructions in the [the CO2SYS-MATLAB repo](https://github.com/jamesorr/CO2SYS-MATLAB), you should also consider citing the original work by [Lewis and Wallace (1998)](https://pyco2sys.readthedocs.io/en/latest/refs/#l).

## Installation

**The current version of PyCO2SYS is not compatible with Numpy v2.  The problem lies with Autograd, rather than PyCO2SYS itself.  We are currently working on an updated version of PyCO2SYS which will avoid this problem and work with Numpy v2.**

If you manage Python with conda, we recommend that you first install NumPy, pandas and xarray into the environment where PyCO2SYS is to be installed with conda.

### With pip

Install from the [Python Package Index](https://pypi.org/project/PyCO2SYS/):

    pip install PyCO2SYS

Update an existing installation:

    pip install PyCO2SYS --upgrade --no-cache-dir

### With conda/mamba

Install from the [conda-forge channel](https://anaconda.org/conda-forge/pyco2sys):

    conda install PyCO2SYS -c conda-forge

## Documentation

Documentation for the current release, based on the `main` branch, is available at [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/) and at [mvdh.xyz/PyCO2SYS](https://mvdh.xyz/PyCO2SYS/).

## Basic use

The only function you need is `pyco2.sys`.  To solve the marine carbonate system from two of its parameters (`par1` and `par2`), just use:

```python
import PyCO2SYS as pyco2
results = pyco2.sys(par1, par2, par1_type, par2_type, **kwargs)
```

The keys to the `results` dict are described in the [documentation](https://pyco2sys.readthedocs.io/en/latest/co2sys_nd/#results).  Arguments should be provided as scalars or NumPy arrays in any mutually broadcastable combination.  A large number of optional `kwargs` can be provided to specify everything beyond the carbonate system parameters — [read the docs!](https://pyco2sys.readthedocs.io/en/latest/co2sys_nd/).

You can also look at the [examples Notebooks](https://github.com/mvdh7/PyCO2SYS-examples) that you can try out without needing to install anything on your computer.

## About

PyCO2SYS is maintained by [Dr Matthew Humphreys](https://mvdh.xyz/) at the [NIOZ (Royal Netherlands Institute for Sea Research)](https://www.nioz.nl/en) with the support of the main developers of all previous versions of CO<sub>2</sub>SYS.

Contributions are welcome - please check the [guidelines](https://github.com/mvdh7/PyCO2SYS/blob/main/CONTRIBUTING.md) before setting to work.

## License

PyCO2SYS is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).
