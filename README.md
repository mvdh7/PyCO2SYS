# PyCO2SYS

[![PyPI version](https://badge.fury.io/py/PyCO2SYS.svg)](https://badge.fury.io/py/PyCO2SYS)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.3744275-informational)](https://zenodo.org/badge/latestdoi/237243120)
[![Docs](https://readthedocs.org/projects/pyco2sys/badge/?version=develop&style=flat)](https://pyco2sys.readthedocs.io/en/develop/)
[![Build Status](https://travis-ci.org/mvdh7/PyCO2SYS.svg?branch=develop)](https://travis-ci.org/mvdh7/PyCO2SYS)
[![Coverage](https://github.com/mvdh7/PyCO2SYS/blob/develop/misc/coverage.svg)](https://github.com/mvdh7/PyCO2SYS/blob/develop/misc/coverage.txt)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PyCO2SYS** is a Python implementation of CO2SYS, based on the [MATLAB v2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB) but also including the updates made for the forthcoming [MATLAB CO2SYS v3](https://github.com/jonathansharp/CO2-System-Extd) as well as some additional related calculations.  PyCO2SYS solves the full marine carbonate system from the values of any two of its parameters.

Every combination of input parameters has been tested, with differences in the results small enough to be attributable to floating point errors and iterative solver endpoint differences (i.e. negligible).  See the scripts in [validate](https://github.com/mvdh7/PyCO2SYS/tree/master/validate) to see how and check this for yourself, and their [discussion](https://pyco2sys.readthedocs.io/en/latest/validate/) in the online docs.  **Please [let us know](https://github.com/mvdh7/PyCO2SYS/issues) ASAP if you discover a discrepancy that we have not spotted!**

Documentation is available online at [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/).

There are also some usage examples that you can either download or run live in your web browser (with no Python installation required) at [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples#pyco2sys-examples).

## Citation

The citation for PyCO2SYS alone is:

> Humphreys, M. P., Gregor, L., Pierrot, D., van Heuven, S. M. A. C., Lewis, E. R., and Wallace, D. W. R. (2020).  PyCO2SYS: marine carbonate system calculations in Python.  *Zenodo.*  [doi:10.5281/zenodo.3744275](https://doi.org/10.5281/zenodo.3744275).

The DOI above refers to all versions of PyCO2SYS.  Please also specify the version number that you used.  You can find this in Python with:

```python
import PyCO2SYS as pyco2
pyco2.say_hello()
```

As per the instructions in the [the CO2SYS-MATLAB repo](https://github.com/jamesorr/CO2SYS-MATLAB), you should also cite the original work by [Lewis and Wallace (1998)](https://pyco2sys.readthedocs.io/en/latest/refs/#l).

## Installation

Install from the Python Package Index:

    pip install PyCO2SYS

Update an existing installation:

    pip install PyCO2SYS --upgrade --no-cache-dir

## Basic use

The default API has been kept as close to the MATLAB version as possible, although the first output is now a dict for convenience.  If you want the MATLAB-like experience, simplest recommended usage is therefore:

```python
from PyCO2SYS import CO2SYS
CO2dict = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN, PRESOUT,
    SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS, NH3=0.0, H2S=0.0)
```

Each field in the output `CO2dict` corresponds to a column in the original MATLAB output `DATA`.  The keys to the dict come from the original MATLAB output `HEADERS`.  Vector inputs should be provided as Numpy arrays.  Everything gets flattened with `ravel()`.  Single-value inputs are fine - they are automatically cast into correctly-sized arrays.  The extra inputs not present in the MATLAB version are all optional and explained [in the documentation](https://pyco2sys.readthedocs.io/en/latest/co2sys/#inputs).  There are more of these than appear in the example above.

It is also possible to provide Pandas Series and Xarray DataArrays using the wrapper functions provided.  For this and a more detailed explanation of all the inputs and outputs, see the [Pythonic API documentation](https://pyco2sys.readthedocs.io/en/latest/co2sys/).

You can also look at the [example scripts](https://github.com/mvdh7/PyCO2SYS/tree/master/examples) here in the repo, or there are also some [examples as Jupyter Notebooks](https://github.com/mvdh7/PyCO2SYS-examples) that you can try out without needing to install anything on your computer.

## About

PyCO2SYS is maintained by [Dr Matthew Humphreys](https://mvdh.xyz/) of the [NIOZ (Royal Netherlands Institute for Sea Research)](https://www.nioz.nl/en) with support from the main developers of all previous versions of CO<sub>2</sub>SYS.

Contributions are welcome; please check the [guidelines](https://github.com/mvdh7/PyCO2SYS/blob/master/CONTRIBUTING.md) before setting to work.

## License

PyCO2SYS is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).
