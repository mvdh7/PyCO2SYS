# PyCO2SYS

[![PyPI version](https://badge.fury.io/py/PyCO2SYS.svg)](https://badge.fury.io/py/PyCO2SYS)
[![DOI](https://zenodo.org/badge/237243120.svg)](https://zenodo.org/badge/latestdoi/237243120)

PyCO2SYS is a Python toolbox for solving the marine carbonate system and calculating related seawater properties.  Its core is a Python implementation of CO<sub>2</sub>SYS for MATLAB[^1].

## Installation

Install from the [Python Package Index](https://pypi.org/project/PyCO2SYS/):

    pip install PyCO2SYS

Update an existing installation:

    pip install PyCO2SYS --upgrade --no-cache-dir

## Basic use

The import convention for PyCO2SYS will be:

    :::python
    import PyCO2SYS as pyco2

However, the modules and functions contained within are not yet fully documented.  We therefore recommend that you just [do it like in MATLAB](co2sys) for now:

!!! tip "Do it like in MATLAB"
    If you are familiar with CO<sub>2</sub>SYS for MATLAB and wish to use PyCO2SYS in exactly the same way, with extra optional inputs for total ammonia and sulfide:

        :::python
        from PyCO2SYS import CO2SYS
        CO2dict = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT,
            PRESIN, PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS,
            NH3=0.0, H2S=0.0)

    The output `CO2dict` is a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) containing all the calculated variables as [NumPy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html).  Its keys are named following the `HEADERS` output from the original MATLAB program.  See [Calculate everything!](co2sys) for all the details about the inputs and outputs.

## Citation

If you use PyCO2SYS in your work, please cite it as:

!!! note "PyCO2SYS citation"
    Humphreys, M.P., Gregor, L., Pierrot, D., van Heuven, S.M.A.C., Lewis, E., and Wallace, D.W.R. (2020).  PyCO2SYS: marine carbonate system calculations in Python.  *Zenodo.*  [doi:10.5281/zenodo.3744275](http://doi.org/10.5281/zenodo.3744275).

The DOI refers to all versions of PyCO2SYS.  Please be sure to update the version number if necessary.  You can find the current version that you are using in Python with:

    :::python
    import PyCO2SYS as pyco2
    pyco2.say_hello()

As per the instructions in the [the CO2SYS-MATLAB repo](https://github.com/jamesorr/CO2SYS-MATLAB), you should also cite the original work by [Lewis and Wallace (1998)](refs/#l).

Additionally, for the MATLAB programs:

  * If you use `CO2SYS.m`, please cite [van Heuven et al. (2011)](refs/#h).
  * If you use `errors.m` or `derivnum.m`, please cite [Orr et al. (2018)](refs/#o).

## About

PyCO2SYS is maintained primarily by [Dr Matthew Humphreys](https://mvdh.xyz/) of [NIOZ Royal Netherlands Institute for Sea Research](https://www.nioz.nl/en)/[Utrecht University](https://www.uu.nl/en) with support from the main developers of all previous versions of CO<sub>2</sub>SYS.

### History

The original CO<sub>2</sub>SYS program for DOS was written by Ernie Lewis and Doug Wallace.  This was translated into MATLAB by Denis Pierrot and subsequently optimised by Steven van Heuven.  Jim Orr and co-authors added further sets of equilibrium constants and implemented error propagation in a separate program.  The latest MATLAB version was translated into Python as PyCO2SYS by Matthew Humphreys.  Further (ongoing) modifications and additions to PyCO2SYS have been made by Matthew Humphreys and Luke Gregor.

### License

PyCO2SYS is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).

[^1]: For CO<sub>2</sub>SYS for MATLAB refer to [LW98](refs/#l), [HPR11](refs/#h) and [OEDG18](refs/#o).

## Contributing

Suggestions for new features, bug reports and contributions to PyCO2SYS are very welcome!  Please follow the [contribution guidelines](https://github.com/mvdh7/PyCO2SYS/blob/master/CONTRIBUTING.md).
