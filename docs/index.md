# PyCO2SYS

[![PyPI version](https://badge.fury.io/py/PyCO2SYS.svg)](https://badge.fury.io/py/PyCO2SYS) [![DOI](https://zenodo.org/badge/237243120.svg)](https://zenodo.org/badge/latestdoi/237243120)

PyCO2SYS is a Python implementation of CO<sub>2</sub>SYS [[LW98](refs/#LW98), [HPR11](refs/#HPR11), [OEDG18](refs/#OEDG18)], the MATLAB toolbox for marine carbonate system calculations.

## Installation and use

Install from the Python Package Index:

    pip install PyCO2SYS

Update an existing installation:

    pip install PyCO2SYS --upgrade --no-cache-dir

Import and use very much like in MATLAB with:

    :::python
    from PyCO2SYS import CO2SYS
    CO2dict = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN, PRESOUT,
        SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS, NH3=0.0, H2S=0.0, KFCONSTANT=1)

For a more Pythonic experience, the import convention is:

    :::python
    import PyCO2SYS as pyco2


See the [Github repo README](https://github.com/mvdh7/pyco2sys#pyco2sys) for more details on the inputs and outputs.

## Citation

!!! note "PyCO2SYS citation"
    Humphreys, M. P., Pierrot, D., van Heuven, S. M. A. C., Lewis, E., & Wallace, D. W. R. (2020). PyCO2SYS v1.3.0: marine carbonate system calculations in Python. *Zenodo.* [doi:10.5281/zenodo.3746347](http://doi.org/10.5281/zenodo.3746347).

The DOI above refers to all versions of PyCO2SYS. Please be sure to update the version number if necessary. You can find the current version that you are using in Python with:

    :::python
    from PyCO2SYS.meta import version

As per the instructions in the [the CO2SYS-MATLAB repo](https://github.com/jamesorr/CO2SYS-MATLAB), you should also cite the original work by [Lewis and Wallace (1998)](refs/#LW98).

Additionally, for the MATLAB programs:

  * If you use `CO2SYS.m`, please cite [van Heuven et al. (2011)](refs/#HPR11).
  * If you use `errors.m` or `derivnum.m`, please cite [Orr et al. (2018)](refs/#OEDG18).

## About

PyCO2SYS is maintained by [Dr Matthew P. Humphreys](https://mvdh.xyz/) at NIOZ Royal Netherlands Institute for Sea Research, Department of Ocean Systems (OCS), and Utrecht University, Texel, the Netherlands.
