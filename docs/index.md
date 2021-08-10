# PyCO2SYS

PyCO2SYS is a Python toolbox for solving the marine carbonate system and calculating related seawater properties.  Its core is a Python implementation of CO2SYS for MATLAB[^1].

## Installation

Install from the [Python Package Index](https://pypi.org/project/PyCO2SYS/):

    pip install PyCO2SYS

Update an existing installation:

    pip install PyCO2SYS --upgrade --no-cache-dir

## Basic use

The import convention for PyCO2SYS is:

```python
import PyCO2SYS as pyco2
```

and the only function you need is:

```python
results = pyco2.sys(**kwargs)
```

Read [How to use PyCO2SYS](co2sys_nd) for more on `pyco2.sys`, its `kwargs` and its `results`.


### Examples

You can see some working examples of PyCO2SYS in action on Github at [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples).  You can run all of the notebooks there live in your browser [via Binder](https://mybinder.org/v2/gh/mvdh7/PyCO2SYS-examples/master), without installing anything on your computer.

Adding your notebooks showcasing PyCO2SYS to [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples) is welcomed!

## About

PyCO2SYS is maintained primarily by [Dr Matthew Humphreys](https://www.nioz.nl/en/about/organisation/staff/matthew-humphreys) of NIOZ Royal Netherlands Institute for Sea Research (Texel) with support from the main developers of all previous versions of CO2SYS.

### Citation

A paper describing PyCO2SYS is in preparation, so please check back here for updates.  In the meantime, if you use PyCO2SYS in your work, please cite it as:

!!! note "PyCO2SYS citation"
    Humphreys, M. P., Schiller, A. J., Sandborn, D. E., Gregor, L., Pierrot, D., van Heuven, S. M. A. C., Lewis, E. R., and Wallace, D. W. R. (2021).  PyCO2SYS: marine carbonate system calculations in Python.  *Zenodo.*  [doi:10.5281/zenodo.3744275](http://doi.org/10.5281/zenodo.3744275).

The DOI refers to all versions of PyCO2SYS.  Please specify which version of PyCO2SYS you used.  You can find the version number that you are using in Python with:

```python
import PyCO2SYS as pyco2
pyco2.hello()
```

You should also cite the original work by [Lewis and Wallace (1998)](refs/#l), and specify which [optional sets of constants](co2sys_nd/#settings) you used in your calculations.

### History

The original CO2SYS program for DOS was created by Ernie Lewis and Doug Wallace ([LW98](refs/#l)).  This was translated into MATLAB by Denis Pierrot and subsequently optimised by Steven van Heuven ([HPR11](refs/#h)).  Jim Orr and co-authors added further sets of equilibrium constants and implemented error propagation in a separate program ([OEDG18](refs/#o)).  The latest MATLAB version was translated into Python as PyCO2SYS by Matthew Humphreys, benefitting enormously from all this previous work.  Further (ongoing) modifications and additions to PyCO2SYS have been made by Matthew Humphreys, Luke Gregor and Daniel Sandborn ([HSG21](refs/#h)).

### License

PyCO2SYS is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contributing

Suggestions for new features, bug reports and contributions to PyCO2SYS are very welcome.  Please follow the [contributing guidelines](https://github.com/mvdh7/PyCO2SYS/blob/master/CONTRIBUTING.md).

To add a notebook to PyCO2SYS-examples, please follow the [contributing guidelines](https://github.com/mvdh7/PyCO2SYS-examples#contributing) of that repo.

[^1]: For CO2SYS for MATLAB refer to [LW98](refs/#l), [HPR11](refs/#h) and [OEDG18](refs/#o).
