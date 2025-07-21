# PyCO2SYS

!!! info "PyCO2SYS v2"

    These are the docs for the forthcoming PyCO2SYS v2!

    **These instructions will not work for the current version 1.8** that can be installed through pip and conda - please see [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/) for documentation for the latest release.

    **If you are here to test PyCO2SYS v2**, then create a test environment with Python v3.10 or greater, and then in that environment run

        pip install git+https://github.com/mvdh7/PyCO2SYS@v2.0.0-b3

    This installs PyCO2SYS and its core requirements ([JAX](https://jax.readthedocs.io/en/latest/index.html) and [NetworkX](https://networkx.org/)).  If you wish to try out using PyCO2SYS with pandas and/or xarray, you'll need to install those into the environment separately.

!!! tip "JAX double precision"

    On import, PyCO2SYS should automatically [set JAX in double precision mode](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).  However, it's possible that this won't always happen, which will adversely affect the results.  If a warning about this appears when running code, or to be on the safe side, set the environment variable `JAX_ENABLE_X64=True` to enforce this behaviour, for example:

        conda env config vars set JAX_ENABLE_X64=True

PyCO2SYS is a Python toolbox for solving the marine carbonate system and calculating related seawater properties.  It was originally based on CO2SYS for MATLAB[^1].

## Installation

!!! warning "Don't follow the instructions below"

    The installation instructions below are not yet active - see the boxes above if you're here to test PyCO2SYS v2!

### With pip

Install from the [Python Package Index](https://pypi.org/project/PyCO2SYS/):

    pip install PyCO2SYS

### With conda/mamba

!!! warning "Use pip on Windows"

    PyCO2SYS cannot currently be installed through conda on Windows – use pip instead.

Install from the [conda-forge channel](https://anaconda.org/conda-forge/pyco2sys):

    conda install conda-forge::PyCO2SYS

## How to use PyCO2SYS

Start with the

  * [Quick-start guide](quick.md)

before moving on to

  * [Arguments and results](detail.md),
  * [Adjust conditions](adjust.md) and
  * [Uncertainty propagation](uncertainty.md).

If you're really interested, also look at

  * [Advanced tips and tricks](advanced.md).

If you're already familiar with PyCO2SYS v1, then

  * [Switching from v1 to v2](v1_to_v2.md)

may help you to transition.

### Examples

You can see some working examples of PyCO2SYS in action on Github at [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples).  You can run all of the notebooks there live in your browser [via Binder](https://mybinder.org/v2/gh/mvdh7/PyCO2SYS-examples/master), without installing anything on your computer.

Adding your notebooks showcasing PyCO2SYS to [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples) is welcomed!

## About

PyCO2SYS is maintained primarily by [Dr Matthew P. Humphreys](https://www.nioz.nl/en/about/organisation/staff/matthew-humphreys) of NIOZ Royal Netherlands Institute for Sea Research (Texel) with support from the main developers of all previous versions of CO2SYS.

### Citation

A paper describing PyCO2SYS is freely available:

!!! note "PyCO2SYS manuscript"
    Humphreys, M. P., Lewis, E. R., Sharp, J. D., and Pierrot, D. (2022).  PyCO2SYS v1.8: marine carbonate system calculations in Python.  *Geoscientific Model Development* 15, 15-43.  [doi:10.5194/gmd-15-15-2022](https://doi.org/10.5194/gmd-15-15-2022).

To cite the PyCO2SYS software itself:

!!! note "PyCO2SYS code citation"
    Humphreys, M. P., Cala, B. A., Schiller, A. J., Sandborn, D. E., Gregor, L., Pierrot, D., van Heuven, S. M. A. C., Lewis, E. R., and Wallace, D. W. R. (2024).  PyCO2SYS: marine carbonate system calculations in Python.  *Zenodo.*  [doi:10.5281/zenodo.3744275](http://doi.org/10.5281/zenodo.3744275).

The DOI above refers to all versions of PyCO2SYS.  Please specify which version of PyCO2SYS you used.  You can find the version number that you are using in Python with:

```python
import PyCO2SYS as pyco2
pyco2.hello()
```

You should also consider citing the original work by [Lewis and Wallace (1998)](refs.md/#l), and specify which [optional sets of constants](detail.md/#settings) you used in your calculations.

### History

The original CO2SYS program for DOS was created by Ernie Lewis and Doug Wallace ([LW98](refs.md/#l)).  This was translated into MATLAB by Denis Pierrot and subsequently optimised by Steven van Heuven ([HPR11](refs.md/#h)).  Jim Orr and co-authors added further sets of equilibrium constants and implemented error propagation in a separate program ([OEDG18](refs.md/#o)).  The latest MATLAB version was translated into Python as PyCO2SYS by Matthew Humphreys, benefitting enormously from all this previous work.  Further (ongoing) modifications and additions to the PyCO2SYS code and documentation have been made by Matthew Humphreys, Luke Gregor, Daniel Sandborn and Abigail Schiller ([HSS21](refs.md/#h)).

### License

PyCO2SYS is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contributing

Suggestions for new features, bug reports and contributions to PyCO2SYS are very welcome.  Please follow the [contributing guidelines](https://github.com/mvdh7/PyCO2SYS/blob/master/CONTRIBUTING.md).

To add a notebook to PyCO2SYS-examples, please follow the [contributing guidelines](https://github.com/mvdh7/PyCO2SYS-examples#contributing) for that repo.

[^1]: For CO2SYS for MATLAB refer to [LW98](refs.md/#l), [HPR11](refs.md/#h) and [OEDG18](refs.md/#o).
