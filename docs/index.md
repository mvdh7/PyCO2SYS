!!! danger "PyCO2SYS v2 beta"

    These are the docs for the beta release of PyCO2SYS v2!

    If you're here to test it, then thank you! and please report any issues via [the GitHub repo](https://github.com/mvdh7/PyCO2SYS/issues).

    **These instructions will not work for the current version 1.8** that can be installed through `pip` and `conda` - please see [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/) for documentation for the latest stable release.

# PyCO2SYS

PyCO2SYS is a Python toolbox for solving the marine carbonate system and calculating related seawater properties.

## Installation

Create an environment with Python v3.11 or greater and run

    pip install git+https://github.com/mvdh7/PyCO2SYS@v2.0.0b7

This installs the latest "stable" beta release of PyCO2SYS v2 and its core requirements ([JAX](https://jax.readthedocs.io/en/latest/index.html) and [NetworkX](https://networkx.org/)).  To try out using PyCO2SYS with pandas and/or xarray, you'll need to install those separately.

As bugs are fixed and beta features finalised, new beta versions will be released ad hoc and the install link above updated.  New version releases will be announced on [SystemCO2.net](https://systemco2.net).

!!! tip "JAX double precision"

    On import, PyCO2SYS v2 should automatically [set JAX in double precision mode](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).  However, it's possible that this won't always happen, which will adversely affect the results.  If a warning about this appears when running code, or to be on the safe side, set the environment variable `JAX_ENABLE_X64=True` to enforce this behaviour, for example:

        conda env config vars set JAX_ENABLE_X64=True

<!--
### With pip

Install from the [Python Package Index](https://pypi.org/project/PyCO2SYS/):

    pip install PyCO2SYS

### With conda/mamba

!!! warning "Use pip on Windows"

    PyCO2SYS cannot currently be installed through conda on Windows – use pip instead.

Install from the [conda-forge channel](https://anaconda.org/conda-forge/pyco2sys):

    conda install conda-forge::PyCO2SYS
-->

## How to use PyCO2SYS v2

Start with the

  * [Quick-start guide](quick.md)

before moving on to

  * [Arguments and results](detail.md),
  * [Adjust conditions](adjust.md),
  * [Uncertainty propagation](uncertainty.md), and
  * [Validity range checker](validity.md).

If you're really interested, also look at

  * [Advanced tips and tricks](advanced.md).

If you're already familiar with PyCO2SYS v1, then

  * [Switching from v1 to v2](v1_to_v2.md)

may help you to transition.

<!--
### Examples

You can see some working examples of PyCO2SYS in action on Github at [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples).  You can run all of the notebooks there live in your browser [via Binder](https://mybinder.org/v2/gh/mvdh7/PyCO2SYS-examples/master), without installing anything on your computer.
-->

## About

PyCO2SYS is developed and maintained primarily by [Dr Matthew P. Humphreys](https://www.nioz.nl/en/about/organisation/staff/matthew-humphreys) at NIOZ Royal Netherlands Institute for Sea Research (Texel), with support from a range of other contributers and the developers of several other CO2SYS implementations.

### Citation

A paper describing PyCO2SYS is freely available:

!!! note "PyCO2SYS manuscript"
    Humphreys, M. P., Lewis, E. R., Sharp, J. D., and Pierrot, D. (2022).  PyCO2SYS v1.8: marine carbonate system calculations in Python.  *Geoscientific Model Development* 15, 15-43.  [doi:10.5194/gmd-15-15-2022](https://doi.org/10.5194/gmd-15-15-2022).

Although this is about PyCO2SYS v1.8, it is similar enough that it could still be used as a citation for now.  A paper specifically describing PyCO2SYS v2 is in preparation.

To cite the PyCO2SYS software itself:

!!! note "PyCO2SYS code citation"
    Humphreys, M. P., Martin-Mayor, M., Cala, B. A., Schiller, A. J., Sandborn, D. E., Gregor, L., Pierrot, D., van Heuven, S. M. A. C., Lewis, E. R., and Wallace, D. W. R. (2024).  PyCO2SYS: marine carbonate system calculations in Python.  *Zenodo.*  [doi:10.5281/zenodo.3744275](http://doi.org/10.5281/zenodo.3744275).

The DOI above refers to all versions of PyCO2SYS.  Please specify which version of PyCO2SYS you used.  You can find the version number that you are using in Python with:

```python
import PyCO2SYS as pyco2
pyco2.hello()
```

You should also consider citing the original work by [Lewis and Wallace (1998)](refs/#l), plus which [optional sets of constants](detail/#settings) you used in your calculations.

### History

The original CO2SYS program for DOS was created by Ernie Lewis and Doug Wallace ([LW98](refs/#l)).  This was translated into MATLAB by Denis Pierrot and subsequently optimised by Steven van Heuven ([HPR11](refs/#h)).  Jim Orr and co-authors added further sets of equilibrium constants and implemented error propagation in a separate program ([OEDG18](refs/#o)).  The latest MATLAB version was translated into Python as PyCO2SYS by Matthew Humphreys, benefitting enormously from all this previous work ([HLSP22](refs/#h)).  Further (ongoing) modifications and additions to the PyCO2SYS code and documentation have been made by Matthew Humphreys, Luke Gregor, Daniel Sandborn, Abigail Schiller, Ben Cala and Macarena Martin-Mayor.

### License

PyCO2SYS is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contributing

Please open a discussion at [SystemCO2.net](https://systemco2.net) if you have any suggestions for new features, bug reports or are planning a contribution to PyCO2SYS, all of which are very welcome.
