# Contributing to PyCO2SYS

Contributions to PyCO2SYS from anyone are very welcome, but please read this first!

- [Contributing to PyCO2SYS](#contributing-to-pyco2sys)
  - [How does PyCO2SYS work?](#how-does-pyco2sys-work)
    - [Version 1](#version-1)
    - [Version 2](#version-2)
  - [Ideas and bug reporting](#ideas-and-bug-reporting)
  - [Adding or editing code](#adding-or-editing-code)
    - [Branches](#branches)
    - [Code style](#code-style)
    - [Credit](#credit)
  - [Documentation](#documentation)

## How does PyCO2SYS work?

### Version 1

Version 1 followed a similar approach to CO2SYS-MATLAB, which allowed all parameters and options to be provided as scalars or arrays, and where every possible parameter of interest was always computed and provided in the results dict.  There were a few important differences from CO2SYS-MATLAB, especially using automatic differentiation (with [Autograd](https://github.com/HIPS/autograd)) for the pH solvers and calculating buffer factors (but not for uncertainty propagation).  Its workings were described and results validated by [Humphreys et al. (2022)](https://doi.org/10.5194/gmd-15-15-2022).

### Version 2

Version 2 completely overhauls the internal mechanism for calculations.  Some flexibility has been lost, primarily, all of the optional settings (any argument beginning with `opt_`) all need to be scalars instead of arrays.  However, this has been done to allow results to be computed only when requested, instead of every possible parameter at once, resulting in considerable speed-ups in computations.

Calculations are based around the `CO2System` class.  A `CO2System` is constructed by providing a set of `values`, containing potentially multidimensional parameters such as the known marine carbonate system parameters, temperature, salinity, pressure and nutrients, and a set of `opts`, which determine (e.g.) which parameterisations are used for the various equilibrium constants.  

## Ideas and bug reporting

If you would like a new feature to be added to PyCO2SYS, or if you find a bug or error in any of its calculations, then please first [share this as an issue](https://github.com/mvdh7/PyCO2SYS/issues). Please do this regardless of whether you are able to solve the issue yourself, to help to avoid duplicate work.

## Adding or editing code

If you would like to add or edit something directly then please make a fork of PyCO2SYS, make your changes, and submit the updates back with a pull request, noting the comments below.  If you are frequently making lots of contributions, you could also be given direct access to the main repo.

Please add a comment on the corresponding [issue](https://github.com/mvdh7/PyCO2SYS/issues) to say that you are working on that problem.

### Branches

The *main* branch contains the most recent release, and nothing more.  Please do not submit pull requests directly to *main*.

The *develop* branch is where the next version is being prepared.  When you have something ready to add, please submit your pull request to *develop*.  You may also wish to make your new fork from *develop* to be sure you are using the latest version.

### Code style

Every module and function should have at least a simple docstring.  I am in the process of updating the docstrings to follow the [numpydoc](https://numpydoc.readthedocs.io/en/stable/format.html) format, so please follow this for any new additions.

Functions that are "private" and not intended to be used by the typical end user should begin with an underscore.  These should still have a docstring.

For readable consistency with minimal effort, everything in PyCO2SYS will be reformatted by [Ruff](https://github.com/astral-sh/ruff) before each new release.

### Credit

Anyone making a substantial contribution will be invited to join the list of authors for the [Zenodo citation](https://doi.org/10.5281/zenodo.3744275).

## Documentation

Documentation is available at [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/).  This site is automatically generated after each commit from the files in the [docs](https://github.com/mvdh7/PyCO2SYS/tree/main/docs) directory on `main` using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).  The docs on *develop* are also automatically generated after each commit to that branch and can be seen at [pyco2sys.hseao3.group](https://pyco2sys.hseao3.group).

There is a repo of PyCO2SYS examples as [Jupyter Notebooks](https://jupyter.org/), which you can add to at [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples).

If you add new features to PyCO2SYS, please also propose some sort of documentation for them in one of these formats.

Any changes that you make should be added to appropriate set of release notes in the [version history](https://github.com/mvdh7/PyCO2SYS/blob/develop/docs/versions.md) and any related citations added to the [references](https://github.com/mvdh7/PyCO2SYS/blob/develop/docs/refs.md).
