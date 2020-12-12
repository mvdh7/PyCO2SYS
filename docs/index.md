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

Only the top-level "calculate everything" functions and uncertainty propagation tools are documented so far.  There are two main interfaces through which to run PyCO2SYS.

The first is to [use `pyco2.sys`](co2sys_nd):

!!! tip "Calculate everything with `pyco2.sys`"

    As a minimum, you only need to provide values and types for two carbonate system parameters, and everything else gets assigned a default value unless you specify something different with the `kwargs`.  Arguments can be scalar or multi-dimensional [NumPy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html).

        :::python
        import PyCO2SYS as pyco2
        results = pyco2.sys(par1, par2, par1_type, par2_type, **kwargs)

    The output `results` are a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) containing all the calculated variables as scalars or [NumPy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html), matching the argument dimensions.  For more information on the optional `kwargs` and names of the output keys, see [Interfaces/New-style `pyco2.sys`](co2sys_nd).

The second way is to [do it like in MATLAB](co2sys):

!!! tip "Do it like in MATLAB"

    If you are familiar with CO2SYS v1 and/or v2 for MATLAB and wish to use PyCO2SYS in exactly the same way, with extra optional inputs for total ammonia and sulfide:

        :::python
        import PyCO2SYS as pyco2
        CO2dict = pyco2.CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT,
            PRESIN, PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS,
            NH3=0.0, H2S=0.0)

    The output `CO2dict` is a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) containing all the calculated variables as one-dimensional [NumPy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html).  Its keys are named following the `HEADERS` output from the original MATLAB program.  See [Interfaces/MATLAB-style CO2SYS](co2sys) for all the details about the inputs and outputs.

    You could also use the adjusted syntax of [CO2SYS v3 for MATLAB](https://github.com/jonathansharp/CO2-System-Extd):

        :::python
        import PyCO2SYS as pyco2
        CO2dict = pyco2.CO2SYS_MATLABv3(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT,
            PRESIN, PRESOUT, SI, PO4, NH3, H2S, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANT,
            KFCONSTANT, BORON)

    Like the MATLAB v3 program, and different from `PyCO2SYS.CO2SYS` and MATLAB v1/v2, this interface uses the updated gas constant value by default.  In both cases, [this can be changed with the kwarg `WhichR`](co2sys/#settings).

Regardless of which interface you use, the underlying calculations are identical and the same inputs will return the same results.

### Examples

#### Suitable for anyone

You can see some working examples of PyCO2SYS in action on Github at [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples).  You can run all of the notebooks there live in your browser [via Binder](https://mybinder.org/v2/gh/mvdh7/PyCO2SYS-examples/master), without installing anything on your computer.

Adding your notebooks showcasing PyCO2SYS to [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples) is welcomed!

#### For Python users

There are also Python scripts containing examples of using PyCO2SYS on Github in [examples](https://github.com/mvdh7/PyCO2SYS/tree/master/examples) and in [validate](https://github.com/mvdh7/PyCO2SYS/tree/master/validate).  The code in the latter directory was used to generate the tables and figures discussed here in the [validation](validate) section (aka: should you trust PyCO2SYS?).

## About

PyCO2SYS is maintained primarily by [Dr Matthew Humphreys](https://mvdh.xyz/) of [NIOZ Royal Netherlands Institute for Sea Research](https://www.nioz.nl/en)/[Utrecht University](https://www.uu.nl/en) with support from the main developers of all previous versions of CO2SYS.

### Citation

If you use PyCO2SYS in your work, please cite it as:

!!! note "PyCO2SYS citation"
    Humphreys, M. P., Gregor, L., Pierrot, D., van Heuven, S. M. A. C., Lewis, E. R., and Wallace, D. W. R. (2020).  PyCO2SYS: marine carbonate system calculations in Python.  *Zenodo.*  [doi:10.5281/zenodo.3744275](http://doi.org/10.5281/zenodo.3744275).

The DOI refers to all versions of PyCO2SYS.  Please specify which version of PyCO2SYS you used.  You can find the version number that you are using in Python with:

    :::python
    import PyCO2SYS as pyco2
    pyco2.hello()

As per the instructions in the [the CO2SYS-MATLAB repo](https://github.com/jamesorr/CO2SYS-MATLAB), you should also cite the original work by [Lewis and Wallace (1998)](refs/#l).

Additionally, for the MATLAB programs:

  * If you use `CO2SYS.m`, please cite [van Heuven et al. (2011)](refs/#h).
  * If you use `errors.m` or `derivnum.m`, please cite [Orr et al. (2018)](refs/#o).

### History

The original CO2SYS program for DOS was created by Ernie Lewis and Doug Wallace ([LW98](refs/#l)).  This was translated into MATLAB by Denis Pierrot and subsequently optimised by Steven van Heuven ([HPR11](refs/#h)).  Jim Orr and co-authors added further sets of equilibrium constants and implemented error propagation in a separate program ([OEDG18](refs/#o)).  The latest MATLAB version was translated into Python as PyCO2SYS by Matthew Humphreys, benefitting enormously from all this previous work.  Further (ongoing) modifications and additions to PyCO2SYS have been made by Matthew Humphreys and Luke Gregor ([HGP20](refs/#h)).

### License

PyCO2SYS is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contributing

Suggestions for new features, bug reports and contributions to PyCO2SYS are very welcome.  Please follow the [contributing guidelines](https://github.com/mvdh7/PyCO2SYS/blob/master/CONTRIBUTING.md).

To add a notebook to PyCO2SYS-examples, please follow the [contributing guidelines](https://github.com/mvdh7/PyCO2SYS-examples#contributing) of that repo.

[^1]: For CO2SYS for MATLAB refer to [LW98](refs/#l), [HPR11](refs/#h) and [OEDG18](refs/#o).
