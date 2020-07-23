# Version history

!!! info "Semantics"
    Version numbering aims to follow [semantic versioning](https://semver.org/). Therefore:

      * New *patch* versions (e.g. 1.1.**0** to 1.1.**1**) make minor changes that do not alter fuctionality or calculated results.
      * New *minor* versions (e.g. 1.**0**.1 to 1.**1**.0) add new functionality, but will not break your code.  They will not alter the results of calculations with default settings (except for in the hopefully rare case of correcting a bug or typo).
      * New *major* versions (e.g. **1**.1.1 to **2**.0.0) may break your code and require you to rewrite things.  They may alter the results of calculations with default settings.

!!! warning
    *Will (not) break your code* refers **only** to the functions covered in this documentation.

    For the main CO2SYS function as imported with

        :::python
        from PyCO2SYS import CO2SYS
        CO2dict = CO2SYS(*args, **kwargs)

    the only things that may change, in at least a *minor* version release, are:

      1. Additional inputs may be added to the `kwargs`, but always with default values such that the results do not change if they are not explicitly set.
      2. Additional calculated variables may be returned in the output `CO2dict`.

    The structure of the underlying modules and their functions is not yet totally stable and, for now, may change in any version increment.  Such changes will be described in the release notes below.

## 1.4

Enables uncertainty propagation with forward finite-difference derivatives.

### 1.4.4 (forthcoming)

!!! example "Changes in v1.4.4"

    ***Internal reorganisation***

    * All 2-to-3 functions in `PyCO2SYS.solve.get` now have a more consistent set of inputs.
    * Revised pH scale conversion functions for consistency and added tests for their internal accuracy.

### 1.4.3 (16 July 2020)

!!! example "Changes in v1.4.3"

    ***Bug fixes***

    * Corrected missing a pH scale conversion in [SLH20](../refs/#s) option for carbonic acid dissociation.  **Any calculations with this option in PyCO2SYS v1.4.1 or v1.4.2 should be updated!**

    ***Validation***

    * Results validated against new release candidate version of CO2SYS-MATLAB v3.

    ***New API***

    * New wrapper function with the same input order and default gas constant option as the new CO2SYS-MATLAB v3 available in `PyCO2SYS.api.CO2SYS_MATLABv3`.

    ***Internal reorganisation***

    * `_approx` function inputs in `PyCO2SYS.solve.delta` updated to match the exact Autograd functions for easier switching.


### 1.4.2 (9 July 2020)

!!! example "Changes in v1.4.2"

    ***Bug fixes***

    * Swapped order of `totals` and `Ks` arguments for all functions in `PyCO2SYS.solve.delta` for consistency with other modules.
    * Inverted the alkalinity-pH residual equations in `PyCO2SYS.solve.delta`.

    ***Reorganisation***

    * Broke out some parts of `PyCO2SYS.equilibria.assemble` into separate functions.

### 1.4.1 (1 July 2020)

!!! example "Changes in v1.4.1"

    ***Extra calculation options***

    * Added the [2018 CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?r) value for the universal gas constant *R* as an option for consistency with forthcoming CO2SYS-MATLAB v3.  The original DOEv2 version remains default.
    * Added the [SLH20](../refs/#s) equations as option `16` for the carbonic acid dissociation constants.

### 1.4.0 (9 June 2020)

!!! example "Changes in v1.4.0"

    ***New features***

    * Added `uncertainty` module with functions to evaluate derivatives of PyCO2SYS outputs with respect to inputs, along with corresponding [documentation](../uncertainty).
    * Specific input values can optionally be provided for all total concentrations and equilibrium constants that are estimated internally from salinity, temperature and pressure.

    ***General improvements***

    * Added basic sanity checking to prevent some invalid marine carbonate system parameter input values.
    * Nutrient concentrations have always been set to zero internally for `K1K2CONSTANTS` options `6` and `8`, and salinity too for `8`, regardless of the input values.  This is now reflected in the output values of these variables in the `CO2dict`.

    ***New outputs***

    * Substrate:inhibitor ratio (SIR) of [B15](../refs/#b), calculated with `SIratio` in new module `bio`.
    * Inputs `PAR1` and `PAR2`.
    * The "Peng correction" factor.
    * The fugacity factor for converting between CO<sub>2</sub> partial pressure and fugacity.
    * The activity coefficient of the H<sup>+</sup> ion for NBS pH scale conversions.

    ***Validation***

    * Calculations compare very favourably against the forthcoming [CO2SYS for MATLAB v3](https://github.com/jonathansharp/CO2-System-Extd) - see [Validation](../validate/#co2sys-for-matlab) for discussion of the results.

## 1.3

Adds bicarbonate ion and aqueous CO<sub>2</sub> as inputs from which the carbonate system can be solved.  Continues to reorganise code behind the scenes.  Makes almost everything [Autograd](https://github.com/HIPS/autograd)-able and uses this approach to calculate buffer constants.  Validates results against CO2SYS for MATLAB.

### 1.3.0 (1 May 2020)

!!! example "Changes in v1.3.0"

    ***New features***

    * Added bicarbonate ion (type `7`) and aqueous CO<sub>2</sub> (type `8`) as options for known input marine carbonate system variables.
    * Added module `test` with functions to perform internal consistency checks on `PyCO2SYS.CO2SYS` calculations and compare results with those from other sources.
    * Added module `api` with a wrapper for `PyCO2SYS.CO2SYS` to allow inputs as Pandas Series and/or Xarray DataArrays.

    ***Improved calculations***

    * The Revelle factor and all other buffer factors added in v1.2 are now evaluated using automatic differentiation, which means that the effects of all equilibrating species are taken into account.
        * The original, non-automatic functions that do not account for nutrient effects are still available in `buffers.explicit`.
        * Can switch between calculation methods using new optional input `buffers_mode`.
    * Corrected Revelle factor calculations:
        * Added missing "Peng correction" to Revelle factor calculation at output conditions.  *Note that this correction is currently also missing from CO2SYS for MATLAB!*
        * Decreased DIC perturbation size for more accurate finite-difference "explicit" evaluation.
        * Finite-difference calculation now references the correct DIC value.
    * Implemented better initial guesses for pH in all iterative solvers in `solve.get` following [M13](../refs/#m) and [OE15](../refs/#o).
    * Switched to using exact slopes in iterative solvers in `solve.get`, evaluated using Autograd in new submodule `solve.delta`.
    * Updated entire package to be [Autograd](https://github.com/HIPS/autograd)-able.
    * Return NaN instead of negative DIC if an impossible pH-alkalinity combination is given as input (i.e. pH is too high).
    * Return NaN where DIC and one of its components is given if the component is impossibly large.

    ***Internal reorganisation***

    * Major internal reorganisation that is probably not fully captured in these notes.
    * Renamed modules:
        * `assemble` is now `engine`.
        * `concentrations` is now `salts`.
        * `extra` is now `buffers.explicit`.
    * Module `equilibria` now contains sub-modules:
        * `p1atm` for calculating constants at atmospheric pressure.
        * `pcx` for determining pressure correction factors.
        * `pressured` for calculating constants at given pressure.
    * Module `solve` now contains sub-modules:
        * `initialise` to generate first-guess estimates of pH for the TA-pH solvers.
        * `get` to calculate a new system variable from various input pairs.
    * Added module `solubility` for mineral solubility calculations.
    * Relocated `_CaSolubility` function from root to `solubility.CaCO3`.
        * Separated out its internal calculations into a set of subfunctions also in the `solubility` module.
        * Added calcium molinity `TCa` as an input, instead of being evaluated internally.
    * Added calcium molininty `TCa` (estimated from salinity) into the main `CO2dict` output from `PyCO2SYS.CO2SYS`.
    * Relocated `_RevelleFactor` function from root to `buffers.RevelleFactor`.
    * Relocated `_FindpHOnAllScales` function from root to `convert.pH2allscales`.
    * Added module `constants` for storing values of universal physical constants.
    * Lists of equilibrium constants and total concentrations now passed around internally as dicts, for safety.
    * Total sulfate and bisulfate dissociation constant renamed from `TS` and `KS` to `TSO4` and `KSO4` internally to avoid confusion with sulfide species.
    * The as-close-as-possible MATLAB clone in `PyCO2SYS.original` no longer produces a dict but just the original `DATA`, `HEADERS` and `NICEHEADERS` outputs.

    ***Miscellaneous***

    * Documentation substantially expanded and switched to using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
        * Validation section added with internal consistency checks and an external comparison of PyCO2SYS calculations with CO2SYS for MATLAB.
    * All code now formatted with [Black](https://github.com/psf/black) (except for module `original`).
    * Version number now reported with `PyCO2SYS.say_hello()` in homage to the original MS-DOS program.

## 1.2

Adds additional buffer factor calculations that are not currently included in CO2SYS for MATLAB.  New releases are henceforth assigned DOIs from Zenodo.

### 1.2.1 (9 Apr 2020)

!!! example "Changes in v1.2.1"
    * Fixed typo in [ESM10](../refs/#ESM10) equations that had been carried through into `extra.buffers_ESM10` function (thanks [Jim Orr](https://twitter.com/James1Orr/status/1248216403355803648)!).

### 1.2.0 (8 Apr 2020)

!!! example "Changes in v1.2.0"
    * Added module `extra` containing functions to calculate variables not included in CO2SYS for MATLAB:
      * `buffers_ESM10` calculates the buffer factors of [ESM10](../refs/#ESM10), corrected for the typos noted by [RAH18](../refs/#RAH18).
      * `bgc_isocap` calculates the "exact" isocapnic quotient of [HDW18](../refs/#HDW18), Eq. 8.
      * `bgc_isocap_approx` calculates the approximate isocapnic quotient of [HDW18](../refs/#HDW18), Eq. 7.
      * `psi` calculates the $\psi$ factor of [FCG94](../refs/#FCG94).
    * Added all functions in `extra` to the `CO2dict` output of the main `CO2SYS` function, and documented in the [Github repo README](https://github.com/mvdh7/PyCO2SYS#pyco2sys).

## 1.1

Adds extra optional inputs for consistency with Pierrot et al.'s forthcoming MATLAB "v1.21".  Continues to reorganise subfunctions into more Pythonic modules, while avoiding changing the actual mechanics of calculations.

### 1.1.1 (20 Mar 2020)

!!! example "Changes in v1.1.1"
    * Removed unnecessary `WhoseTB` input to `assemble.equilibria`.

### 1.1.0 (19 Mar 2020)

!!! example "Changes in v1.1.0"
    * Updated pH-solving iterative functions so that iteration stops separately for each row once it reaches the tolerance threshold.
    * Extracted all functions for solving the CO<sub>2</sub> system into a separate module (`solve`).
    * Extracted other key subfunctions into module `assemble`.
    * Added total ammonium (`NH3`) and hydrogen sulfide (`H2S`) concentrations as optional inputs to be included in the alkalinity model.
    * Added optional input to choose between different equations for hydrogen fluoride dissociation constant (`KFCONSTANT`).
    * Added functions to enable carbonate ion as an input carbonate system variable.
    * Output is now only the `CO2dict` dict, not the original `DATA`, `HEADERS` and `NICEHEADERS`.
    * Eliminated all global variables throughout the entire program.

## 1.0

### 1.0.1 (28 Feb 2020)

Starts to make things more Pythonic.

!!! example "Changes in v1.0.1"
      * Extracted all equations for concentrations and equilibrium constants into functions in separate modules (`concentrations` and `equilibria`).
      * Eliminated all global variables from the `_Constants` function.
      * Moved the as-close-as-possible version into module `original`. The default `from PyCO2SYS import CO2SYS` now imports the more Pythonic implementation.

### 1.0.0 (3 Feb 2020)

An as-close-as-possible clone of [MATLAB CO2SYS v2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB).

!!! example "Release notes for v1.0.0"
      * The first output `DICT` is new: a dict containing a separate entry for each variable in the original output `DATA`, with the keys named following the original output `HEADERS`.
      * The output `DATA` is transposed relative to the MATLAB version because Numpy is row-major while MATLAB is column-major.
      * Every combination of input options was tested against the MATLAB version with no significant differences (i.e. all differences can be attributed to floating point errors).
