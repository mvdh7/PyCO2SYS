# Version history

!!! info "Semantics"
    Version numbering aims to follow [semantic versioning](https://semver.org/). Therefore:

      * New *patch* versions (e.g. 1.1.**0** to 1.1.**1**) make minor changes that do not alter fuctionality or calculated results.
      * New *minor* versions (e.g. 1.**0**.1 to 1.**1**.0) add new functionality, but will not break your code.  They will not alter the results of calculations with default settings (except for in the hopefully rare case of correcting a bug or typo).
      * New *major* versions (e.g. **1**.1.1 to **2**.0.0) may break your code and require you to rewrite things.  They may significantly alter the results of calculations with default settings.

    We will always add aliases for existing functions if their API is updated, to avoid unforseen breaking changes wherever possible.

!!! warning
    *Will (not) break your code* refers **only** to the functions covered in this documentation.

    For the main CO2SYS function as imported with

    ```python
    import PyCO2SYS as pyco2
    
    co2s = pyco2.sys(**kwargs)
    ```

    the only things that may change, in at least a *minor* version release, are:

      1. Additional inputs may be added to the `kwargs`, but always with default values such that the results do not change if they are not explicitly set.
      2. Additional calculated variables may be returned in the output `results`.

    The structure of the underlying modules and their functions is not yet totally stable and, for now, may change in any version increment.  Such changes will be described in the release notes below.

## 2.0 (forthcoming)

Switches from Autograd to JAX for automatic differentiation.  Internal mechanism rebuilt for significantly more memory-efficient and faster calculations.

!!! new-version "Changes in v2.0"

    * Validity range checker implemented.
    * Nitrous acid equilibrium at 1 atm pressure included.
    * Calculations performed only when needed for specifically requested parameters.
    * Only one combination of known marine carbonate system parameters allowed per calculation.
    * Only one combination of optional settings allowed per calculation.
    * Optional settings each only affect one parameterisation, so there are more of them.  (In v1, `opt_k_carbonic` could alter several other parameterisations beyond just the carbonic acid equilibrium.)
    * "Input" and "output" conditions deprecated in favour of the `adjust` method.
    * Uncertainty propagation uses automatic differentiation instead of finite differences.
    * Simple tests suggest calculations including iterative pH solving are on the order of 100 times faster and have about 100 times lower peak memory demand.
    * Differences in calculated values from v1.8 should all be at the level of computer precision (i.e., negligible).

## 1.8

Adds atmospheric pressure input for *p*CO<sub>2</sub>-*f*CO<sub>2</sub>-*x*CO<sub>2</sub> interconversions and (from v1.8.2) optional hydrostatic pressure effect on CO<sub>2</sub> solubility and fugacity.  Uncertainty analysis updated for more reproducible results.  Rigorously validated and fully documented in peer-reviewed journal article ([Humphreys et al., 2022](https://doi.org/10.5194/gmd-15-15-2022)).

### 1.8.3 (16 February 2024)

!!! new-version "Changes in v1.8.3"

    ***New features***

    * Added `"dlnfCO2_dT"` and `"dlnpCO2_dT"` results, the theoretical effect of temperature on the natural log of <i>ƒ</i>CO<sub>2</sub> and <i>p</i>CO<sub>2</sub>.
    * Added the [PLR18](refs.md/#p) parameterisation of the carbonic acid constants for sea-ice brines.

    ***Default options***

    * Reverted default `opt_k_carbonic` to `10` (i.e., [LDK00](refs.md/#l)) for consistency with the best practice guide.

    ***Bug fixes***
    
    * Updated `pyco2.equilibria.p1atm.kH2CO3_NBS_MCHP73` (used for `opt_k_carbonic` options `6` and `7`) to update any salinity values less than 10<sup>–16</sup> to be 10<sup>–16</sup>, because zero salinities give a NaN for <i>K</i><sub>2</sub>, which causes problems for Autograd.  This should not make any practical difference, because the parameterisation is only valid for salinities above 19.
    * Added `opt_pressured_kCO2` to results dict and incorporated it correctly into the uncertainty propagation functions.

    ***Technical***

    * Updated from building with setup.py to pyproject.toml.
    * PyCO2SYS can now be installed with conda/mamba (via conda-forge).

### 1.8.2 (19 January 2023)

!!! new-version "Changes in v1.8.2"

    ***New features***

    * Added `opt_pressured_kCO2` to enable pressure corrections for the fugacity factor and CO<sub>2</sub> solubility constant following [W74](refs.md/#w).  These have been added to CO2SYS-MATLAB by Jon Sharp at the same time with consistent results (differences less than 10<sup>−4</sup> %).  These pressure corrections are not enabled by default, for consistency with previous versions.

    ***Bug fixes***

    * Fixed logicals in `solve.core()` that had meant no results were returned for parameter type combinations `7, 10`, `7, 11`, `8, 10` and `8, 11`.
    * Updated Autograd version for compatibility with Python 3.11.

### 1.8.1 (18 May 2022)

!!! new-version "Changes in v1.8.1"

    ***Breaking changes***

    * For consistency with other settings, `buffers_mode` kwarg key changed to `opt_buffers_mode` and its values are now integers rather than strings.

    ***New features***

    * Adds new `par1_type` / `par2_type` options `10` and `11` for saturation states with respect to calcite and aragonite.
    * Adds [KSK18](refs.md/#k) parameterisation for estimating total borate from salinity.

    ***Dependencies***

    * Switched to Autograd v1.4.

### 1.8.0 (27 October 2021)

!!! new-version "Changes in v1.8.0"

    ***New features***

    * Adds `pressure_atmosphere` and `pressure_atmosphere_out` arguments, rather than assuming 1 atm total barometric pressure.

    ***Behind-the-scenes improvements***

    * Adds additional constraint to the initial pH estimate for more robust results for the alkalinity-CO<sub>2</sub> fugacity parameter pair.
    * Difference derivatives for uncertainties now have a fixed step size for each argument, instead of scaling depending on arguments, for more reproducible results.

## 1.7

Adds new syntax to return equilibrium constants and total salts without needing to solve the full carbonate system.  Fully documented in manuscript in review ([Humphreys et al., 2021, *Geosci. Model Dev. Discuss.*](https://doi.org/10.5194/gmd-2021-159)).

### 1.7.1 (10 August 2021)

!!! new-version "Changes in v1.7.1"

    ***Bug fixes***

    * Improved handling of zero-valued inputs.
    * Adjusted `CO2SYS_wrap` to work with latest pandas release.

### 1.7.0 (13 May 2021)

!!! new-version "Changes in v1.7.0"

    ***New features***

    * Can now run `pyco2.sys` with no carbonate system parameter arguments provided, to just return all the equilibrium constants etc. under the specified conditions.
    * Can also run `pyco2.sys` with only one carbonate system parameter argument.  This does not solve the carbonate system, but does calculate all that can be calculated with that parameter.
    * Added carbonic acid constants parameterisation of [SB21](refs.md/#s).
    * Added bisulfate dissociation constant parameterisation of [WM13](refs.md/#w)/[WMW14](refs.md/#w).
    * Added spreadsheet-to-spreadsheet function `pyco2.ezio` (with thanks to [Daniel Sandborn](https://github.com/d-sandborn)).
    * Integrated uncertainty propagation into the main `pyco2.sys` function and expanded its capabilities.

    ***Internal updates***

    * Switched default first-guess pH for solving from the alkalinity-carbonate ion parameter pair at low alkalinity from 10 to 3.
    * Renamed various internal functions and variables for better consistency with the Pythonic `pyco2.sys` i/o syntax.
    * Removed the `PyCO2SYS.test` module, instead defining the round-robin test functions it contained directly in the test suite.
    * Added various internal settings for testing and validation against older CO2SYS-MATLAB versions.
    * Adjust aqueous CO<sub>2</sub> calculation for better consistency with CO2SYS-MATLAB (but negligible changes in the results).
    * Can now use `PyCO2SYS.hello()` to find version number and credits (alias for `PyCO2SYS.say_hello()`).
    * The final component of DIC (or DIC itself) to be calculated is now always computed by difference from the known components.
    * Various functions in `convert` module renamed.

    ***Validation***

    * Rigorous validation against various CO2SYS-MATLAB versions performed, as described in forthcoming PyCO2SYS manuscript (Humphreys et al., in prep.).

    ***Bug fixes***

    * `par1`, `par2`, `par1_type` and `par2_type` arguments now always get broadcasted to the maximum size, even if they are scalar.
    * Erroneous `"k_phosphate_*"` keys corrected to `"k_phosphoric_"`.
    * Override values for equilibrium constants under output conditions now assigned correctly.
    * Fixed minor errors in initial pH estimates when solving from alkalinity and either DIC or [CO$_2$(aq)].

## 1.6

Adds extra alkalinity components with arbitrary p*K* values.

### 1.6.0 (26 October 2020)

!!! new-version "Changes in v1.6.0"

    ***Bug fixes***

    * Updates the total alkalinity equation to fix minor error in pH scale conversions inherited from CO2SYS-MATLAB (see related note in [v1.5.0 release notes](#150-29-july-2020)).

    ***New inputs and outputs***

    * Enables inputting total molalities and equilibrium constants for up to two additional contributors to total alkalinity.
    * Full chemical speciation returned in the output dict of `pyco2.sys`, not just the alkalinity components as before.

    ***New syntax***

    * Adds `sys` as an alias for `CO2SYS_nd` at the top level.  Recommended Python-style syntax is thus now `pyco2.sys`.

## 1.5

Introduces a more Pythonic top-level function that accepts multidimensional arguments and that only returns results at "output" conditions if explicitly specified.

### 1.5.1 (30 July 2020)

!!! new-version "Changes in v1.5.1"

    ***Bug fixes***

    * Switched `dx`-scaling function in `PyCO2SYS.uncertainties` to use `numpy.nanmedian` instead of `numpy.median`.
    * Fixed `PyCO2SYS.uncertainties.propagate_nd` bug that prevented calculations on non-scalar arguments.

### 1.5.0 (29 July 2020)

!!! new-version "Changes in v1.5.0"

    ***New top-level functions***

    * Adds `PyCO2SYS.CO2SYS_nd` top-level function with a more Pythonic interface and with NumPy broadcasting of $n$-dimensional inputs.
    * In `PyCO2SYS.CO2SYS_nd`, results at "output" conditions are only calculated if output temperature or pressure is provided.
    * Adds corresponding `PyCO2SYS.uncertainty.forward_nd` and `PyCO2SYS.uncertainty.propagate_nd` functions for uncertainty propagation.

    ***Alternative calculations***

    * New alkalinity equation fixing pH scale conversion bug inherited from CO2SYS-MATLAB is available, but not yet implemented by default.

    ***Extra arguments and results***

    * Solubility constants for aragonite and calcite available directly as outputs from `PyCO2SYS.CO2SYS` and `PyCO2SYS.CO2SYS_nd`
    * Explicit values for the solubility constants can be given as arguments to override the default internal calculation.

    ***Internal reorganisation***

    * All 2-to-3 functions in `PyCO2SYS.solve.get` now have a more consistent set of inputs.
    * Revised pH scale conversion functions for consistency and added tests for their internal accuracy.
    * Switched preallocations to use `np.shape` not `np.size` in preparation for working with $n$-dimensional inputs.
    * Updated style to import the whole NumPy module as `np` instead of individual functions separately.
    * Converted `PyCO2SYS.engine` to a sub-module.

## 1.4

Enables uncertainty propagation with forward finite-difference derivatives.

### 1.4.3 (16 July 2020)

!!! new-version "Changes in v1.4.3"

    ***Bug fixes***

    * Corrected missing a pH scale conversion in [SLH20](refs.md/#s) option for carbonic acid dissociation.  **Any calculations with this option in PyCO2SYS v1.4.1 or v1.4.2 should be updated!**

    ***Validation***

    * Results validated against new release candidate version of CO2SYS-MATLAB v3.

    ***New API***

    * New wrapper function with the same input order and default gas constant option as the new CO2SYS-MATLAB v3 available in `PyCO2SYS.api.CO2SYS_MATLABv3`.

    ***Internal reorganisation***

    * `_approx` function inputs in `PyCO2SYS.solve.delta` updated to match the exact Autograd functions for easier switching.


### 1.4.2 (9 July 2020)

!!! new-version "Changes in v1.4.2"

    ***Bug fixes***

    * Swapped order of `totals` and `Ks` arguments for all functions in `PyCO2SYS.solve.delta` for consistency with other modules.
    * Inverted the alkalinity-pH residual equations in `PyCO2SYS.solve.delta`.

    ***Reorganisation***

    * Broke out some parts of `PyCO2SYS.equilibria.assemble` into separate functions.

### 1.4.1 (1 July 2020)

!!! new-version "Changes in v1.4.1"

    ***Extra calculation options***

    * Added the [2018 CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?r) value for the universal gas constant *R* as an option for consistency with forthcoming CO2SYS-MATLAB v3.  The original DOEv2 version remains default.
    * Added the [SLH20](refs.md/#s) equations as option `16` for the carbonic acid dissociation constants.

### 1.4.0 (9 June 2020)

!!! new-version "Changes in v1.4.0"

    ***New features***

    * Added `uncertainty` module with functions to evaluate derivatives of PyCO2SYS outputs with respect to inputs, along with corresponding [documentation](uncertainty.md).
    * Specific input values can optionally be provided for all total concentrations and equilibrium constants that are estimated internally from salinity, temperature and pressure.

    ***General improvements***

    * Added basic sanity checking to prevent some invalid marine carbonate system parameter input values.
    * Nutrient concentrations have always been set to zero internally for `K1K2CONSTANTS` options `6` and `8`, and salinity too for `8`, regardless of the input values.  This is now reflected in the output values of these variables in the `CO2dict`.

    ***New outputs***

    * Substrate:inhibitor ratio (SIR) of [B15](refs.md/#b), calculated with `SIratio` in new module `bio`.
    * Inputs `PAR1` and `PAR2`.
    * The "Peng correction" factor.
    * The fugacity factor for converting between CO<sub>2</sub> partial pressure and fugacity.
    * The activity coefficient of the H<sup>+</sup> ion for NBS pH scale conversions.

    ***Validation***

    * Calculations compare very favourably against the forthcoming [CO2SYS for MATLAB v3](https://github.com/jonathansharp/CO2-System-Extd).

## 1.3

Adds bicarbonate ion and aqueous CO<sub>2</sub> as inputs from which the carbonate system can be solved.  Continues to reorganise code behind the scenes.  Makes almost everything [Autograd](https://github.com/HIPS/autograd)-able and uses this approach to calculate buffer constants.  Validates results against CO2SYS for MATLAB.

### 1.3.0 (1 May 2020)

!!! new-version "Changes in v1.3.0"

    ***New features***

    * Added bicarbonate ion (type `7`) and aqueous CO<sub>2</sub> (type `8`) as options for known input marine carbonate system variables.
    * Added module `test` with functions to perform internal consistency checks on `PyCO2SYS.CO2SYS` calculations and compare results with those from other sources.
    * Added module `api` with a wrapper for `PyCO2SYS.CO2SYS` to allow inputs as Pandas Series and/or Xarray DataArrays.

    ***Improved calculations***

    * The Revelle factor and all other buffer factors added in v1.2 are now evaluated using automatic differentiation, which means that the effects of all equilibrating species are taken into account.
        * The original, non-automatic functions that do not account for nutrient effects are still available in `buffers.explicit`.
        * Can switch between calculation methods using new optional input `opt_buffers_mode`.
    * Corrected Revelle factor calculations:
        * Added missing "Peng correction" to Revelle factor calculation at output conditions.  *Note that this correction is currently also missing from CO2SYS for MATLAB!*
        * Decreased DIC perturbation size for more accurate finite-difference "explicit" evaluation.
        * Finite-difference calculation now references the correct DIC value.
    * Implemented better initial guesses for pH in all iterative solvers in `solve.get` following [M13](refs.md/#m) and [OE15](refs.md/#o).
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

Calculates a wider variety of chemical buffer factors.

### 1.2.1 (9 April 2020)

!!! new-version "Changes in v1.2.1"
    * Fixed typo in [ESM10](refs.md/#e) equations that had been carried through into `extra.buffers_ESM10` function (thanks [Jim Orr](https://twitter.com/James1Orr/status/1248216403355803648)!).

### 1.2.0 (8 April 2020)

!!! new-version "Changes in v1.2.0"
    * Added module `extra` containing functions to calculate variables not included in CO2SYS for MATLAB:
      * `buffers_ESM10` calculates the buffer factors of [ESM10](refs.md/#e), corrected for the typos noted by [RAH18](refs.md/#r).
      * `bgc_isocap` calculates the "exact" isocapnic quotient of [HDW18](refs.md/#h), Eq. 8.
      * `bgc_isocap_approx` calculates the approximate isocapnic quotient of [HDW18](refs.md/#h), Eq. 7.
      * `psi` calculates the $\psi$ factor of [FCG94](refs.md/#f).
    * Added all functions in `extra` to the `CO2dict` output of the main `CO2SYS` function, and documented in the [Github repo README](https://github.com/mvdh7/PyCO2SYS#pyco2sys).

## 1.1

Adds optional inputs of total ammonium, hydrogen sulfide, and carbonate ion molinities for consistency with forthcoming MATLAB "v1.21".

### 1.1.1 (20 March 2020)

!!! new-version "Changes in v1.1.1"
    * Removed unnecessary `WhoseTB` input to `assemble.equilibria`.

### 1.1.0 (19 March 2020)

!!! new-version "Changes in v1.1.0"
    * Updated pH-solving iterative functions so that iteration stops separately for each row once it reaches the tolerance threshold.
    * Extracted all functions for solving the CO<sub>2</sub> system into a separate module (`solve`).
    * Extracted other key subfunctions into module `assemble`.
    * Added total ammonium (`NH3`) and hydrogen sulfide (`H2S`) concentrations as optional inputs to be included in the alkalinity model.
    * Added optional input to choose between different equations for hydrogen fluoride dissociation constant (`KFCONSTANT`).
    * Added functions to enable carbonate ion as an input carbonate system variable.
    * Output is now only the `CO2dict` dict, not the original `DATA`, `HEADERS` and `NICEHEADERS`.
    * Eliminated all global variables throughout the entire program.

## 1.0

### 1.0.1 (28 February 2020)

Starts to make things more Pythonic.

!!! new-version "Changes in v1.0.1"
      * Extracted all equations for concentrations and equilibrium constants into functions in separate modules (`concentrations` and `equilibria`).
      * Eliminated all global variables from the `_Constants` function.
      * Moved the as-close-as-possible version into module `original`. The default `from PyCO2SYS import CO2SYS` now imports the more Pythonic implementation.

### 1.0.0 (3 February 2020)

An as-close-as-possible clone of [MATLAB CO2SYS v2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB).

!!! new-version "Release notes for v1.0.0"
      * The first output `DICT` is new: a dict containing a separate entry for each variable in the original output `DATA`, with the keys named following the original output `HEADERS`.
      * The output `DATA` is transposed relative to the MATLAB version because Numpy is row-major while MATLAB is column-major.
      * Every combination of input options was tested against the MATLAB version with no significant differences (i.e. all differences can be attributed to floating point errors).
