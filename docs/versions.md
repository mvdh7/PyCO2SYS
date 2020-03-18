# Version history

## 1.1.0

Adding extra inputs and options for consistency with Pierrot et al.'s new MATLAB "v1.21".

**Release date:** forthcoming

  * Extracted all functions for solving the CO<sub>2</sub> system into a separate module (`solve`).
  * Added total ammonium and hydrogen sulfide concentrations as inputs to be included in the alkalinity model.
  * Added functions to enable carbonate ion as an input carbonate system variable.
  * Output is now only the `DICT` variable, not the original `DATA`, `HEADERS` and `NICEHEADERS`.
  * Eliminated all global variables throughout the entire program.

## 1.0.1

**Release date:** 28 Feb 2020

Starting to make things more Pythonic.

  * Extracted all equations for concentrations and equilibrium constants into functions in separate modules (`concentrations` and `equilibria`).
  * Eliminated all global variables from the `_Constants` function.
  * Moved the as-close-as-possible version into module `original`. The default `from PyCO2SYS import CO2SYS` now imports the more Pythonic implementation.

## 1.0.0

**Release date:** 3 Feb 2020

An as-close-as-possible clone of MATLAB CO2SYS v2.0.5, obtained from [github.com/jamesorr/CO2SYS-MATLAB](https://github.com/jamesorr/CO2SYS-MATLAB).

  * The first output `DICT` is new: a dict containing a separate entry for each variable in the original output `DATA`, with the keys named following the original output `HEADERS`.
  * The output `DATA` is transposed relative to the MATLAB version because Numpy is row-major while MATLAB is column-major.
  * Every combination of input options was tested against the MATLAB version with no significant differences (i.e. all differences can be attributed to floating point errors).
