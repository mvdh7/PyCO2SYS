# Version history

## 1.0.1

**Release date:** forthcoming

Making things more Pythonic.

  * Extracted all equations for concentrations and equilibrium constants into functions in separate modules.

  * Eliminated all global variables from the `_Constants` function.

  * Moved the as-close-as-possible version into module `original`, and removed the new `DICT` output from it. The default `from PyCO2SYS import CO2SYS` now imports the more Pythonic implementation.

## 1.0.0

**Release date:** 3 Feb 2020

An as-close-as-possible clone of MATLAB CO2SYS v2.0.5, obtained from [github.com/jamesorr/CO2SYS-MATLAB](https://github.com/jamesorr/CO2SYS-MATLAB).

  * The first output `DICT` is new: a dict containing a separate entry for each variable in the original output `DATA`, with the keys named following the original output `HEADERS`.

  * The output `DATA` is transposed relative to the MATLAB version because Numpy is row-major while MATLAB is column-major.

  * Every combination of input options was tested against the MATLAB version with no significant differences (i.e. all differences can be attributed to floating point errors).
