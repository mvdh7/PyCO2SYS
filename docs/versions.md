# Version history

## 1.0.0

An as-close-as-possible clone of MATLAB CO2SYS v2.0.5, obtained from [github.com/jamesorr/CO2SYS-MATLAB](https://github.com/jamesorr/CO2SYS-MATLAB).

  * The first output `DICT` is new: a dict containing a separate entry for each variable in the original output `DATA`, with the keys named following the original output `HEADERS`.

  * The output `DATA` is transposed relative to the MATLAB version because Numpy is row-major while MATLAB is column-major.

  * Every combination of input options was tested against the MATLAB version with no significant differences (i.e. all differences can be attributed to floating point errors).
