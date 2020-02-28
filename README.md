# PyCO2SYS v1.1.0-dev

**PyCO2SYS** is a Python implementation of CO2SYS, based on the [MATLAB version 2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB). This software calculates the full marine carbonate system from values of any two of its variables.

Every combination of input parameters has been tested, with differences in the results small enough to be attributable to floating point errors (i.e. negligible). See the scripts `CO2SYStest.m` and `CO2SYStest.py` to see how and check this for yourself. **Please [let me know](https://mvdh.xyz/contact) ASAP if you discover a discrepancy that I have not spotted!**

Documentation is under construction at [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/).

## Installation

    pip install PyCO2SYS

## Usage

Usage has been kept as close to the MATLAB version as possible, although the first output is now a dict for convenience. Recommended usage is therefore:

```python
from PyCO2SYS import CO2SYS
DICT = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL,
    TEMPIN, TEMPOUT, PRESIN, PRESOUT, SI, PO4,
    pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS)[0]
```

To get all the MATLAB outputs (noting that `DATA` and `DICT` contain the exact same information, [just in a different format](#outputs)):

```python
DICT, DATA, HEADERS, NICEHEADERS = CO2SYS(
    PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL,
    TEMPIN, TEMPOUT, PRESIN, PRESOUT, SI, PO4,
    pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS)
```

Vector inputs should be provided as Numpy arrays (either row or column, makes no difference which).

See also the example scripts in the repo.

## Inputs

Taken directly from the MATLAB version:

```matlab
% INPUT:
%
%   PAR1  (some unit) : scalar or vector of size n
%   PAR2  (some unit) : scalar or vector of size n
%   PAR1TYPE       () : scalar or vector of size n (*)
%   PAR2TYPE       () : scalar or vector of size n (*)
%   SAL            () : scalar or vector of size n
%   TEMPIN  (degr. C) : scalar or vector of size n
%   TEMPOUT (degr. C) : scalar or vector of size n
%   PRESIN     (dbar) : scalar or vector of size n
%   PRESOUT    (dbar) : scalar or vector of size n
%   SI    (umol/kgSW) : scalar or vector of size n
%   PO4   (umol/kgSW) : scalar or vector of size n
%   pHSCALEIN         : scalar or vector of size n (**)
%   K1K2CONSTANTS     : scalar or vector of size n (***)
%   KSO4CONSTANTS     : scalar or vector of size n (****)
%
%  (*) Each element must be an integer,
%      indicating that PAR1 (or PAR2) is of type:
%  1 = Total Alkalinity
%  2 = DIC
%  3 = pH
%  4 = pCO2
%  5 = fCO2
%
%  (**) Each element must be an integer,
%       indicating that the pH-input (PAR1 or PAR2, if any) is at:
%  1 = Total scale
%  2 = Seawater scale
%  3 = Free scale
%  4 = NBS scale
%
%  (***) Each element must be an integer,
%        indicating the K1 K2 dissociation constants that are to be used:
%   1 = Roy, 1993                                         T:    0-45  S:  5-45. Total scale. Artificial seawater.
%   2 = Goyet & Poisson                                   T:   -1-40  S: 10-50. Seaw. scale. Artificial seawater.
%   3 = HANSSON              refit BY DICKSON AND MILLERO T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
%   4 = MEHRBACH             refit BY DICKSON AND MILLERO T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
%   5 = HANSSON and MEHRBACH refit BY DICKSON AND MILLERO T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
%   6 = GEOSECS (i.e., original Mehrbach)                 T:    2-35  S: 19-43. NBS scale.   Real seawater.
%   7 = Peng    (i.e., original Mehrbach but without XXX) T:    2-35  S: 19-43. NBS scale.   Real seawater.
%   8 = Millero, 1979, FOR PURE WATER ONLY (i.e., Sal=0)  T:    0-50  S:     0.
%   9 = Cai and Wang, 1998                                T:    2-35  S:  0-49. NBS scale.   Real and artificial seawater.
%  10 = Lueker et al, 2000                                T:    2-35  S: 19-43. Total scale. Real seawater.
%  11 = Mojica Prieto and Millero, 2002.                  T:    0-45  S:  5-42. Seaw. scale. Real seawater
%  12 = Millero et al, 2002                               T: -1.6-35  S: 34-37. Seaw. scale. Field measurements.
%  13 = Millero et al, 2006                               T:    0-50  S:  1-50. Seaw. scale. Real seawater.
%  14 = Millero        2010                               T:    0-50  S:  1-50. Seaw. scale. Real seawater.
%  15 = Waters, Millero, & Woosley 2014                   T:    0-50  S:  1-50. Seaw. scale. Real seawater.
%
%  (****) Each element must be an integer that
%         indicates the KSO4 dissociation constants that are to be used,
%         in combination with the formulation of the borate-to-salinity ratio to be used.
%         Having both these choices in a single argument is somewhat awkward,
%         but it maintains syntax compatibility with the previous version.
%  1 = KSO4 of Dickson 1990a   & TB of Uppstrom 1974  (PREFERRED)
%  2 = KSO4 of Khoo et al 1977 & TB of Uppstrom 1974
%  3 = KSO4 of Dickson 1990a   & TB of Lee 2010
%  4 = KSO4 of Khoo et al 1977 & TB of Lee 2010
```

## Outputs

The keys of the output `DICT`, and rows of `DATA`, correspond to the  variables in the list below. For example, to access the bicarbonate ion concentrations under the input conditions, we could use either of the following options:

```python
bicarb_in = DICT['HCO3in']
bicarb_in = DATA[5]
```

The other outputs (`DATA`, `HEADERS` and `NICEHEADERS`) are ~the same as in the MATLAB (except as noted [here](#differences-from-the-matlab-original)).

  * 0 - `TAlk` - total alkalinity (umol/kgSW)
  * 1 - `TCO2` - dissolved inorganic carbon (umol/kgSW)
  * 2 - `pHin` - pH on the input scale and conditions ()
  * 3 - `pCO2in` - seawater CO<sub>2</sub> partial pressure, input conditions (uatm)
  * 4 - `fCO2in` - seawater CO<sub>2</sub> fugacity, input conditions (uatm)
  * 5 - `HCO3in` - bicarbonate ion concentration, input conditions (umol/kgSW)
  * 6 - `CO3in` - carbonate ion concentration, input conditions (umol/kgSW)
  * 7 - `CO2in` - dissolved CO<sub>2</sub> concentration, input conditions (umol/kgSW)
  * 8 - `BAlkin` - borate alkalinity, input conditions (umol/kgSW)
  * 9 - `OHin` - hydroxide ion concentration, input conditions (umol/kgSW)
  * 10 - `PAlkin` - phosphate alkalinity, input conditions (umol/kgSW)
  * 11 - `SiAlkin` - silicate alkalinity, input conditions (umol/kgSW)
  * 12 - `Hfreein` - "free" hydrogen ion concentration, input conditions (umol/kgSW)
  * 13 - `RFin` - Revelle Factor, input conditions ()
  * 14 - `OmegaCAin` - calcite saturation state, input conditions ()
  * 15 - `OmegaARin` - aragonite saturation state, input conditions ()
  * 16 - `xCO2in` - CO<sub>2</sub> mole fraction, input conditions (ppm)
  * 17 - `pHout` - pH on the output scale and conditions ()
  * 18 - `pCO2out` - seawater CO<sub>2</sub> partial pressure, output conditions (uatm)
  * 19 - `fCO2out` - seawater CO<sub>2</sub> fugacity, output conditions (uatm)
  * 20 - `HCO3out` - bicarbonate ion concentration, output conditions (umol/kgSW)
  * 21 - `CO3out` - carbonate ion concentration, output conditions (umol/kgSW)
  * 22 - `CO2out` - dissolved CO<sub>2</sub> concentration, output conditions (umol/kgSW)
  * 23 - `BAlkout` - borate alkalinity, output conditions (umol/kgSW)
  * 24 - `OHout` - hydroxide ion concentration, output conditions (umol/kgSW)
  * 25 - `PAlkout` - phosphate alkalinity, output conditions (umol/kgSW)
  * 26 - `SiAlkout` - silicate alkalinity, output conditions (umol/kgSW)
  * 27 - `Hfreeout` - "free" hydrogen ion concentration, output conditions (umol/kgSW)
  * 28 - `RFout` - Revelle Factor, output conditions ()
  * 29 - `OmegaCAout` - calcite saturation state, output conditions ()
  * 30 - `OmegaARout` - aragonite saturation state, output conditions ()
  * 31 - `xCO2out` - CO<sub>2</sub> mole fraction, output conditions (ppm)
  * 32 - `pHinTOTAL` - Total scale pH, input conditions ()
  * 33 - `pHinSWS` - Seawater scale pH, input conditions ()
  * 34 - `pHinFREE` - Free scale pH, input conditions ()
  * 35 - `pHinNBS` - NBS scale pH, input conditions ()
  * 36 - `pHoutTOTAL` - Total scale pH, output conditions ()
  * 37 - `pHoutSWS` - Seawater scale pH, output conditions ()
  * 38 - `pHoutFREE` - Free scale pH, output conditions ()
  * 39 - `pHoutNBS` - NBS scale pH, output conditions ()
  * 40 - `TEMPIN` - input temperature (deg C)
  * 41 - `TEMPOUT` - output temperature (deg C)
  * 42 - `PRESIN` - input pressure (dbar or m)
  * 43 - `PRESOUT` - output pressure (dbar or m)
  * 44 - `PAR1TYPE` - input parameter 1 type (integer)
  * 45 - `PAR2TYPE` - input parameter 2 type (integer)
  * 46 - `K1K2CONSTANTS` - carbonic acid constants option (integer)
  * 47 - `KSO4CONSTANTS` - bisulfate dissociation option(integer)
  * 48 - `pHSCALEIN` - input pH scale (integer)
  * 49 - `SAL` - salinity(psu)
  * 50 - `PO4` - phosphate concentration (umol/kgSW)
  * 51 - `SI` - silicate concentration (umol/kgSW)
  * 52 - `K0input` - Henry's constant for CO2, input conditions ()
  * 53 - `K1input` - first carbonic acid dissociation constant, input conditions ()
  * 54 - `K2input` - second carbonic acid dissociation constant, input conditions ()
  * 55 - `pK1input` - -log<sub>10</sub>(`K1input`) ()
  * 56 - `pK2input` - -log<sub>10</sub>(`K2input`) ()
  * 57 - `KWinput` - water dissociation constant, input conditions ()
  * 58 - `KBinput` - boric acid dissociation constant, input conditions ()
  * 59 - `KFinput` - hydrogen fluoride dissociation constant, input conditions ()
  * 60 - `KSinput` - bisulfate dissociation constant, input conditions ()
  * 61 - `KP1input` - first phosphoric acid dissociation constant, input conditions ()
  * 62 - `KP2input` - second phosphoric acid dissociation constant, input conditions ()
  * 63 - `KP3input` - third phosphoric acid dissociation constant, input conditions ()
  * 64 - `KSiinput` - silica acid dissociation constant, input conditions ()    
  * 65 - `K0output` - Henry's constant for CO2, output conditions ()
  * 66 - `K1output` - first carbonic acid dissociation constant, output conditions ()
  * 67 - `K2output` - second carbonic acid dissociation constant, output conditions ()
  * 68 - `pK1output` - -log<sub>10</sub>(`K1output`) ()
  * 69 - `pK2output` - -log<sub>10</sub>(`K2output`) ()
  * 70 - `KWoutput` - water dissociation constant, output conditions ()
  * 71 - `KBoutput` - boric acid dissociation constant, output conditions ()
  * 72 - `KFoutput` - hydrogen fluoride dissociation constant, output conditions ()
  * 73 - `KSoutput` - bisulfate dissociation constant, output conditions ()
  * 74 - `KP1output` - first phosphoric acid dissociation constant, output conditions ()
  * 75 - `KP2output` - second phosphoric acid dissociation constant, output conditions ()
  * 76 - `KP3output` - third phosphoric acid dissociation constant, output conditions ()
  * 77 - `KSioutput` - silica acid dissociation constant, output conditions ()   
  * 78 - `TB` - total borate concentration (umol/kgSW)
  * 79 - `TF` - total fluoride concentration (umol/kgSW)
  * 80 - `TS` - total sulfate concentration (umol/kgSW)

## Differences from the MATLAB original

  * Inputs are the same as in the MATLAB version, with vectors of input values provided as Numpy arrays.
  * Outputs are also the same, with the exception that an extra output `DICT` comes before the MATLAB three (`DATA`, `HEADERS` and `NICEHEADERS`) - this contains the numerical results in `DATA` but in a dict with the names in `HEADERS` as the keys.
  * `DATA` in the Python version is the transpose of the same variable in the MATLAB version. Note that the row number for each variable in `DATA` is offset by 1 from the corresponding column of the equivalent MATLAB variable because of Python's zero-indexing.

## Citation

See [the original MATLAB repo](https://github.com/jamesorr/CO2SYS-MATLAB) for more detailed information on versions and citation.

  * If you use any CO2SYS related software, please cite the original work by Lewis and Wallace (1998).
  * If you use CO2SYS.m, please cite van Heuven et al (2011).
  * If you use errors.m or derivnum.m, please cite Orr et al. (2018).
  * If you use CO2SYS.py, please mention it somewhere with a link to this repository.
