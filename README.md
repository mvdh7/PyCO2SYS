# PyCO2SYS v0.1

**PyCO2SYS** is a Python implementation of CO2SYS, based on the [MATLAB version 2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB). This software calculates the full marine carbonate system from values of any two of its variables.

> **Some basic comparisons have not shown any differences in the results, but a thorough intercomparison between the MATLAB and Python has not yet been carried out - so use at your own risk (for now)!**

## Installation

    pip install PyCO2SYS

## Usage

Usage has been kept as close to the MATLAB version as possible, although the first output is now a dict for convenience:

```python
from PyCO2SYS import CO2SYS
DICT, DATA, HEADERS, NICEHEADERS = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL,
    TEMPIN, TEMPOUT, PRESIN, PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS)[0]
```

Vector inputs should be provided as Numpy arrays (either row or column, makes no difference which).

See also the example scripts in the repo.

## Output contents

The keys of the output `DICT`, and rows of `DATA`, correspond to the following variables:

    * 00 - `TAlk` - total alkalinity (umol/kgSW)
    * 01 - `TCO2` - dissolved inorganic carbon (umol/kgSW)
    * 02 - `pHin` - pH on the input scale and conditions ()
    * 03 - `pCO2in` - seawater CO2 partial pressure, input conditions (uatm)
    * 04 - `fCO2in` - seawater CO2 fugacity, input conditions (uatm)
    * 05 - `HCO3in` - bicarbonate ion concentration, input conditions (umol/kgSW)
    * 06 - `CO3in` - carbonate ion concentration, input conditions (umol/kgSW)
    * 07 - `CO2in` - dissolved CO2 concentration, input conditions (umol/kgSW)
    * 08 - `BAlkin` - borate alkalinity, input conditions (umol/kgSW)
    * 09 - `OHin` - hydroxide ion concentration, input conditions (umol/kgSW)
    * 10 - `PAlkin` - phosphate alkalinity, input conditions (umol/kgSW)
    * 11 - `SiAlkin` - silicate alkalinity, input conditions (umol/kgSW)
    * 12 - `Hfreein` - "free" hydrogen ion concentration, input conditions (umol/kgSW)
    * 13 - `RFin` - Revelle Factor, input conditions ()
    * 14 - `OmegaCAin` - calcite saturation state, input conditions ()
    * 15 - `OmegaARin` - aragonite saturation state, input conditions ()
    * 16 - `xCO2in` - mole fraction CO2, input conditions (ppm)
    * 17 - `pHout` - pH on the output scale and conditions ()
    * 18 - `pCO2out` - seawater CO2 partial pressure, output conditions (uatm)
    * 19 - `fCO2out` - seawater CO2 fugacity, output conditions (uatm)
    * 20 - `HCO3out` - bicarbonate ion concentration, output conditions (umol/kgSW)
    * 21 - `CO3out` - carbonate ion concentration, output conditions (umol/kgSW)
    * 22 - `CO2out` - dissolved CO2 concentration, output conditions (umol/kgSW)
    * 23 - `BAlkout` - borate alkalinity, output conditions (umol/kgSW)
    * 24 - `OHout` - hydroxide ion concentration, output conditions (umol/kgSW)
    * 25 - `PAlkout` - phosphate alkalinity, output conditions (umol/kgSW)
    * 26 - `SiAlkout` - silicate alkalinity, output conditions (umol/kgSW)
    * 27 - `Hfreeout` - "free" hydrogen ion concentration, output conditions (umol/kgSW)
    * 28 - `RFout` - Revelle Factor, output conditions ()
    * 29 - `OmegaCAout` - calcite saturation state, output conditions ()
    * 30 - `OmegaARout` - aragonite saturation state, output conditions ()
    * 31 - `xCO2out` - mole fraction CO2, output conditions (ppm)
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
    * 54 - `K2input`            ()          
    * 55 - `pK1input`            ()          
    * 56 - `pK2input`            ()          
    * 57 - `KWinput`            ()          
    * 58 - `KBinput`            ()          
    * 59 - `KFinput`            ()          
    * 60 - `KSinput`            ()          
    * 61 - `KP1input`            ()          
    * 62 - `KP2input`            ()          
    * 63 - `KP3input`            ()          
    * 64 - `KSiinput`            ()              
    * 65 - `K0output`           ()          
    * 66 - `K1output`           ()          
    * 67 - `K2output`           ()          
    * 68 - `pK1output`           ()          
    * 69 - `pK2output`           ()          
    * 70 - `KWoutput`           ()          
    * 71 - `KBoutput`           ()          
    * 72 - `KFoutput`           ()          
    * 73 - `KSoutput`           ()          
    * 74 - `KP1output`           ()          
    * 75 - `KP2output`           ()          
    * 76 - `KP3output`           ()          
    * 77 - `KSioutput`           ()              
    * 78 - `TB` - total borate concentration (umol/kgSW)
    * 79 - `TF` - total fluoride concentration (umol/kgSW)
    * 80 - `TS` - total sulfate concentration (umol/kgSW)

## Differences from the MATLAB original

Inputs are the same as in the MATLAB version, with vectors of input values provided as Numpy arrays. Outputs are also the same, with the exception that an extra output `DICT` comes before the MATLAB three (`DATA`, `HEADERS` and `NICEHEADERS`) - this contains the numerical results in `DATA` but in a dict with the names in `HEADERS` as the keys. Note also that `DATA` in the Python version is the transpose of the same variable in the MATLAB version.

## Citation

See [the original MATLAB repo](https://github.com/jamesorr/CO2SYS-MATLAB) for more detailed information on versions and citation.

  * If you use any CO2SYS related software, please cite the original work by Lewis and Wallace (1998).
  * If you use CO2SYS.m, please cite van Heuven et al (2011).
  * If you use errors.m or derivnum.m, please cite Orr et al. (2018).
