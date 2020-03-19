# PyCO2SYS

[![PyPI version](https://badge.fury.io/py/PyCO2SYS.svg)](https://badge.fury.io/py/PyCO2SYS)

**PyCO2SYS** is a Python implementation of CO<sub>2</sub>SYS, based on the [MATLAB v2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB) but also including the updates made for tentatively forthcoming MATLAB v1.21. This software calculates the full marine carbonate system from values of any two of its variables.

Every combination of input parameters has been tested, with differences in the results small enough to be attributable to floating point errors and iterative solver endpoint differences (i.e. negligible). See the scripts in the [compare](compare) directory to see how and check this for yourself. **Please [let me know](https://mvdh.xyz/contact) ASAP if you discover a discrepancy that I have not spotted!**

Documentation is under construction at [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/).

## Installation

Install from the Python Package Index:

    pip install PyCO2SYS

Update an existing installation:

    pip install PyCO2SYS --upgrade --no-cache-dir    

## Use

The API has been kept as close to the MATLAB version as possible, although the first output is now a dict for convenience. Recommended usage is therefore:

```python
from PyCO2SYS import CO2SYS
CO2dict = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN, PRESOUT,
    SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS, NH3=0.0, H2S=0.0, KFCONSTANT=1)
```

Each field in the output `CO2dict` corresponds to a column in the original MATLAB output `DATA`. The keys to the dict come from the original MATLAB output `HEADERS`.

Vector inputs should be provided as Numpy arrays. Everything gets flattened with `ravel`. Single-value inputs are fine, they are automatically cast into correctly-sized arrays.

See also the example scripts here in the repo.

## Inputs

### Required inputs

The required inputs are identical to [the MATLAB version](https://github.com/jamesorr/CO2SYS-MATLAB):

  * `PAR1` - first known carbonate system parameter value.
  * `PAR2` - second known carbonate system parameter value.
  * `PAR1TYPE` - integer identifying which parameters `PAR1` are.
  * `PAR2TYPE` - integer identifying which parameters `PAR2` are.

The possible known carbonate system parameters are `1`: total alkalinity in μmol·kg<sup>−1</sup>, `2`: dissolved inorganic carbon in μmol·kg<sup>−1</sup>, `3`: pH (dimensionless), `4`: dissolved CO<sub>2</sub> partial pressure in μatm, `5`: dissolved CO<sub>2</sub> fugacity in μatm, and `6`: carbonate ion concentration in μmol·kg<sup>−1</sup>.

Here and throughout the inputs and outputs, "kg<sup>−1</sup>" refers to the total mass of seawater (solvent + solutes), not just the mass of H<sub>2</sub>O.

  * `SAL` - practical salinity.
  * `TEMPIN` - temperature of input carbonate system parameters.
  * `TEMPOUT` - temperature at which to calculate outputs.
  * `PRESIN` - pressure of input carbonate system parameters.
  * `PRESOUT` - pressure at which to calculate outputs.

All temperatures are in °C and pressures are in dbar. Pressure is within the water column as typically measured by a CTD sensor, i.e. not including atmospheric pressure. The 'input' conditions could represent conditions in the laboratory during a measurement, while the 'output' conditions could be those observed in situ during sample collection.

  * `SI` - total silicate concentration.
  * `PO4` - total phosphate concentration.

Nutrient concentrations are all in μmol·kg<sup>−1</sup>.

  * `pHSCALEIN` - pH scale(s) that pH values in `PAR1` or `PAR2` are on.

The options are `1`: Total scale, `2`: Seawater scale, `3`: Free scale, and `4`: NBS scale, as defined by [ZW01](https://pyco2sys.readthedocs.io/en/latest/refs/#ZW01).

  * `K1K2CONSTANTS` - which set of constants to use for carbonic acid dissociation.

The options are integers from `1` to `15` inclusive. From the original MATLAB documentation:

  ```matlab
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
```

  * `KSO4CONSTANTS` - which sets of constants to use for bisulfate dissociation and borate:chlorinity ratio.

The options are integers from `1` to `4` inclusive. From the original MATLAB documentation:

```matlab
%  1 = KSO4 of Dickson 1990a   & TB of Uppstrom 1974  (PREFERRED)
%  2 = KSO4 of Khoo et al 1977 & TB of Uppstrom 1974
%  3 = KSO4 of Dickson 1990a   & TB of Lee et al. 2010
%  4 = KSO4 of Khoo et al 1977 & TB of Lee et al. 2010
```

### Optional inputs

There are also some optional keyword inputs for consistency with a tentatively forthcoming new MATLAB version:

  * `NH3` - total ammonia concentration.
  * `H2S` - total hydrogen sulfide concentration.

As for all other concentrations, these are in μmol·kg<sup>−1</sup>. If no values are provided, these default to zero (i.e. consistent with [MATLAB v2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB)).

  * `KFCONSTANT` - which constant to use for hydrogen fluoride dissociation.

The options are `1`: [DR79](https://pyco2sys.readthedocs.io/en/latest/refs/#DR79), and `2`: [PF87](https://pyco2sys.readthedocs.io/en/latest/refs/#PF87). If nothing is provided, the default is `1` for consistency with [MATLAB v2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB).

## Outputs

The keys of the output `DICT` correspond to the variables in the lists below.

### Outputs also in [MATLAB v2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB)

  * `TAlk` - total alkalinity (μmol·kg<sup>−1</sup>).
  * `TCO2` - dissolved inorganic carbon (μmol·kg<sup>−1</sup>).
  * `pHin` - pH on the input scale and conditions.
  * `pCO2in` - seawater CO<sub>2</sub> partial pressure, input conditions (μatm).
  * `fCO2in` - seawater CO<sub>2</sub> fugacity, input conditions (μatm).
  * `HCO3in` - bicarbonate ion concentration, input conditions (μmol·kg<sup>−1</sup>).
  * `CO3in` - carbonate ion concentration, input conditions (μmol·kg<sup>−1</sup>).
  * `CO2in` - dissolved CO<sub>2</sub> concentration, input conditions (μmol·kg<sup>−1</sup>).
  * `BAlkin` - borate alkalinity, input conditions (μmol·kg<sup>−1</sup>).
  * `OHin` - hydroxide ion concentration, input conditions (μmol·kg<sup>−1</sup>).
  * `PAlkin` - phosphate alkalinity, input conditions (μmol·kg<sup>−1</sup>).
  * `SiAlkin` - silicate alkalinity, input conditions (μmol·kg<sup>−1</sup>).
  * `Hfreein` - "free" hydrogen ion concentration, input conditions (μmol·kg<sup>−1</sup>).
  * `RFin` - Revelle Factor, input conditions.
  * `OmegaCAin` - calcite saturation state, input conditions.
  * `OmegaARin` - aragonite saturation state, input conditions.
  * `xCO2in` - CO<sub>2</sub> mole fraction, input conditions (ppm).
  * `pHout` - pH on the output scale and conditions.
  * `pCO2out` - seawater CO<sub>2</sub> partial pressure, output conditions (μatm).
  * `fCO2out` - seawater CO<sub>2</sub> fugacity, output conditions (μatm).
  * `HCO3out` - bicarbonate ion concentration, output conditions (μmol·kg<sup>−1</sup>).
  * `CO3out` - carbonate ion concentration, output conditions (μmol·kg<sup>−1</sup>).
  * `CO2out` - dissolved CO<sub>2</sub> concentration, output conditions (μmol·kg<sup>−1</sup>).
  * `BAlkout` - borate alkalinity, output conditions (μmol·kg<sup>−1</sup>).
  * `OHout` - hydroxide ion concentration, output conditions (μmol·kg<sup>−1</sup>).
  * `PAlkout` - phosphate alkalinity, output conditions (μmol·kg<sup>−1</sup>).
  * `SiAlkout` - silicate alkalinity, output conditions (μmol·kg<sup>−1</sup>).
  * `Hfreeout` - "free" hydrogen ion concentration, output conditions (μmol·kg<sup>−1</sup>).
  * `RFout` - Revelle Factor, output conditions.
  * `OmegaCAout` - calcite saturation state, output conditions.
  * `OmegaARout` - aragonite saturation state, output conditions.
  * `xCO2out` - CO<sub>2</sub> mole fraction, output conditions (ppm).
  * `pHinTOTAL` - Total scale pH, input conditions.
  * `pHinSWS` - Seawater scale pH, input conditions.
  * `pHinFREE` - Free scale pH, input conditions.
  * `pHinNBS` - NBS scale pH, input conditions.
  * `pHoutTOTAL` - Total scale pH, output conditions.
  * `pHoutSWS` - Seawater scale pH, output conditions.
  * `pHoutFREE` - Free scale pH, output conditions.
  * `pHoutNBS` - NBS scale pH, output conditions.
  * `TEMPIN` - input temperature (deg C).
  * `TEMPOUT` - output temperature (deg C).
  * `PRESIN` - input pressure (dbar or m).
  * `PRESOUT` - output pressure (dbar or m).
  * `PAR1TYPE` - input parameter 1 type (integer).
  * `PAR2TYPE` - input parameter 2 type (integer).
  * `K1K2CONSTANTS` - carbonic acid constants option (integer).
  * `KSO4CONSTANT` - bisulfate dissociation and borate:chlorinity option (integer).
  * `pHSCALEIN` - input pH scale (integer).
  * `SAL` - practical salinity.
  * `PO4` - phosphate concentration (μmol·kg<sup>−1</sup>).
  * `SI` - silicate concentration (μmol·kg<sup>−1</sup>).
  * `K0input` - Henry's constant for CO<sub>2</sub>, input conditions.
  * `K1input` - first carbonic acid dissociation constant, input conditions.
  * `K2input` - second carbonic acid dissociation constant, input conditions.
  * `pK1input` - -log<sub>10</sub>(`K1input`).
  * `pK2input` - -log<sub>10</sub>(`K2input`).
  * `KWinput` - water dissociation constant, input conditions.
  * `KBinput` - boric acid dissociation constant, input conditions.
  * `KFinput` - hydrogen fluoride dissociation constant, input conditions.
  * `KSinput` - bisulfate dissociation constant, input conditions.
  * `KP1input` - first phosphoric acid dissociation constant, input conditions.
  * `KP2input` - second phosphoric acid dissociation constant, input conditions.
  * `KP3input` - third phosphoric acid dissociation constant, input conditions.
  * `KSiinput` - silica acid dissociation constant, input conditions.
  * `K0output` - Henry's constant for CO<sub>2</sub>, output conditions.
  * `K1output` - first carbonic acid dissociation constant, output conditions.
  * `K2output` - second carbonic acid dissociation constant, output conditions.
  * `pK1output` - -log<sub>10</sub>(`K1output`).
  * `pK2output` - -log<sub>10</sub>(`K2output`).
  * `KWoutput` - water dissociation constant, output conditions.
  * `KBoutput` - boric acid dissociation constant, output conditions.
  * `KFoutput` - hydrogen fluoride dissociation constant, output conditions.
  * `KSoutput` - bisulfate dissociation constant, output conditions.
  * `KP1output` - first phosphoric acid dissociation constant, output conditions.
  * `KP2output` - second phosphoric acid dissociation constant, output conditions.
  * `KP3output` - third phosphoric acid dissociation constant, output conditions.
  * `KSioutput` - silica acid dissociation constant, output conditions.
  * `TB` - total borate concentration (μmol·kg<sup>−1</sup>).
  * `TF` - total fluoride concentration (μmol·kg<sup>−1</sup>).
  * `TS` - total sulfate concentration (μmol·kg<sup>−1</sup>).

### New outputs

  * `KSO4CONSTANT` - bisulfate dissociation option (integer).
  * `KFCONSTANT` - hydrogen sulfide dissociation option (integer).  
  * `BORON` - boron:chlorinity option (integer).
  * `NH3` - total ammonium concentration (μmol·kg<sup>−1</sup>).
  * `H2S` - total sulfide concentration (μmol·kg<sup>−1</sup>).
  * `NH3Alkin` - ammonia alkalinity, input conditions (μmol·kg<sup>−1</sup>).
  * `H2SAlkin` - hydrogen sulfide alkalinity, input conditions (μmol·kg<sup>−1</sup>).
  * `NH3Alkout` - ammonia alkalinity, output conditions (μmol·kg<sup>−1</sup>).
  * `H2SAlkout` - hydrogen sulfide alkalinity, output conditions (μmol·kg<sup>−1</sup>).
  * `KNH3input`: ammonium equilibrium constant, input conditions.
  * `KH2Sinput`: hydrogen sulfide equilibrium constant, input conditions.
  * `KNH3output`: ammonium equilibrium constant, output conditions.
  * `KH2Soutput`: hydrogen sulfide equilibrium constant, output conditions.

## Citation

See [the original MATLAB repo](https://github.com/jamesorr/CO2SYS-MATLAB) for more detailed information on versions and citation.

  * If you use any CO<sub2</sub>SYS-related software, please cite the original work by [Lewis and Wallace (1998)](https://pyco2sys.readthedocs.io/en/latest/refs/#LW98).
  * If you use CO2SYS.m, please cite [van Heuven et al. (2011)](https://pyco2sys.readthedocs.io/en/latest/refs/#HPR11).
  * If you use errors.m or derivnum.m, please cite [Orr et al. (2018)](https://pyco2sys.readthedocs.io/en/latest/refs/#OEDG18).
  * If you use PyCO2SYS, please mention it somewhere with a link to this repository, but check back here first to see if a proper citation is available.

Please mention which version of PyCO2SYS you used. You can find this in Python with:

```python
from PyCO2SYS.meta import version
print('This is PyCO2SYS v{}'.format(version))
```
