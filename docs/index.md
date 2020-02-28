# PyCO2SYS v1.0.1

PyCO2SYS is a Python implementation of CO2SYS [[LW98](refs/#LW98), [HPR11](refs/#HPR11), [OEDG18](refs/#OEDG18)], the MATLAB toolbox for marine carbonate system calculations.

## Installation and use

Install from the Python Package Index:

    pip install PyCO2SYS

Import and use CO2SYS just like in MATLAB:

```python
from PyCO2SYS import CO2SYS
DICT = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL,
    TEMPIN, TEMPOUT, PRESIN, PRESOUT, SI, PO4,
    pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS)[0]
```

The entries of `DICT` contain all the variables from the original MATLAB output `DATA`, with the keys corresponding to the MATLAB `HEADERS`.
