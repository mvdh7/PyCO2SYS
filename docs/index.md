# PyCO2SYS v1.0.1

PyCO2SYS is a Python implementation of CO2SYS [[LW98](refs/#LW98), [HPR11](refs/#HPR11), [OEDG18](refs/#OEDG18)], the MATLAB toolbox for marine carbonate system calculations.

## Installation and use

Install from the Python Package Index:

    pip install PyCO2SYS

Import and use just like in MATLAB:

```python
from PyCO2SYS import CO2SYS
DICT = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL,
    TEMPIN, TEMPOUT, PRESIN, PRESOUT, SI, PO4,
    pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS)[0]
```

The entries of `DICT` contain all the variables from the original MATLAB output `DATA`, with the keys corresponding to the MATLAB `HEADERS`.

## Citation

  * If you use any CO2SYS-related software, please cite the original work by Lewis and Wallace (1998) [[LW98](refs/#LW98)].
  * If you use CO2SYS.m, please cite van Heuven et al. (2011) [[HPR11](refs/#HPR11)].
  * If you use errors.m or derivnum.m, please cite Orr et al. (2018) [[OEDG18](refs/#OEDG18)].
  * If you use PyCO2SYS, please mention it with a link to the Github repository: [github.com/mvdh7/PyCO2SYS](https://github.com/mvdh7/PyCO2SYS).
