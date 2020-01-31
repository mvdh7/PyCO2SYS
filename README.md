# PyCO2SYS

**PyCO2SYS** is a Python implementation of CO2SYS, based on the [MATLAB version 2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB). This software calculates the full marine carbonate system from values of any two of its variables.

## Installation and usage

    pip install PyCO2SYS
    
```python
from PyCO2SYS import CO2SYS
co2dict = CO2SYS(par1, par2, par1type, par2type, sal, tempin, tempout, 
                 presin, presout, sil, po4, pHscale, k1k2c, kso4c)[0]
```

See also the example scripts in the repo.

## Differences from the MATLAB original

Inputs are the same as in the MATLAB version, with vectors of input values provided as Numpy arrays. Outputs are also the same, with the exception that an extra output `DICT` comes before the MATLAB three (`DATA`, `HEADERS` and `NICEHEADERS`) - this contains the numerical results in `DATA` but in a dict with the names in `HEADERS` as the keys. Note also that `DATA` in the Python version is the transpose of the same variable in the MATLAB version.

## Citation

See [the original MATLAB repo](https://github.com/jamesorr/CO2SYS-MATLAB) for more detailed information on versions and citation.

  * If you use any CO2SYS related software, please cite the original work by Lewis and Wallace (1998).
  * If you use CO2SYS.m, please cite van Heuven et al (2011).
  * If you use errors.m or derivnum.m, please cite Orr et al. (2018).
