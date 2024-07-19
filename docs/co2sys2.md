# PyCO2SYS v2

## Syntax

We create a `CO2System` by providing values of all known parameters (`values`) as a dict:

```python
from PyCO2SYS import CO2System

values = dict(
    alkalinity=2250,
    pH=8.1,
    temperature=12.5,
    salinity=32.4,
)

sys = CO2System(values)
```
