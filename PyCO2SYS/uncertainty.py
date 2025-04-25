# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Define standard uncertainties for propagation."""

from jax import numpy as np

# Define default uncertainties in pK values following OEDG18
pKs_OEDG18 = {
    "pk_CO2": 0.002,
    "pk_H2CO3": 0.0075,
    "pk_HCO3": 0.015,
    "pk_BOH3": 0.01,
    "pk_H2O": 0.01,
    "k_aragonite": 0.02,
    "k_calcite": 0.02,
}
# OEDG18 with fractional uncertainty in total_borate too
all_OEDG18 = pKs_OEDG18.copy()
all_OEDG18.update({"total_borate__f": 0.02})

# Total salts
u_cf_total_sulfate = np.array(
    [
        [(1e6 * (0.00023 / 96.062) / 1.80655) ** 2, 0],
        [0, 0],
    ]
)
