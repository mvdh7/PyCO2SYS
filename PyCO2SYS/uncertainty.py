# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2026  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Define standard uncertainties for propagation."""

# Define default uncertainties in pK values following OEDG18.
# They are squared because PyCO2SYS requires variances, not standard
# deviations.
pks_OEDG18 = {
    "pk_CO2": 0.002**2,
    "pk_H2CO3": 0.0075**2,
    "pk_HCO3": 0.015**2,
    "pk_BOH3": 0.01**2,
    "pk_H2O": 0.01**2,
    "pk_aragonite": 0.02**2,
    "pk_calcite": 0.02**2,
}
# OEDG18 defines a fractional uncertainty in total_borate too
total_borate_pct_OEDG18 = 0.02


def set_u_OEDG18(co2s):
    co2s.set_u_coeffs_from_single(**pks_OEDG18)
    co2s.set_u(
        coeffs_total_borate=(
            total_borate_pct_OEDG18 * co2s.coeffs_total_borate
        )
        ** 2
    )
    return co2s
