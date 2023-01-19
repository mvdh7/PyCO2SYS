# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Minimalistic functions that can be used with xarray.apply_ufunc()."""

from . import equilibria, salts, solve


def pH_from_alkalinity_dic(
    alkalinity,
    dic,
    opt_gas_constant=3,
    opt_k_bisulfate=1,
    opt_k_carbonic=10,
    opt_k_fluoride=1,
    opt_pH_scale=3,
    opt_total_borate=1,
    opt_pressured_kCO2=0,
    pressure_atmosphere=1,
    pressure=0,
    salinity=35,
    temperature=25,
    total_ammonia=0,
    total_phosphate=0,
    total_silicate=0,
    total_sulfide=0,
):
    """Calculate pH from total alkalinity and dissolved inorganic carbon."""
    totals = salts.assemble(
        salinity,
        total_silicate,
        total_phosphate,
        total_ammonia,
        total_sulfide,
        opt_k_carbonic,
        opt_total_borate,
    )
    k_constants = equilibria.assemble(
        temperature,
        pressure,
        totals,
        opt_pH_scale,
        opt_k_carbonic,
        opt_k_bisulfate,
        opt_k_fluoride,
        opt_gas_constant,
        pressure_atmosphere=pressure_atmosphere,
        opt_pressured_kCO2=opt_pressured_kCO2,
    )
    pH = solve.get.pHfromTATC(alkalinity * 1e-6, dic * 1e-6, totals, k_constants)
    return pH
