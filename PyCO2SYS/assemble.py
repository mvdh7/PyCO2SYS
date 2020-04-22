# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
from autograd.numpy import array, exp, full, nan, size, unique, where
from . import convert, salts
from .equilibria import p1atm as eq
from .constants import RGasConstant

def inputs(input_locals):
    """Condition inputs for use with CO2SYS (sub)functions."""
    # Determine and check lengths of input vectors
    veclengths = array([size(v) for v in input_locals.values()])
    assert size(unique(veclengths[veclengths != 1])) <= 1, \
        'CO2SYS function inputs must all be of same length, or of length 1.'
    # Make vectors of all inputs
    ntps = max(veclengths)
    args = {k: full(ntps, v) if size(v)==1 else v.ravel()
            for k, v in input_locals.items()}
    # Convert to float where appropriate
    float_vars = ['SAL', 'TEMPIN', 'TEMPOUT', 'PRESIN', 'PRESOUT', 'SI', 'PO4',
                  'NH3', 'H2S', 'PAR1', 'PAR2']
    for k in args.keys():
        if k in float_vars:
            args[k] = args[k].astype('float64')
    return args, ntps

def concs_TB(Sal, WhichKs, WhoseTB):
    """Calculate total borate from salinity for the given options."""
    TB = where(WhichKs==8, 0.0, nan) # pure water
    TB = where((WhichKs==6) | (WhichKs==7), salts.borate_C65(Sal), TB)
    F = (WhichKs!=6) & (WhichKs!=7) & (WhichKs!=8)
    TB = where(F & (WhoseTB==1), salts.borate_U74(Sal), TB)
    TB = where(F & (WhoseTB==2), salts.borate_LKB10(Sal), TB)
    return TB

def concs_TCa(Sal, WhichKs):
    """Calculate total calcium from salinity for the given options."""
    F = (WhichKs==6) | (WhichKs==7) # GEOSECS values
    TCa = where(F, salts.calcium_C65(Sal), salts.calcium_RT67(Sal))
    return TCa

def concentrations(Sal, WhichKs, WhoseTB):
    """Estimate total concentrations of borate, fluoride and sulfate from
    salinity.

    Inputs must first be conditioned with inputs().

    Based on a subset of Constants, version 04.01, 10-13-97, by Ernie Lewis.
    """
    TB = concs_TB(Sal, WhichKs, WhoseTB)
    TF = salts.fluoride_R65(Sal)
    TS = salts.sulfate_MR66(Sal)
    TCa = concs_TCa(Sal, WhichKs)
    # Return equilibrating results as a dict for stability
    return TCa, {
        'TB': TB,
        'TF': TF,
        'TSO4': TS,
    }
