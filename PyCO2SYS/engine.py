# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
from autograd.numpy import array, full, size, unique

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
