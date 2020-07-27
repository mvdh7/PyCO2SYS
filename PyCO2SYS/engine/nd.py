# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Carbonate system solving in N dimensions."""

from autograd import numpy as np

# Define function input keys that should be converted to floats
input_floats = {
    "SAL",
    "TEMPIN",
    "TEMPOUT",
    "PRESIN",
    "PRESOUT",
    "SI",
    "PO4",
    "NH3",
    "H2S",
    "PAR1",
    "PAR2",
    "TA",
    "TC",
    "PH",
    "PC",
    "FC",
    "CARB",
    "HCO3",
    "CO2",
    "TempC",
    "Pdbar",
    "TSi",
    "TPO4",
    "TNH3",
    "TH2S",
    "TB",
    "TF",
    "TS",
    "TCa",
    "K0",
    "K1",
    "K2",
    "KW",
    "KB",
    "KF",
    "KS",
    "KP1",
    "KP2",
    "KP3",
    "KSi",
    "KNH3",
    "KH2S",
    "RGas",
}


def condition(inputs, to_shape=None):
    """Condition n-d inputs for PyCO2SYS.
    
    If NumPy can broadcast the inputs together, they are a valid combination, and they
    will be combined following NumPy broadcasting rules.

    All array-like inputs will be broadcast into the same shape.
    Any scalar inputs will be left as scalars.
    """
    try:  # check all inputs can be broadcast together
        inputs_broadcast = np.broadcast(*inputs.values())
        if to_shape is not None:
            try:  # check inputs can be broadcast to to_shape, if provided
                np.broadcast(np.ones(to_shape), np.ones(inputs_broadcast.shape))
                inputs_broadcast_shape = to_shape
            except ValueError:
                print("PyCO2SYS error: inputs are not broadcastable to to_shape.")
                return
        else:
            inputs_broadcast_shape = inputs_broadcast.shape
        # Broadcast the non-scalar inputs to a consistent shape
        inputs_conditioned = {
            k: np.broadcast_to(v, inputs_broadcast_shape) if not np.isscalar(v) else v
            for k, v in inputs.items()
        }
        # Convert to float, where needed
        inputs_conditioned = {
            k: np.float64(v) if k in input_floats else v
            for k, v in inputs_conditioned.items()
        }
    except ValueError:
        print("PyCO2SYS error: input shapes cannot be broadcast together.")
        return
    return inputs_conditioned
