# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Carbonate system solving in N dimensions."""

from autograd import numpy as np
from .. import equilibria, salts

# Define function input keys that should be converted to floats
input_floats = {
    "par1",
    "par2",
    "salinity",
    "temperature",
    "temperature_output",
    "pressure",
    "pressure_output",
    "total_ammonia",
    "total_phosphate",
    "total_silicate",
    "total_sulfide",
}


def condition(inputs, to_shape=None):
    """Condition n-d inputs for PyCO2SYS.
    
    If NumPy can broadcast the inputs together, they are a valid combination, and they
    will be combined following NumPy broadcasting rules.

    All array-like inputs will be broadcast into the same shape.
    Any scalar inputs will be left as scalars.
    """
    try:  # check all inputs can be broadcast together
        inputs = {k: v for k, v in inputs.items() if v is not None}
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


def CO2SYS(
    par1,
    par2,
    par1_type,
    par2_type,
    salinity=35,
    temperature=25,
    temperature_output=None,
    pressure=0,
    pressure_output=None,
    total_ammonia=0,
    total_phosphate=0,
    total_silicate=0,
    total_sulfide=0,
    borate_opt=2,
    bisulfate_opt=1,
    carbonic_opt=16,
    fluoride_opt=1,
    gas_constant_opt=3,
    pH_scale_opt=1,
    buffers_mode="auto",
):
    inputs = condition(locals())
    totals = salts.assemble(
        inputs["salinity"],
        inputs["total_silicate"],
        inputs["total_phosphate"],
        inputs["total_ammonia"],
        inputs["total_sulfide"],
        inputs["carbonic_opt"],
        inputs["borate_opt"],
    )
    kconstants = equilibria.assemble(
        inputs["temperature"],
        inputs["pressure"],
        totals,
        inputs["pH_scale_opt"],
        inputs["carbonic_opt"],
        inputs["bisulfate_opt"],
        inputs["fluoride_opt"],
        inputs["gas_constant_opt"],
    )
    return totals, kconstants
