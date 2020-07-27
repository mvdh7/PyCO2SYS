# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Carbonate system solving in N dimensions."""

from autograd import numpy as np
from .. import equilibria, salts, solve

# Define function input keys that should be converted to floats
input_floats = {
    "par1",
    "par2",
    "salinity",
    "temperature",
    "temperature_out",
    "pressure",
    "pressure_out",
    "total_ammonia",
    "total_phosphate",
    "total_silicate",
    "total_sulfate",
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
    temperature_out=None,
    pressure=0,
    pressure_out=None,
    total_ammonia=0,
    total_borate=None,
    total_calcium=None,
    total_fluoride=None,
    total_phosphate=0,
    total_silicate=0,
    total_sulfate=None,
    total_sulfide=0,
    fugacity_factor=None,
    fugacity_factor_out=None,
    gas_constant=None,
    gas_constant_out=None,
    k_ammonia=None,
    k_ammonia_out=None,
    k_borate=None,
    k_borate_out=None,
    k_bisulfate=None,
    k_bisulfate_out=None,
    k_carbon_dioxide=None,
    k_carbon_dioxide_out=None,
    k_carbonic_1=None,
    k_carbonic_1_out=None,
    k_carbonic_2=None,
    k_carbonic_2_out=None,
    k_fluoride=None,
    k_fluoride_out=None,
    k_phosphate_1=None,
    k_phosphate_1_out=None,
    k_phosphate_2=None,
    k_phosphate_2_out=None,
    k_phosphate_3=None,
    k_phosphate_3_out=None,
    k_silicate=None,
    k_silicate_out=None,
    k_sulfide=None,
    k_sulfide_out=None,
    k_water=None,
    k_water_out=None,
    borate_opt=2,
    bisulfate_opt=1,
    carbonic_opt=16,
    fluoride_opt=1,
    gas_constant_opt=3,
    pH_scale_opt=1,
    buffers_mode="auto",
):
    """Efficiently run CO2SYS with n-dimensional inputs allowed."""
    inputs = condition(locals())
    # Prepare totals dict
    totals_optional = {
        "total_borate": "TB",
        "total_calcium": "TCa",
        "total_fluoride": "TF",
        "total_sulfate": "TSO4",
    }
    if np.any(np.isin(list(inputs.keys()), list(totals_optional.keys()))):
        totals = {
            totals_optional[k]: v * 1e-6
            for k, v in inputs.items()
            if k in totals_optional
        }
    else:
        totals = None
    totals = salts.assemble(
        inputs["salinity"],
        inputs["total_silicate"],
        inputs["total_phosphate"],
        inputs["total_ammonia"],
        inputs["total_sulfide"],
        inputs["carbonic_opt"],
        inputs["borate_opt"],
        totals=totals,
    )
    # Prepare equilibrium constants dict (input conditions)
    k_constants_optional = {
        "fugacity_factor": "FugFac",
        "gas_constant": "RGas",
        "k_ammonia": "KNH3",
        "k_borate": "KB",
        "k_bisulfate": "KSO4",
        "k_carbon_dioxide": "K0",
        "k_carbonic_1": "K1",
        "k_carbonic_2": "K2",
        "k_fluoride": "KF",
        "k_phosphate_1": "KP1",
        "k_phosphate_2": "KP2",
        "k_phosphate_3": "KP3",
        "k_silicate": "KSi",
        "k_sulfide": "KH2S",
        "k_water": "KW",
    }
    if np.any(np.isin(list(inputs.keys()), list(k_constants_optional.keys()))):
        k_constants_in = {
            k_constants_optional[k]: v
            for k, v in inputs.items()
            if k in k_constants_optional
        }
    else:
        k_constants_in = None
    k_constants_in = equilibria.assemble(
        inputs["temperature"],
        inputs["pressure"],
        totals,
        inputs["pH_scale_opt"],
        inputs["carbonic_opt"],
        inputs["bisulfate_opt"],
        inputs["fluoride_opt"],
        inputs["gas_constant_opt"],
        Ks=k_constants_in,
    )
    # Solve the core marine carbonate system at input conditions
    core_in = solve.core(
        inputs["par1"],
        inputs["par2"],
        inputs["par1_type"],
        inputs["par2_type"],
        totals,
        k_constants_in,
        convert_units=True,
    )
    # Calculate the rest at input conditions
    others_in = solve.others(
        core_in,
        inputs["temperature"],
        inputs["pressure"],
        totals,
        k_constants_in,
        inputs["pH_scale_opt"],
        inputs["carbonic_opt"],
        inputs["buffers_mode"],
    )
    # If requested, solve the core marine carbonate system at output conditions
    if "pressure_out" in inputs.keys() or "temperature_out" in inputs.keys():
        # Make sure we've got output values for both temperature and pressure
        if "pressure_out" in inputs.keys():
            if "temperature_out" not in inputs.keys():
                inputs["temperature_out"] = inputs["temperature"]
        if "temperature_out" in inputs.keys():
            if "pressure_out" not in inputs.keys():
                inputs["pressure_out"] = inputs["pressure"]
        # Prepare equilibrium constants dict (output conditions)
        k_constants_optional_out = {
            "{}_out".format(k): v for k, v in k_constants_optional.items()
        }
        if np.any(np.isin(list(inputs.keys()), k_constants_optional_out)):
            k_constants_out = {
                k_constants_optional_out[k]: v
                for k, v in inputs.items()
                if k in k_constants_optional_out
            }
        else:
            k_constants_out = None
        k_constants_out = equilibria.assemble(
            inputs["temperature_out"],
            inputs["pressure_out"],
            totals,
            inputs["pH_scale_opt"],
            inputs["carbonic_opt"],
            inputs["bisulfate_opt"],
            inputs["fluoride_opt"],
            inputs["gas_constant_opt"],
            Ks=k_constants_out,
        )
        # Solve the core marine carbonate system at output conditions
        core_out = solve.core(
            core_in["TA"],
            core_in["TC"],
            1,
            2,
            totals,
            k_constants_out,
            convert_units=False,
        )
        # Calculate the rest at output conditions
        others_out = solve.others(
            core_out,
            inputs["temperature_out"],
            inputs["pressure_out"],
            totals,
            k_constants_out,
            inputs["pH_scale_opt"],
            inputs["carbonic_opt"],
            inputs["buffers_mode"],
        )
    else:
        core_out = None
        others_out = None
    return inputs, core_in, others_in, core_out, others_out
