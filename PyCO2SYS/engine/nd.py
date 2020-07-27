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


def condition(args, to_shape=None):
    """Condition n-d args for PyCO2SYS.
    
    If NumPy can broadcast the args together, they are a valid combination, and they
    will be combined following NumPy broadcasting rules.

    All array-like args will be broadcast into the same shape.
    Any scalar args will be left as scalars.
    """
    try:  # check all args can be broadcast together
        args = {k: v for k, v in args.items() if v is not None}
        args_broadcast = np.broadcast(*args.values())
        if to_shape is not None:
            try:  # check args can be broadcast to to_shape, if provided
                np.broadcast(np.ones(to_shape), np.ones(args_broadcast.shape))
                args_broadcast_shape = to_shape
            except ValueError:
                print("PyCO2SYS error: args are not broadcastable to to_shape.")
                return
        else:
            args_broadcast_shape = args_broadcast.shape
        # Broadcast the non-scalar args to a consistent shape
        args_conditioned = {
            k: np.broadcast_to(v, args_broadcast_shape) if not np.isscalar(v) else v
            for k, v in args.items()
        }
        # Convert to float, where needed
        args_conditioned = {
            k: np.float64(v) if k in input_floats else v
            for k, v in args_conditioned.items()
        }
    except ValueError:
        print("PyCO2SYS error: input shapes cannot be broadcast together.")
        return
    return args_conditioned


def _get_in_out(core, others, k_constants, suffix=""):
    return {
        "pH{}".format(suffix): core["PH"],
        "pCO2{}".format(suffix): core["PC"] * 1e6,
        "fCO2{}".format(suffix): core["FC"] * 1e6,
        "bicarbonate{}".format(suffix): core["HCO3"] * 1e6,
        "carbonate{}".format(suffix): core["CARB"] * 1e6,
        "aqueous_CO2{}".format(suffix): core["CO2"] * 1e6,
        "alkalinity_borate{}".format(suffix): others["BAlk"] * 1e6,
        "hydroxide{}".format(suffix): others["OH"] * 1e6,
        "alkalinity_phosphate{}".format(suffix): others["PAlk"] * 1e6,
        "alkalinity_silicate{}".format(suffix): others["SiAlk"] * 1e6,
        "alkalinity_ammonia{}".format(suffix): others["NH3Alk"] * 1e6,
        "alkalinity_sulfide{}".format(suffix): others["H2SAlk"] * 1e6,
        "hydrogen_free{}".format(suffix): others["Hfree"] * 1e6,
        "revelle_factor{}".format(suffix): others["Revelle"],
        "saturation_calcite{}".format(suffix): others["OmegaCa"],
        "saturation_aragonite{}".format(suffix): others["OmegaAr"],
        "xCO2{}".format(suffix): others["xCO2dry"] * 1e6,
        "pH_total{}".format(suffix): others["pHT"],
        "pH_sws{}".format(suffix): others["pHS"],
        "pH_free{}".format(suffix): others["pHF"],
        "pH_nbs{}".format(suffix): others["pHN"],
        "k_carbon_dioxide{}".format(suffix): k_constants["K0"],
        "k_carbonic_1{}".format(suffix): k_constants["K1"],
        "k_carbonic_2{}".format(suffix): k_constants["K2"],
        "k_water{}".format(suffix): k_constants["KW"],
        "k_borate{}".format(suffix): k_constants["KB"],
        "k_bisulfate{}".format(suffix): k_constants["KSO4"],
        "k_fluoride{}".format(suffix): k_constants["KF"],
        "k_phosphoric_1{}".format(suffix): k_constants["KP1"],
        "k_phosphoric_2{}".format(suffix): k_constants["KP2"],
        "k_phosphoric_3{}".format(suffix): k_constants["KP3"],
        "k_silicate{}".format(suffix): k_constants["KSi"],
        "k_ammonia{}".format(suffix): k_constants["KNH3"],
        "k_sulfide{}".format(suffix): k_constants["KH2S"],
        "gamma_dic{}".format(suffix): others["gammaTC"],
        "beta_dic{}".format(suffix): others["betaTC"],
        "omega_dic{}".format(suffix): others["omegaTC"],
        "gamma_alk{}".format(suffix): others["gammaTA"],
        "beta_alk{}".format(suffix): others["betaTA"],
        "omega_alk{}".format(suffix): others["omegaTA"],
        "isocapnic_quotient{}".format(suffix): others["isoQ"],
        "isocapnic_quotient_approx{}".format(suffix): others["isoQx"],
        "psi{}".format(suffix): others["psi"],
        "substrate_inhibitor_ratio{}".format(suffix): others["SIR"],
        "fugacity_factor{}".format(suffix): k_constants["FugFac"],
        "fH{}".format(suffix): k_constants["fH"],
    }


def _get_results_dict(
    args,
    totals,
    core_in,
    others_in,
    k_constants_in,
    core_out,
    others_out,
    k_constants_out,
):
    """Assemble the results dict for CO2SYS."""
    results = {
        "par1": args["par1"],
        "par2": args["par2"],
        "salinity": totals["Sal"],
        "temperature": args["temperature"],
        "pressure": args["pressure"],
        "total_ammonia": totals["TNH3"] * 1e6,
        "total_borate": totals["TB"] * 1e6,
        "total_calcium": totals["TCa"] * 1e6,
        "total_fluoride": totals["TF"] * 1e6,
        "total_phosphate": totals["TPO4"] * 1e6,
        "total_silicate": totals["TSi"] * 1e6,
        "total_sulfate": totals["TSO4"] * 1e6,
        "total_sulfide": totals["TH2S"] * 1e6,
        "PengCorrection": totals["PengCorrection"] * 1e6,
        "gas_constant": k_constants_in["RGas"],
        "alkalinity": core_in["TA"] * 1e6,
        "dic": core_in["TC"] * 1e6,
    }
    results.update(_get_in_out(core_in, others_in, k_constants_in, suffix=""))
    if core_out is not None:
        results.update(
            {
                "temperature_out": args["temperature_out"],
                "pressure_out": args["pressure_out"],
            }
        )
        results.update(
            _get_in_out(core_out, others_out, k_constants_out, suffix="_out")
        )
    return results


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
    """Efficiently run CO2SYS with n-dimensional args allowed."""
    args = condition(locals())
    # Prepare totals dict
    totals_optional = {
        "total_borate": "TB",
        "total_calcium": "TCa",
        "total_fluoride": "TF",
        "total_sulfate": "TSO4",
    }
    if np.any(np.isin(list(args.keys()), list(totals_optional.keys()))):
        totals = {
            totals_optional[k]: v * 1e-6
            for k, v in args.items()
            if k in totals_optional
        }
    else:
        totals = None
    totals = salts.assemble(
        args["salinity"],
        args["total_silicate"],
        args["total_phosphate"],
        args["total_ammonia"],
        args["total_sulfide"],
        args["carbonic_opt"],
        args["borate_opt"],
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
    if np.any(np.isin(list(args.keys()), list(k_constants_optional.keys()))):
        k_constants_in = {
            k_constants_optional[k]: v
            for k, v in args.items()
            if k in k_constants_optional
        }
    else:
        k_constants_in = None
    k_constants_in = equilibria.assemble(
        args["temperature"],
        args["pressure"],
        totals,
        args["pH_scale_opt"],
        args["carbonic_opt"],
        args["bisulfate_opt"],
        args["fluoride_opt"],
        args["gas_constant_opt"],
        Ks=k_constants_in,
    )
    # Solve the core marine carbonate system at input conditions
    core_in = solve.core(
        args["par1"],
        args["par2"],
        args["par1_type"],
        args["par2_type"],
        totals,
        k_constants_in,
        convert_units=True,
    )
    # Calculate the rest at input conditions
    others_in = solve.others(
        core_in,
        args["temperature"],
        args["pressure"],
        totals,
        k_constants_in,
        args["pH_scale_opt"],
        args["carbonic_opt"],
        args["buffers_mode"],
    )
    # If requested, solve the core marine carbonate system at output conditions
    if "pressure_out" in args.keys() or "temperature_out" in args.keys():
        # Make sure we've got output values for both temperature and pressure
        if "pressure_out" in args.keys():
            if "temperature_out" not in args.keys():
                args["temperature_out"] = args["temperature"]
        if "temperature_out" in args.keys():
            if "pressure_out" not in args.keys():
                args["pressure_out"] = args["pressure"]
        # Prepare equilibrium constants dict (output conditions)
        k_constants_optional_out = {
            "{}_out".format(k): v for k, v in k_constants_optional.items()
        }
        if np.any(np.isin(list(args.keys()), k_constants_optional_out)):
            k_constants_out = {
                k_constants_optional_out[k]: v
                for k, v in args.items()
                if k in k_constants_optional_out
            }
        else:
            k_constants_out = None
        k_constants_out = equilibria.assemble(
            args["temperature_out"],
            args["pressure_out"],
            totals,
            args["pH_scale_opt"],
            args["carbonic_opt"],
            args["bisulfate_opt"],
            args["fluoride_opt"],
            args["gas_constant_opt"],
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
            args["temperature_out"],
            args["pressure_out"],
            totals,
            k_constants_out,
            args["pH_scale_opt"],
            args["carbonic_opt"],
            args["buffers_mode"],
        )
    else:
        core_out = None
        others_out = None
        k_constants_out = None
    return _get_results_dict(
        args,
        totals,
        core_in,
        others_in,
        k_constants_in,
        core_out,
        others_out,
        k_constants_out,
    )
