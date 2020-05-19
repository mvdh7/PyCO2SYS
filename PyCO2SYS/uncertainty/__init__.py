# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Uncertainty propagation."""

from scipy.misc import derivative
from autograd.numpy import array, isin
from autograd.numpy import all as np_all
from . import automatic
from .. import engine

__all__ = ["automatic"]


def co2inputs(co2dict, grads_of, grads_wrt, dx=1e-8, use_explicit=True, verbose=True):
    """Get derivatives of `co2dict` values w.r.t. the main function inputs.

    `co2dict` is output by `PyCO2SYS.CO2SYS`.
    `grads_of` is a list of keys from `co2dict` that you want to calculate the
    derivatives of, or a single key, or `"all"`.
    `grads_wrt` is a list of `PyCO2SYS.CO2SYS` input variable names that you want to
    calculate the derivatives with respect to, or a single name, or `"all"`.
    """

    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    inputs_wrt = [
        "PAR1",
        "PAR2",
        "SAL",
        "TEMPIN",
        "TEMPOUT",
        "PRESIN",
        "PRESOUT",
        "SI",
        "PO4",
        "NH3",
        "H2S",
    ]
    if isinstance(grads_wrt, str):
        assert (grads_wrt == "all") or grads_of in inputs_wrt
        if grads_wrt == "all":
            grads_wrt = inputs_wrt
        else:
            grads_wrt = [grads_wrt]
    assert np_all(isin(list(grads_wrt), inputs_wrt,)), "Invalid `grads_wrt` requested."
    if isinstance(grads_of, str):
        assert (grads_of == "all") or grads_of in engine.gradables
        if grads_of == "all":
            grads_of = engine.gradables
        else:
            grads_of = [grads_of]
    assert np_all(isin(grads_of, engine.gradables,)), "Invalid `grads_of` requested."
    co2args = {
        arg: co2dict[arg]
        for arg in [
            "PAR1",
            "PAR2",
            "PAR1TYPE",
            "PAR2TYPE",
            "SAL",
            "TEMPIN",
            "TEMPOUT",
            "PRESIN",
            "PRESOUT",
            "SI",
            "PO4",
            "NH3",
            "H2S",
            "pHSCALEIN",
            "K1K2CONSTANTS",
            "KSO4CONSTANT",
            "KFCONSTANT",
            "BORON",
            "buffers_mode",
            "KSO4CONSTANTS",
        ]
    }
    # Get the gradients
    co2deriv = {}
    # Define gradients that we have explicit methods for, if requested
    if use_explicit:
        # Automatic derivatives for PAR1/PAR2 propagation into core MCS
        pars_requested = [grad for grad in grads_wrt if grad in ["PAR1", "PAR2"]]
        p1p2u = automatic.pars2core(co2dict, pars_requested)
        for p in pars_requested:
            co2deriv[p] = {}
            for k, v in p1p2u[p].items():
                co2deriv[p][k] = v
    # Get central difference derivatives for the rest
    for grad in grads_wrt:
        printv("Computing derivatives w.r.t. {}...".format(grad))
        if grad not in co2deriv:
            co2deriv[grad] = {}
        for output in grads_of:
            if output not in co2deriv[grad]:

                def kfunc(v, co2args):
                    co2args[grad] = v
                    return engine._CO2SYS(**co2args)[output]

                co2deriv[grad][output] = derivative(
                    kfunc, co2args[grad], dx=dx, args=[co2args]
                )
    # Convert derivatives arrays to Jacobians
    co2jacs = {}
    for output in grads_of:
        co2jacs[output] = array([co2deriv[grad][output] for grad in grads_wrt]).T
    return co2jacs, grads_wrt
