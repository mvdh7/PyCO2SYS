import numpy as np, PyCO2SYS as pyco2

# Calculate initial results
kwargs = {
    "salinity": 32,
    "k_carbonic_1": 1e-6,
}
par1 = np.linspace(2000, 2100, 11)
par2 = np.vstack(np.linspace(2300, 2400, 21))
par1_type = 2
par2_type = 1
results = pyco2.sys(par1, par2, par1_type, par2_type, **kwargs)

# Get gradients
grads_of = ["pH", "k_carbonic_1"]
grads_wrt = ["par1", "par2", "k_carbonic_1", "pk_carbonic_1", "temperature"]
CO2SYS_derivs, dxs = pyco2.uncertainty.forward_nd(
    results, grads_of, grads_wrt, **kwargs
)

# Do independent uncertainty propagation
uncertainties_into = ["pH", "isocapnic_quotient", "dic"]
uncertainties_from = {
    "par1": 2,
    "par2": 2,
    "pk_carbonic_1": 0.02,
}
uncertainties, components = pyco2.uncertainty.propagate_nd(
    results,
    uncertainties_into,
    uncertainties_from,
    **kwargs,
)


# Compare with CO2SYS v1.4 API
par1g = np.broadcast_to(par1, np.broadcast(par1, par2).shape)
par2g = np.broadcast_to(par2, np.broadcast(par1, par2).shape)
co2dict = pyco2.CO2SYS(
    par1g,
    par2g,
    par1_type,
    par2_type,
    kwargs["salinity"],
    results["temperature"],
    results["temperature"],
    results["pressure"],
    results["pressure"],
    results["total_silicate"],
    results["total_phosphate"],
    results["opt_pH_scale"],
    results["opt_k_carbonic"],
    pyco2.convert.options_new2old(
        results["opt_k_bisulfate"], results["opt_total_borate"]
    ),
    WhichR=3,
    equilibria_in={"K1": kwargs["k_carbonic_1"]},
)
uncert_old, components_old = pyco2.uncertainty.propagate(
    co2dict,
    ["pHin", "isoQin", "TCO2"],
    {
        "PAR1": 2,
        "PAR2": 2,
        "pK1input": 0.02,
    },
    equilibria_in={"K1": kwargs["k_carbonic_1"]},
)


def test_old_new():
    assert np.allclose(co2dict["pHin"], results["pH"].ravel())


def test_uncertainty_old_new():
    assert np.allclose(uncert_old["pHin"], uncertainties["pH"].ravel())


# test_old_new()
# test_uncertainty_old_new()

# # Try out the standard uncertainties of OEDG18
# uncertainties_pk, components_pk = pyco2.uncertainty.propagate_nd(
#     results, uncertainties_into, pyco2.uncertainty.pKs_OEDG18, **kwargs
# )
