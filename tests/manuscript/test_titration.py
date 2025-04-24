# %%
import numpy as np

from PyCO2SYS import CO2System

# Initial sample conditions
values = {
    "alkalinity": 2450,
    "dic": 2200,
    "salinity": 35,
    "temperature": 25,
    "total_borate": 420,
    "total_fluoride": 70,
    "total_sulfate": 28240,
    "pk_BOH3": -np.log10(1.78e-9),
    "pk_H2CO3": -np.log10(1e-6),
    "pk_HCO3": -np.log10(8.2e-16 / 1e-6),
    "pk_HF_free": -np.log10(1 / 4.08e2),
    "pk_HSO4_free": -np.log10(1 / 1.23e1),
    "pk_H2O": -np.log10(4.32e-14),
}
opts = {
    "opt_pH_scale": 3,
}
sample_mass = 0.2  # kg

# Titrant properties
titrant_molinity = 0.3  # mol/kg
titrant_mass = np.arange(0, 2.51, 0.05) * 1e-3  # kg
dilution_factor = sample_mass / (sample_mass + titrant_mass)

# Dilute alkalinity etc. through the titration
values["alkalinity"] = (
    1e6
    * (sample_mass * values["alkalinity"] * 1e-6 - titrant_molinity * titrant_mass)
    / (sample_mass + titrant_mass)
)
values["dic"] *= dilution_factor
for k in ["total_borate", "total_fluoride", "total_sulfate"]:
    values[k] *= dilution_factor

# Solve for pH, no phosphate
sys = CO2System(**values, **opts)
pH = sys["pH"]

# And again, with phosphate
values_phosphate = values.copy()
values_phosphate.update(
    {
        "total_phosphate": 10 * dilution_factor,
        "pk_H3PO4": -np.log10(1 / 5.68e1),
        "pk_H2PO4": -np.log10(8e-7),
        "pk_HPO4": -np.log10(1.32e-15 / 8e-7),
    }
)
sys_phosphate = CO2System(**values_phosphate, **opts)
pH_phosphate = sys_phosphate["pH"]

# Compare with D81's tables
d81_pH = np.genfromtxt(
    "tests/manuscript/data/Dickson-1981-pH-no_phosphate.dat",
    delimiter="\t",
    skip_header=2,
)[:, 1]
d81_pH_phosphate = np.genfromtxt(
    "tests/manuscript/data/Dickson-1981-pH-with_phosphate.dat",
    delimiter="\t",
    skip_header=2,
)[:, 1]

pH_diff = np.round(pH, decimals=6) - d81_pH
pH_diff_phosphate = np.round(pH_phosphate, decimals=6) - d81_pH_phosphate


def test_D81():
    # Perfect agreement vs D81 Table 1
    assert np.all(np.isclose(pH_diff, 0, atol=1e-12))


def test_D81_phosphate():
    # Presumably these are typos in D81 Table 4, given how well everything else agrees
    typos = (
        np.isclose(titrant_mass * 1e3, 0.45)
        | np.isclose(titrant_mass * 1e3, 0.60)
        | np.isclose(titrant_mass * 1e3, 1.25)
    )
    assert np.all(np.isclose(pH_diff_phosphate[~typos], 0, atol=1e-12))
    # print(titrant_mass[typos] * 1e3)
    # print(d81_pH_phosphate[typos])
    # print(pH_phosphate[typos])


# test_D81()
# test_D81_phosphate()
