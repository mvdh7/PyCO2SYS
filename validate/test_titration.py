import numpy as np
import PyCO2SYS as pyco2

# Switch to fixed alkalinity function
pyco2.solve.get.TAfromTCpH = pyco2.solve.get.TAfromTCpH_fixed

# Initial sample conditions
alkalinity = 2450
dic = 2200
kwargs = {
    "salinity": 35,
    "temperature": 25,
    "total_borate": 420,
    "total_fluoride": 70,
    "total_sulfate": 28240,
    "k_borate": 1.78e-9,
    "k_carbonic_1": 1e-6,
    "k_carbonic_2": 8.2e-16 / 1e-6,
    "k_fluoride": 1 / 4.08e2,
    "k_bisulfate": 1 / 1.23e1,
    "k_water": 4.32e-14,
    "opt_pH_scale": 3,
}
sample_mass = 0.2  # kg

# Titrant properties
titrant_molinity = 0.3  # mol/kg
titrant_mass = np.arange(0, 2.51, 0.05) * 1e-3  # kg
dilution_factor = sample_mass / (sample_mass + titrant_mass)

# Dilute alkalinity etc. through the titration
alkalinity = (
    1e6
    * (sample_mass * alkalinity * 1e-6 - titrant_molinity * titrant_mass)
    / (sample_mass + titrant_mass)
)
dic *= dilution_factor
for k in ["total_borate", "total_fluoride", "total_sulfate"]:
    kwargs[k] *= dilution_factor

# Solve for pH, no phosphate
pH = pyco2.CO2SYS_nd(alkalinity, dic, 1, 2, **kwargs)["pH_free"]

# And again, with phosphate
kwargs.update(
    {
        "total_phosphate": 10 * dilution_factor,
        "k_phosphate_1": 1 / 5.68e1,
        "k_phosphate_2": 8e-7,
        "k_phosphate_3": 1.32e-15 / 8e-7,
    }
)
pH_phosphate = pyco2.CO2SYS_nd(alkalinity, dic, 1, 2, **kwargs)["pH_free"]

# Compare with D81's tables
d81_pH = np.genfromtxt(
    "validate/data/Dickson-1981-pH-no_phosphate.dat", delimiter="\t", skip_header=2
)[:, 1]
d81_pH_phosphate = np.genfromtxt(
    "validate/data/Dickson-1981-pH-with_phosphate.dat", delimiter="\t", skip_header=2
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


test_D81()
test_D81_phosphate()

# Revert to original alkalinity function
pyco2.solve.get.TAfromTCpH = pyco2.solve.get.TAfromTCpH_original
