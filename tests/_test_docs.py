# %%
import PyCO2SYS as pyco2


# Set up the CO2System
co2s = pyco2.sys(alkalinity=2250, dic=2100)

# Solve for and return pH
pH = co2s["pH"]
# Solve for and return pH and pCO2
results = co2s[["pH", "pCO2"]]
co2s.solve(parameters=None)
co2s = (
    pyco2.sys(alkalinity=2300, pH=8.1, temperature=25)
    .set_u(alkalinity=2, pH=0.01, temperature=0.01)
    .adjust(temperature=12, pressure=1200)
    .solve(["saturation_aragonite", "fCO2"])
    .prop("saturation_aragonite")
)
