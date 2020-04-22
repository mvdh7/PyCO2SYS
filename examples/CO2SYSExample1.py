import numpy as np
from matplotlib import pyplot as plt
from PyCO2SYS import CO2SYS

# This is an example of the use of CO2SYS. Have a look at the code
# Steven van Heuven. svheuven@gmail.com
#
# Converted to Python by Matthew P. Humphreys [2020-01-31]

print(" ")
print("This is an example of the use of CO2SYS.m")
print("that uses its ability to process vectors of data.")
print(" ")
print("We will generate a figure that shows the sensitivity of pH and pCO2")
print(" to changes in DIC, while keeping everything else constant")
print(" ")
print("(Addional info: alk=2400, si=50, po4=2, dissociation constats: Mehrbach Refit)")
print(" ")

par1type = 1  # The first parameter supplied is of type "1", which is "alkalinity"
par1 = 2400  # value of the first parameter
par2type = 2  # The first parameter supplied is of type "1", which is "DIC"
par2 = np.arange(2100, 2300, 5)
# ^ value of the second parameter, which is a long vector of different DIC's!
sal = 35  # Salinity of the sample
tempin = 10  # Temperature at input conditions
presin = 0  # Pressure    at input conditions
tempout = 0  # Temperature at output conditions - doesn't matter in this example
presout = 0  # Pressure    at output conditions - doesn't matter in this example
sil = 50  # Concentration of silicate  in the sample (in umol/kg)
po4 = 2  # Concentration of phosphate in the sample (in umol/kg)
pHscale = 1  # pH scale at which the input pH is reported ("1" means "Total Scale")  - doesn't matter in this example
k1k2c = 4  # Choice of H2CO3 and HCO3- dissociation constants K1 and K2 ("4" means "Mehrbach refit")
kso4c = 1  # Choice of HSO4- dissociation constants KSO4 ("1" means "Dickson")

# Do the calculation. See CO2SYS's help for syntax and output format
CO2dict = CO2SYS(
    par1,
    par2,
    par1type,
    par2type,
    sal,
    tempin,
    tempout,
    presin,
    presout,
    sil,
    po4,
    pHscale,
    k1k2c,
    kso4c,
)

# Draw the figure
fig, ax = plt.subplots(2, 1, figsize=(6, 10))

# The calculated pCO2's are in the field 'pCO2in' of the output DICT of CO2SYS
ax[0].plot(par2, CO2dict["pCO2in"], c="r", marker="o")
# marker=(:circle, stroke(:red)))
ax[0].set_xlabel("DIC")
ax[0].set_ylabel("pCO2 [uatm]")

# The calculated pH's are in the field 'pHin' of the output DICT of CO2SYS
ax[1].plot(par2, CO2dict["pHin"], c="r", marker="o")
ax[1].set_xlabel("DIC")
ax[1].set_ylabel("pH")

print("DONE!")
print(" ")
print("See CO2SYSExample1.py to see what the syntax for this calculation was.")
print(" ")
