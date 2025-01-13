# %%
import PyCO2SYS as pyco2

# Set starting conditions
start = pyco2.sys(
    alkalinity=2280,
    dic=2000,
    total_silicate=0,
)
start.solve("pCO2")
si_factor = 1

# Increase alkalinity and silicate by adding "ground rock"
rock_alkalinity = start.dic * 2
add_rock = pyco2.sys(
    alkalinity=start.alkalinity + rock_alkalinity,
    dic=start.dic,
    total_silicate=(start.total_silicate + rock_alkalinity / 2) * si_factor,
)
add_rock.solve("pCO2")

# Equilibrate to original pCO2
add_rock_eq = pyco2.sys(
    alkalinity=add_rock.alkalinity,
    pCO2=start.pCO2,
    total_silicate=add_rock.total_silicate,
)
add_rock_eq.solve("dic")

# Precipitate CaCO3 to return to original alkalinity
remove_CaCO3 = pyco2.sys(
    alkalinity=start.alkalinity,
    dic=add_rock_eq.dic - rock_alkalinity / 2,
    total_silicate=add_rock.total_silicate,
)
remove_CaCO3.solve("pCO2")

# Equilibrate to original pCO2
remove_CaCO3_eq = pyco2.sys(
    alkalinity=start.alkalinity,
    pCO2=start.pCO2,
    total_silicate=remove_CaCO3.total_silicate,
)
remove_CaCO3_eq.solve("dic")

# Calculate CO2 taken out of atmosphere in each step
add_rock_CO2 = 0.0
add_rock_eq_CO2 = add_rock_eq.dic - add_rock.dic
remove_CaCO3_CO2 = 0.0
remove_CaCO3_eq_CO2 = remove_CaCO3_eq.dic - remove_CaCO3.dic
total_uptake = add_rock_eq_CO2 + remove_CaCO3_eq_CO2

print(add_rock_CO2, add_rock_eq_CO2, remove_CaCO3_CO2, remove_CaCO3_eq_CO2)
print(total_uptake)
print("")
print(
    start.dic,
    add_rock.dic,
    add_rock_eq.dic,
    remove_CaCO3.dic,
    remove_CaCO3_eq.dic,
)
print(
    start.pCO2,
    add_rock.pCO2,
    add_rock_eq.pCO2,
    remove_CaCO3.pCO2,
    remove_CaCO3_eq.pCO2,
)
print(
    start.pH,
    add_rock.pH,
    add_rock_eq.pH,
    remove_CaCO3.pH,
    remove_CaCO3_eq.pH,
)
