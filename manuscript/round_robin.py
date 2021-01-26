import numpy as np
import PyCO2SYS as pyco2

# Define round-robin test conditions
par1 = 2300  # parameter 1, here total alkalinity in μmol/kg-sw
par2 = 2100  # parameter 2, here dissolved inorganic carbon in μmol/kg-sw
par1_type = 1  # "parameter 1 is total alkalinity"
par2_type = 2  # "parameter 2 is DIC"
kwargs = dict(
    salinity=33,  # practical salinity
    temperature=22,  # temperature in °C
    pressure=1234,  # pressure in dbar
    total_silicate=10,  # total silicate in μmol/kg-sw
    total_phosphate=1,  # total phosphate in μmol/kg-sw
    total_ammonia=2,  # total ammonia in μmol/kg-sw
    total_sulfide=3,  # total sulfide in μmol/kg-sw
    total_alpha=5,
    k_alpha=1e-4,
    total_beta=6,
    k_beta=1e-8,
    buffers_mode="none",  # don't bother calculating buffer factors
)

# Solve the system initially
results = pyco2.sys(par1, par2, par1_type, par2_type, **kwargs)


# Define parameter types and names in output
partypes = {
    1: "alkalinity",
    2: "dic",
    3: "pH",
    4: "pCO2",
    5: "fCO2",
    6: "carbonate",
    7: "bicarbonate",
    8: "aqueous_CO2",
}


def rr_par_combos(par1_type, par2_type):
    """Get all possible valid pairs of parameter types, excluding the input pair."""
    # Get all possible combinations of parameter type numbers
    allpars = list(partypes.keys())
    par1_types, par2_types = np.meshgrid(allpars, allpars)
    par1_types = par1_types.ravel()
    par2_types = par2_types.ravel()
    # Select only valid combinations and cut out input combination
    allIcases = pyco2.solve.getIcase(par1_types, par2_types, checks=False)
    inputIcase = pyco2.solve.getIcase(par1_type, par2_type, checks=False)
    valid = (par1_types != par2_types) & ~np.isin(allIcases, [45, 48, 58, inputIcase])
    par1_types = par1_types[valid]
    par2_types = par2_types[valid]
    return par1_types, par2_types


# Prepare round-robin inputs
rr_par1_type, rr_par2_type = rr_par_combos(par1_type, par2_type)
rr_par1 = np.array([results[partypes[t]] for t in rr_par1_type])
rr_par2 = np.array([results[partypes[t]] for t in rr_par2_type])

# Do the round-robin calculations and compare
rr_results = pyco2.sys(rr_par1, rr_par2, rr_par1_type, rr_par2_type, **kwargs)
rr_diff = {k: rr_results[k] - results[k] for k in partypes.values()}


def test_round_robin():
    for k, v in rr_diff.items():
        # print(k, np.max(np.abs(v)))
        print(pyco2.solve.get.pH_tolerance)
        assert np.all(
            np.isclose(np.max(np.abs(v)), 0, rtol=0, atol=pyco2.solve.get.pH_tolerance)
        )


# test_round_robin()

# # Generate a LaTeX table of the results
# if True:
#     varnames = {
#         "alkalinity": r"$\ta$",
#         "dic": r"$\dic$",
#         "pH": r"pH$_{\mathrm{T}}$",
#         "pCO2": "$\pCOtwo$",
#         "fCO2": "$\fCOtwo$",
#         "carbonate": "$[\carb$]",
#         "bicarbonate": "$[\bicarb]$",
#         "aqueous_CO2": "$[\chem{CO_2(aq)}]$",
#     }
#     with open("manuscript/html/round_robin.tex", "w") as f:
#         for var, name in varnames.items():
#             if var == "pH":
#                 fmt = ".3"
#             else:
#                 fmt = ".1"
#             f.write(
#                 ("{} & {:" + fmt + "f} & {:.2e} \\\\\n").format(
#                     name, results[var], np.max(np.abs(rr_diff[var]))
#                 )
#             )


# # Generate the HTML table for the docs, if requested
# if True:
#     varnames = {
#         "alkalinity": "Total alkalinity / μmol/kg-sw",
#         "dic": "Dissolved inorganic carbon / μmol/kg-sw",
#         "pH": "pH (Total scale)",
#         "pCO2": "<i>p</i>CO<sub>2</sub> / μatm",
#         "fCO2": "<i>f</i>CO<sub>2</sub> / μatm",
#         "carbonate": "Carbonate ion / μmol/kg-sw",
#         "bicarbonate": "Bicarbonate ion / μmol/kg-sw",
#         "aqueous_CO2": "Aqueous CO<sub>2</sub> / μmol/kg-sw",
#     }
#     with open("manuscript/html/round_robin.md", "w") as f:
#         f.write("<!-- HTML for table generated with manuscript/round-robin.py -->\n")
#         f.write("<table><tr>\n")
#         f.write('<th style="text-align:right">Carbonate system parameter</th>\n')
#         f.write('<th style="text-align:center">Mean result</th>\n')
#         f.write('<th style="text-align:center">Max. abs. diff.</th></tr>\n')
#         for var, name in varnames.items():
#             f.write("</tr><tr>\n")
#             vmad = "{:.2e}".format(np.max(np.abs(rr_diff[var])))
#             vmad = vmad.replace("e-", "·10<sup>−") + "</sup>"
#             f.write('<td style="text-align:right">{}</td>\n'.format(name))
#             f.write(
#                 '<td style="text-align:center">{:.1f}</td>\n'.format(
#                     np.mean(rr_results[var])
#                 )
#             )
#             f.write('<td style="text-align:center">{}</td>\n'.format(vmad))
#         f.write("</tr></table>\n")
