# Import packages
import PyCO2SYS as pyco2
import pandas as pd

# Why not
pyco2.say_hello()

# Define test conditions
par1 = 2300  # parameter 1, here total alkalinity in μmol/kg-sw
par2 = 8.1  # parameter 2, here pH on the Total scale
par1type = 1  # "parameter 1 is total alkalinity"
par2type = 3  # "parameter 2 is pH"
sal = 33  # practical salinity
temp = 22  # temperature in °C
pres = 1000  # pressure in dbar
si = 10  # total silicate in μmol/kg-sw
phos = 1  # total phosphate in μmol/kg-sw
nh3 = 2  # total ammonia in μmol/kg-sw
h2s = 3  # total sulfide in μmol/kg-sw
pHscale = 1  # "input pH is on the Total scale"
k1k2c = 10  # "use LDK00 constants for carbonic acid dissociation"
kso4c = 3  # "use D90a for bisulfate dissociation and LKB10 for borate:salinity"

# Run the test
res, diff = pyco2.test.roundrobin(
    par1,
    par2,
    par1type,
    par2type,
    sal,
    temp,
    pres,
    si,
    phos,
    pHscale,
    k1k2c,
    kso4c,
    NH3=nh3,
    H2S=h2s,
    buffers_mode="none",
)

# Convert results to pandas DataFrames
res = pd.DataFrame(res)  # raw values
diff = pd.DataFrame(diff)  # differences between input pairs

# Print out key results
keyvars = ["TAlk", "TCO2", "pHin", "pCO2in", "fCO2in", "CO3in", "HCO3in", "CO2in"]
mean_res = res[keyvars].mean()
max_abs_diff = diff[keyvars].abs().max()
print(max_abs_diff)  # biggest differences across all input pair combinations

# Generate the HTML table for the docs if requested
if False:
    varnames = {
        "TAlk": "Total alkalinity / μmol/kg-sw",
        "TCO2": "Dissolved inorganic carbon / μmol/kg-sw",
        "pHin": "pH (Total scale)",
        "pCO2in": "<i>p</i>CO<sub>2</sub> / μatm",
        "fCO2in": "<i>f</i>CO<sub>2</sub> / μatm",
        "CO3in": "Carbonate ion / μmol/kg-sw",
        "HCO3in": "Bicarbonate ion / μmol/kg-sw",
        "CO2in": "Aqueous CO<sub>2</sub> / μmol/kg-sw",
    }
    with open("validate/html/round_robin.md", "w") as f:
        f.write("<!-- HTML for table generated with examples/round-robin.py -->\n")
        f.write("<table><tr>\n")
        f.write('<th style="text-align:right">Carbonate system parameter</th>\n')
        f.write('<th style="text-align:center">Mean result</th>\n')
        f.write('<th style="text-align:center">Max. abs. diff.</th></tr>\n')
        for var in keyvars:
            f.write("</tr><tr>\n")
            vmad = "{:.2e}".format(max_abs_diff[var])
            vmad = vmad.replace("e-", "·10<sup>−") + "</sup>"
            f.write('<td style="text-align:right">{}</td>\n'.format(varnames[var]))
            f.write('<td style="text-align:center">{:.1f}</td>\n'.format(mean_res[var]))
            f.write('<td style="text-align:center">{}</td>\n'.format(vmad))
        f.write("</tr></table>\n")


def test_roundrobin():
    assert max_abs_diff.max() < 1e-10
