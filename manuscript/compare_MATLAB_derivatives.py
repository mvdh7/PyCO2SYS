from os import listdir
import pandas as pd, numpy as np
import PyCO2SYS as pyco2

# Get list of derivatives and their files
dpath = "manuscript/data/CO2SYSv3_2_0/derivatives/"
dfiles = [f for f in listdir(dpath) if f.endswith("_v3.csv")]
dtypes = [f.split("_")[0] for f in dfiles]

# Mapper from MATLAB filenames to PyCO2SYS keys
m2p = dict(
    bor="total_borate",
    k0="k_CO2",
    k1="k_carbonic_1",
    k2="k_carbonic_2",
    kb="k_borate",
    kw="k_water",
    par1="par1",
    par2="par2",
    phos="total_phosphate",
    sal="salinity",
    sil="total_silicate",
    temp="temperature",
)

# Calculate the derivatives with PyCO2SYS
co2matlab = pd.read_csv(dpath + "compare_MATLABv3_2_0__derivs.csv")
co2inputs = [
    co2matlab[var].values
    for var in [
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
    ]
]
results = pd.DataFrame(
    pyco2.sys(
        co2inputs[0],
        co2inputs[1],
        co2inputs[2],
        co2inputs[3],
        salinity=co2inputs[4],
        temperature=co2inputs[5],
        temperature_out=co2inputs[6],
        pressure=co2inputs[7],
        pressure_out=co2inputs[8],
        total_silicate=co2inputs[9],
        total_phosphate=co2inputs[10],
        total_ammonia=co2inputs[11],
        total_sulfide=co2inputs[12],
        opt_pH_scale=co2inputs[13],
        opt_k_carbonic=co2inputs[14],
        opt_k_bisulfate=co2inputs[15],
        opt_k_fluoride=co2inputs[16],
        opt_total_borate=co2inputs[17],
        opt_gas_constant=3,
        grads_of=[
            "alkalinity",
            "dic",
            "pCO2",
            "fCO2",
            "bicarbonate",
            "carbonate",
            "aqueous_CO2",
            "xCO2",
            "pCO2_out",
            "fCO2_out",
            "bicarbonate_out",
            "carbonate_out",
            "aqueous_CO2_out",
            "xCO2_out",
        ],
        grads_wrt=["temperature", "salinity", "k_CO2", "k_carbonic_1", "k_carbonic_2"],
    )
)

# Import derivatives from CO2SYS-MATLAB files and get differences
m_derivs = {}
diff_derivs = {}
for f, t in zip(dfiles, dtypes):
    # Import from MATLAB
    m_derivs[m2p[t]] = md = pd.read_csv(dpath + f)
    mapper = {
        c: (
            c.replace("dTAlk", "d_alkalinity")
            .replace("dTCO2", "d_dic")
            .replace("dpCO2in", "d_pCO2")
            .replace("dfCO2in", "d_fCO2")
            .replace("dHCO3in", "d_bicarbonate")
            .replace("dCO3in", "d_carbonate")
            .replace("dCO2in", "d_aqueous_CO2")
            .replace("dxCO2in", "d_xCO2")
            .replace("dpCO2out", "d_pCO2_out")
            .replace("dfCO2out", "d_fCO2_out")
            .replace("dHCO3out", "d_bicarbonate_out")
            .replace("dCO3out", "d_carbonate_out")
            .replace("dCO2out", "d_aqueous_CO2_out")
            .replace("dxCO2out", "d_xCO2_out")
            .replace("_dT", "__d_temperature")
            .replace("_dS", "__d_salinity")
            .replace("_dKW", "__d_k_water")
            .replace("_dK0", "__d_k_CO2")
            .replace("_dK1", "__d_k_carbonic_1")
            .replace("_dK2", "__d_k_carbonic_2")
        )
        for c in md.columns
    }
    md.rename(columns=mapper, inplace=True)
    # Get differences
    diff_derivs[m2p[t]] = dd = {}
    for c in md.columns:
        if c in results:
            # dd["m_" + c] = md[c].values
            # dd["p_" + c] = results[c].values
            # Percentages:
            dd_c = 100 * (md[c] - results[c]) / results[c]
            dd_c = np.where(results[c] == 0, md[c], dd_c)
            dd[c] = dd_c
            # # Absolutes:
            # dd[c] = md[c] - results[c]
    diff_derivs[m2p[t]] = pd.DataFrame(dd)
# dds = diff_derivs["salinity"]
