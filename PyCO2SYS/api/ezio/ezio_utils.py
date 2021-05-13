# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
#
"""EZIO: Easy Input/Output of CO2SYS.xlsx-style spreadsheets"""
"""Utilities"""

import pandas as pd
import os
from ...engine.nd import CO2SYS as CO2SYS_nd


def get_spreadsheet(path):  # Input file MUST be the same format as in CO2SYS Excel.
    """
    Receives filepath for .csv or .xlsx spreadsheet structured as in Excel CO2SYS.
    First two lines are not read (the 'START' button and header groups in Excel).

    Parameters
    ----------
    path : string filepath
        Filepath in local system to 'CO2SYS.xlsx'-style spreadsheet.
        Spreadsheet can be either .csv (comma delimited) or .xlsx (Microsoft Excel).

    Returns
    -------
    input_file : Pandas dataframe
        read_csv() or read_excel() output.

    """
    head, tail = os.path.splitext(path)  # Which filetype?
    input_file = pd.DataFrame()
    if tail == ".csv":  # comma-separated, first two rows ignored
        input_file = pd.read_csv(path, header=2, sep=",", dtype="np.float64")
    elif tail == ".xlsx":  # first two rows ignored
        input_file = pd.read_excel(path, header=2)
    else:
        print("ERROR: File could not be read.")
    return input_file


def save_output(path, df):
    """
    The resulting dataframe is saved as a .csv file in the same directory
    as the input file, with the tag '_processed.csv' appended.

    Parameters
    ----------
    path : string filepath
        Filepath in local system to 'CO2SYS.xlsx'-style spreadsheet.
    df : Pandas dataframe
        Result of PyCO2SYS solving algorithm, structured as in CO2SYS Excel.

    Returns
    -------
    None.

    """
    head, tail = os.path.splitext(path)
    newtail = "_processed.csv"  # adds "_processed" tag to filename
    newpath = head + newtail
    df.to_csv(
        newpath, encoding="utf-8-sig"
    )  # encoding ensures that symbols are output correctly


def EZIO_calculate(
    input_file,
    pH_scale=1,  # default values match those at https://pyco2sys.readthedocs.io/en/latest/co2sys_nd/
    k_bisulfate=1,
    k_carbonic=16,
    k_fluoride=1,
    total_borate=1,
):
    """
    EZIO wrapper function around pyco2.sys().  Solves the inorganic carbonate
    system given the inputs of a CO2SYS Excel-structured spreadsheet.  Outputs
    in similar fashion.

    Parameters
    ----------
    input_file : Pandas dataframe
        read_csv() or read_excel() output.
    opt_pH_scale : int, optional
        pH scale as defined by ZW01. The default is 1 (Total).
    opt_k_carbonic : int, optional
        Carbonic acid dissociation constants. The default is 16 (SLH20).
    opt_k_fluoride : int, optional
        Fluoride equilibrium constant. The default is 1 (DR79).
    opt_total_borate : int, optional
        Total borate to salinity constant. The default is 1 (U74).

    Returns
    -------
    output_df : Pandas dataframe
        Result of PyCO2SYS solving algorithm, structured as in CO2SYS Excel.

    """
    output_df = pd.DataFrame(  # Initiate empty dataframe matching output format of CO2SYS Excel
        {
            "Salinity": [],  # Would this be better as a class?
            "t(°C) in": [],
            "P in (dbar)": [],
            "Total P (μmol/kgW)": [],
            "Total Si (μmol/kgW)": [],
            "TA in (μmol/kgW)": [],
            "TCO2 in (μmol/kgW)": [],
            "pH in": [],
            "fCO2 in (μatm)": [],
            "pCO2 in (μatm)": [],
            "HCO3 in (μmol/kgW)": [],
            "CO3 in (μmol/kgW)": [],
            "CO2 in (μmol/kgW)": [],
            "B Alk in (μmol/kgW)": [],
            "OH in (μmol/kgW)": [],
            "P Alk in (μmol/kgW)": [],
            "Si Alk in (μmol/kgW) ": [],
            "Revelle in": [],
            "ΩCa in": [],
            "ΩAr in": [],
            "xCO2 in (dry at 1 atm) (ppm)": [],
            "-": [],
            "t(°C) out": [],
            "P out (dbar)": [],
            "pH out": [],
            "fCO2 out (μatm)": [],
            "pCO2 out (μatm)": [],
            "HCO3 out (μmol/kgW)": [],
            "CO3 out (μmol/kgW)": [],
            "CO2 out (μmol/kgW)": [],
            "B Alk out (μmol/kgW)": [],
            "OH out (μmol/kgW)": [],
            "P Alk out (μmol/kgW)": [],
            "Si Alk out (μmol/kgW)": [],
            "Revelle out": [],
            "ΩCa out": [],
            "ΩAr out": [],
            "xCO2 out (dry at 1 atm) (ppm)": [],
        }
    )

    for i in range(len(input_file)):
        pars = ~input_file.iloc[
            i, [7, 8, 9, 10, 11]
        ].isna()  # Select among carbonate parameters.
        if sum(pars) != 2:  # Under- or over-constrained systems rejected.
            print("Incorrect number of input parameters provided in row", i + 1)
            break  # No calculations performed, but processed file still output.
        values = input_file.iloc[i, [7, 8, 9, 10, 11]].dropna()
        par1 = values[0]
        par2 = values[1]

        # Check which parameters were used for this sample.  Not very robust.
        par_names = input_file.iloc[0, [7, 8, 9, 10, 11]].index[pars].tolist()
        if any(
            c in par_names[0]
            for c in ("DIC", "TCO2", "dic", "dissolved inorganic carbon", "CT", "Ct")
        ):
            par1_type = 2
        elif any(
            c in par_names[0]
            for c in (
                "TA",
                "Alk",
                "ALK",
                "alkalinity",
                "alk",
                "Total Alkalinity",
                "total alkalinity",
            )
        ):
            par1_type = 1
        elif "pH" in par_names[0]:
            par1_type = 3
        elif "fCO2" in par_names[0]:
            par1_type = 5
        elif "pCO2" in par_names[0]:
            par1_type = 4
        if any(
            c in par_names[1]
            for c in ("DIC", "TCO2", "dic", "dissolved inorganic carbon", "CT", "Ct")
        ):
            par2_type = 2
        elif any(
            c in par_names[1]
            for c in (
                "TA",
                "Alk",
                "ALK",
                "alkalinity",
                "alk",
                "Total Alkalinity",
                "total alkalinity",
            )
        ):
            par2_type = 1
        elif "pH" in par_names[1]:
            par2_type = 3
        elif "fCO2" in par_names[1]:
            par2_type = 5
        elif "pCO2" in par_names[1]:
            par2_type = 4

        # Non-carbonate parameters: VERY sensitive to column placement.
        # MUST match CO2SYS Excel.
        isalinity = input_file.iloc[i, [0]]
        itemperature = input_file.iloc[i, [1]]
        ipressure = input_file.iloc[i, [2]]
        itemperature_out = input_file.iloc[i, [5]]
        ipressure_out = input_file.iloc[i, [6]]
        itotal_silicate = input_file.iloc[i, [4]]
        itotal_phosphate = input_file.iloc[i, [3]]

        results = CO2SYS_nd(
            par1,
            par2,
            par1_type,
            par2_type,  # as in https://pyco2sys.readthedocs.io/en/latest/co2sys_nd/
            salinity=isalinity,
            temperature=itemperature,
            pressure=ipressure,
            temperature_out=itemperature_out,
            pressure_out=ipressure_out,
            total_silicate=itotal_silicate,
            total_phosphate=itotal_phosphate,
            opt_k_carbonic=k_carbonic,
            opt_k_bisulfate=k_bisulfate,
            opt_total_borate=total_borate,
            opt_k_fluoride=k_fluoride,
            opt_pH_scale=pH_scale,
        )
        output_newrow = pd.DataFrame(
            {
                "Salinity": results["salinity"],
                "t(°C) in": results["temperature"],
                "P in (dbar)": results["pressure"],
                "Total P (μmol/kgW)": results["total_phosphate"],
                "Total Si (μmol/kgW)": results["total_silicate"],
                "TA in (μmol/kgW)": results["alkalinity"],
                "TCO2 in (μmol/kgW)": results["dic"],
                "pH in": results["pH"],
                "fCO2 in (μatm)": results["fCO2"],
                "pCO2 in (μatm)": results["pCO2"],
                "HCO3 in (μmol/kgW)": results["HCO3"],
                "CO3 in (μmol/kgW)": results["CO3"],
                "CO2 in (μmol/kgW)": results["CO2"],
                "B Alk in (μmol/kgW)": results["alkalinity_borate"],
                "OH in (μmol/kgW)": results["OH"],
                "P Alk in (μmol/kgW)": results["alkalinity_phosphate"],
                "Si Alk in (μmol/kgW) ": results["alkalinity_silicate"],
                "Revelle in": results["revelle_factor"],
                "ΩCa in": results["saturation_calcite"],
                "ΩAr in": results["saturation_aragonite"],
                "xCO2 in (dry at 1 atm) (ppm)": results["xCO2"],
                "-": ["-"],
                "t(°C) out": results["temperature_out"],
                "P out (dbar)": results["pressure_out"],
                "pH out": results["pH_out"],
                "fCO2 out (μatm)": results["fCO2_out"],
                "pCO2 out (μatm)": results["pCO2_out"],
                "HCO3 out (μmol/kgW)": results["HCO3_out"],
                "CO3 out (μmol/kgW)": results["CO3_out"],
                "CO2 out (μmol/kgW)": results["CO2_out"],
                "B Alk out (μmol/kgW)": results["alkalinity_borate_out"],
                "OH out (μmol/kgW)": results["OH_out"],
                "P Alk out (μmol/kgW)": results["alkalinity_phosphate_out"],
                "Si Alk out (μmol/kgW)": results["alkalinity_silicate_out"],
                "Revelle out": results["revelle_factor_out"],
                "ΩCa out": results["saturation_calcite_out"],
                "ΩAr out": results["saturation_aragonite_out"],
                "xCO2 out (dry at 1 atm) (ppm)": results["xCO2_out"],
            }
        )
        output_df = output_df.append(
            output_newrow
        )  # Add the new row to the growing output file, run again for subsequent samples.
    return output_df
