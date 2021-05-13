# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
#
"""EZIO: Easy Input/Output of 'CO2SYS.xlsx'-style spreadsheets"""

import pandas as pd
from .ezio_utils import get_spreadsheet
from .ezio_utils import EZIO_calculate
from .ezio_utils import save_output


def ezio(
    path,
    opt_pH_scale=1,  # default values match those at https://pyco2sys.readthedocs.io/en/latest/co2sys_nd/
    opt_k_bisulfate=1,
    opt_k_carbonic=16,
    opt_k_fluoride=1,
    opt_total_borate=1,
):
    """The main function for easy input/output of 'CO2SYS.xlsx'-style spreadsheets.
    Takes a spreadsheet as input, which MUST be formatted with the same columns
    as the input columns in CO2SYS.xlsx, in the following order:
        Salinity	
        t(°C)	
        P (dbars)	
        Total P (μmol/kgSW)	
        Total Si (μmol/kgSW)	
        t(oC)	
        P (dbars)	
        TA (μmol/kgSW)	
        TCO2 (μmol/kgSW)	
        pH   (Chosen Scale)	
        fCO2 (μatm)	
        pCO2 (μatm)
    
    This is accomplished by simply copying the leftmost cells (col A-L) from
    'CO2SYS.xlsx' (or the template provided at pyco2sys.readthedocs.io) and
    pasting the cells into a blank speadsheet to be saved as .csv or .xlsx.
    
    The resulting dataframe is saved as a .csv file in the same directory
    as the input file, with the tag '_processed.csv' appended.

    Parameters
    ----------
    path : string filepath
        Filepath in local system to 'CO2SYS.xlsx'-style spreadsheet.  
        Spreadsheet can be either .csv (comma delimited) or .xlsx (Microsoft Excel).
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
        Solved carbonate system parameters.

    """
    input_file = get_spreadsheet(
        path
    )  # compatible with both .csv and .xlsx, see input template.
    output_df = EZIO_calculate(
        input_file,
        pH_scale=opt_pH_scale,
        k_bisulfate=opt_k_bisulfate,
        k_carbonic=opt_k_carbonic,
        k_fluoride=opt_k_fluoride,
        total_borate=opt_total_borate,
    )
    save_output(path, output_df)  # .csv output, appends "_processed" to path.
    if isinstance(output_df, pd.DataFrame):
        print("Calculation completed and output .csv file created.")
    return output_df
