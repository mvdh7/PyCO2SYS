# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
# Author of this function: Daniel Sandborn
"""EZIO: Easy Input/Output of CO2SYS.xlsx-style spreadsheets"""

from ezio_utils import get_spreadsheet
from ezio_utils import EZIO_calkulate
from ezio_utils import save_output


def ezio(path,
         opt_pH_scale=1, #default values match those at https://pyco2sys.readthedocs.io/en/latest/co2sys_nd/
         opt_k_bisulfate=1,
         opt_k_carbonic=16,
         opt_k_fluoride=1,
         opt_total_borate=1):
    input_file = get_spreadsheet(path) #compatible with both .csv and .xlsx, see input template.
    output_df = EZIO_calkulate(input_file,
                               pH_scale = opt_pH_scale,
                               k_bisulfate = opt_k_bisulfate,
                               k_carbonic = opt_k_carbonic,
                               k_fluoride = opt_k_fluoride,
                               total_borate = opt_total_borate)
    save_output(path, output_df) #.csv output, appends "_processed" to path.
    print("Calculation completed and output .csv file created.")