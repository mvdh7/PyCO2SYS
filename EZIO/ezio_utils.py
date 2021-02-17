# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
# Author of this function: Daniel Sandborn
"""EZIO: Easy Input/Output of CO2SYS.xlsx-style spreadsheets"""
"""Utilities"""

import numpy as np
import pandas as pd
import os
import PyCO2SYS as pyco2

def get_spreadsheet(path):
    head, tail = os.path.splitext(path)
    input_file = pd.DataFrame()
    if tail == ".csv":
        input_file = pd.read_csv(path, header=2, sep = ',', dtype = 'np.float64')
    elif tail == ".xlsx":
        input_file = pd.read_excel(path, header=2)
    else:
        print("ERROR: File could not be read.")
    return input_file

def save_output(path, df):
    head, tail = os.path.splitext(path)
    newtail = "_processed.csv"
    newpath = head+newtail
    df.to_csv(newpath)

def EZIO_calkulate(input_file, 
                   pH_scale = 1, 
                   k_bisulfate = 1, 
                   k_carbonic = 16, 
                   k_fluoride = 1, 
                   total_borate = 1):
    output_df = pd.DataFrame(
        {"Salinity" : [],
         "t(°C) in" : [],
         "P in (dbar)" : [],
         "Total P (μmol/kgW)" : [],
         "Total Si (μmol/kgW)" : [],
         "TA in (μmol/kgW)" : [],
         "TCO2 in (μmol/kgW)" : [],
         "pH in" : [],
         "fCO2 in (μatm)" : [],
         "pCO2 in (μatm)" : [],
         "HCO3 in (μmol/kgW)" : [],
         "CO3 in (μmol/kgW)" : [],
         "CO2 in (μmol/kgW)" : [],
         "B Alk in (μmol/kgW)" : [],
         "OH in (μmol/kgW)" : [],
         "P Alk in (μmol/kgW)" : [],
         "Si Alk in (μmol/kgW) " : [],
         "Revelle in" : [],
         "ΩCa in" : [],
         "ΩAr in" : [],
         "xCO2 in (dry at 1 atm) (ppm)" : [],
         "t(°C) out" : [],
         "P out (dbar)" : [],
         "pH out" : [],
         "fCO2 out (μatm)" : [],
         "pCO2 out (μatm)" : [],
         "HCO3 out (μmol/kgW)" : [],
         "CO3 out (μmol/kgW)" : [],
         "CO2 out (μmol/kgW)" : [],
         "B Alk out (μmol/kgW)" : [],
         "OH out (μmol/kgW)" : [],
         "P Alk out (μmol/kgW)" : [],
         "Si Alk out (μmol/kgW)" : [],
         "Revelle out" : [],
         "ΩCa out" : [],
         "ΩAr out" : [],
         "xCO2 out (dry at 1 atm) (ppm)" : []})
    
    for i in range(len(input_file)):
        pars = ~input_file.iloc[i,[7,8,9,10,11]].isna()
        if sum(pars) != 2:
            print("Incorrect number of input parameters provided in row", i+1)
            break
        values = input_file.iloc[i,[7,8,9,10,11]].dropna()
        par1 = values[0]
        par2 = values[1]
        
        par_names = input_file.iloc[0,[7,8,9,10,11]].index[pars].tolist()
        if ("TCO2"  in par_names[0]):
            par1_type = 2
        elif ("TA"  in par_names[0]):
            par1_type = 1
        elif ("pH"  in par_names[0]):
            par1_type = 3
        elif ("fCO2"  in par_names[0]):
            par1_type = 5
        elif ("pCO2"  in par_names[0]):
            par1_type = 4
        if ("TCO2"  in par_names[1]):
            par2_type = 2
        elif ("TA"  in par_names[1]):
            par2_type = 1
        elif ("pH"  in par_names[1]):
            par2_type = 3
        elif ("fCO2"  in par_names[1]):
            par2_type = 5
        elif ("pCO2"  in par_names[1]):
            par2_type = 4
        
        isalinity = input_file.iloc[i,[0]]
        itemperature = input_file.iloc[i,[1]]
        ipressure = input_file.iloc[i,[2]]
        itemperature_out = input_file.iloc[i,[5]]
        ipressure_out = input_file.iloc[i,[6]]
        itotal_silicate = input_file.iloc[i,[4]]
        itotal_phosphate = input_file.iloc[i,[3]]
        
        results = pyco2.sys(par1, par2, par1_type, par2_type, 
                            salinity = isalinity, temperature = itemperature, 
                            pressure = ipressure, temperature_out = itemperature_out, 
                            pressure_out = ipressure_out, total_silicate = itotal_silicate, 
                            total_phosphate = itotal_phosphate, opt_k_carbonic = k_carbonic, 
                            opt_k_bisulfate = k_bisulfate, opt_total_borate = total_borate, 
                            opt_k_fluoride = k_fluoride, opt_pH_scale = pH_scale)
        output_newrow = pd.DataFrame(
            {"Salinity" : [results['salinity']],
             "t(°C) in" : [results['temperature']],
             "P in (dbar)" : [results['pressure']],
             "Total P (μmol/kgW)" : [results['total_phosphate']],
             "Total Si (μmol/kgW)" : [results['total_silicate']],
             "TA in (μmol/kgW)" : [results['alkalinity']],
             "TCO2 in (μmol/kgW)" : [results['dic']],
             "pH in" : [results['pH']],
             "fCO2 in (μatm)" : [results['fCO2']],
             "pCO2 in (μatm)" : [results['pCO2']],
             "HCO3 in (μmol/kgW)" : [results['HCO3']],
             "CO3 in (μmol/kgW)" : [results['CO3']],
             "CO2 in (μmol/kgW)" : [results['CO2']],
             "B Alk in (μmol/kgW)" : [results['alkalinity_borate']],
             "OH in (μmol/kgW)" : [results['OH']],
             "P Alk in (μmol/kgW)" : [results['alkalinity_phosphate']],
             "Si Alk in (μmol/kgW) " : [results['alkalinity_silicate']],
             "Revelle in" : [results['revelle_factor']],
             "ΩCa in" : [results['saturation_calcite']],
             "ΩAr in" : [results['saturation_aragonite']],
             "xCO2 in (dry at 1 atm) (ppm)" : [results['xCO2']],
             "t(°C) out" : [results['temperature_out']],
             "P out (dbar)" : [results['pressure_out']],
             "pH out" : [results['pH_out']],
             "fCO2 out (μatm)" : [results['fCO2_out']],
             "pCO2 out (μatm)" : [results['pCO2_out']],
             "HCO3 out (μmol/kgW)" : [results['HCO3_out']],
             "CO3 out (μmol/kgW)" : [results['CO3_out']],
             "CO2 out (μmol/kgW)" : [results['CO2_out']],
             "B Alk out (μmol/kgW)" : [results['alkalinity_borate_out']],
             "OH out (μmol/kgW)" : [results['OH_out']],
             "P Alk out (μmol/kgW)" : [results['alkalinity_phosphate_out']],
             "Si Alk out (μmol/kgW)" : [results['alkalinity_silicate_out']],
             "Revelle out" : [results['revelle_factor_out']],
             "ΩCa out" : [results['saturation_calcite_out']],
             "ΩAr out" : [results['saturation_aragonite_out']],
             "xCO2 out (dry at 1 atm) (ppm)" : [results['xCO2_out']]})
        output_df = output_df.append(output_newrow)
    return output_df

