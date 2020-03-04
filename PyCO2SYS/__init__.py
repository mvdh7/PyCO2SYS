from . import (
    concentrations,
    convert,
    equilibria,
    meta,
    original,
)
__all__ = [
    'concentrations',
    'convert',
    'equilibria',
    'meta',
    'original',
]

__author__ = 'Matthew P. Humphreys'
__version__ = meta.version

# Shorthand module names
conc = concentrations
eq = equilibria

#**************************************************************************
#
# First   CO2SYS.m version: 1.1 (Sep 2011)
# MATLAB  CO2SYS.m version: 2.0 (20 Dec 2016)
#
# CO2SYS is a MATLAB-version of the original CO2SYS for DOS. 
# CO2SYS calculates and returns the state of the carbonate system of 
#    oceanographic water samples, if supplied with enough input.
# PyCO2SYS has been converted from MATLAB to Python.
#
# Please note that this software is intended to be exactly identical to the 
#    DOS and Excel versions that have been released previously, meaning that
#    results obtained should be very nearly indentical for identical input.
# Additionally, several of the dissociation constants K1 and K2 that have 
#    been published since the original DOS version was written are implemented.
#    For a complete list of changes since version 1.0, see below.
#
# For much more info please have a look at:
#    Lewis, E., and D. W. R. Wallace. 1998. Program Developed for
#    CO2 System Calculations. ORNL/CDIAC-105. Carbon Dioxide Information
#    Analysis Center, Oak Ridge National Laboratory, U.S. Department of Energy,
#    Oak Ridge, Tennessee. 
#    http://cdiac.ornl.gov/oceans/co2rprt.html
#
#**************************************************************************
#
# SYNTAX AND EXAMPLES IN THIS SECTION ARE FOR THE MATLAB VERSION ONLY
#
#  **** SYNTAX:
#  [RESULT,HEADERS,NICEHEADERS]=CO2SYS(PAR1,PAR2,PAR1TYPE,PAR2TYPE,...
#        ...SAL,TEMPIN,TEMPOUT,PRESIN,PRESOUT,SI,PO4,pHSCALEIN,...
#        ...K1K2CONSTANTS,KSO4CONSTANTS)
# 
#  **** SYNTAX EXAMPLES:
#  [Result]                     = CO2SYS(2400,2200,1,2,35,0,25,4200,0,15,1,1,4,1)
#  [Result,Headers]             = CO2SYS(2400,   8,1,3,35,0,25,4200,0,15,1,1,4,1)
#  [Result,Headers,Niceheaders] = CO2SYS( 500,   8,5,3,35,0,25,4200,0,15,1,1,4,1)
#  [A]                          = CO2SYS(2400,2000:10:2400,1,2,35,0,25,4200,0,15,1,1,4,1)
#  [A]                          = CO2SYS(2400,2200,1,2,0:1:35,0,25,4200,0,15,1,1,4,1)
#  [A]                          = CO2SYS(2400,2200,1,2,35,0,25,0:100:4200,0,15,1,1,4,1)
#  
#  **** APPLICATION EXAMPLE (copy and paste this into command window):
#  tmps=0:40; sals=0:40; [X,Y]=meshgrid(tmps,sals);
#  A = CO2SYS(2300,2100,1,2,Y(:),X(:),nan,0,nan,1,1,1,9,1);
#  Z=nan(size(X)); Z(:)=A(:,4); figure; contourf(X,Y,Z,20); caxis([0 1200]); colorbar;
#  ylabel('Salinity [psu]'); xlabel('Temperature [degC]'); title('Dependence of pCO2 [uatm] on T and S')
# 
#**************************************************************************
#
# INPUT:
#
#   PAR1  (some unit) : scalar or vector of size n
#   PAR2  (some unit) : scalar or vector of size n
#   PAR1TYPE       () : scalar or vector of size n (*)
#   PAR2TYPE       () : scalar or vector of size n (*)
#   SAL            () : scalar or vector of size n
#   TEMPIN  (degr. C) : scalar or vector of size n 
#   TEMPOUT (degr. C) : scalar or vector of size n 
#   PRESIN     (dbar) : scalar or vector of size n 
#   PRESOUT    (dbar) : scalar or vector of size n
#   SI    (umol/kgSW) : scalar or vector of size n
#   PO4   (umol/kgSW) : scalar or vector of size n
#   pHSCALEIN         : scalar or vector of size n (**)
#   K1K2CONSTANTS     : scalar or vector of size n (***)
#   KSO4CONSTANTS     : scalar or vector of size n (****)
#
#  (*) Each element must be an integer, 
#      indicating that PAR1 (or PAR2) is of type: 
#  1 = Total Alkalinity
#  2 = DIC
#  3 = pH
#  4 = pCO2
#  5 = fCO2
# 
#  (**) Each element must be an integer, 
#       indicating that the pH-input (PAR1 or PAR2, if any) is at:
#  1 = Total scale
#  2 = Seawater scale
#  3 = Free scale
#  4 = NBS scale
# 
#  (***) Each element must be an integer, 
#        indicating the K1 K2 dissociation constants that are to be used:
#   1 = Roy, 1993											T:    0-45  S:  5-45. Total scale. Artificial seawater.
#   2 = Goyet & Poisson										T:   -1-40  S: 10-50. Seaw. scale. Artificial seawater.
#   3 = HANSSON              refit BY DICKSON AND MILLERO	T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
#   4 = MEHRBACH             refit BY DICKSON AND MILLERO	T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
#   5 = HANSSON and MEHRBACH refit BY DICKSON AND MILLERO	T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
#   6 = GEOSECS (i.e., original Mehrbach)					T:    2-35  S: 19-43. NBS scale.   Real seawater.
#   7 = Peng	(i.e., originam Mehrbach but without XXX)	T:    2-35  S: 19-43. NBS scale.   Real seawater.
#   8 = Millero, 1979, FOR PURE WATER ONLY (i.e., Sal=0)	T:    0-50  S:     0. 
#   9 = Cai and Wang, 1998									T:    2-35  S:  0-49. NBS scale.   Real and artificial seawater.
#  10 = Lueker et al, 2000									T:    2-35  S: 19-43. Total scale. Real seawater.
#  11 = Mojica Prieto and Millero, 2002.					T:    0-45  S:  5-42. Seaw. scale. Real seawater
#  12 = Millero et al, 2002									T: -1.6-35  S: 34-37. Seaw. scale. Field measurements.
#  13 = Millero et al, 2006									T:    0-50  S:  1-50. Seaw. scale. Real seawater.
#  14 = Millero        2010  								T:    0-50  S:  1-50. Seaw. scale. Real seawater.
#  15 = Waters, Millero, & Woosley 2014  					T:    0-50  S:  1-50. Seaw. scale. Real seawater.
# 
#  (****) Each element must be an integer that 
#         indicates the KSO4 dissociation constants that are to be used,
#         in combination with the formulation of the borate-to-salinity ratio to be used.
#         Having both these choices in a single argument is somewhat awkward, 
#         but it maintains syntax compatibility with the previous version.
#  1 = KSO4 of Dickson 1990a   & TB of Uppstrom 1974  (PREFERRED) 
#  2 = KSO4 of Khoo et al 1977 & TB of Uppstrom 1974
#  3 = KSO4 of Dickson 1990a   & TB of Lee 2010
#  4 = KSO4 of Khoo et al 1977 & TB of Lee 2010
#
#**************************************************************************#
#
# OUTPUT: * an array containing the following parameter values (one column per sample):
#         *  a cell-array containing crudely formatted headers
#         *  a cell-array containing nicely formatted headers
#
#    POS  PARAMETER        UNIT
#
#    01 - TAlk                 (umol/kgSW)
#    02 - TCO2                 (umol/kgSW)
#    03 - pHin                 ()
#    04 - pCO2 input           (uatm)
#    05 - fCO2 input           (uatm)
#    06 - HCO3 input           (umol/kgSW)
#    07 - CO3 input            (umol/kgSW)
#    08 - CO2 input            (umol/kgSW)
#    09 - BAlk input           (umol/kgSW)
#    10 - OH input             (umol/kgSW)
#    11 - PAlk input           (umol/kgSW)
#    12 - SiAlk input          (umol/kgSW)
#    13 - Hfree input          (umol/kgSW)
#    14 - RevelleFactor input  ()
#    15 - OmegaCa input        ()
#    16 - OmegaAr input        ()
#    17 - xCO2 input           (ppm)
#    18 - pH output            ()
#    19 - pCO2 output          (uatm)
#    20 - fCO2 output          (uatm)
#    21 - HCO3 output          (umol/kgSW)
#    22 - CO3 output           (umol/kgSW)
#    23 - CO2 output           (umol/kgSW)
#    24 - BAlk output          (umol/kgSW)
#    25 - OH output            (umol/kgSW)
#    26 - PAlk output          (umol/kgSW)
#    27 - SiAlk output         (umol/kgSW)
#    28 - Hfree output         (umol/kgSW)
#    29 - RevelleFactor output ()
#    30 - OmegaCa output       ()
#    31 - OmegaAr output       ()
#    32 - xCO2 output          (ppm)
#    33 - pH input (Total)     ()          
#    34 - pH input (SWS)       ()          
#    35 - pH input (Free)      ()          
#    36 - pH input (NBS)       ()          
#    37 - pH output (Total)    ()          
#    38 - pH output (SWS)      ()          
#    39 - pH output (Free)     ()          
#    40 - pH output (NBS)      () 
#    41 - TEMP input           (deg C)     ***    
#    42 - TEMPOUT              (deg C)     ***
#    43 - PRES input           (dbar or m) ***
#    44 - PRESOUT              (dbar or m) ***
#    45 - PAR1TYPE             (integer)   ***
#    46 - PAR2TYPE             (integer)   ***
#    47 - K1K2CONSTANTS        (integer)   ***
#    48 - KSO4CONSTANTS        (integer)   *** 
#    49 - pHSCALE of input     (integer)   ***
#    50 - SAL                  (psu)       ***
#    51 - PO4                  (umol/kgSW) ***
#    52 - SI                   (umol/kgSW) ***
#    53 - K0  input            ()          
#    54 - K1  input            ()          
#    55 - K2  input            ()          
#    56 - pK1 input            ()          
#    57 - pK2 input            ()          
#    58 - KW  input            ()          
#    59 - KB  input            ()          
#    60 - KF  input            ()          
#    61 - KS  input            ()          
#    62 - KP1 input            ()          
#    63 - KP2 input            ()          
#    64 - KP3 input            ()          
#    65 - KSi input            ()              
#    66 - K0  output           ()          
#    67 - K1  output           ()          
#    68 - K2  output           ()          
#    69 - pK1 output           ()          
#    70 - pK2 output           ()          
#    71 - KW  output           ()          
#    72 - KB  output           ()          
#    73 - KF  output           ()          
#    74 - KS  output           ()          
#    75 - KP1 output           ()          
#    76 - KP2 output           ()          
#    77 - KP3 output           ()          
#    78 - KSi output           ()              
#    79 - TB                   (umol/kgSW) 
#    80 - TF                   (umol/kgSW) 
#    81 - TS                   (umol/kgSW) 
#    82 - TP                   (umol/kgSW) 
#    83 - TSi                  (umol/kgSW)
#
#    *** SIMPLY RESTATES THE INPUT BY USER 
#
# In all the above, the terms "input" and "output" may be understood
#    to refer to the 2 scenarios for which CO2SYS performs calculations, 
#    each defined by its own combination of temperature and pressure.
#    For instance, one may use CO2SYS to calculate, from measured DIC and TAlk,
#    the pH that that sample will have in the lab (e.g., T=25 degC, P=0 dbar),
#    and what the in situ pH would have been (e.g., at T=1 degC, P=4500).
#    A = CO2SYS(2400,2200,1,2,35,25,1,0,4200,1,1,1,4,1)
#    pH_lab = A(3);  # 7.84
#    pH_sea = A(18); # 8.05
# 
#**************************************************************************
#
# This is version 2.0 (uploaded to CDIAC at SEP XXth, 2011):
#
# **** Changes since 2.0
#	- slight changes to allow error propagation
#	- new option to choose K1 & K2 from Waters et al. (2014): fixes inconsistencies with Millero (2010) identified by Orr et al. (2015)
#
# **** Changes since 1.01 (uploaded to CDIAC at June 11th, 2009):
# - Function cleans up its global variables when done (if you loose variables, this may be the cause -- see around line 570)
# - Added the outputting of K values
# - Implementation of constants of Cai and Wang, 1998
# - Implementation of constants of Lueker et al., 2000
# - Implementation of constants of Mojica-Prieto and Millero, 2002
# - Implementation of constants of Millero et al., 2002 (only their eqs. 19, 20, no TCO2 dependency)
# - Implementation of constants of Millero et al., 2006
# - Implementation of constants of Millero et al., 2010
# - Properly listed Sal and Temp limits for the available constants
# - added switch for using the new Lee et al., (2010) formulation of Total Borate (see KSO4CONSTANTS above)
# - Minor corrections to the GEOSECS constants (gave NaN for some output in earlier version)
# - Fixed decimal point error on [H+] (did not get converted to umol/kgSW from mol/kgSW).
# - Changed 'Hfreein' to 'Hfreeout' in the 'NICEHEADERS'-output (typo)
#
# **** Changes since 1.00 (uploaded to CDIAC at May 29th, 2009):
# - added a note explaining that all known bugs were removed before release of 1.00
#
#**************************************************************************
#
# CO2SYS originally by Lewis and Wallace 1998
# Converted to MATLAB by Denis Pierrot at
# CIMAS, University of Miami, Miami, Florida
# Vectorization, internal refinements and speed improvements by
# Steven van Heuven, University of Groningen, The Netherlands.
# Questions, bug reports et cetera (MATLAB): svheuven@gmail.com
# Conversion to Python by Matthew Humphreys, NIOZ Royal Netherlands Institute
# for Sea Research, Texel, and Utrecht University, the Netherlands.
# Questions, bug reports et cetera (Python): m.p.humphreys@icloud.com
#
#**************************************************************************


#**************************************************************************
# NOTHING BELOW THIS SHOULD REQUIRE EDITING BY USER!
#**************************************************************************

from copy import deepcopy
from numpy import (array, exp, full, log, log10, logical_and, logical_or, nan,
                   ones, size, sqrt, unique, vstack, where, zeros)
from numpy import abs as np_abs
from numpy import any as np_any
from numpy import min as np_min
from numpy import max as np_max

def _Constants(TempC, Pdbar, pHScale, WhichKs, WhoseKSO4, ntps, TP, TSi, Sal):
    """Evaluate all stoichiometric equilibrium constants, converted to the
    chosen pH scale, and corrected for pressure.
    """
# SUB Constants, version 04.01, 10-13-97, written by Ernie Lewis.
# Converted from MATLAB to Python 2020-01-29 by Matthew Humphreys.
# Inputs: pHScale#, WhichKs#, WhoseKSO4#, Sali, TempCi, Pdbar
# Outputs: K0, K(), T(), fH, FugFac, VPFac
# This finds the Constants of the CO2 system in seawater or freshwater,
# corrects them for pressure, and reports them on the chosen pH scale.
# The process is as follows: the Constants (except KS, KF which stay on the
# free scale - these are only corrected for pressure) are
#       1) evaluated as they are given in the literature
#       2) converted to the SWS scale in mol/kg-SW or to the NBS scale
#       3) corrected for pressure
#       4) converted to the SWS pH scale in mol/kg-SW
#       5) converted to the chosen pH scale
#
#       PROGRAMMER'S NOTE: all logs are log base e
#       PROGRAMMER'S NOTE: all Constants are converted to the pH scale
#               pHScale# (the chosen one) in units of mol/kg-SW
#               except KS and KF are on the free scale
#               and KW is in units of (mol/kg-SW)^2
    
    RGasConstant = 83.1451 # ml bar-1 K-1 mol-1, DOEv2
    # RGasConstant = 83.14472 # # ml bar-1 K-1 mol-1, DOEv3
    TempK = TempC + 273.15
    RT = RGasConstant*TempK
    logTempK = log(TempK)
    Pbar = Pdbar/10.0

    # Generate empty vectors for holding results
    TB = full(ntps, nan)
    TF = full(ntps, nan)
    TS = full(ntps, nan)

    # Calculate total borate
    F = WhichKs==8
    if any(F): # Pure water
        TB[F] = 0.0
    F = logical_or(WhichKs==6, WhichKs==7)
    if any(F):
        TB[F] = conc.borate_C65(Sal[F])
    F = logical_and.reduce((WhichKs!=6, WhichKs!=7, WhichKs!=8))
    if any(F): # All other cases
        FF = logical_and(F, logical_or(WhoseKSO4==1, WhoseKSO4==2)) 
        if any(FF): # If user opted for Uppstrom's values
            TB[FF] = conc.borate_U74(Sal[FF])
        FF = logical_and(F, logical_or(WhoseKSO4==3, WhoseKSO4==4)) 
        if any(FF): # If user opted for the new Lee values
            TB[FF] = conc.borate_LKB10(Sal[FF])

    # Calculate total fluoride and sulfate and ionic strength
    TF = conc.fluoride_R65(Sal)
    TS = conc.sulfate_MR66(Sal)

    # Calculate K0 (Henry's constant for CO2)
    K0 = eq.kCO2_W74(TempK, Sal)

    # Calculate KS (bisulfate ion dissociation constant)
    KS = full(ntps, nan)
    F = logical_or(WhoseKSO4==1, WhoseKSO4==3)
    if any(F):
        KS[F] = eq.kHSO4_FREE_D90a(TempK[F], Sal[F])
    F = logical_or(WhoseKSO4==2, WhoseKSO4==4)
    if any(F):
        KS[F] = eq.kHSO4_FREE_KRCB77(TempK[F], Sal[F])

    # Calculate KF (hydrogen fluoride dissociation constant)
    KF = eq.kHF_FREE_DR79(TempK, Sal)

    # Calculate pH scale conversion factors:
    # These are NOT pressure-corrected.
    SWStoTOT = convert.sws2tot(TS, KS, TF, KF)
    FREEtoTOT = convert.free2tot(TS, KS)

    # Calculate fH
    fH = full(ntps, nan)
    # Use GEOSECS's value for cases 1,2,3,4,5 (and 6) to convert pH scales.
    F = WhichKs==8
    if any(F):
        fH[F] = 1.0 # this shouldn't occur in the program for this case
    F = WhichKs==7
    if any(F):
        fH[F] = convert.fH_PTBO87(TempK[F], Sal[F])
    F = logical_and(WhichKs!=7, WhichKs!=8)
    if any(F):
        fH[F] = convert.fH_TWB82(TempK[F], Sal[F])

    # Calculate boric acid dissociation constant (KB)
    KB = full(ntps, nan)
    F = WhichKs==8 # Pure water case
    if any(F):
        KB[F] = 0.0
    F = logical_or(WhichKs==6, WhichKs==7)
    if any(F):
        KB[F] = eq.kBOH3_NBS_LTB69(TempK[F], Sal[F])
        KB[F] /= fH[F] # Convert NBS to SWS
    F = logical_and.reduce((WhichKs!=6, WhichKs!=7, WhichKs!=8))
    if any(F):
        KB[F] = eq.kBOH3_TOT_D90b(TempK[F], Sal[F])
        KB[F] /= SWStoTOT[F] # Convert TOT to SWS

    # Calculate water dissociation constant (KW)
    KW = full(ntps, nan)
    F = WhichKs==7
    if any(F):
        KW[F] = eq.kH2O_SWS_M79(TempK[F], Sal[F])
    F = WhichKs==8
    if any(F):
        KW[F] = eq.kH2O_SWS_HO58_M79(TempK[F], Sal[F])
    F = logical_and.reduce((WhichKs!=6, WhichKs!=7, WhichKs!=8))
    if any(F):
        KW[F] = eq.kH2O_SWS_M95(TempK[F], Sal[F])
    # KW is on the SWS pH scale in (mol/kg-SW)**2
    F = WhichKs==6
    if any(F):
        KW[F] = 0 # GEOSECS doesn't include OH effects

    # Calculate phosphate and silicate dissociation constants
    KP1 = full(ntps, nan)
    KP2 = full(ntps, nan)
    KP3 = full(ntps, nan)
    KSi = full(ntps, nan)
    F = WhichKs==7
    if any(F):
        KP1[F], KP2[F], KP3[F] = eq.kH3PO4_NBS_KP67(TempK[F], Sal[F])
        # KP1 is already on SWS!
        KP2[F] /= fH[F] # Convert NBS to SWS
        KP3[F] /= fH[F] # Convert NBS to SWS
        KSi[F] = eq.kSi_NBS_SMB64(TempK[F], Sal[F])
        KSi[F] /= fH[F] # Convert NBS to SWS
    F = logical_or(WhichKs==6, WhichKs==8)
    if any(F):
        # Neither the GEOSECS choice nor the freshwater choice
        # include contributions from phosphate or silicate.
        KP1[F] = 0.0
        KP2[F] = 0.0
        KP3[F] = 0.0
        KSi[F] = 0.0
    F = logical_and.reduce((WhichKs!=6, WhichKs!=7, WhichKs!=8))
    if any(F):
        KP1[F], KP2[F], KP3[F] = eq.kH3PO4_SWS_YM95(TempK[F], Sal[F])
        KSi[F] = eq.kSi_SWS_YM95(TempK[F], Sal[F])

    # Calculate carbonic acid dissociation constants (K1 and K2)
    K1 = full(ntps, nan)
    K2 = full(ntps, nan)
    F = WhichKs==1
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_TOT_RRV93(TempK[F], Sal[F])
        K1[F] /= SWStoTOT[F] # Convert TOT to SWS
        K2[F] /= SWStoTOT[F] # Convert TOT to SWS
    F = WhichKs==2
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_GP89(TempK[F], Sal[F])
    F = WhichKs==3
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_H73_DM87(TempK[F], Sal[F])
    F = WhichKs==4
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_MCHP73_DM87(TempK[F], Sal[F])
    F = WhichKs==5
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_HM_DM87(TempK[F], Sal[F])
    F = logical_or(WhichKs==6, WhichKs==7)
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_NBS_MCHP73(TempK[F], Sal[F])
        K1[F] /= fH[F] # Convert NBS to SWS
        K2[F] /= fH[F] # Convert NBS to SWS
    F = WhichKs==8
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_M79(TempK[F], Sal[F])
    F = WhichKs==9
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_NBS_CW98(TempK[F], Sal[F])
        K1[F] /= fH[F] # Convert NBS to SWS
        K2[F] /= fH[F] # Convert NBS to SWS
    F = WhichKs==10
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_TOT_LDK00(TempK[F], Sal[F])
        K1[F] /= SWStoTOT[F] # Convert TOT to SWS
        K2[F] /= SWStoTOT[F] # Convert TOT to SWS
    F = WhichKs==11
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_MM02(TempK[F], Sal[F])
    F = WhichKs==12
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_MPL02(TempK[F], Sal[F])
    F = WhichKs==13
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_MGH06(TempK[F], Sal[F])
    F = WhichKs==14
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_M10(TempK[F], Sal[F])
    F = WhichKs==15
    # Added by J. C. Orr on 4 Dec 2016
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_WMW14(TempK[F], Sal[F])

    # From CO2SYS_v1_21.m: calculate KH2S and KNH3
    KH2S = full(ntps, nan)
    KNH3 = full(ntps, nan)
    F = logical_or.reduce((WhichKs==6, WhichKs==7, WhichKs==8))
    # Contributions from Ammonium and H2S not included for these options.
    if any(F):
        KH2S[F] = 0.0
        KNH3[F] = 0.0
    F = logical_and.reduce((WhichKs!=6, WhichKs!=7, WhichKs!=8))
    if any(F):
        KH2S[F] = eq.kH2S_TOT_YM95(TempK[F], Sal[F])
        KNH3[F] = eq.kNH3_TOT_CW95(TempK[F], Sal[F])
        KH2S[F] /= SWStoTOT[F] # Convert TOT to SWS
        KNH3[F] /= SWStoTOT[F] # Convert TOT to SWS
        
#****************************************************************************
# Correct dissociation constants for pressure
# Currently: For WhichKs# = 1 to 7, all Ks (except KF and KS, which are on
#       the free scale) are on the SWS scale.
#       For WhichKs# = 6, KW set to 0, KP1, KP2, KP3, KSi don't matter.
#       For WhichKs# = 8, K1, K2, and KW are on the "pH" pH scale
#       (the pH scales are the same in this case); the other Ks don't
#       matter.
#
# No salinity dependence is given for the pressure coefficients here.
# It is assumed that the salinity is at or very near Sali = 35.
# These are valid for the SWS pH scale, but the difference between this and
# the total only yields a difference of .004 pH units at 1000 bars, much
# less than the uncertainties in the values.
#****************************************************************************
# The sources used are:
# Millero, 1995:
#       Millero, F. J., Thermodynamics of the carbon dioxide system in the
#       oceans, Geochemica et Cosmochemica Acta 59:661-677, 1995.
#       See table 9 and eqs. 90-92, p. 675.
#       TYPO: a factor of 10^3 was left out of the definition of Kappa
#       TYPO: the value of R given is incorrect with the wrong units
#       TYPO: the values of the a's for H2S and H2O are from the 1983
#                values for fresh water
#       TYPO: the value of a1 for B(OH)3 should be +.1622
#        Table 9 on p. 675 has no values for Si.
#       There are a variety of other typos in Table 9 on p. 675.
#       There are other typos in the paper, and most of the check values
#       given don't check.
# Millero, 1992:
#       Millero, Frank J., and Sohn, Mary L., Chemical Oceanography,
#       CRC Press, 1992. See chapter 6.
#       TYPO: this chapter has numerous typos (eqs. 36, 52, 56, 65, 72,
#               79, and 96 have typos).
# Millero, 1983:
#       Millero, Frank J., Influence of pressure on chemical processes in
#       the sea. Chapter 43 in Chemical Oceanography, eds. Riley, J. P. and
#       Chester, R., Academic Press, 1983.
#       TYPO: p. 51, eq. 94: the value -26.69 should be -25.59
#       TYPO: p. 51, eq. 95: the term .1700t should be .0800t
#       these two are necessary to match the values given in Table 43.24
# Millero, 1979:
#       Millero, F. J., The thermodynamics of the carbon dioxide system
#       in seawater, Geochemica et Cosmochemica Acta 43:1651-1661, 1979.
#       See table 5 and eqs. 7, 7a, 7b on pp. 1656-1657.
# Takahashi et al, in GEOSECS Pacific Expedition, v. 3, 1982.
#       TYPO: the pressure dependence of K2 should have a 16.4, not 26.4
#       This matches the GEOSECS results and is in Edmond and Gieskes.
# Culberson, C. H. and Pytkowicz, R. M., Effect of pressure on carbonic acid,
#       boric acid, and the pH of seawater, Limnology and Oceanography
#       13:403-417, 1968.
# Edmond, John M. and Gieskes, J. M. T. M., The calculation of the degree of
#       seawater with respect to calcium carbonate under in situ conditions,
#       Geochemica et Cosmochemica Acta, 34:1261-1291, 1970.
#****************************************************************************
# These references often disagree and give different fits for the same thing.
# They are not always just an update either; that is, Millero, 1995 may agree
#       with Millero, 1979, but differ from Millero, 1983.
# For WhichKs# = 7 (Peng choice) I used the same factors for KW, KP1, KP2,
#       KP3, and KSi as for the other cases. Peng et al didn't consider the
#       case of P different from 0. GEOSECS did consider pressure, but didn't
#       include Phos, Si, or OH, so including the factors here won't matter.
# For WhichKs# = 8 (freshwater) the values are from Millero, 1983 (for K1, K2,
#       and KW). The other aren't used (TB = TS = TF = TP = TSi = 0.), so
#       including the factors won't matter.
#****************************************************************************
#       deltaVs are in cm3/mole
#       Kappas are in cm3/mole/bar
#****************************************************************************

    # Correct K1, K2 and KB for pressure:
    deltaV = full(ntps, nan)
    Kappa = full(ntps, nan)
    lnK1fac = full(ntps, nan)
    lnK2fac = full(ntps, nan)
    lnKBfac = full(ntps, nan)
    F = WhichKs==8
    if any(F):
        # Pressure effects on K1 in freshwater: this is from Millero, 1983.
        deltaV[F]  = -30.54 + 0.1849 *TempC[F] - 0.0023366*TempC[F]**2
        Kappa[F]   = (-6.22 + 0.1368 *TempC[F] - 0.001233 *TempC[F]**2)/1000
        lnK1fac[F] = (-deltaV[F] + 0.5*Kappa[F]*Pbar[F])*Pbar[F]/RT[F]
        # Pressure effects on K2 in freshwater: this is from Millero, 1983.
        deltaV[F]  = -29.81 + 0.115*TempC[F] - 0.001816*TempC[F]**2
        Kappa[F]   = (-5.74 + 0.093*TempC[F] - 0.001896*TempC[F]**2)/1000
        lnK2fac[F] = (-deltaV[F] + 0.5*Kappa[F]*Pbar[F])*Pbar[F]/RT[F]
        lnKBfac[F] = 0 #; this doesn't matter since TB = 0 for this case
    F = logical_or(WhichKs==6, WhichKs==7)
    if any(F):
        # GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
        # Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
        # Culberson and Pytkowicz, L and O 13:403-417, 1968:
        # but the fits are the same as those in
        # Edmond and Gieskes, GCA, 34:1261-1291, 1970
        # who in turn quote Li, personal communication
        lnK1fac[F] = (24.2 - 0.085*TempC[F])*Pbar[F]/RT[F]
        lnK2fac[F] = (16.4 - 0.04 *TempC[F])*Pbar[F]/RT[F]
        #               Takahashi et al had 26.4, but 16.4 is from Edmond and Gieskes
        #               and matches the GEOSECS results
        lnKBfac[F] = (27.5 - 0.095*TempC[F])*Pbar[F]/RT[F]
    F=logical_and.reduce((WhichKs!=6, WhichKs!=7, WhichKs!=8))
    if any(F):
        #***PressureEffectsOnK1:
        #               These are from Millero, 1995.
        #               They are the same as Millero, 1979 and Millero, 1992.
        #               They are from data of Culberson and Pytkowicz, 1968.
        deltaV[F]  = -25.5 + 0.1271*TempC[F]
        #                 'deltaV = deltaV - .151*(Sali - 34.8); # Millero, 1979
        Kappa[F]   = (-3.08 + 0.0877*TempC[F])/1000
        #                 'Kappa = Kappa  - .578*(Sali - 34.8)/1000.; # Millero, 1979
        lnK1fac[F] = (-deltaV[F] + 0.5*Kappa[F]*Pbar[F])*Pbar[F]/RT[F]
        #               The fits given in Millero, 1983 are somewhat different.

        #***PressureEffectsOnK2:
        #               These are from Millero, 1995.
        #               They are the same as Millero, 1979 and Millero, 1992.
        #               They are from data of Culberson and Pytkowicz, 1968.
        deltaV[F]  = -15.82 - 0.0219*TempC[F]
        #                  'deltaV = deltaV + .321*(Sali - 34.8); # Millero, 1979
        Kappa[F]   = (1.13 - 0.1475*TempC[F])/1000
        #                 'Kappa = Kappa - .314*(Sali - 34.8)/1000: # Millero, 1979
        lnK2fac[F] = (-deltaV[F] + 0.5*Kappa[F]*Pbar[F])*Pbar[F]/RT[F]
        #               The fit given in Millero, 1983 is different.
        #               Not by a lot for deltaV, but by much for Kappa. #

        #***PressureEffectsOnKB:
        #               This is from Millero, 1979.
        #               It is from data of Culberson and Pytkowicz, 1968.
        deltaV[F]  = -29.48 + 0.1622*TempC[F] - 0.002608*TempC[F]**2
        #               Millero, 1983 has:
        #                 'deltaV = -28.56 + .1211*TempCi - .000321*TempCi*TempCi
        #               Millero, 1992 has:
        #                 'deltaV = -29.48 + .1622*TempCi + .295*(Sali - 34.8)
        #               Millero, 1995 has:
        #                 'deltaV = -29.48 - .1622*TempCi - .002608*TempCi*TempCi
        #                 'deltaV = deltaV + .295*(Sali - 34.8); # Millero, 1979
        Kappa[F]   = -2.84/1000 # Millero, 1979
        #               Millero, 1992 and Millero, 1995 also have this.
        #                 'Kappa = Kappa + .354*(Sali - 34.8)/1000: # Millero,1979
        #               Millero, 1983 has:
        #                 'Kappa = (-3 + .0427*TempCi)/1000
        lnKBfac[F] = (-deltaV[F] + 0.5*Kappa[F]*Pbar[F])*Pbar[F]/RT[F]

    # CorrectKWForPressure:
    lnKWfac = full(ntps, nan)
    F=(WhichKs==8)
    if any(F):
        # PressureEffectsOnKWinFreshWater:
        #               This is from Millero, 1983.
        deltaV[F]  =  -25.6 + 0.2324*TempC[F] - 0.0036246*TempC[F]**2
        Kappa[F]   = (-7.33 + 0.1368*TempC[F] - 0.001233 *TempC[F]**2)/1000
        lnKWfac[F] = (-deltaV[F] + 0.5*Kappa[F]*Pbar[F])*Pbar[F]/RT[F]

        #               NOTE the temperature dependence of KappaK1 and KappaKW
        #               for fresh water in Millero, 1983 are the same.
    F=(WhichKs!=8)
    if any(F):
        # GEOSECS doesn't include OH term, so this won't matter.
        # Peng et al didn't include pressure, but here I assume that the KW correction
        #       is the same as for the other seawater cases.
        # PressureEffectsOnKW:
        #               This is from Millero, 1983 and his programs CO2ROY(T).BAS.
        deltaV[F]  = -20.02 + 0.1119*TempC[F] - 0.001409*TempC[F]**2
        #               Millero, 1992 and Millero, 1995 have:
        Kappa[F]   = (-5.13 + 0.0794*TempC[F])/1000 # Millero, 1983
        #               Millero, 1995 has this too, but Millero, 1992 is different.
        lnKWfac[F] = (-deltaV[F] + 0.5*Kappa[F]*Pbar[F])*Pbar[F]/RT[F]
        #               Millero, 1979 does not list values for these.

    # PressureEffectsOnKF:
    #       This is from Millero, 1995, which is the same as Millero, 1983.
    #       It is assumed that KF is on the free pH scale.
    deltaV = -9.78 - 0.009*TempC - 0.000942*TempC**2
    Kappa = (-3.91 + 0.054*TempC)/1000
    lnKFfac = (-deltaV + 0.5*Kappa*Pbar)*Pbar/RT
    # PressureEffectsOnKS:
    #       This is from Millero, 1995, which is the same as Millero, 1983.
    #       It is assumed that KS is on the free pH scale.
    deltaV = -18.03 + 0.0466*TempC + 0.000316*TempC**2
    Kappa = (-4.53 + 0.09*TempC)/1000
    lnKSfac = (-deltaV + 0.5*Kappa*Pbar)*Pbar/RT

    # CorrectKP1KP2KP3KSiForPressure:
    # These corrections don't matter for the GEOSECS choice (WhichKs# = 6) and
    #       the freshwater choice (WhichKs# = 8). For the Peng choice I assume
    #       that they are the same as for the other choices (WhichKs# = 1 to 5).
    # The corrections for KP1, KP2, and KP3 are from Millero, 1995, which are the
    #       same as Millero, 1983.
    # PressureEffectsOnKP1:
    deltaV = -14.51 + 0.1211*TempC - 0.000321*TempC**2
    Kappa  = (-2.67 + 0.0427*TempC)/1000
    lnKP1fac = (-deltaV + 0.5*Kappa*Pbar)*Pbar/RT
    # PressureEffectsOnKP2:
    deltaV = -23.12 + 0.1758*TempC - 0.002647*TempC**2
    Kappa  = (-5.15 + 0.09  *TempC)/1000
    lnKP2fac = (-deltaV + 0.5*Kappa*Pbar)*Pbar/RT
    # PressureEffectsOnKP3:
    deltaV = -26.57 + 0.202 *TempC - 0.003042*TempC**2
    Kappa  = (-4.08 + 0.0714*TempC)/1000
    lnKP3fac = (-deltaV + 0.5*Kappa*Pbar)*Pbar/RT
    # PressureEffectsOnKSi:
    #  The only mention of this is Millero, 1995 where it is stated that the
    #    values have been estimated from the values of boric acid. HOWEVER,
    #    there is no listing of the values in the table.
    #    I used the values for boric acid from above.
    deltaV = -29.48 + 0.1622*TempC - 0.002608*TempC**2
    Kappa  = -2.84/1000
    lnKSifac = (-deltaV + 0.5*Kappa*Pbar)*Pbar/RT

    # CorrectKNH3KH2SForPressure:
    # The corrections are from Millero, 1995, which are the
    #       same as Millero, 1983.
    # PressureEffectsOnKNH3:
    deltaV = -26.43 + 0.0889*TempC - 0.000905*TempC**2
    Kappa = (-5.03 + 0.0814*TempC)/1000
    lnKNH3fac = (-deltaV + 0.5*Kappa*Pbar)*Pbar/RT
    # PressureEffectsOnKH2S:
    # Millero 1995 gives values for deltaV in fresh water instead of SW.
    # Millero 1995 gives -b0 as -2.89 instead of 2.89
    # Millero 1983 is correct for both
    deltaV = -11.07 - 0.009*TempC - 0.000942*TempC**2
    Kappa = (-2.89 + 0.054*TempC)/1000
    lnKH2Sfac = (-deltaV + 0.5*Kappa*Pbar)*Pbar/RT

    # CorrectKsForPressureHere:
    K1 *= exp(lnK1fac)
    K2 *= exp(lnK2fac)
    KW *= exp(lnKWfac)
    KB *= exp(lnKBfac)
    KF *= exp(lnKFfac)
    KS *= exp(lnKSfac)
    KP1 *= exp(lnKP1fac)
    KP2 *= exp(lnKP2fac)
    KP3 *= exp(lnKP3fac)
    KSi *= exp(lnKSifac)
    KNH3 *= exp(lnKNH3fac)
    KH2S *= exp(lnKH2Sfac)

    # CorrectpHScaleConversionsForPressure:
    # fH has been assumed to be independent of pressure.
    SWStoTOT = convert.sws2tot(TS, KS, TF, KF)
    FREEtoTOT = convert.free2tot(TS, KS)

    #  The values KS and KF are already pressure-corrected, so the pH scale
    #  conversions are now valid at pressure.

    # Find pH scale conversion factor: this is the scale they will be put on
    pHfactor = full(ntps, nan)
    F = pHScale==1 # Total
    pHfactor[F] = SWStoTOT[F]
    F = pHScale==2 # SWS, they are all on this now
    pHfactor[F] = 1.0
    F = pHScale==3 # pHfree
    pHfactor[F] = SWStoTOT[F]/FREEtoTOT[F]
    F = pHScale==4 # pHNBS
    pHfactor[F] = fH[F]

    # Convert from SWS pH scale to chosen scale
    K1 *= pHfactor
    K2 *= pHfactor
    KW *= pHfactor
    KB *= pHfactor
    KP1 *= pHfactor
    KP2 *= pHfactor
    KP3 *= pHfactor
    KSi *= pHfactor

    # CalculateFugacityConstants:
    # This assumes that the pressure is at one atmosphere, or close to it.
    # Otherwise, the Pres term in the exponent affects the results.
    #       Weiss, R. F., Marine Chemistry 2:203-215, 1974.
    #       Delta and B in cm3/mol
    FugFac = ones(ntps)
    Delta = (57.7 - 0.118*TempK)
    b = -1636.75 + 12.0408*TempK - 0.0327957*TempK**2 + 3.16528*0.00001*TempK**3
    # For a mixture of CO2 and air at 1 atm (at low CO2 concentrations):
    P1atm = 1.01325 # in bar
    FugFac = exp((b + 2*Delta)*P1atm/RT)
    F=logical_or(WhichKs==6, WhichKs==7) # GEOSECS and Peng assume pCO2 = fCO2, or FugFac = 1
    FugFac[F] = 1.0
    # CalculateVPFac:
    # Weiss, R. F., and Price, B. A., Nitrous oxide solubility in water and
    #       seawater, Marine Chemistry 8:347-359, 1980.
    # They fit the data of Goff and Gratch (1946) with the vapor pressure
    #       lowering by sea salt as given by Robinson (1954).
    # This fits the more complicated Goff and Gratch, and Robinson equations
    #       from 273 to 313 deg K and 0 to 40 Sali with a standard error
    #       of .015#, about 5 uatm over this range.
    # This may be on IPTS-29 since they didn't mention the temperature scale,
    #       and the data of Goff and Gratch came before IPTS-48.
    # The references are:
    # Goff, J. A. and Gratch, S., Low pressure properties of water from -160 deg
    #       to 212 deg F, Transactions of the American Society of Heating and
    #       Ventilating Engineers 52:95-122, 1946.
    # Robinson, Journal of the Marine Biological Association of the U. K.
    #       33:449-455, 1954.
    #       This is eq. 10 on p. 350.
    #       This is in atmospheres.
    VPWP = exp(24.4543 - 67.4509*(100/TempK) - 4.8489*log(TempK/100))
    VPCorrWP = exp(-0.000544*Sal)
    VPSWWP = VPWP*VPCorrWP
    VPFac = 1.0 - VPSWWP # this assumes 1 atmosphere
    return (K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KH2S, KNH3, TB, TF, TS,
            RGasConstant, RT, K0, fH, FugFac, VPFac, TempK, logTempK, Pbar)

def _CalculatepHfromTATC(TAx, TCx):
    global pHScale, WhichKs, WhoseKSO4, sqrSal, Pbar, RT
    global K0, fH, FugFac, VPFac, ntps, TempK, logTempK
    global K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S
    global TB, TF, TS, TP, TSi, TNH3, TH2S, F
    #Outputs pH
    # SUB CalculatepHfromTATC, version 04.01, 10-13-96, written by Ernie Lewis.
    # Inputs: TA, TC, K(), T()
    # Output: pH
    # This calculates pH from TA and TC using K1 and K2 by Newton's method.
    # It tries to solve for the pH at which Residual = 0.
    # The starting guess is pH = 8.
    # Though it is coded for H on the total pH scale, for the pH values occuring
    # in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    # negligible) as long as the K Constants are on that scale.
    #
    # Made this to accept vectors. It will continue iterating until all
    # values in the vector are "abs(deltapH) < pHTol". SVH2007
    K1F=K1[F];   K2F=K2[F];   KWF =KW[F];
    KP1F=KP1[F]; KP2F=KP2[F]; KP3F=KP3[F];  TPF=TP[F];
    TSiF=TSi[F]; KSiF=KSi[F]; TBF =TB[F];   KBF=KB[F];
    TSF =TS[F];  KSF =KS[F];  TFF =TF[F];   KFF=KF[F];
    TNH3F = TNH3[F]; TH2SF = TH2S[F]; KNH3F = KNH3[F]; KH2SF = KH2S[F]
    vl          = sum(F)  # VectorLength
    pHGuess     = 8       # this is the first guess
    pHTol       = 0.0001  # tolerance for iterations end
    ln10        = log(10) #
    pHx         = full(vl, pHGuess) # creates a vector holding the first guess for all samples
    deltapH     = pHTol+1
    while np_any(np_abs(deltapH) > pHTol):
        H         = 10.0**(-pHx)
        Denom     = (H*H + K1F*H + K1F*K2F)
        CAlk      = TCx*K1F*(H + 2*K2F)/Denom
        BAlk      = TBF*KBF/(KBF + H)
        OH        = KWF/H
        PhosTop   = KP1F*KP2F*H + 2*KP1F*KP2F*KP3F - H*H*H
        PhosBot   = H*H*H + KP1F*H*H + KP1F*KP2F*H + KP1F*KP2F*KP3F
        PAlk      = TPF*PhosTop/PhosBot
        SiAlk     = TSiF*KSiF/(KSiF + H)
        NH3Alk    = TNH3F*KNH3F/(KNH3F + H)
        H2SAlk    = TH2SF*KH2SF/(KH2SF + H)
        FREEtoTOT = convert.free2tot(TSF, KSF) # pH scale conversion factor
        Hfree     = H/FREEtoTOT # for H on the total scale
        HSO4      = TSF/(1 + KSF/Hfree) # since KS is on the free scale
        HF        = TFF/(1 + KFF/Hfree) # since KF is on the free scale
        Residual  = (TAx - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk +
                     Hfree + HSO4 + HF)
        # find Slope dTA/dpH;
        # (this is not exact, but keeps all important terms);
        Slope     = ln10*(TCx*K1F*H*(H*H + K1F*K2F + 4*H*K2F)/Denom/Denom + BAlk*H/(KBF + H) + OH + H)
        deltapH   = Residual/Slope # this is Newton's method
        # to keep the jump from being too big;
        while any(np_abs(deltapH) > 1):
            FF=np_abs(deltapH)>1; deltapH[FF]=deltapH[FF]/2
        pHx = pHx + deltapH # Is on the same scale as K1 and K2 were calculated...
    return pHx

def _CalculatefCO2fromTCpH(TCx, pHx):
    global K0, K1, K2, F
# SUB CalculatefCO2fromTCpH, version 02.02, 12-13-96, written by Ernie Lewis.
# Inputs: TC, pH, K0, K1, K2
# Output: fCO2
# This calculates fCO2 from TC and pH, using K0, K1, and K2.
    H = 10.0**(-pHx)
    fCO2x = TCx*H*H/(H*H + K1[F]*H + K1[F]*K2[F])/K0[F]
    return fCO2x

def _CalculatepHfCO2fromTATC(TAx, TCx):
    global FugFac, F
# Outputs pH fCO2, in that order
# SUB FindpHfCO2fromTATC, version 01.02, 10-10-97, written by Ernie Lewis.
# Inputs: pHScale%, WhichKs%, WhoseKSO4%, TA, TC, Sal, K(), T(), TempC, Pdbar
# Outputs: pH, fCO2
# This calculates pH and fCO2 from TA and TC at output conditions.
    pHx = _CalculatepHfromTATC(TAx, TCx) # pH is returned on the scale requested in "pHscale" (see 'constants'...)
    fCO2x = _CalculatefCO2fromTCpH(TCx, pHx)
    return pHx, fCO2x

def _CalculateTCfromTApH(TAx, pHx):
    global K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S
    global TB, TF, TS, TP, TSi, TNH3, TH2S, F
    K1F=K1[F];   K2F=K2[F];   KWF=KW[F];
    KP1F=KP1[F]; KP2F=KP2[F]; KP3F=KP3[F]; TPF=TP[F];
    TSiF=TSi[F]; KSiF=KSi[F]; TBF=TB[F];   KBF=KB[F];
    TSF=TS[F];   KSF=KS[F];   TFF=TF[F];   KFF=KF[F];
    TNH3F = TNH3[F]; TH2SF = TH2S[F]; KNH3F = KNH3[F]; KH2SF = KH2S[F]
# SUB CalculateTCfromTApH, version 02.03, 10-10-97, written by Ernie Lewis.
# Inputs: TA, pH, K(), T()
# Output: TC
# This calculates TC from TA and pH.
# Though it is coded for H on the total pH scale, for the pH values occuring
# in seawater (pH > 6) it will be equally valid on any pH scale (H terms
# negligible) as long as the K Constants are on that scale.
    H         = 10.0**(-pHx)
    BAlk      = TBF*KBF/(KBF + H)
    OH        = KWF/H
    PhosTop   = KP1F*KP2F*H + 2*KP1F*KP2F*KP3F - H*H*H
    PhosBot   = H*H*H + KP1F*H*H + KP1F*KP2F*H + KP1F*KP2F*KP3F
    PAlk      = TPF*PhosTop/PhosBot
    SiAlk     = TSiF*KSiF/(KSiF + H)
    NH3Alk    = TNH3F*KNH3F/(KNH3F + H)
    H2SAlk    = TH2SF*KH2SF/(KH2SF + H)
    FREEtoTOT = (1 + TSF/KSF) # pH scale conversion factor
    Hfree     = H/FREEtoTOT #' for H on the total scale
    HSO4      = TSF/(1 + KSF/Hfree) #' since KS is on the free scale
    HF        = TFF/(1 + KFF/Hfree) #' since KF is on the free scale
    CAlk      = (TAx - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree +
                 HSO4 + HF)
    TCxtemp   = CAlk*(H*H + K1F*H + K1F*K2F)/(K1F*(H + 2*K2F))
    return TCxtemp

def _CalculatepHfromTAfCO2(TAi, fCO2i):
    global K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S
    global TB, TF, TS, TP, TSi, TNH3, TH2S, F
# SUB CalculatepHfromTAfCO2, version 04.01, 10-13-97, written by Ernie Lewis.
# Inputs: TA, fCO2, K0, K(), T()
# Output: pH
# This calculates pH from TA and fCO2 using K1 and K2 by Newton's method.
# It tries to solve for the pH at which Residual = 0.
# The starting guess is pH = 8.
# Though it is coded for H on the total pH scale, for the pH values occuring
# in seawater (pH > 6) it will be equally valid on any pH scale (H terms
# negligible) as long as the K Constants are on that scale.
    K0F=K0[F];   K1F=K1[F];   K2F=K2[F];   KWF=KW[F];
    KP1F=KP1[F]; KP2F=KP2[F]; KP3F=KP3[F]; TPF=TP[F];
    TSiF=TSi[F]; KSiF=KSi[F]; TBF=TB[F];   KBF=KB[F];
    TSF=TS[F];   KSF=KS[F];   TFF=TF[F];   KFF=KF[F];
    TNH3F = TNH3[F]; TH2SF = TH2S[F]; KNH3F = KNH3[F]; KH2SF = KH2S[F]
    vl         = sum(F) # vectorlength
    pHGuess    = 8      # this is the first guess
    pHTol      = 0.0001 # tolerance
    ln10       = log(10)
    pH         = full(vl, pHGuess)
    deltapH = pHTol+pH
    while np_any(np_abs(deltapH) > pHTol):
        H         = 10.0**(-pH)
        HCO3      = K0F*K1F*fCO2i/H
        CO3       = K0F*K1F*K2F*fCO2i/(H*H)
        CAlk      = HCO3 + 2*CO3
        BAlk      = TBF*KBF/(KBF + H)
        OH        = KWF/H
        PhosTop   = KP1F*KP2F*H + 2*KP1F*KP2F*KP3F - H*H*H
        PhosBot   = H*H*H + KP1F*H*H + KP1F*KP2F*H + KP1F*KP2F*KP3F
        PAlk      = TPF*PhosTop/PhosBot
        SiAlk     = TSiF*KSiF/(KSiF + H)
        NH3Alk    = TNH3F*KNH3F/(KNH3F + H)
        H2SAlk    = TH2SF*KH2SF/(KH2SF + H)
        FREEtoTOT = convert.free2tot(TSF, KSF) # ' pH scale conversion factor
        Hfree     = H/FREEtoTOT#' for H on the total scale
        HSO4      = TSF/(1 + KSF/Hfree) #' since KS is on the free scale
        HF        = TFF/(1 + KFF/Hfree)# ' since KF is on the free scale
        Residual  = (TAi - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk +
                     Hfree + HSO4 + HF)
        #               find Slope dTA/dpH
        #               (this is not exact, but keeps all important terms):
        Slope     = ln10*(HCO3 + 4*CO3 + BAlk*H/(KBF + H) + OH + H)
        deltapH   = Residual/Slope # this is Newton's method
        # to keep the jump from being too big:
        while np_any(np_abs(deltapH) > 1):
            FF=np_abs(deltapH)>1; deltapH[FF]=deltapH[FF]/2
        pH = pH + deltapH
    return pH

def _CalculateTAfromTCpH(TCi, pHi):
    global K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S
    global TB, TF, TS, TP, TSi, TNH3, TH2S, F
# SUB CalculateTAfromTCpH, version 02.02, 10-10-97, written by Ernie Lewis.
# Inputs: TC, pH, K(), T()
# Output: TA
# This calculates TA from TC and pH.
# Though it is coded for H on the total pH scale, for the pH values occuring
# in seawater (pH > 6) it will be equally valid on any pH scale (H terms
# negligible) as long as the K Constants are on that scale.
    K1F=K1[F];   K2F=K2[F];   KWF=KW[F];
    KP1F=KP1[F]; KP2F=KP2[F]; KP3F=KP3[F]; TPF=TP[F];
    TSiF=TSi[F]; KSiF=KSi[F]; TBF=TB[F];   KBF=KB[F];
    TSF=TS[F];   KSF=KS[F];   TFF=TF[F];   KFF=KF[F];
    TNH3F = TNH3[F]; TH2SF = TH2S[F]; KNH3F = KNH3[F]; KH2SF = KH2S[F]
    H         = 10.0**(-pHi)
    CAlk      = TCi*K1F*(H + 2*K2F)/(H*H + K1F*H + K1F*K2F)
    BAlk      = TBF*KBF/(KBF + H)
    OH        = KWF/H
    PhosTop   = KP1F*KP2F*H + 2*KP1F*KP2F*KP3F - H*H*H
    PhosBot   = H*H*H + KP1F*H*H + KP1F*KP2F*H + KP1F*KP2F*KP3F
    PAlk      = TPF*PhosTop/PhosBot
    SiAlk     = TSiF*KSiF/(KSiF + H)
    NH3Alk    = TNH3F*KNH3F/(KNH3F + H)
    H2SAlk    = TH2SF*KH2SF/(KH2SF + H)
    FREEtoTOT = (1 + TSF/KSF) # ' pH scale conversion factor
    Hfree     = H/FREEtoTOT #' for H on the total scale
    HSO4      = TSF/(1 + KSF/Hfree)# ' since KS is on the free scale
    HF        = TFF/(1 + KFF/Hfree)# ' since KF is on the free scale
    TActemp   = (CAlk + BAlk + OH + PAlk + SiAlk - NH3Alk - H2SAlk - Hfree -
                 HSO4 - HF)
    return TActemp

def _CalculatepHfromTCfCO2(TCi, fCO2i):
    global K0, K1, K2, F
# SUB CalculatepHfromTCfCO2, version 02.02, 11-12-96, written by Ernie Lewis.
# Inputs: TC, fCO2, K0, K1, K2
# Output: pH
# This calculates pH from TC and fCO2 using K0, K1, and K2 by solving the
#       quadratic in H: fCO2*K0 = TC*H*H/(K1*H + H*H + K1*K2).
# if there is not a real root, then pH is returned as missingn.
    RR = K0[F]*fCO2i/TCi
    #       if RR >= 1
    #          varargout{1}= missingn;
    #          disp('nein!');return;
    #       end
    # check after sub to see if pH = missingn.
    Discr = (K1[F]*RR)*(K1[F]*RR) + 4*(1 - RR)*(K1[F]*K2[F]*RR)
    H     = 0.5*(K1[F]*RR + sqrt(Discr))/(1 - RR)
    #       if (H <= 0)
    #           pHctemp = missingn;
    #       else
    pHctemp = log(H)/log(0.1)
    #       end
    return pHctemp

def _CalculateTCfrompHfCO2(pHi, fCO2i):
    global K0, K1, K2, F
# SUB CalculateTCfrompHfCO2, version 01.02, 12-13-96, written by Ernie Lewis.
# Inputs: pH, fCO2, K0, K1, K2
# Output: TC
# This calculates TC from pH and fCO2, using K0, K1, and K2.
    H       = 10.0**(-pHi)
    TCctemp = K0[F]*fCO2i*(H*H + K1[F]*H + K1[F]*K2[F])/(H*H)
    return TCctemp

def _CalculatepHfromTACarb(TAi, CARBi):
    global K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S
    global TB, TF, TS, TP, TSi, F, TNH3, TH2S
    # SUB CalculatepHfromTACarb, version 01.0, 06-12-2019, written by Denis Pierrot.
    # Inputs: TA, Carbonate, K(), T()
    # Output: pH
    # This calculates pH from TA and Carb using K1 and K2 by Newton's method.
    # It tries to solve for the pH at which Residual = 0.
    # The starting guess is pH = 8.
    # Though it is coded for H on the total pH scale, for the pH values occuring
    # in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    # negligible) as long as the K constants are on that scale.
    K2F=K2[F];   KWF=KW[F];
    KP1F=KP1[F]; KP2F=KP2[F]; KP3F=KP3[F]; TPF=TP[F];
    TSiF=TSi[F]; KSiF=KSi[F]; TBF=TB[F];   KBF=KB[F];
    TSF=TS[F];   KSF=KS[F];   TFF=TF[F];   KFF=KF[F];
    TNH3F=TNH3[F]; KNH3F=KNH3[F]; TH2SF=TH2S[F];   KH2SF=KH2S[F];
    vl         = sum(F) # vectorlength
    pHGuess    = 8.0    # this is the first guess
    pHTol      = 0.0001 # tolerance
    ln10       = log(10)
    pH         = full(vl, pHGuess)
    deltapH = pHTol+pH
    loopc = 0
    nF0 = F
    K2x=K2[F];   KWx=KW[F];
    KP1x=KP1[F]; KP2x=KP2[F]; KP3x=KP3[F]; TPx=TP[F];
    TSix=TSi[F]; KSix=KSi[F]; TBx=TB[F];   KBx=KB[F];
    TSx=TS[F];   KSx=KS[F];   TFx=TF[F];   KFx=KF[F];
    TNH3x=TNH3[F]; KNH3x=KNH3[F]; TH2Sx=TH2S[F];   KH2Sx=KH2S[F];
    nF = np_abs(deltapH) > pHTol
    while any(nF):    
        if sum(nF0) > sum(nF):
            K2x=K2F[nF];   KWx=KWF[nF];
            KP1x=KP1F[nF]; KP2x=KP2F[nF]; KP3x=KP3F[nF]; TPx=TPF[nF];
            TSix=TSiF[nF]; KSix=KSiF[nF]; TBx=TBF[nF];   KBx=KBF[nF];
            TSx=TSF[nF];   KSx=KSF[nF];   TFx=TFF[nF];   KFx=KFF[nF];
            TNH3x=TNH3F[nF]; KNH3x=KNH3F[nF]; TH2Sx=TH2SF[nF];   KH2Sx=KH2SF[nF];
        nF0 = deepcopy(nF)
        pHx = pH[nF]
        CARBix = CARBi[nF]
        TAix = TAi[nF]
        H         = 10.0**-pHx
        CAlk      = CARBix*(H+2*K2x)/K2x
        BAlk      = TBx*KBx/(KBx + H)
        OH        = KWx/H
        PhosTop   = KP1x*KP2x*H + 2*KP1x*KP2x*KP3x - H*H*H
        PhosBot   = H*H*H + KP1x*H*H + KP1x*KP2x*H + KP1x*KP2x*KP3x
        PAlk      = TPx*PhosTop/PhosBot
        SiAlk     = TSix*KSix/(KSix + H)
        NH3Alk     = TNH3x*KNH3x/(KNH3x + H)
        H2SAlk     = TH2Sx*KH2Sx/(KH2Sx + H)
        FREEtoTOT = convert.free2tot(TSx, KSx) # pH scale conversion factor
        Hfree     = H/FREEtoTOT # for H on the total scale
        HSO4      = TSx/(1 + KSx/Hfree) # since KS is on the free scale
        HF        = TFx/(1 + KFx/Hfree) # since KF is on the free scale
        Residual  = (TAix - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk -
                     H2SAlk + Hfree + HSO4 + HF)
        #               find Slope dTA/dpH
        #               (this is not exact, but keeps all important terms):
        Slope = ln10*(-CARBix*H/K2x + BAlk*H/(KBx + H) + OH + H)
        deltapHn = Residual/Slope # this is Newton's method
        # to keep the jump from being too big:
        deltapHn[np_abs(deltapHn) > 1] = 0.9
        pHx += deltapHn
        deltapH[nF] = deltapHn
        pH[nF] = pHx
        loopc += 1
        nF = np_abs(deltapH) > pHTol
        if loopc>10000:
            Fr = where(F)
            pH[nF] = nan
            print('pH value did not converge for data on row(s): {}'.format(Fr))
            deltapH[nF] = pHTol*0.9
            nF= np_abs(deltapH) > pHTol
    return pH

def _CalculatepHfromTCCarb(TCi, Carbi):
    global K1, K2, F
# SUB CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, written by Denis Pierrot.
# Inputs: Carbonate Ions, TC, K1, K2
# Output: pH
# This calculates pH from Carbonate and TC using K1, and K2 by solving the
#       quadratic in H: TC * K1 * K2= Carb * (H * H + K1 * H +  K1 * K2).
    RR = 1 - TCi/Carbi
    Discr = K1[F]**2 - 4*K1[F]*K2[F]*RR
    H = (-K1[F] + sqrt(Discr))/2
    pHctemp = log(H)/log(0.1)
    return pHctemp

def _CalculatefCO2frompHCarb(pHx, Carbx):
    global K0, K1, K2, F
# SUB CalculatefCO2frompHCarb, version 01.0, 06-12-2019, written by Denis Pierrot
# Inputs: Carb, pH, K0, K1, K2
# Output: fCO2
# This calculates fCO2 from Carb and pH, using K0, K1, and K2.
    H = 10.0**-pHx
    fCO2x = Carbx*H**2/(K0[F]*K1[F]*K2[F])
    return fCO2x

def _CalculatepHfromfCO2Carb(fCO2i, Carbi):
    global K0, K1, K2, F
# SUB CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, written by Denis Pierrot.
# Inputs: Carbonate Ions, fCO2, K0, K1, K2
# Output: pH
# This calculates pH from Carbonate and fCO2 using K0, K1, and K2 by solving the
#       equation in H: fCO2 * K0 * K1* K2 = Carb * H * H
    RR = (K0[F]*K1[F]*K2[F]*fCO2i)/Carbi
    H = sqrt(RR)
    pHctemp = log(H)/log(0.1)
    return pHctemp

def _CalculateCarbfromTCpH(TCx, pHx):
    global K1, K2, F
# SUB CalculateCarbfromTCpH, version 01.0, 06-12-2019, written by Denis Pierrot.
# Inputs: TC, pH, K0, K1, K2
# Output: Carbonate
# This calculates Carbonate from TC and pH, using K0, K1, and K2.
    H = 10.0**-pHx
    Carbx = TCx*K1[F]*K2[F]/(H*H + K1[F]*H + K1[F]*K2[F])
    return Carbx

def _CalculateAlkParts(pHx, TCx):
    global K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S
    global TB, TF, TS, TP, TSi, TNH3, TH2S, F
# SUB CalculateAlkParts, version 01.03, 10-10-97, written by Ernie Lewis.
# Inputs: pH, TC, K(), T()
# Outputs: HCO3, CO3, BAlk, OH, PAlk, SiAlk, Hfree, HSO4, HF
# This calculates the various contributions to the alkalinity.
# Though it is coded for H on the total pH scale, for the pH values occuring
# in seawater (pH > 6) it will be equally valid on any pH scale (H terms
# negligible) as long as the K Constants are on that scale.
    H         = 10.0**(-pHx)
    HCO3      = TCx*K1*H  /(K1*H + H*H + K1*K2)
    CO3       = TCx*K1*K2/(K1*H + H*H + K1*K2)
    BAlk      = TB*KB/(KB + H)
    OH        = KW/H
    PhosTop   = KP1*KP2*H + 2*KP1*KP2*KP3 - H*H*H
    PhosBot   = H*H*H + KP1*H*H + KP1*KP2*H + KP1*KP2*KP3
    PAlk      = TP*PhosTop/PhosBot
    # this is good to better than .0006*TP:
    # PAlk = TP*(-H/(KP1+H) + KP2/(KP2+H) + KP3/(KP3+H))
    SiAlk     = TSi*KSi/(KSi + H)
    NH3Alk    = TNH3*KNH3/(KNH3 + H)
    H2SAlk    = TH2S*KH2S/(KH2S + H)
    FREEtoTOT = convert.free2tot(TS, KS) # pH scale conversion factor
    Hfree     = H/FREEtoTOT          #' for H on the total scale
    HSO4      = TS/(1 + KS/Hfree) #' since KS is on the free scale
    HF        = TF/(1 + KF/Hfree) #' since KF is on the free scale
    return HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF

def _RevelleFactor(TAi, TCi):
# global WhichKs;
# SUB RevelleFactor, version 01.03, 01-07-97, written by Ernie Lewis.
# Inputs: WhichKs#, TA, TC, K0, K(), T()
# Outputs: Revelle
# This calculates the Revelle factor (dfCO2/dTC)|TA/(fCO2/TC).
# It only makes sense to talk about it at pTot = 1 atm, but it is computed
#       here at the given K(), which may be at pressure <> 1 atm. Care must
#       thus be used to see if there is any validity to the number computed.
    TC0 = deepcopy(TCi)
    dTC = 0.000001 # ' 1 umol/kg-SW
    # ' Find fCO2 at TA, TC + dTC
    TCi = TC0 + dTC
    pHc= _CalculatepHfromTATC(TAi, TCi)
    fCO2c= _CalculatefCO2fromTCpH(TCi, pHc)
    fCO2plus = deepcopy(fCO2c)
    # ' Find fCO2 at TA, TC - dTC
    TCi = TC0 - dTC
    pHc= _CalculatepHfromTATC(TAi, TCi)
    fCO2c= _CalculatefCO2fromTCpH(TCi, pHc)
    fCO2minus = deepcopy(fCO2c)
    # CalculateRevelleFactor:
    Revelle = (fCO2plus - fCO2minus)/dTC/((fCO2plus + fCO2minus)/TCi);
    return Revelle

def _CaSolubility(Sal, TempC, Pdbar, TC, pH):
    global K1, K2, TempK, logTempK, sqrSal, Pbar, RT, WhichKs, ntps
# global PertK    # Id of perturbed K
# global Perturb  # perturbation
# ***********************************************************************
# SUB CaSolubility, version 01.05, 05-23-97, written by Ernie Lewis.
# Inputs: WhichKs#, Sal, TempCi, Pdbari, TCi, pHi, K1, K2
# Outputs: OmegaCa, OmegaAr
# This calculates omega, the solubility ratio, for calcite and aragonite.
# This is defined by: Omega = [CO3--]*[Ca++]/Ksp,
#       where Ksp is the solubility product (either KCa or KAr).
# ***********************************************************************
# These are from:
# Mucci, Alphonso, The solubility of calcite and aragonite in seawater
#       at various salinities, temperatures, and one atmosphere total
#       pressure, American Journal of Science 283:781-799, 1983.
# Ingle, S. E., Solubility of calcite in the ocean,
#       Marine Chemistry 3:301-319, 1975,
# Millero, Frank, The thermodynamics of the carbonate system in seawater,
#       Geochemica et Cosmochemica Acta 43:1651-1661, 1979.
# Ingle et al, The solubility of calcite in seawater at atmospheric pressure
#       and 35#o salinity, Marine Chemistry 1:295-307, 1973.
# Berner, R. A., The solubility of calcite and aragonite in seawater in
#       atmospheric pressure and 34.5#o salinity, American Journal of
#       Science 276:713-730, 1976.
# Takahashi et al, in GEOSECS Pacific Expedition, v. 3, 1982.
# Culberson, C. H. and Pytkowicz, R. M., Effect of pressure on carbonic acid,
#       boric acid, and the pHi of seawater, Limnology and Oceanography
#       13:403-417, 1968.
# '***********************************************************************
    Ca = full(ntps, nan)
    # Ar = full(ntps, nan)
    KCa = full(ntps, nan)
    KAr = full(ntps, nan)
    F=logical_and(WhichKs!=6, WhichKs!=7)
    if any(F):
    # (below here, F isn't used, since almost always all rows match the above criterium,
    #  in all other cases the rows will be overwritten later on).
        # CalculateCa:
        #       Riley, J. P. and Tongudai, M., Chemical Geology 2:263-269, 1967:
        #       this is .010285*Sali/35
        Ca = 0.02128/40.087*(Sal/1.80655)# ' in mol/kg-SW
        # CalciteSolubility:
        #       Mucci, Alphonso, Amer. J. of Science 283:781-799, 1983.
        logKCa = -171.9065 - 0.077993*TempK + 2839.319/TempK
        logKCa = logKCa + 71.595*logTempK/log(10)
        logKCa = logKCa + (-0.77712 + 0.0028426*TempK + 178.34/TempK)*sqrSal
        logKCa = logKCa - 0.07711*Sal + 0.0041249*sqrSal*Sal
        #       sd fit = .01 (for Sal part, not part independent of Sal)
        KCa = 10.0**(logKCa)# ' this is in (mol/kg-SW)^2
        # AragoniteSolubility:
        #       Mucci, Alphonso, Amer. J. of Science 283:781-799, 1983.
        logKAr = -171.945 - 0.077993*TempK + 2903.293/TempK
        logKAr = logKAr + 71.595*logTempK/log(10)
        logKAr = logKAr + (-0.068393 + 0.0017276*TempK + 88.135/TempK)*sqrSal
        logKAr = logKAr - 0.10018*Sal + 0.0059415*sqrSal*Sal
        #       sd fit = .009 (for Sal part, not part independent of Sal)
        KAr    = 10.0**(logKAr)# ' this is in (mol/kg-SW)^2
        # PressureCorrectionForCalcite:
        #       Ingle, Marine Chemistry 3:301-319, 1975
        #       same as in Millero, GCA 43:1651-1661, 1979, but Millero, GCA 1995
        #       has typos (-.5304, -.3692, and 10^3 for Kappa factor)
        deltaVKCa = -48.76 + 0.5304*TempC
        KappaKCa  = (-11.76 + 0.3692*TempC)/1000
        lnKCafac  = (-deltaVKCa + 0.5*KappaKCa*Pbar)*Pbar/RT
        KCa       = KCa*exp(lnKCafac)
        # PressureCorrectionForAragonite:
        #       Millero, Geochemica et Cosmochemica Acta 43:1651-1661, 1979,
        #       same as Millero, GCA 1995 except for typos (-.5304, -.3692,
        #       and 10^3 for Kappa factor)
        deltaVKAr = deltaVKCa + 2.8
        KappaKAr  = KappaKCa
        lnKArfac  = (-deltaVKAr + 0.5*KappaKAr*Pbar)*Pbar/RT
        KAr       = KAr*exp(lnKArfac)
    F=logical_or(WhichKs==6, WhichKs==7)
    if any(F):
        #
        # *** CalculateCaforGEOSECS:
        # Culkin, F, in Chemical Oceanography, ed. Riley and Skirrow, 1965:
        # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982)
        Ca[F] = 0.01026*Sal[F]/35
        # Culkin gives Ca = (.0213/40.078)*(Sal/1.80655) in mol/kg-SW
        # which corresponds to Ca = .01030*Sal/35.
        #
        # *** CalculateKCaforGEOSECS:
        # Ingle et al, Marine Chemistry 1:295-307, 1973 is referenced in
        # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982
        # but the fit is actually from Ingle, Marine Chemistry 3:301-319, 1975)
        KCa[F] = 0.0000001*(-34.452 - 39.866*Sal[F]**(1/3) +
            110.21*log(Sal[F])/log(10) - 0.0000075752*TempK[F]**2)
        # this is in (mol/kg-SW)^2
        #
        # *** CalculateKArforGEOSECS:
        # Berner, R. A., American Journal of Science 276:713-730, 1976:
        # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982)
        KAr[F] = 1.45*KCa[F]# ' this is in (mol/kg-SW)^2
        # Berner (p. 722) states that he uses 1.48.
        # It appears that 1.45 was used in the GEOSECS calculations
        #
        # *** CalculatePressureEffectsOnKCaKArGEOSECS:
        # Culberson and Pytkowicz, Limnology and Oceanography 13:403-417, 1968
        # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982
        # but their paper is not even on this topic).
        # The fits appears to be new in the GEOSECS report.
        # I can't find them anywhere else.
        KCa[F] = KCa[F]*exp((36   - 0.2 *TempC[F])*Pbar[F]/RT[F])
        KAr[F] = KAr[F]*exp((33.3 - 0.22*TempC[F])*Pbar[F]/RT[F])
    # CalculateOmegasHere:
    H = 10.0**(-pH)
    CO3 = TC*K1*K2/(K1*H + H*H + K1*K2)
    return CO3*Ca/KCa, CO3*Ca/KAr # OmegaCa, OmegaAr: both dimensionless

def _FindpHOnAllScales(pH):
    global pHScale, K, T, TS, KS, TF, KF, fH, ntps
# SUB FindpHOnAllScales, version 01.02, 01-08-97, written by Ernie Lewis.
# Inputs: pHScale#, pH, K(), T(), fH
# Outputs: pHNBS, pHfree, pHTot, pHSWS
# This takes the pH on the given scale and finds the pH on all scales.
#  TS = T(3); TF = T(2);
#  KS = K(6); KF = K(5);# 'these are at the given T, S, P
    FREEtoTOT = (1 + TS/KS)# ' pH scale conversion factor
    SWStoTOT  = (1 + TS/KS)/(1 + TS/KS + TF/KF)# ' pH scale conversion factor
    factor=full(ntps, nan)
    F=pHScale==1  #'"pHtot"
    factor[F] = 0
    F=pHScale==2 # '"pHsws"
    factor[F] = -log(SWStoTOT[F])/log(0.1)
    F=pHScale==3 # '"pHfree"
    factor[F] = -log(FREEtoTOT[F])/log(0.1)
    F=pHScale==4  #'"pHNBS"
    factor[F] = -log(SWStoTOT[F])/log(0.1) + log(fH[F])/log(0.1)
    pHtot  = pH    - factor;    # ' pH comes into this sub on the given scale
    pHNBS  = pHtot - log(SWStoTOT) /log(0.1) + log(fH)/log(0.1)
    pHfree = pHtot - log(FREEtoTOT)/log(0.1)
    pHsws  = pHtot - log(SWStoTOT) /log(0.1)
    return pHtot, pHsws, pHfree, pHNBS

def CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN,
        PRESOUT, SI, PO4, NH3, H2S, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS):
    global pHScale, WhichKs, WhoseKSO4, Pbar
    global Sal, sqrSal, TempK, logTempK, TempCi, TempCo, Pdbari, Pdbaro
    global FugFac, VPFac, PengCorrection, ntps, RGasConstant
    global fH, RT
    global K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S
    global TB, TF, TS, TP, TSi, TNH3, TH2S, F

    # Input conditioning.
    args = [PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN,
            PRESOUT, SI, PO4, NH3, H2S, pHSCALEIN, K1K2CONSTANTS,
            KSO4CONSTANTS]

    # Determine lengths of input vectors.
    veclengths = [size(arg) for arg in args]
    if size(unique(veclengths)) > 2:
        print('*** INPUT ERROR: Input vectors must all be of same length, ' +
              'or of length 1. ***')
        return

    # Make row vectors of all inputs.
    ntps = max(veclengths)
    args = [full(ntps, arg) if size(arg)==1 else arg.ravel()
            for arg in args]
    (PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN, PRESOUT,
        SI, PO4, NH3, H2S, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS) = args
    
    # Convert any integer inputs to floats.
    SAL = SAL.astype('float64')
    TEMPIN = TEMPIN.astype('float64')
    TEMPOUT = TEMPOUT.astype('float64')
    PRESIN = PRESIN.astype('float64')
    PRESOUT = PRESOUT.astype('float64')
    SI = SI.astype('float64')
    PO4 = PO4.astype('float64')
    NH3 = NH3.astype('float64')
    H2S = H2S.astype('float64')

    # Assign input to the 'historical' variable names.
    pHScale = pHSCALEIN
    WhichKs = K1K2CONSTANTS
    WhoseKSO4 = KSO4CONSTANTS
    p1 = PAR1TYPE
    p2 = PAR2TYPE
    TempCi = TEMPIN
    TempCo = TEMPOUT
    Pdbari = PRESIN
    Pdbaro = PRESOUT
    Sal = SAL
    sqrSal = sqrt(SAL)
    TP = PO4
    TSi = SI
    TNH3 = NH3
    TH2S = H2S

    # Generate empty vectors for...
    TA = full(ntps, nan) # Talk
    TC = full(ntps, nan) # DIC
    PH = full(ntps, nan) # pH
    PC = full(ntps, nan) # pCO2
    FC = full(ntps, nan) # fCO2
    CARB = full(ntps, nan) # CO3 ions

    # Assign values to empty vectors.
    F = p1==1; TA[F] = PAR1[F]/1e6 # Convert from micromol/kg to mol/kg
    F = p1==2; TC[F] = PAR1[F]/1e6 # Convert from micromol/kg to mol/kg
    F = p1==3; PH[F] = PAR1[F]
    F = p1==4; PC[F] = PAR1[F]/1e6 # Convert from microatm. to atm.
    F = p1==5; FC[F] = PAR1[F]/1e6 # Convert from microatm. to atm.
    F = p1==6; CARB[F] = PAR1[F]/1e6 # Convert from micromol/kg to mol/kg
    F = p2==1; TA[F] = PAR2[F]/1e6 # Convert from micromol/kg to mol/kg
    F = p2==2; TC[F] = PAR2[F]/1e6 # Convert from micromol/kg to mol/kg
    F = p2==3; PH[F] = PAR2[F]
    F = p2==4; PC[F] = PAR2[F]/1e6 # Convert from microatm. to atm.
    F = p2==5; FC[F] = PAR2[F]/1e6 # Convert from microatm. to atm.
    F = p2==6; CARB[F] = PAR2[F]/1e6 # Convert from micromol/kg to mol/kg

    # Generate the columns holding Si, Phos and Sal.
    # Pure Water case:
    F = WhichKs==8
    Sal[F] = 0.0
    # GEOSECS and Pure Water:
    F = logical_or(WhichKs==8, WhichKs==6)
    TP[F] = 0.0
    TSi[F] = 0.0
    TNH3[F] = 0.0
    TH2S[F] = 0.0
    # All other cases
    F = ~F
    TP[F] /= 1e6
    TSi[F] /= 1e6
    TNH3[F] /= 1e6
    TH2S[F] /= 1e6

    # The vector 'PengCorrection' is used to modify the value of TA, for those
    # cases where WhichKs==7, since PAlk(Peng) = PAlk(Dickson) + TP.
    # Thus, PengCorrection is 0 for all cases where WhichKs is not 7
    PengCorrection = zeros(ntps)
    F = WhichKs==7
    PengCorrection[F] = TP[F]

    # Calculate the constants for all samples at input conditions
    # The constants calculated for each sample will be on the appropriate pH
    # scale!
    ConstPuts = (pHScale, WhichKs, WhoseKSO4, ntps, TP, TSi, Sal)
    (K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KH2S, KNH3, TB, TF, TS,
            RGasConstant, RT, K0, fH, FugFac, VPFac, TempK, logTempK, Pbar) = \
        _Constants(TempCi, Pdbari, *ConstPuts)

    # Make sure fCO2 is available for each sample that has pCO2.
    F = logical_or(p1==4, p2==4)
    FC[F] = PC[F]*FugFac[F]

    # Generate vector for results, and copy the raw input values into them. This
    # copies ~60% NaNs, which will be replaced for calculated values later on.
    TAc = deepcopy(TA)
    TCc = deepcopy(TC)
    PHic = deepcopy(PH)
    PCic = deepcopy(PC)
    FCic = deepcopy(FC)
    CARBic = deepcopy(CARB)

    # Generate vector describing the combination of input parameters
    # So, the valid ones are: 12,13,15,23,25,35
    Icase = (10*np_min(array([p1, p2]), axis=0) +
        np_max(array([p1, p2]), axis=0))

    # Calculate missing values for AT, CT, PH, FC:
    # pCO2 will be calculated later on, routines work with fCO2.
    F = Icase==12 # input TA, TC
    if any(F):
        PHic[F], FCic[F] = _CalculatepHfCO2fromTATC(TAc[F]-PengCorrection[F],
                                                    TCc[F])
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F])
    F = Icase==13 # input TA, pH
    if any(F):
        TCc[F] = _CalculateTCfromTApH(TAc[F]-PengCorrection[F], PHic[F])
        FCic[F] = _CalculatefCO2fromTCpH(TCc[F], PHic[F])
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F])
    F = logical_or(Icase==14, Icase==15) # input TA, (pCO2 or fCO2)
    if any(F):
        PHic[F] = _CalculatepHfromTAfCO2(TAc[F]-PengCorrection[F], FCic[F])
        TCc[F] = _CalculateTCfromTApH(TAc[F]-PengCorrection[F], PHic[F])
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F])
    F = Icase==16 # input TA, CARB
    if any(F):
        PHic[F] = _CalculatepHfromTACarb(TAc[F]-PengCorrection[F], CARBic[F])
        TCc[F] = _CalculateTCfromTApH(TAc[F]-PengCorrection[F], PHic[F])
        FCic[F] = _CalculatefCO2fromTCpH(TCc[F], PHic[F])
    F = Icase==23 # input TC, pH
    if any(F):
        TAc[F] = _CalculateTAfromTCpH  (TCc[F], PHic[F]) + PengCorrection[F]
        FCic[F] = _CalculatefCO2fromTCpH(TCc[F], PHic[F])
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F])
    F = logical_or(Icase==24, Icase==25) # input TC, (pCO2 or fCO2)
    if any(F):
        PHic[F] = _CalculatepHfromTCfCO2(TCc[F], FCic[F])
        TAc[F] = _CalculateTAfromTCpH(TCc[F], PHic[F]) + PengCorrection[F]
    F = Icase==26 # input TC, CARB
    if any(F):
        PHic[F] = _CalculatepHfromTCCarb(TCc[F], CARBic[F])
        FCic[F] = _CalculatefCO2fromTCpH(TCc[F], PHic[F])
        TAc[F] = _CalculateTAfromTCpH(TCc[F], PHic[F]) + PengCorrection[F]
    F = logical_or(Icase==34, Icase==35) # input pH, (pCO2 or fCO2)
    if any(F):
        TCc[F] = _CalculateTCfrompHfCO2(PHic[F], FCic[F])
        TAc[F] = _CalculateTAfromTCpH(TCc[F],  PHic[F]) + PengCorrection[F]
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F])
    F = Icase==36 # input pH, CARB
    if any(F):
        FCic[F] = _CalculatefCO2frompHCarb(PHic[F], CARBic[F])
        TCc[F] = _CalculateTCfrompHfCO2(PHic[F], FCic[F])
        TAc[F] = _CalculateTAfromTCpH(TCc[F], PHic[F]) + PengCorrection[F]
    F = logical_or(Icase==46, Icase==56) # input (pCO2 or fCO2), CARB
    if any(F):
        PHic[F] = _CalculatepHfromfCO2Carb(FCic[F], CARBic[F])
        TCc[F] = _CalculateTCfrompHfCO2(PHic[F], FCic[F])
        TAc[F] = _CalculateTAfromTCpH(TCc[F], PHic[F]) + PengCorrection[F]
    # By now, an fCO2 value is available for each sample.
    # Generate the associated pCO2 value:
    PCic = FCic/FugFac

    # CalculateOtherParamsAtInputConditions:
    (HCO3inp, CO3inp, BAlkinp, OHinp, PAlkinp, SiAlkinp, NH3Alkinp, H2SAlkinp,
        Hfreeinp, HSO4inp, HFinp) = _CalculateAlkParts(PHic, TCc)
    PAlkinp += PengCorrection
    CO2inp = TCc - CO3inp - HCO3inp
    F = full(ntps, True) # i.e., do for all samples:
    Revelleinp = _RevelleFactor(TAc-PengCorrection, TCc)
    OmegaCainp, OmegaArinp = _CaSolubility(Sal, TempCi, Pdbari, TCc, PHic)
    xCO2dryinp = PCic/VPFac # this assumes pTot = 1 atm

    # Just for reference, convert pH at input conditions to the other scales, too.
    pHicT, pHicS, pHicF, pHicN = _FindpHOnAllScales(PHic)

    # Merge the Ks at input into an array. Ks at output will be glued to this later.
    KIVEC = array([K0, K1, K2, -log10(K1), -log10(K2), KW, KB, KF, KS, KP1,
                   KP2, KP3, KSi, KNH3, KH2S])

    # Calculate the constants for all samples at output conditions
    (K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KH2S, KNH3, TB, TF, TS,
            RGasConstant, RT,K0, fH, FugFac, VPFac, TempK, logTempK, Pbar) = \
        _Constants(TempCo, Pdbaro, *ConstPuts)

    # Calculate, for output conditions, using conservative TA and TC, pH, fCO2 and pCO2
    F = full(ntps, True) # i.e., do for all samples:
    PHoc, FCoc = _CalculatepHfCO2fromTATC(TAc-PengCorrection, TCc)
    CARBoc = _CalculateCarbfromTCpH(TCc, PHoc)
    PCoc = FCoc/FugFac

    # Calculate Other Stuff At Output Conditions:
    (HCO3out, CO3out, BAlkout, OHout, PAlkout, SiAlkout, NH3Alkout, H2SAlkout,
        Hfreeout, HSO4out, HFout) = _CalculateAlkParts(PHoc, TCc)
    PAlkout += PengCorrection
    CO2out = TCc - CO3out - HCO3out
    Revelleout = _RevelleFactor(TAc, TCc)
    OmegaCaout, OmegaArout = _CaSolubility(Sal, TempCo, Pdbaro, TCc, PHoc)
    xCO2dryout = PCoc/VPFac # this assumes pTot = 1 atm

    # Just for reference, convert pH at output conditions to the other scales, too.
    pHocT, pHocS, pHocF, pHocN = _FindpHOnAllScales(PHoc)

    KOVEC = array([K0, K1, K2, -log10(K1), -log10(K2), KW, KB, KF, KS, KP1,
                   KP2, KP3, KSi, KNH3, KH2S])
    TVEC = array([TB, TF, TS])

    # Saving data in array, 81 columns, as many rows as samples input
    DATA = vstack((array([
        TAc*1e6      ,  TCc*1e6        , PHic          , PCic*1e6       , FCic*1e6,
        HCO3inp*1e6  ,  CARBic*1e6     , CO2inp*1e6    , BAlkinp*1e6    , OHinp*1e6,
        PAlkinp*1e6  ,  SiAlkinp*1e6   , NH3Alkinp*1e6 , H2SAlkinp*1e6  ,
        Hfreeinp*1e6  , Revelleinp     , OmegaCainp, # Multiplied Hfreeinp *1e6, svh20100827
        OmegaArinp   ,  xCO2dryinp*1e6 , PHoc          , PCoc*1e6       , FCoc*1e6,
        HCO3out*1e6  ,  CARBoc*1e6     , CO2out*1e6    , BAlkout*1e6    , OHout*1e6,
        PAlkout*1e6  ,  SiAlkout*1e6   , NH3Alkout*1e6 , H2SAlkout*1e6  ,
        Hfreeout*1e6  , Revelleout     , OmegaCaout, # Multiplied Hfreeout *1e6, svh20100827
        OmegaArout   ,  xCO2dryout*1e6 , pHicT         , pHicS          , pHicF,
        pHicN        ,  pHocT          , pHocS         , pHocF          , pHocN,
        TEMPIN       ,  TEMPOUT        , PRESIN        , PRESOUT        , PAR1TYPE,
        PAR2TYPE     ,  K1K2CONSTANTS  , KSO4CONSTANTS , pHSCALEIN      , SAL,
        PO4          ,  SI             ,
    ]), KIVEC, KOVEC, TVEC*1e6))

    HEADERS = array(['TAlk', 'TCO2', 'pHin', 'pCO2in', 'fCO2in',
        'HCO3in', 'CO3in', 'CO2in', 'BAlkin', 'OHin',
        'PAlkin', 'SiAlkin', 'NH3Alkin', 'H2SAlkin',
        'Hfreein', 'RFin', 'OmegaCAin',
        'OmegaARin', 'xCO2in', 'pHout', 'pCO2out', 'fCO2out',
        'HCO3out', 'CO3out', 'CO2out', 'BAlkout', 'OHout',
        'PAlkout', 'SiAlkout', 'NH3Alkout', 'H2SAlkout',
        'Hfreeout', 'RFout', 'OmegaCAout',
        'OmegaARout', 'xCO2out', 'pHinTOTAL', 'pHinSWS', 'pHinFREE',
        'pHinNBS', 'pHoutTOTAL', 'pHoutSWS', 'pHoutFREE', 'pHoutNBS',
        'TEMPIN', 'TEMPOUT', 'PRESIN', 'PRESOUT', 'PAR1TYPE',
        'PAR2TYPE', 'K1K2CONSTANTS', 'KSO4CONSTANTS', 'pHSCALEIN', 'SAL',
        'PO4', 'SI',
        'K0input', 'K1input', 'K2input', 'pK1input', 'pK2input',
        'KWinput', 'KBinput', 'KFinput', 'KSinput', 'KP1input',
        'KP2input', 'KP3input', 'KSiinput', 'KNH3input', 'KH2Sinput',
        'K0output', 'K1output', 'K2output', 'pK1output', 'pK2output',
        'KWoutput', 'KBoutput', 'KFoutput', 'KSoutput', 'KP1output',
        'KP2output', 'KP3output', 'KSioutput', 'KNH3output', 'KH2Soutput',
        'TB', 'TF', 'TS'])

    del F, K2, KP3, Pdbari, Sal, TS, VPFac, ntps
    del FugFac, KB, KS, Pdbaro, TSi, WhichKs, pHScale
    del KF, KSi, PengCorrection, TB, TempCi, WhoseKSO4, sqrSal
    del K0, KP1, KW, RGasConstant, TF, TempCo, fH
    del K1, KP2, Pbar, RT, TP, TempK, logTempK
    del KNH3, KH2S, TNH3, TH2S

    DICT = {HEADERS[i]: DATA[i] for i in range(len(DATA))}

    return DICT, DATA, HEADERS
