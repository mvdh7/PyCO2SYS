# NEXT STEPS:
# - Tidy up _Constants() i/o.
# - Work out and fix CO3out, pCO2out, etc. discrepancy.
# - Eliminate all overwriting of variables during switch from input to output
#   conditions.
# - Extract CO2 system components from _CalkulateAlkParts()
# - Extract subfunctions from _CaSolubility() into relevant modules.
# - Add references from _CaSolubility() to docs.
# - Relocate all _CalculateX()s into a module (e.g. PyCO2SYS.solve).
# - Use assert to check input vector lengths, not an if.

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

from copy import deepcopy
from numpy import (array, exp, full, full_like, log, log10, logical_and,
                   logical_or, nan, size, sqrt, unique, zeros)
from numpy import abs as np_abs
from numpy import any as np_any
from numpy import min as np_min
from numpy import max as np_max

def _Concentrations(ntps, WhichKs, WhoseTB, Sal):
    """Estimate total concentrations of borate, fluoride and sulfate from
    salinity.
    """
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
        FF = logical_and(F, WhoseTB==1) 
        if any(FF): # If user opted for Uppstrom's values
            TB[FF] = conc.borate_U74(Sal[FF])
        FF = logical_and(F, WhoseTB==2) 
        if any(FF): # If user opted for the new Lee values
            TB[FF] = conc.borate_LKB10(Sal[FF])
    # Calculate total fluoride and sulfate
    TF = conc.fluoride_R65(Sal)
    TS = conc.sulfate_MR66(Sal)    
    return TB, TF, TS

def _Constants(TempC, Pdbar, pHScale, WhichKs, WhoseKSO4, WhoseKF, WhoseTB,
        ntps, TP, TSi, Sal, TF, TS):
    """Evaluate all stoichiometric equilibrium constants, converted to the
    chosen pH scale, and corrected for pressure.
    
    This finds the Constants of the CO2 system in seawater or freshwater,
    corrects them for pressure, and reports them on the chosen pH scale.
    The process is as follows: the Constants (except KS, KF which stay on the
    free scale - these are only corrected for pressure) are:
          1) evaluated as they are given in the literature,
          2) converted to the SWS scale in mol/kg-SW or to the NBS scale,
          3) corrected for pressure,
          4) converted to the SWS pH scale in mol/kg-SW,
          5) converted to the chosen pH scale.
    
    Based on Constants, version 04.01, 10-13-97, by Ernie Lewis.
    """
    # PROGRAMMER'S NOTE: all logs are log base e
    # PROGRAMMER'S NOTE: all Constants are converted to the pH scale
    #     pHScale# (the chosen one) in units of mol/kg-SW
    #     except KS and KF are on the free scale
    #     and KW is in units of (mol/kg-SW)^2
    
    RGasConstant = 83.1451 # ml bar-1 K-1 mol-1, DOEv2
    # RGasConstant = 83.14472 # # ml bar-1 K-1 mol-1, DOEv3
    TempK = TempC + 273.15
    RT = RGasConstant*TempK
    Pbar = Pdbar/10.0

    # Calculate K0 (Henry's constant for CO2)
    K0 = eq.kCO2_W74(TempK, Sal)

    # Calculate KS (bisulfate ion dissociation constant)
    KS = full(ntps, nan)
    F = WhoseKSO4==1
    if any(F):
        KS[F] = eq.kHSO4_FREE_D90a(TempK[F], Sal[F])
    F = WhoseKSO4==2
    if any(F):
        KS[F] = eq.kHSO4_FREE_KRCB77(TempK[F], Sal[F])

    # Calculate KF (hydrogen fluoride dissociation constant)
    KF = full(ntps, nan)
    F = WhoseKF==1
    if any(F):
        KF[F] = eq.kHF_FREE_DR79(TempK[F], Sal[F])
    F = WhoseKF==2
    if any(F):
        KF[F] = eq.kHF_FREE_PF87(TempK[F], Sal[F])

    # Calculate pH scale conversion factors - these are NOT pressure-corrected
    SWStoTOT = convert.sws2tot(TS, KS, TF, KF)
    # Calculate fH
    fH = full(ntps, nan)
    # Use GEOSECS's value for cases 1-6 to convert pH scales
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
    if any(F):
        K1[F], K2[F] = eq.kH2CO3_SWS_WMW14(TempK[F], Sal[F])

    # From CO2SYS_v1_21.m: calculate KH2S and KNH3
    KH2S = full(ntps, nan)
    KNH3 = full(ntps, nan)
    F = logical_or.reduce((WhichKs==6, WhichKs==7, WhichKs==8))
    # Contributions from NH3 and H2S not included for these options.
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
    KNH3 *= pHfactor
    KH2S *= pHfactor
    
    return (K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S, RT, fH,
            RGasConstant)

def _Fugacity(ntps, TempK, Sal, WhichKs, RT):
    # CalculateFugacityConstants:
    # This assumes that the pressure is at one atmosphere, or close to it.
    # Otherwise, the Pres term in the exponent affects the results.
    #       Weiss, R. F., Marine Chemistry 2:203-215, 1974.
    #       Delta and B in cm3/mol
    Delta = (57.7 - 0.118*TempK)
    b = (-1636.75 + 12.0408*TempK - 0.0327957*TempK**2 +
         3.16528*0.00001*TempK**3)
    # For a mixture of CO2 and air at 1 atm (at low CO2 concentrations):
    P1atm = 1.01325 # in bar
    FugFac = exp((b + 2*Delta)*P1atm/RT)
    # GEOSECS and Peng assume pCO2 = fCO2, or FugFac = 1
    F = logical_or(WhichKs==6, WhichKs==7)
    if any(F):
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
    return FugFac, VPFac

def _CalculateAlkParts(pHx, TCx,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
    """Calculate the different components of total alkalinity from pH and
    dissolved inorganic carbon.
    
    Although coded for H on the total pH scale, for the pH values occuring in
    seawater (pH > 6) this will be equally valid on any pH scale (i.e. H terms
    are negligible) as long as the K Constants are on that scale.
    
    Based on CalculateAlkParts, version 01.03, 10-10-97, by Ernie Lewis.
    """
    H = 10.0**-pHx
    HCO3 = TCx*K1*H/(K1*H + H**2 + K1*K2)
    CO3 = TCx*K1*K2/(K1*H + H**2 + K1*K2)
    BAlk = TB*KB/(KB + H)
    OH = KW/H
    PAlk = (TP*(KP1*KP2*H + 2*KP1*KP2*KP3 - H**3)/
            (H**3 + KP1*H**2 + KP1*KP2*H + KP1*KP2*KP3))
    SiAlk = TSi*KSi/(KSi + H)
    NH3Alk = TNH3*KNH3/(KNH3 + H)
    H2SAlk = TH2S*KH2S/(KH2S + H)
    FREEtoTOT = convert.free2tot(TS, KS)
    Hfree = H/FREEtoTOT # for H on the Total scale
    HSO4 = TS/(1 + KS/Hfree) # since KS is on the Free scale
    HF = TF/(1 + KF/Hfree) # since KF is on the Free scale
    return HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF

def _CalculatepHfromTATC(TAx, TCx,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
    """Calculate pH from total alkalinity and dissolved inorganic carbon.
    
    This calculates pH from TA and TC using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess is pH = 8.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.
    
    Based on CalculatepHfromTATC, version 04.01, 10-13-96, by Ernie Lewis.
    SVH2007: Made this to accept vectors. It will continue iterating until all
    values in the vector are "abs(deltapH) < pHTol".
    """
    pHGuess = 8.0 # this is the first guess
    pHx = full_like(TAx, pHGuess) # first guess for all samples
    pHTol = 1e-4 # tolerance for ending iterations
    deltapH = 1 + pHTol
    ln10 = log(10)
    while np_any(np_abs(deltapH) > pHTol):
        HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
            _CalculateAlkParts(pHx, TCx,
                K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
                TB, TF, TS, TP, TSi, TNH3, TH2S)
        CAlk = HCO3 + 2*CO3
        H = 10.0**-pHx
        Denom = H**2 + K1*H + K1*K2
        Residual = (TAx - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk +
                    Hfree + HSO4 + HF)
        # Find slope dTA/dpH (this is not exact, but keeps important terms):
        Slope = ln10*(TCx*K1*H*(H**2 + K1*K2 + 4*H*K2)/Denom**2 +
                      BAlk*H/(KB + H) + OH + H)
        deltapH = Residual/Slope # this is Newton's method
        # To keep the jump from being too big:
        while any(np_abs(deltapH) > 1):
            FF = np_abs(deltapH) > 1
            deltapH[FF] /= 2.0
        pHx += deltapH
        # ^pHx is on the same scale as K1 and K2 were calculated.
    return pHx

def _CalculatefCO2fromTCpH(TCx, pHx, K0, K1, K2):
    """Calculate CO2 fugacity from dissolved inorganic carbon and pH.
    
    Based on CalculatefCO2fromTCpH, version 02.02, 12-13-96, by Ernie Lewis.
    """
    H = 10.0**-pHx
    fCO2x = TCx*H**2/(H**2 + K1*H + K1*K2)/K0
    return fCO2x

def _CalculateTCfromTApH(TAx, pHx,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
    """Calculate dissolved inorganic carbon from total alkalinity and pH.
 
    This calculates TC from TA and pH.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.
    
    Based on CalculateTCfromTApH, version 02.03, 10-10-97, by Ernie Lewis.
    """
    H = 10.0**-pHx
    HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
        _CalculateAlkParts(pHx, 0.0,
            K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
            TB, TF, TS, TP, TSi, TNH3, TH2S)
    CAlk = (TAx - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 +
            HF)
    TCx = CAlk*(H**2 + K1*H + K1*K2)/(K1*(H + 2*K2))
    return TCx

def _CalculatepHfromTAfCO2(TAi, fCO2i, K0,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
    """Calculate pH from total alkalinity and CO2 fugacity.
    
    This calculates pH from TA and fCO2 using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess is pH = 8.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.
    
    Based on CalculatepHfromTAfCO2, version 04.01, 10-13-97, by Ernie Lewis.
    """
    pHGuess = 8.0 # this is the first guess
    pH = full_like(TAi, pHGuess) # first guess for all samples
    pHTol = 1e-4 # tolerance for ending iterations
    deltapH = 1 + pHTol
    ln10 = log(10)
    while np_any(np_abs(deltapH) > pHTol):
        H = 10.0**-pH
        HCO3 = K0*K1*fCO2i/H
        CO3 = K0*K1*K2*fCO2i/H**2
        CAlk = HCO3 + 2*CO3
        _, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
            _CalculateAlkParts(pH, 0.0,
                K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
                TB, TF, TS, TP, TSi, TNH3, TH2S)
        Residual = (TAi - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk +
                    Hfree + HSO4 + HF)
        # Find Slope dTA/dpH (this is not exact, but keeps all important terms)
        Slope = ln10*(HCO3 + 4*CO3 + BAlk*H/(KB + H) + OH + H)
        deltapH = Residual/Slope # this is Newton's method
        # To keep the jump from being too big:
        while np_any(np_abs(deltapH) > 1):
            FF = np_abs(deltapH) > 1
            if any(FF):
                deltapH[FF] /= 2
        pH += deltapH
    return pH

def _CalculateTAfromTCpH(TCi, pHi,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
    """Calculate total alkalinity from dissolved inorganic carbon and pH.
    
    This calculates TA from TC and pH.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.
    
    Based on CalculateTAfromTCpH, version 02.02, 10-10-97, by Ernie Lewis.
    """
    HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
        _CalculateAlkParts(pHi, TCi,
            K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
            TB, TF, TS, TP, TSi, TNH3, TH2S)
    CAlk = HCO3 + 2*CO3
    TAc = (CAlk + BAlk + OH + PAlk + SiAlk + NH3Alk + H2SAlk - Hfree -
           HSO4 - HF)
    return TAc

def _CalculatepHfromTCfCO2(TCi, fCO2i, K0, K1, K2):
    """Calculate pH from dissolved inorganic carbon and CO2 fugacity.
    
    This calculates pH from TC and fCO2 using K0, K1, and K2 by solving the
    quadratic in H: fCO2*K0 = TC*H*H/(K1*H + H*H + K1*K2).
    If there is not a real root, then pH is returned as NaN.
    
    Based on CalculatepHfromTCfCO2, version 02.02, 11-12-96, by Ernie Lewis.
    """
    RR = K0*fCO2i/TCi
    Discr = (K1*RR)**2 + 4*(1 - RR)*K1*K2*RR
    H = 0.5*(K1*RR + sqrt(Discr))/(1 - RR)
    H[H < 0] = nan
    pHc = log(H)/log(0.1)
    return pHc

def _CalculateTCfrompHfCO2(pHi, fCO2i, K0, K1, K2):
    """Calculate dissolved inorganic carbon from pH and CO2 fugacity.
    
    Based on CalculateTCfrompHfCO2, version 01.02, 12-13-96, by Ernie Lewis.
    """
    H = 10.0**-pHi
    TCc = K0*fCO2i*(H**2 + K1*H + K1*K2)/H**2
    return TCc

def _CalculatepHfromTACarb(TAi, CARBi,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
    """Calculate pH from total alkalinity and carbonate ion.

    This calculates pH from TA and Carb using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess is pH = 8.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K constants are on that scale.
    
    Based on CalculatepHfromTACarb, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    pHGuess = 8.0 # this is the first guess
    pH = full_like(TAi, pHGuess) # first guess for all samples
    pHTol = 1e-4 # tolerance for ending iterations
    deltapH = 1 + pHTol
    ln10 = log(10)
    while np_any(np_abs(deltapH) > pHTol):
        H = 10.0**-pH
        CAlk = CARBi*(H + 2*K2)/K2
        _, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
            _CalculateAlkParts(pH, 0.0,
                K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
                TB, TF, TS, TP, TSi, TNH3, TH2S)
        Residual = (TAi - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk -
                    H2SAlk + Hfree + HSO4 + HF)
        # Find Slope dTA/dpH (this is not exact, but keeps all important terms)
        Slope = ln10*(-CARBi*H/K2 + BAlk*H/(KB + H) + OH + H)
        deltapH = Residual/Slope # this is Newton's method
        # To keep the jump from being too big:
        while np_any(np_abs(deltapH) > 1):
            FF = np_abs(deltapH) > 1
            if any(FF):
                deltapH[FF] /= 2
        pH += deltapH
    return pH

def _CalculatepHfromTCCarb(TCi, Carbi, K1, K2):
    """Calculate pH from dissolved inorganic carbon and carbonate ion.
    
    This calculates pH from Carbonate and TC using K1, and K2 by solving the
    quadratic in H: TC * K1 * K2= Carb * (H * H + K1 * H +  K1 * K2).
    
    Based on CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, by Denis
    Pierrot.
    """
    RR = 1 - TCi/Carbi
    Discr = K1**2 - 4*K1*K2*RR
    H = (-K1 + sqrt(Discr))/2
    pHc = log(H)/log(0.1)
    return pHc

def _CalculatefCO2frompHCarb(pHx, Carbx, K0, K1, K2):
    """Calculate CO2 fugacity from pH and carbonate ion.
    
    Based on CalculatefCO2frompHCarb, version 01.0, 06-12-2019, by Denis
    Pierrot.
    """
    H = 10.0**-pHx
    fCO2x = Carbx*H**2/(K0*K1*K2)
    return fCO2x

def _CalculatepHfromfCO2Carb(fCO2i, Carbi, K0, K1, K2):
    """Calculate pH from CO2 fugacity and carbonate ion.
    
    This calculates pH from Carbonate and fCO2 using K0, K1, and K2 by solving
    the equation in H: fCO2 * K0 * K1* K2 = Carb * H * H
    
    Based on CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, by Denis
    Pierrot.
    """
    RR = K0*K1*K2*fCO2i/Carbi
    H = sqrt(RR)
    pHc = log(H)/log(0.1)
    return pHc

def _CalculateCarbfromTCpH(TCx, pHx, K1, K2):
    """Calculate carbonate ion from dissolved inorganic carbon and pH.
    
    Based on CalculateCarbfromTCpH, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    H = 10.0**-pHx
    CARBx = TCx*K1*K2/(H**2 + K1*H + K1*K2)
    return CARBx

def _RevelleFactor(TAi, TCi, K0,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
    """Calculate the Revelle Factor from total alkalinity and dissolved
    inorganic carbon.
    
    This calculates the Revelle factor (dfCO2/dTC)|TA/(fCO2/TC).
    It only makes sense to talk about it at pTot = 1 atm, but it is computed
    here at the given K(), which may be at pressure <> 1 atm. Care must
    thus be used to see if there is any validity to the number computed.
    
    Based on RevelleFactor, version 01.03, 01-07-97, by Ernie Lewis.
    """
    Ts = [TB, TF, TS, TP, TSi, TNH3, TH2S]
    Ks = [K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S]
    TC0 = deepcopy(TCi)
    dTC = 1e-6 # 1 umol/kg-SW
    # Find fCO2 at TA, TC+dTC
    TCi = TC0 + dTC
    pHc = _CalculatepHfromTATC(TAi, TCi, *Ks, *Ts)
    fCO2c = _CalculatefCO2fromTCpH(TCi, pHc, K0, K1, K2)
    fCO2plus = deepcopy(fCO2c)
    # Find fCO2 at TA, TC-dTC
    TCi = TC0 - dTC
    pHc = _CalculatepHfromTATC(TAi, TCi, *Ks, *Ts)
    fCO2c = _CalculatefCO2fromTCpH(TCi, pHc, K0, K1, K2)
    fCO2minus = deepcopy(fCO2c)
    # Calculate Revelle Factor
    Revelle = (fCO2plus - fCO2minus)/dTC / ((fCO2plus + fCO2minus)/TCi)
    return Revelle

def _CaSolubility(Sal, TempC, Pdbar, TC, pH, WhichKs, K1, K2, RT):
    """Calculate calcite and aragonite solubility.
    
    This calculates omega, the solubility ratio, for calcite and aragonite.
    This is defined by: Omega = [CO3--]*[Ca++]/Ksp,
          where Ksp is the solubility product (either KCa or KAr).

    These are from:
    Mucci, Alphonso, The solubility of calcite and aragonite in seawater
          at various salinities, temperatures, and one atmosphere total
          pressure, American Journal of Science 283:781-799, 1983.
    Ingle, S. E., Solubility of calcite in the ocean,
          Marine Chemistry 3:301-319, 1975,
    Millero, Frank, The thermodynamics of the carbonate system in seawater,
          Geochemica et Cosmochemica Acta 43:1651-1661, 1979.
    Ingle et al, The solubility of calcite in seawater at atmospheric pressure
          and 35#o salinity, Marine Chemistry 1:295-307, 1973.
    Berner, R. A., The solubility of calcite and aragonite in seawater in
          atmospheric pressure and 34.5#o salinity, American Journal of
          Science 276:713-730, 1976.
    Takahashi et al, in GEOSECS Pacific Expedition, v. 3, 1982.
    Culberson, C. H. and Pytkowicz, R. M., Effect of pressure on carbonic acid,
          boric acid, and the pHi of seawater, Limnology and Oceanography
          13:403-417, 1968.
    
    Based on CaSolubility, version 01.05, 05-23-97, written by Ernie Lewis.
    """
    TempK = TempC + 273.15
    Pbar = Pdbar/10.0
    Ca = full_like(Sal, nan)
    # Ar = full(ntps, nan)
    KCa = full_like(Sal, nan)
    KAr = full_like(Sal, nan)
    # CalculateCa:
    #       Riley, J. P. and Tongudai, M., Chemical Geology 2:263-269, 1967:
    #       this is .010285*Sali/35
    Ca = 0.02128/40.087*(Sal/1.80655)# ' in mol/kg-SW
    # CalciteSolubility:
    #       Mucci, Alphonso, Amer. J. of Science 283:781-799, 1983.
    logKCa = -171.9065 - 0.077993*TempK + 2839.319/TempK
    logKCa = logKCa + 71.595*log(TempK)/log(10)
    logKCa = logKCa + (-0.77712 + 0.0028426*TempK + 178.34/TempK)*sqrt(Sal)
    logKCa = logKCa - 0.07711*Sal + 0.0041249*sqrt(Sal)*Sal
    #       sd fit = .01 (for Sal part, not part independent of Sal)
    KCa = 10.0**(logKCa)# ' this is in (mol/kg-SW)^2
    # AragoniteSolubility:
    #       Mucci, Alphonso, Amer. J. of Science 283:781-799, 1983.
    logKAr = -171.945 - 0.077993*TempK + 2903.293/TempK
    logKAr = logKAr + 71.595*log(TempK)/log(10)
    logKAr = logKAr + (-0.068393 + 0.0017276*TempK + 88.135/TempK)*sqrt(Sal)
    logKAr = logKAr - 0.10018*Sal + 0.0059415*sqrt(Sal)*Sal
    #       sd fit = .009 (for Sal part, not part independent of Sal)
    KAr    = 10.0**logKAr # this is in (mol/kg-SW)^2
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
    # Now overwrite GEOSECS values:
    F = logical_or(WhichKs==6, WhichKs==7)
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
        KAr[F] = 1.45*KCa[F] # this is in (mol/kg-SW)^2
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
    # Calculate Omegas here:
    H = 10.0**-pH
    CO3 = TC*K1*K2/(K1*H + H**2 + K1*K2)
    return CO3*Ca/KCa, CO3*Ca/KAr # OmegaCa, OmegaAr: both dimensionless

def _FindpHOnAllScales(pH, pHScale, KS, KF, TS, TF, fH):
    """Calculate pH on all scales.
    
    This takes the pH on the given scale and finds the pH on all scales.
    
    Based on FindpHOnAllScales, version 01.02, 01-08-97, by Ernie Lewis.
    """
    FREEtoTOT = convert.free2tot(TS, KS)
    SWStoTOT = convert.sws2tot(TS, KS, TF, KF)
    factor = full_like(pH, nan)
    F = pHScale==1 # Total scale
    factor[F] = 0
    F = pHScale==2 # Seawater scale
    factor[F] = -log(SWStoTOT[F])/log(0.1)
    F = pHScale==3 # Free scale
    factor[F] = -log(FREEtoTOT[F])/log(0.1)
    F = pHScale==4 # NBS scale
    factor[F] = -log(SWStoTOT[F])/log(0.1) + log(fH[F])/log(0.1)
    pHtot = pH - factor # pH comes into this sub on the given scale
    pHNBS  = pHtot - log(SWStoTOT) /log(0.1) + log(fH)/log(0.1)
    pHfree = pHtot - log(FREEtoTOT)/log(0.1)
    pHsws  = pHtot - log(SWStoTOT) /log(0.1)
    return pHtot, pHsws, pHfree, pHNBS

def CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN,
        PRESOUT, SI, PO4, NH3, H2S, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANT,
        KFCONSTANT, BORON):

    # Input conditioning.
    args = [PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN,
        PRESOUT, SI, PO4, NH3, H2S, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANT,
        KFCONSTANT, BORON]

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
    (PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN,
        PRESOUT, SI, PO4, NH3, H2S, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANT,
        KFCONSTANT, BORON) = args
    
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
    WhoseKSO4 = KSO4CONSTANT
    WhoseKF = KFCONSTANT
    WhoseTB = BORON
    p1 = PAR1TYPE
    p2 = PAR2TYPE
    TempCi = TEMPIN
    TempCo = TEMPOUT
    Pdbari = PRESIN
    Pdbaro = PRESOUT
    Sal = deepcopy(SAL)
    TP = deepcopy(PO4)
    TSi = deepcopy(SI)
    TNH3 = deepcopy(NH3)
    TH2S = deepcopy(H2S)

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
    TB, TF, TS = _Concentrations(ntps, WhichKs, WhoseTB, Sal)
    Ts = [TB, TF, TS, TP, TSi, TNH3, TH2S]

    # The vector 'PengCorrection' is used to modify the value of TA, for those
    # cases where WhichKs==7, since PAlk(Peng) = PAlk(Dickson) + TP.
    # Thus, PengCorrection is 0 for all cases where WhichKs is not 7
    PengCorrection = zeros(ntps)
    F = WhichKs==7
    PengCorrection[F] = TP[F]

    # Calculate the constants for all samples at input conditions
    # The constants calculated for each sample will be on the appropriate pH
    # scale!
    ConstPuts = (pHScale, WhichKs, WhoseKSO4, WhoseKF, WhoseTB, ntps, TP, TSi,
                 Sal, TF, TS)
    (K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S, RT, fH,
        RGasConstant) = _Constants(TempCi, Pdbari, *ConstPuts)
    Ks = [K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S]
    TempK = TempCi + 273.15
    FugFac, VPFac = _Fugacity(ntps, TempK, Sal, WhichKs, RT)

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
    # So, the valid ones are: 12,13,14,15,16,23,24,25,26,34,35,36,46,56
    Icase = (10*np_min(array([p1, p2]), axis=0) +
        np_max(array([p1, p2]), axis=0))

    # Calculate missing values for AT, CT, PH, FC:
    # pCO2 will be calculated later on, routines work with fCO2.
    F = Icase==12 # input TA, TC
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PHic[F] = _CalculatepHfromTATC(TAc[F]-PengCorrection[F], TCc[F],
                                       *KFs, *TFs)
        # ^pH is returned on the scale requested in "pHscale" (see 'constants')
        FCic[F] = _CalculatefCO2fromTCpH(TCc[F], PHic[F], K0[F], K1[F], K2[F])
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F], K1[F], K2[F])
    F = Icase==13 # input TA, pH
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        TCc[F] = _CalculateTCfromTApH(TAc[F]-PengCorrection[F], PHic[F],
                                      *KFs, *TFs)
        FCic[F] = _CalculatefCO2fromTCpH(TCc[F], PHic[F], K0[F], K1[F], K2[F])
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F], K1[F], K2[F])
    F = logical_or(Icase==14, Icase==15) # input TA, (pCO2 or fCO2)
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PHic[F] = _CalculatepHfromTAfCO2(TAc[F]-PengCorrection[F], FCic[F],
                                         K0[F], *KFs, *TFs)
        TCc[F] = _CalculateTCfromTApH(TAc[F]-PengCorrection[F], PHic[F],
                                      *KFs, *TFs)
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F], K1[F], K2[F])
    F = Icase==16 # input TA, CARB
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PHic[F] = _CalculatepHfromTACarb(TAc[F]-PengCorrection[F], CARBic[F],
                                         *KFs, *TFs)
        TCc[F] = _CalculateTCfromTApH(TAc[F]-PengCorrection[F], PHic[F],
                                      *KFs, *TFs)
        FCic[F] = _CalculatefCO2fromTCpH(TCc[F], PHic[F], K0[F], K1[F], K2[F])
    F = Icase==23 # input TC, pH
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        TAc[F] = (_CalculateTAfromTCpH(TCc[F], PHic[F], *KFs, *TFs) +
                  PengCorrection[F])
        FCic[F] = _CalculatefCO2fromTCpH(TCc[F], PHic[F], K0[F], K1[F], K2[F])
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F], K1[F], K2[F])
    F = logical_or(Icase==24, Icase==25) # input TC, (pCO2 or fCO2)
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PHic[F] = _CalculatepHfromTCfCO2(TCc[F], FCic[F], K0[F], K1[F], K2[F])
        TAc[F] = (_CalculateTAfromTCpH(TCc[F], PHic[F], *KFs, *TFs) +
                  PengCorrection[F])
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F], K1[F], K2[F])
    F = Icase==26 # input TC, CARB
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PHic[F] = _CalculatepHfromTCCarb(TCc[F], CARBic[F], K1[F], K2[F])
        FCic[F] = _CalculatefCO2fromTCpH(TCc[F], PHic[F], K0[F], K1[F], K2[F])
        TAc[F] = (_CalculateTAfromTCpH(TCc[F], PHic[F], *KFs, *TFs) +
                  PengCorrection[F])
    F = logical_or(Icase==34, Icase==35) # input pH, (pCO2 or fCO2)
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        TCc[F] = _CalculateTCfrompHfCO2(PHic[F], FCic[F], K0[F], K1[F], K2[F])
        TAc[F] = (_CalculateTAfromTCpH(TCc[F], PHic[F], *KFs, *TFs) +
                  PengCorrection[F])
        CARBic[F] = _CalculateCarbfromTCpH(TCc[F], PHic[F], K1[F], K2[F])
    F = Icase==36 # input pH, CARB
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        FCic[F] = _CalculatefCO2frompHCarb(PHic[F], CARBic[F],
                                           K0[F], K1[F], K2[F])
        TCc[F] = _CalculateTCfrompHfCO2(PHic[F], FCic[F], K0[F], K1[F], K2[F])
        TAc[F] = (_CalculateTAfromTCpH(TCc[F], PHic[F], *KFs, *TFs) +
                  PengCorrection[F])
    F = logical_or(Icase==46, Icase==56) # input (pCO2 or fCO2), CARB
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PHic[F] = _CalculatepHfromfCO2Carb(FCic[F], CARBic[F],
                                           K0[F], K1[F], K2[F])
        TCc[F] = _CalculateTCfrompHfCO2(PHic[F], FCic[F], K0[F], K1[F], K2[F])
        TAc[F] = (_CalculateTAfromTCpH(TCc[F], PHic[F], *KFs, *TFs) +
                  PengCorrection[F])
    # By now, an fCO2 value is available for each sample.
    # Generate the associated pCO2 value:
    PCic = FCic/FugFac

    # CalculateOtherParamsAtInputConditions:
    (HCO3inp, CO3inp, BAlkinp, OHinp, PAlkinp, SiAlkinp, NH3Alkinp, H2SAlkinp,
        Hfreeinp, HSO4inp, HFinp) = _CalculateAlkParts(PHic, TCc,
            K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S, *Ts)
    PAlkinp += PengCorrection
    CO2inp = TCc - CO3inp - HCO3inp
    F = full(ntps, True) # i.e., do for all samples:
    Revelleinp = _RevelleFactor(TAc-PengCorrection, TCc, K0, *Ks, *Ts)
    OmegaCainp, OmegaArinp = _CaSolubility(Sal, TempCi, Pdbari, TCc, PHic,
                                           WhichKs, K1, K2, RT)
    xCO2dryinp = PCic/VPFac # this assumes pTot = 1 atm

    # Just for reference, convert pH at input conditions to the other scales, too.
    pHicT, pHicS, pHicF, pHicN = _FindpHOnAllScales(PHic, pHScale, KS, KF,
                                                    TS, TF, fH)

    # Save the Ks at input
    K0in = deepcopy(K0)
    K1in = deepcopy(K1)
    K2in = deepcopy(K2)
    pK1in = -log10(K1in)
    pK2in = -log10(K2in)
    KWin = deepcopy(KW)
    KBin = deepcopy(KB)
    KFin = deepcopy(KF)
    KSin = deepcopy(KS)
    KP1in = deepcopy(KP1)
    KP2in = deepcopy(KP2)
    KP3in = deepcopy(KP3)
    KSiin = deepcopy(KSi)
    KNH3in = deepcopy(KNH3)
    KH2Sin = deepcopy(KH2S)

    # Calculate the constants for all samples at output conditions
    (K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S, RT, fH,
        RGasConstant) = _Constants(TempCo, Pdbaro, *ConstPuts)
    Ks = [K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S]
    TempK = TempCo + 273.15
    FugFac, VPFac = _Fugacity(ntps, TempK, Sal, WhichKs, RT)

    # Calculate, for output conditions, using conservative TA and TC, pH, fCO2 and pCO2
    F = full(ntps, True) # i.e., do for all samples:
    PHoc = _CalculatepHfromTATC(TAc-PengCorrection, TCc, *Ks, *Ts)
    # ^pH is returned on the scale requested in "pHscale" (see 'constants')
    FCoc = _CalculatefCO2fromTCpH(TCc, PHoc, K0, K1, K2)
    CARBoc = _CalculateCarbfromTCpH(TCc, PHoc, K1, K2)
    PCoc = FCoc/FugFac

    # Calculate Other Stuff At Output Conditions:
    (HCO3out, CO3out, BAlkout, OHout, PAlkout, SiAlkout, NH3Alkout, H2SAlkout,
        Hfreeout, HSO4out, HFout) = _CalculateAlkParts(PHoc, TCc, *Ks, *Ts)
    PAlkout += PengCorrection
    CO2out = TCc - CO3out - HCO3out
    Revelleout = _RevelleFactor(TAc, TCc, K0, *Ks, *Ts)
    OmegaCaout, OmegaArout = _CaSolubility(Sal, TempCo, Pdbaro, TCc, PHoc,
                                           WhichKs, K1, K2, RT)
    xCO2dryout = PCoc/VPFac # this assumes pTot = 1 atm

    # Just for reference, convert pH at output conditions to the other scales, too.
    pHocT, pHocS, pHocF, pHocN = _FindpHOnAllScales(PHoc, pHScale, KS, KF,
                                                    TS, TF, fH)

    # Save the Ks at output
    K0out = deepcopy(K0)
    K1out = deepcopy(K1)
    K2out = deepcopy(K2)
    pK1out = -log10(K1)
    pK2out = -log10(K2)
    KWout = deepcopy(KW)
    KBout = deepcopy(KB)
    KFout = deepcopy(KF)
    KSout = deepcopy(KS)
    KP1out = deepcopy(KP1)
    KP2out = deepcopy(KP2)
    KP3out = deepcopy(KP3)
    KSiout = deepcopy(KSi)
    KNH3out = deepcopy(KNH3)
    KH2Sout = deepcopy(KH2S)

    # Save data directly as a dict to avoid ordering issues
    DICT = {
        'TAlk': TAc*1e6,
        'TCO2': TCc*1e6,
        'pHin': PHic,
        'pCO2in': PCic*1e6,
        'fCO2in': FCic*1e6,
        'HCO3in': HCO3inp*1e6,
        'CO3in': CARBic*1e6,
        'CO2in': CO2inp*1e6,
        'BAlkin': BAlkinp*1e6,
        'OHin': OHinp*1e6,
        'PAlkin': PAlkinp*1e6,
        'SiAlkin': SiAlkinp*1e6,
        'NH3Alkin': NH3Alkinp*1e6,
        'H2SAlkin': H2SAlkinp*1e6,
        'Hfreein': Hfreeinp*1e6,
        'RFin': Revelleinp,
        'OmegaCAin': OmegaCainp,
        'OmegaARin': OmegaArinp,
        'xCO2in': xCO2dryinp*1e6,
        'pHout': PHoc,
        'pCO2out': PCoc*1e6,
        'fCO2out': FCoc*1e6,
        'HCO3out': HCO3out*1e6,
        'CO3out': CARBoc*1e6,
        'CO2out': CO2out*1e6,
        'BAlkout': BAlkout*1e6,
        'OHout': OHout*1e6,
        'PAlkout': PAlkout*1e6,
        'SiAlkout': SiAlkout*1e6,
        'NH3Alkout': NH3Alkout*1e6,
        'H2SAlkout': H2SAlkout*1e6,
        'Hfreeout': Hfreeout*1e6,
        'RFout': Revelleout,
        'OmegaCAout': OmegaCaout,
        'OmegaARout': OmegaArout,
        'xCO2out': xCO2dryout*1e6,
        'pHinTOTAL': pHicT,
        'pHinSWS': pHicS,
        'pHinFREE': pHicF,
        'pHinNBS': pHicN,
        'pHoutTOTAL': pHocT,
        'pHoutSWS': pHocS,
        'pHoutFREE': pHocF,
        'pHoutNBS': pHocN,
        'TEMPIN': TEMPIN,
        'TEMPOUT': TEMPOUT,
        'PRESIN': PRESIN,
        'PRESOUT': PRESOUT,
        'PAR1TYPE': PAR1TYPE,
        'PAR2TYPE': PAR2TYPE,
        'K1K2CONSTANTS': K1K2CONSTANTS,
        'KSO4CONSTANT': KSO4CONSTANT,
        'KFCONSTANT': KFCONSTANT,
        'BORON': BORON,
        'pHSCALEIN': pHSCALEIN,
        'SAL': SAL,
        'PO4': PO4,
        'SI': SI,
        'NH3': NH3,
        'H2S': H2S,
        'K0input': K0in,
        'K1input': K1in,
        'K2input': K2in,
        'pK1input': pK1in,
        'pK2input': pK2in,
        'KWinput': KWin,
        'KBinput': KBin,
        'KFinput': KFin,
        'KSinput': KSin,
        'KP1input': KP1in,
        'KP2input': KP2in,
        'KP3input': KP3in,
        'KSiinput': KSiin,
        'KNH3input': KNH3in,
        'KH2Sinput': KH2Sin,
        'K0output': K0out,
        'K1output': K1out,
        'K2output': K2out,
        'pK1output': pK1out,
        'pK2output': pK2out,
        'KWoutput': KWout,
        'KBoutput': KBout,
        'KFoutput': KFout,
        'KSoutput': KSout,
        'KP1output': KP1out,
        'KP2output': KP2out,
        'KP3output': KP3out,
        'KSioutput': KSiout,
        'KNH3output': KNH3out,
        'KH2Soutput': KH2Sout,
        'TB': TB*1e6,
        'TF': TF*1e6,
        'TS': TS*1e6,
    }

    return DICT
