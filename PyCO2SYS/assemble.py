from numpy import (array, exp, full, full_like, logical_and, logical_or, nan,
                   size, unique)
from . import concentrations as conc
from . import convert
from . import equilibria as eq

def inputs(input_locals):
    """Condition inputs for use with CO2SYS (sub)functions."""
    # Determine and check lengths of input vectors
    veclengths = array([size(v) for v in input_locals.values()])
    assert size(unique(veclengths[veclengths != 1])) <= 1, \
        'CO2SYS function inputs must all be of same length, or of length 1.'
    # Make vectors of all inputs
    ntps = max(veclengths)
    args = {k: full(ntps, v) if size(v)==1 else v.ravel()
            for k, v in input_locals.items()}
    # Convert to float where appropriate
    float_vars = ['SAL', 'TEMPIN', 'TEMPOUT', 'PRESIN', 'PRESOUT', 'SI', 'PO4',
                  'NH3', 'H2S', 'PAR1', 'PAR2']
    for k in args.keys():
        if k in float_vars:
            args[k] = args[k].astype('float64')
    return args, ntps

def concentrations(Sal, WhichKs, WhoseTB):
    """Estimate total concentrations of borate, fluoride and sulfate from
    salinity.

    Inputs must first be conditioned with inputs().

    Based on a subset of Constants, version 04.01, 10-13-97, by Ernie Lewis.
    """
    # Generate empty vectors for holding results
    TB = full_like(Sal, nan)
    TF = full_like(Sal, nan)
    TS = full_like(Sal, nan)
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

def units(TempC, Pdbar):
    """Convert temperature and pressure units."""
    RGasConstant = 83.1451 # ml bar-1 K-1 mol-1, DOEv2
    # RGasConstant = 83.14472 # # ml bar-1 K-1 mol-1, DOEv3
    TempK = TempC + 273.15
    RT = RGasConstant*TempK
    Pbar = Pdbar/10.0
    return TempK, Pbar, RT

def equilibria(TempC, Pdbar, pHScale, WhichKs, WhoseKSO4, WhoseKF, TP, TSi, Sal,
        TF, TS):
    """Evaluate all stoichiometric equilibrium constants, converted to the
    chosen pH scale, and corrected for pressure.

    Inputs must first be conditioned with inputs().

    This finds the Constants of the CO2 system in seawater or freshwater,
    corrects them for pressure, and reports them on the chosen pH scale.
    The process is as follows: the Constants (except KS, KF which stay on the
    free scale - these are only corrected for pressure) are:
          1) evaluated as they are given in the literature,
          2) converted to the SWS scale in mol/kg-SW or to the NBS scale,
          3) corrected for pressure,
          4) converted to the SWS pH scale in mol/kg-SW,
          5) converted to the chosen pH scale.

    Based on a subset of Constants, version 04.01, 10-13-97, by Ernie Lewis.
    """
    # PROGRAMMER'S NOTE: all logs are log base e
    # PROGRAMMER'S NOTE: all Constants are converted to the pH scale
    #     pHScale# (the chosen one) in units of mol/kg-SW
    #     except KS and KF are on the free scale
    #     and KW is in units of (mol/kg-SW)^2

    TempK, Pbar, RT = units(TempC, Pdbar)

    # Calculate K0 (Henry's constant for CO2)
    K0 = eq.kCO2_W74(TempK, Sal)

    # Calculate KS (bisulfate ion dissociation constant)
    KS = full_like(TempK, nan)
    F = WhoseKSO4==1
    if any(F):
        KS[F] = eq.kHSO4_FREE_D90a(TempK[F], Sal[F])
    F = WhoseKSO4==2
    if any(F):
        KS[F] = eq.kHSO4_FREE_KRCB77(TempK[F], Sal[F])

    # Calculate KF (hydrogen fluoride dissociation constant)
    KF = full_like(TempC, nan)
    F = WhoseKF==1
    if any(F):
        KF[F] = eq.kHF_FREE_DR79(TempK[F], Sal[F])
    F = WhoseKF==2
    if any(F):
        KF[F] = eq.kHF_FREE_PF87(TempK[F], Sal[F])

    # Calculate pH scale conversion factors - these are NOT pressure-corrected
    SWStoTOT = convert.sws2tot(TS, KS, TF, KF)
    # Calculate fH
    fH = full_like(TempC, nan)
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
    KB = full_like(TempC, nan)
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
    KW = full_like(TempC, nan)
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
    KP1 = full_like(TempC, nan)
    KP2 = full_like(TempC, nan)
    KP3 = full_like(TempC, nan)
    KSi = full_like(TempC, nan)
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
    K1 = full_like(TempC, nan)
    K2 = full_like(TempC, nan)
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
    KH2S = full_like(TempC, nan)
    KNH3 = full_like(TempC, nan)
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
    deltaV = full_like(TempC, nan)
    Kappa = full_like(TempC, nan)
    lnK1fac = full_like(TempC, nan)
    lnK2fac = full_like(TempC, nan)
    lnKBfac = full_like(TempC, nan)
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
    lnKWfac = full_like(TempC, nan)
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
    pHfactor = full_like(TempC, nan)
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

    return K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S, fH
