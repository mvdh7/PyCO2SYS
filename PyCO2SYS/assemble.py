# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
from autograd.numpy import array, exp, full, full_like, nan, size, unique, where
from . import concentrations as conc
from . import convert
from . import equilibria as eq
from .constants import RGasConstant

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

def concs_TB(Sal, WhichKs, WhoseTB):
    """Calculate total borate from salinity for the given options."""
    TB = where(WhichKs==8, 0.0, nan) # pure water
    TB = where((WhichKs==6) | (WhichKs==7), conc.borate_C65(Sal), TB)
    F = (WhichKs!=6) & (WhichKs!=7) & (WhichKs!=8)
    TB = where(F & (WhoseTB==1), conc.borate_U74(Sal), TB)
    TB = where(F & (WhoseTB==2), conc.borate_LKB10(Sal), TB)
    return TB

def concentrations(Sal, WhichKs, WhoseTB):
    """Estimate total concentrations of borate, fluoride and sulfate from
    salinity.

    Inputs must first be conditioned with inputs().

    Based on a subset of Constants, version 04.01, 10-13-97, by Ernie Lewis.
    """
    TB = concs_TB(Sal, WhichKs, WhoseTB)
    TF = conc.fluoride_R65(Sal)
    TS = conc.sulfate_MR66(Sal)
    # Return results as a dict for stability
    return {
        'TB': TB,
        'TF': TF,
        'TSO4': TS,
    }

def units(TempC, Pdbar):
    """Convert temperature and pressure units."""
    TempK = convert.TempC2K(TempC)
    Pbar = Pdbar/10.0
    RT = RGasConstant*TempK
    return TempK, Pbar, RT

def lnKfac(deltaV, Kappa, Pbar, TempK):
    """Calculate pressure correction factor for equilibrium constants."""
    return (-deltaV + 0.5*Kappa*Pbar)*Pbar/(RGasConstant*TempK)

def _pcxKfac(deltaV, Kappa, Pbar, TempK):
    """Calculate pressure correction factor for equilibrium constants."""
    return exp((-deltaV + 0.5*Kappa*Pbar)*Pbar/(RGasConstant*TempK))

def _pcxKS(TempK, Pbar):
    """Calculate pressure correction factor for KS."""
    # === CO2SYS.m comments: =======
    # This is from Millero, 1995, which is the same as Millero, 1983.
    # It is assumed that KS is on the free pH scale.
    TempC = convert.TempK2C(TempK)
    deltaV = -18.03 + 0.0466*TempC + 0.000316*TempC**2
    Kappa = (-4.53 + 0.09*TempC)/1000
    return _pcxKfac(deltaV, Kappa, Pbar, TempK)

def eq_KS(TempK, Sal, Pbar, WhoseKSO4):
    """Calculate bisulfate ion dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KS = full(size(TempK), nan)
    KS = where(WhoseKSO4==1, eq.kHSO4_FREE_D90a(TempK, Sal), KS)
    KS = where(WhoseKSO4==2, eq.kHSO4_FREE_KRCB77(TempK, Sal), KS)
    # Now correct for seawater pressure
    KS = KS*_pcxKS(TempK, Pbar)
    return KS

def _pcxKF(TempK, Pbar):
    """Calculate pressure correction factor for KF."""
    # === CO2SYS.m comments: =======
    # This is from Millero, 1995, which is the same as Millero, 1983.
    # It is assumed that KF is on the free pH scale.
    TempC = convert.TempK2C(TempK)
    deltaV = -9.78 - 0.009*TempC - 0.000942*TempC**2
    Kappa = (-3.91 + 0.054*TempC)/1000
    return _pcxKfac(deltaV, Kappa, Pbar, TempK)

def eq_KF(TempK, Sal, Pbar, WhoseKF):
    """Calculate HF dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KF = full(size(TempK), nan)
    KF = where(WhoseKF==1, eq.kHF_FREE_DR79(TempK, Sal), KF)
    KF = where(WhoseKF==2, eq.kHF_FREE_PF87(TempK, Sal), KF)
    # Now correct for seawater pressure
    KF = KF*_pcxKF(TempK, Pbar)
    return KF

def eq_fH(TempK, Sal, WhichKs):
    """Calculate NBS to Seawater pH scale conversion factor for the given
    options.
    """
    fH = where(WhichKs==8, 1.0, nan)
    fH = where(WhichKs==7, convert.fH_PTBO87(TempK, Sal), fH)
    # Use GEOSECS's value for all other cases
    fH = where((WhichKs!=7) & (WhichKs!=8), convert.fH_TWB82(TempK, Sal), fH)
    return fH

def _pcxKB(TempK, Pbar, WhichKs):
    """Calculate pressure correction factor for KB."""
    TempC = convert.TempK2C(TempK)
    deltaV = full(size(TempK), nan)
    Kappa = full(size(TempK), nan)
    KBfac = full(size(TempK), nan) # because GEOSECS doesn't use _pcxKfac eq.
    F = WhichKs==8 # freshwater
    if any(F):
        # this doesn't matter since TB = 0 for this case
        deltaV = where(F, 0.0, deltaV)
        Kappa = where(F, 0.0, Kappa)
    F = (WhichKs==6) | (WhichKs==7)
    if any(F):
        # GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
        # Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
        # Culberson and Pytkowicz, L and O 13:403-417, 1968:
        # but the fits are the same as those in
        # Edmond and Gieskes, GCA, 34:1261-1291, 1970
        # who in turn quote Li, personal communication
        KBfac = where(F,
            exp((27.5 - 0.095*TempC)*Pbar/(RGasConstant*TempK)), KBfac)
        # This one is handled differently because the equation doesn't fit the
        # standard deltaV & Kappa form of _pcxKfac.
    F = (WhichKs!=6) & (WhichKs!=7) & (WhichKs!=8)
    if any(F):
        # This is from Millero, 1979.
        # It is from data of Culberson and Pytkowicz, 1968.
        deltaV = where(F, -29.48 + 0.1622*TempC - 0.002608*TempC**2, deltaV)
        # Millero, 1983 has:
        #   deltaV = -28.56 + .1211*TempCi - .000321*TempCi*TempCi
        # Millero, 1992 has:
        #   deltaV = -29.48 + .1622*TempCi + .295*(Sali - 34.8)
        # Millero, 1995 has:
        #   deltaV = -29.48 - .1622*TempCi - .002608*TempCi*TempCi
        #   deltaV = deltaV + .295*(Sali - 34.8) # Millero, 1979
        Kappa = where(F, -2.84/1000, Kappa) # Millero, 1979
        # Millero, 1992 and Millero, 1995 also have this.
        #   Kappa = Kappa + .354*(Sali - 34.8)/1000: # Millero,1979
        # Millero, 1983 has:
        #   Kappa = (-3 + .0427*TempCi)/1000
    # Now get final KBfac
    F = (WhichKs==6) | (WhichKs==7)
    KBfac = where(F, KBfac, _pcxKfac(deltaV, Kappa, Pbar, TempK))
    return KBfac

def eq_KB(TempK, Sal, Pbar, WhichKs, fH, SWStoTOT0):
    """Calculate boric acid dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KB = full(size(TempK), nan)
    KB = where(WhichKs==8, 0.0, KB) # pure water case
    KB = where((WhichKs==6) | (WhichKs==7),
        eq.kBOH3_NBS_LTB69(TempK, Sal)/fH, KB) # convert NBS to SWS
    KB = where((WhichKs!=6) & (WhichKs!=7) & (WhichKs!=8),
        eq.kBOH3_TOT_D90b(TempK, Sal)/SWStoTOT0, KB) # convert TOT to SWS
    # Now correct for seawater pressure
    KB = KB*_pcxKB(TempK, Pbar, WhichKs)
    return KB

def _pcxKW(TempK, Pbar, WhichKs):
    """Calculate pressure correction factor for KW."""
    TempC = convert.TempK2C(TempK)
    deltaV = full(size(TempK), nan)
    Kappa = full(size(TempK), nan)
    F = WhichKs==8 # freshwater case
    if any(F):
        # This is from Millero, 1983.
        deltaV = where(F, -25.6 + 0.2324*TempC - 0.0036246*TempC**2, deltaV)
        Kappa = where(F, (-7.33 + 0.1368*TempC - 0.001233*TempC**2)/1000, Kappa)
        # Note: the temperature dependence of KappaK1 and KappaKW for freshwater
        # in Millero, 1983 are the same.
    F = WhichKs!=8
    if any(F):
        # GEOSECS doesn't include OH term, so this won't matter.
        # Peng et al didn't include pressure, but here I assume that the KW
        # correction is the same as for the other seawater cases.
        # This is from Millero, 1983 and his programs CO2ROY(T).BAS.
        deltaV = where(F, -20.02 + 0.1119*TempC - 0.001409*TempC**2, deltaV)
        # Millero, 1992 and Millero, 1995 have:
        Kappa = where(F, (-5.13 + 0.0794*TempC)/1000, Kappa) # Millero, 1983
        # Millero, 1995 has this too, but Millero, 1992 is different.
        # Millero, 1979 does not list values for these.
    return _pcxKfac(deltaV, Kappa, Pbar, TempK)

def eq_KW(TempK, Sal, Pbar, WhichKs):
    """Calculate water dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KW = full(size(TempK), nan)
    KW = where(WhichKs==6, 0.0, KW) # GEOSECS doesn't include OH effects
    KW = where(WhichKs==7, eq.kH2O_SWS_M79(TempK, Sal), KW)
    KW = where(WhichKs==8, eq.kH2O_SWS_HO58_M79(TempK, Sal), KW)
    KW = where((WhichKs!=6) & (WhichKs!=7) & (WhichKs!=8),
        eq.kH2O_SWS_M95(TempK, Sal), KW)
    # Now correct for seawater pressure
    KW = KW*_pcxKW(TempK, Pbar, WhichKs)
    return KW

def _pcxKP1(TempK, Pbar):
    """Calculate pressure correction factor for KP1."""
    TempC = convert.TempK2C(TempK)
    deltaV = -14.51 + 0.1211*TempC - 0.000321*TempC**2
    Kappa  = (-2.67 + 0.0427*TempC)/1000
    return _pcxKfac(deltaV, Kappa, Pbar, TempK)

def _pcxKP2(TempK, Pbar):
    """Calculate pressure correction factor for KP2."""
    TempC = convert.TempK2C(TempK)
    deltaV = -23.12 + 0.1758*TempC - 0.002647*TempC**2
    Kappa  = (-5.15 + 0.09  *TempC)/1000
    return _pcxKfac(deltaV, Kappa, Pbar, TempK)

def _pcxKP3(TempK, Pbar):
    """Calculate pressure correction factor for KP3."""
    TempC = convert.TempK2C(TempK)
    deltaV = -26.57 + 0.202 *TempC - 0.003042*TempC**2
    Kappa  = (-4.08 + 0.0714*TempC)/1000
    return _pcxKfac(deltaV, Kappa, Pbar, TempK)

def eq_KP(TempK, Sal, Pbar, WhichKs, fH):
    """Calculate phosphoric acid dissociation constants for the given
    options.
    """
    # Evaluate at atmospheric pressure
    KP1 = full(size(TempK), nan)
    KP2 = full(size(TempK), nan)
    KP3 = full(size(TempK), nan)
    F = WhichKs==7
    if any(F):
        KP1_KP67, KP2_KP67, KP3_KP67 = eq.kH3PO4_NBS_KP67(TempK, Sal)
        KP1 = where(F, KP1_KP67, KP1) # already on SWS!
        KP2 = where(F, KP2_KP67/fH, KP2) # convert NBS to SWS
        KP3 = where(F, KP3_KP67/fH, KP3) # convert NBS to SWS
    F = (WhichKs==6) | (WhichKs==8)
    if any(F):
        # Note: neither the GEOSECS choice nor the freshwater choice include
        # contributions from phosphate or silicate.
        KP1 = where(F, 0.0, KP1)
        KP2 = where(F, 0.0, KP2)
        KP2 = where(F, 0.0, KP2)
    F = (WhichKs!=6) & (WhichKs!=7) & (WhichKs!=8)
    if any(F):
        KP1_YM95, KP2_YM95, KP3_YM95 = eq.kH3PO4_SWS_YM95(TempK, Sal)
        KP1 = where(F, KP1_YM95, KP1)
        KP2 = where(F, KP2_YM95, KP2)
        KP3 = where(F, KP3_YM95, KP3)
    # Now correct for seawater pressure
    # === CO2SYS.m comments: =======
    # These corrections don't matter for the GEOSECS choice (WhichKs = 6) and
    # the freshwater choice (WhichKs = 8). For the Peng choice I assume that
    # they are the same as for the other choices (WhichKs = 1 to 5).
    # The corrections for KP1, KP2, and KP3 are from Millero, 1995, which are
    # the same as Millero, 1983.
    KP1 = KP1*_pcxKP1(TempK, Pbar)
    KP2 = KP2*_pcxKP2(TempK, Pbar)
    KP3 = KP3*_pcxKP3(TempK, Pbar)
    return KP1, KP2, KP3

def _pcxKSi(TempK, Pbar):
    """Calculate pressure correction factor for KSi."""
    # === CO2SYS.m comments: =======
    # The only mention of this is Millero, 1995 where it is stated that the
    # values have been estimated from the values of boric acid. HOWEVER,
    # there is no listing of the values in the table.
    # Here we use the values for boric acid.
    TempC = convert.TempK2C(TempK)
    deltaV = -29.48 + 0.1622*TempC - 0.002608*TempC**2
    Kappa  = -2.84/1000
    return _pcxKfac(deltaV, Kappa, Pbar, TempK)

def eq_KSi(TempK, Sal, Pbar, WhichKs, fH):
    """Calculate silicate dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KSi = full(size(TempK), nan)
    KSi = where(WhichKs==7,
        eq.kSi_NBS_SMB64(TempK, Sal)/fH, KSi) # convert NBS to SWS
    # Note: neither the GEOSECS choice nor the freshwater choice include
    # contributions from phosphate or silicate.
    KSi = where((WhichKs==6) | (WhichKs==8), 0.0, KSi)
    KSi = where((WhichKs!=6) & (WhichKs!=7) & (WhichKs!=8),
        eq.kSi_SWS_YM95(TempK, Sal), KSi)
    # Now correct for seawater pressure
    KSi = KSi*_pcxKSi(TempK, Pbar)
    return KSi

def _pcxKH2S(TempK, Pbar):
    """Calculate pressure correction factor for KH2S."""
    # === CO2SYS.m comments: =======
    # Millero 1995 gives values for deltaV in fresh water instead of SW.
    # Millero 1995 gives -b0 as -2.89 instead of 2.89.
    # Millero 1983 is correct for both.
    TempC = convert.TempK2C(TempK)
    deltaV = -11.07 - 0.009*TempC - 0.000942*TempC**2
    Kappa = (-2.89 + 0.054*TempC)/1000
    return _pcxKfac(deltaV, Kappa, Pbar, TempK)

def eq_KH2S(TempK, Sal, Pbar, WhichKs, SWStoTOT0):
    """Calculate hydrogen disulfide dissociation constant for the given
    options.
    """
    # Evaluate at atmospheric pressure
    KH2S = where((WhichKs==6) | (WhichKs==7) | (WhichKs==8), 0.0,
        eq.kH2S_TOT_YM95(TempK, Sal)/SWStoTOT0) # convert TOT to SWS
    # Now correct for seawater pressure
    KH2S = KH2S*_pcxKH2S(TempK, Pbar)
    return KH2S

def _pcxKNH3(TempK, Pbar):
    """Calculate pressure correction factor for KNH3."""
    # === CO2SYS.m comments: =======
    # The corrections are from Millero, 1995, which are the same as Millero,
    # 1983.
    TempC = convert.TempK2C(TempK)
    deltaV = -26.43 + 0.0889*TempC - 0.000905*TempC**2
    Kappa = (-5.03 + 0.0814*TempC)/1000
    return _pcxKfac(deltaV, Kappa, Pbar, TempK)

def eq_KNH3(TempK, Sal, Pbar, WhichKs, SWStoTOT0):
    """Calculate ammonium dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KNH3 = where((WhichKs==6) | (WhichKs==7) | (WhichKs==8), 0.0,
        eq.kNH3_TOT_CW95(TempK, Sal)/SWStoTOT0) # convert TOT to SWS
    # Now correct for seawater pressure
    KNH3 = KNH3*_pcxKNH3(TempK, Pbar)
    return KNH3

def _pcxK1(TempK, Pbar, WhichKs):
    """Calculate pressure correction factor for K1."""
    TempC = convert.TempK2C(TempK)
    deltaV = full(size(TempK), nan)
    Kappa = full(size(TempK), nan)
    K1fac = full(size(TempK), nan) # because GEOSECS doesn't use _pcxKfac eq.
    F = WhichKs==8 # freshwater
    if any(F):
        # Pressure effects on K1 in freshwater: this is from Millero, 1983.
        deltaV = where(F, -30.54 + 0.1849*TempC - 0.0023366*TempC**2, deltaV)
        Kappa = where(F, (-6.22 + 0.1368*TempC - 0.001233*TempC**2)/1000, Kappa)
    F = (WhichKs==6) | (WhichKs==7)
    if any(F):
        # GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
        # Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
        # Culberson and Pytkowicz, L and O 13:403-417, 1968:
        # but the fits are the same as those in
        # Edmond and Gieskes, GCA, 34:1261-1291, 1970
        # who in turn quote Li, personal communication
        K1fac = where(F,
            exp((24.2 - 0.085*TempC)*Pbar/(RGasConstant*TempK)), K1fac)
        # This one is handled differently because the equation doesn't fit the
        # standard deltaV & Kappa form of _pcxKfac.
    F = (WhichKs!=6) & (WhichKs!=7) & (WhichKs!=8)
    if any(F):
        # These are from Millero, 1995.
        # They are the same as Millero, 1979 and Millero, 1992.
        # They are from data of Culberson and Pytkowicz, 1968.
        deltaV = where(F, -25.5 + 0.1271*TempC, deltaV)
        # deltaV = deltaV - .151*(Sali - 34.8) # Millero, 1979
        Kappa = where(F, (-3.08 + 0.0877*TempC)/1000, Kappa)
        # Kappa = Kappa - .578*(Sali - 34.8)/1000 # Millero, 1979
        # The fits given in Millero, 1983 are somewhat different.
    # Now get final K1fac
    F = (WhichKs==6) | (WhichKs==7)
    K1fac = where(F, K1fac, _pcxKfac(deltaV, Kappa, Pbar, TempK))
    return K1fac

def _pcxK2(TempK, Pbar, WhichKs):
    """Calculate pressure correction factor for K2."""
    TempC = convert.TempK2C(TempK)
    deltaV = full(size(TempK), nan)
    Kappa = full(size(TempK), nan)
    K2fac = full(size(TempK), nan) # because GEOSECS doesn't use _pcxKfac eq.
    F = WhichKs==8 # freshwater
    if any(F):
        # Pressure effects on K2 in freshwater: this is from Millero, 1983.
        deltaV = where(F, -29.81 + 0.115*TempC - 0.001816*TempC**2, deltaV)
        Kappa = where(F, (-5.74 + 0.093*TempC - 0.001896*TempC**2)/1000, Kappa)
    F = (WhichKs==6) | (WhichKs==7)
    if any(F):
        # GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
        # Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
        # Culberson and Pytkowicz, L and O 13:403-417, 1968:
        # but the fits are the same as those in
        # Edmond and Gieskes, GCA, 34:1261-1291, 1970
        # who in turn quote Li, personal communication
        K2fac = where(F,
            exp((16.4 - 0.04*TempC)*Pbar/(RGasConstant*TempK)), K2fac)
        # Takahashi et al had 26.4, but 16.4 is from Edmond and Gieskes
        # and matches the GEOSECS results
        # This one is handled differently because the equation doesn't fit the
        # standard deltaV & Kappa form of _pcxKfac.
    F = (WhichKs!=6) & (WhichKs!=7) & (WhichKs!=8)
    if any(F):
        # These are from Millero, 1995.
        # They are the same as Millero, 1979 and Millero, 1992.
        # They are from data of Culberson and Pytkowicz, 1968.
        deltaV = where(F, -15.82 - 0.0219*TempC, deltaV)
        # deltaV = deltaV + .321*(Sali - 34.8) # Millero, 1979
        Kappa = where(F, (1.13 - 0.1475*TempC)/1000, Kappa)
        # Kappa = Kappa - .314*(Sali - 34.8)/1000 # Millero, 1979
        # The fit given in Millero, 1983 is different.
        # Not by a lot for deltaV, but by much for Kappa.
    # Now get final K2fac
    F = (WhichKs==6) | (WhichKs==7)
    K2fac = where(F, K2fac, _pcxKfac(deltaV, Kappa, Pbar, TempK))
    return K2fac

def _get_eqKC(F, Kfunc, pHcx, K1, K2, TS):
    """Convenience function for getting and setting K1 and K2 values."""
    if any(F):
        K1_F, K2_F = Kfunc(*TS)
        K1 = where(F, K1_F/pHcx, K1)
        K2 = where(F, K2_F/pHcx, K2)
    return K1, K2

def eq_KC(TempK, Sal, Pbar, WhichKs, fH, SWStoTOT0):
    """Calculate carbonic acid dissociation constants for the given options."""
    # Evaluate at atmospheric pressure
    K1 = full(size(TempK), nan)
    K2 = full(size(TempK), nan)
    TS = (TempK, Sal) # for convenience
    K1, K2 = _get_eqKC(WhichKs==1, eq.kH2CO3_TOT_RRV93, SWStoTOT0, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==2, eq.kH2CO3_SWS_GP89, 1.0, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==3, eq.kH2CO3_SWS_H73_DM87, 1.0, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==4, eq.kH2CO3_SWS_MCHP73_DM87, 1.0, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==5, eq.kH2CO3_SWS_HM_DM87, 1.0, K1, K2, TS)
    K1, K2 = _get_eqKC((WhichKs==6) | (WhichKs==7), eq.kH2CO3_NBS_MCHP73, fH,
                       K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==8, eq.kH2CO3_SWS_M79, 1.0, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==9, eq.kH2CO3_NBS_CW98, fH, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==10, eq.kH2CO3_TOT_LDK00, SWStoTOT0, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==11, eq.kH2CO3_SWS_MM02, 1.0, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==12, eq.kH2CO3_SWS_MPL02, 1.0, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==13, eq.kH2CO3_SWS_MGH06, 1.0, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==14, eq.kH2CO3_SWS_M10, 1.0, K1, K2, TS)
    K1, K2 = _get_eqKC(WhichKs==15, eq.kH2CO3_SWS_WMW14, 1.0, K1, K2, TS)
    # Now correct for seawater pressure
    K1 = K1*_pcxK1(TempK, Pbar, WhichKs)
    K2 = K2*_pcxK2(TempK, Pbar, WhichKs)
    return K1, K2

def equilibria(TempC, Pdbar, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF):
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
    # All constants are converted to the pH scale `pHScale` (the chosen one) in
    # units of mol/kg-sw, except KS and KF are on the Free scale, and KW is in
    # units of (mol/kg-sw)**2.
    TempK, Pbar, RT = units(TempC, Pdbar)
    K0 = eq.kCO2_W74(TempK, Sal)
    KS = eq_KS(TempK, Sal, Pbar, WhoseKSO4)
    KF = eq_KF(TempK, Sal, Pbar, WhoseKF)
    # Calculate pH scale conversion factors - these are NOT pressure-corrected
    KS0 = eq_KS(TempK, Sal, 0.0, WhoseKSO4)
    KF0 = eq_KF(TempK, Sal, 0.0, WhoseKF)
    fH = eq_fH(TempK, Sal, WhichKs)
    SWStoTOT0 = convert.sws2tot(totals['TSO4'], KS0, totals['TF'], KF0)
    # Calculate other dissociation constants
    KB = eq_KB(TempK, Sal, Pbar, WhichKs, fH, SWStoTOT0)
    KW = eq_KW(TempK, Sal, Pbar, WhichKs)
    KP1, KP2, KP3 = eq_KP(TempK, Sal, Pbar, WhichKs, fH)
    KSi = eq_KSi(TempK, Sal, Pbar, WhichKs, fH)
    K1, K2 = eq_KC(TempK, Sal, Pbar, WhichKs, fH, SWStoTOT0)
    # From CO2SYS_v1_21.m: calculate KH2S and KNH3
    KH2S = eq_KH2S(TempK, Sal, Pbar, WhichKs, SWStoTOT0)
    KNH3 = eq_KNH3(TempK, Sal, Pbar, WhichKs, SWStoTOT0)
    # Correct pH scale conversions for pressure.
    # fH has been assumed to be independent of pressure.
    SWStoTOT = convert.sws2tot(totals['TSO4'], KS, totals['TF'], KF)
    FREEtoTOT = convert.free2tot(totals['TSO4'], KS)
    # The values KS and KF are already now pressure-corrected, so the pH scale
    # conversions are now valid at pressure.
    # Find pH scale conversion factor: this is the scale they will be put on
    pHfactor = full(size(TempC), nan)
    pHfactor = where(pHScale==1, SWStoTOT, pHfactor) # Total
    pHfactor = where(pHScale==2, 1.0, pHfactor) # Seawater (already on this)
    pHfactor = where(pHScale==3, SWStoTOT/FREEtoTOT, pHfactor) # Free
    pHfactor = where(pHScale==4, fH, pHfactor) # NBS
    # Convert from SWS pH scale to chosen scale
    K1 = K1*pHfactor
    K2 = K2*pHfactor
    KW = KW*pHfactor
    KB = KB*pHfactor
    KP1 = KP1*pHfactor
    KP2 = KP2*pHfactor
    KP3 = KP3*pHfactor
    KSi = KSi*pHfactor
    KNH3 = KNH3*pHfactor
    KH2S = KH2S*pHfactor
    # return K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S, fH
    # Return solution equilibrium constants as a dict
    return K0, fH, {
        'K1': K1,
        'K2': K2,
        'KW': KW,
        'KB': KB,
        'KF': KF,
        'KS': KS,
        'KP1': KP1,
        'KP2': KP2,
        'KP3': KP3,
        'KSi': KSi,
        'KNH3': KNH3,
        'KH2S': KH2S,
    }

# Original notes from CO2SYS-MATLAB regarding pressure corrections (now in
# function `equilibria` above):
#
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
