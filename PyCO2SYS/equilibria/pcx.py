# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Calculate presure correction factors for equilibrium constants."""

from autograd.numpy import exp, full, nan, size, where
from .. import convert
from ..constants import RGasConstant


def Kfac(deltaV, Kappa, Pbar, TempK):
    """Calculate pressure correction factor for equilibrium constants."""
    return exp((-deltaV + 0.5 * Kappa * Pbar) * Pbar / (RGasConstant * TempK))


def KSO4fac(TempK, Pbar):
    """Calculate pressure correction factor for KSO4."""
    # === CO2SYS.m comments: =======
    # This is from Millero, 1995, which is the same as Millero, 1983.
    # It is assumed that KS is on the free pH scale.
    TempC = convert.TempK2C(TempK)
    deltaV = -18.03 + 0.0466 * TempC + 0.000316 * TempC ** 2
    Kappa = (-4.53 + 0.09 * TempC) / 1000
    return Kfac(deltaV, Kappa, Pbar, TempK)


def KFfac(TempK, Pbar):
    """Calculate pressure correction factor for KF."""
    # === CO2SYS.m comments: =======
    # This is from Millero, 1995, which is the same as Millero, 1983.
    # It is assumed that KF is on the free pH scale.
    TempC = convert.TempK2C(TempK)
    deltaV = -9.78 - 0.009 * TempC - 0.000942 * TempC ** 2
    Kappa = (-3.91 + 0.054 * TempC) / 1000
    return Kfac(deltaV, Kappa, Pbar, TempK)


def KBfac(TempK, Pbar, WhichKs):
    """Calculate pressure correction factor for KB."""
    TempC = convert.TempK2C(TempK)
    KBfac = full(size(TempK), nan)  # because GEOSECS doesn't use _pcxKfac p1atm.
    F = WhichKs == 8  # freshwater
    if any(F):
        # this doesn't matter since TB = 0 for this case
        KBfac = where(F, 1.0, KBfac)
    F = (WhichKs == 6) | (WhichKs == 7)
    if any(F):
        # GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
        # Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
        # Culberson and Pytkowicz, L and O 13:403-417, 1968:
        # but the fits are the same as those in
        # Edmond and Gieskes, GCA, 34:1261-1291, 1970
        # who in turn quote Li, personal communication
        KBfac = where(
            F, exp((27.5 - 0.095 * TempC) * Pbar / (RGasConstant * TempK)), KBfac
        )
        # This one is handled differently because the equation doesn't fit the
        # standard deltaV & Kappa form of _pcxKfac.
    F = (WhichKs != 6) & (WhichKs != 7) & (WhichKs != 8)
    if any(F):
        # This is from Millero, 1979.
        # It is from data of Culberson and Pytkowicz, 1968.
        deltaV = -29.48 + 0.1622 * TempC - 0.002608 * TempC ** 2
        # Millero, 1983 has:
        #   deltaV = -28.56 + .1211*TempCi - .000321*TempCi*TempCi
        # Millero, 1992 has:
        #   deltaV = -29.48 + .1622*TempCi + .295*(Sali - 34.8)
        # Millero, 1995 has:
        #   deltaV = -29.48 - .1622*TempCi - .002608*TempCi*TempCi
        #   deltaV = deltaV + .295*(Sali - 34.8) # Millero, 1979
        Kappa = -2.84 / 1000  # Millero, 1979
        # Millero, 1992 and Millero, 1995 also have this.
        #   Kappa = Kappa + .354*(Sali - 34.8)/1000: # Millero,1979
        # Millero, 1983 has:
        #   Kappa = (-3 + .0427*TempCi)/1000
        KBfac = where(F, Kfac(deltaV, Kappa, Pbar, TempK), KBfac)
    return KBfac


def KWfac(TempK, Pbar, WhichKs):
    """Calculate pressure correction factor for KW."""
    TempC = convert.TempK2C(TempK)
    deltaV = full(size(TempK), nan)
    Kappa = full(size(TempK), nan)
    F = WhichKs == 8  # freshwater case
    if any(F):
        # This is from Millero, 1983.
        deltaV = where(F, -25.6 + 0.2324 * TempC - 0.0036246 * TempC ** 2, deltaV)
        Kappa = where(F, (-7.33 + 0.1368 * TempC - 0.001233 * TempC ** 2) / 1000, Kappa)
        # Note: the temperature dependence of KappaK1 and KappaKW for freshwater
        # in Millero, 1983 are the same.
    F = WhichKs != 8
    if any(F):
        # GEOSECS doesn't include OH term, so this won't matter.
        # Peng et al didn't include pressure, but here I assume that the KW
        # correction is the same as for the other seawater cases.
        # This is from Millero, 1983 and his programs CO2ROY(T).BAS.
        deltaV = where(F, -20.02 + 0.1119 * TempC - 0.001409 * TempC ** 2, deltaV)
        # Millero, 1992 and Millero, 1995 have:
        Kappa = where(F, (-5.13 + 0.0794 * TempC) / 1000, Kappa)  # Millero, 1983
        # Millero, 1995 has this too, but Millero, 1992 is different.
        # Millero, 1979 does not list values for these.
    return Kfac(deltaV, Kappa, Pbar, TempK)


def KP1fac(TempK, Pbar):
    """Calculate pressure correction factor for KP1."""
    TempC = convert.TempK2C(TempK)
    deltaV = -14.51 + 0.1211 * TempC - 0.000321 * TempC ** 2
    Kappa = (-2.67 + 0.0427 * TempC) / 1000
    return Kfac(deltaV, Kappa, Pbar, TempK)


def KP2fac(TempK, Pbar):
    """Calculate pressure correction factor for KP2."""
    TempC = convert.TempK2C(TempK)
    deltaV = -23.12 + 0.1758 * TempC - 0.002647 * TempC ** 2
    Kappa = (-5.15 + 0.09 * TempC) / 1000
    return Kfac(deltaV, Kappa, Pbar, TempK)


def KP3fac(TempK, Pbar):
    """Calculate pressure correction factor for KP3."""
    TempC = convert.TempK2C(TempK)
    deltaV = -26.57 + 0.202 * TempC - 0.003042 * TempC ** 2
    Kappa = (-4.08 + 0.0714 * TempC) / 1000
    return Kfac(deltaV, Kappa, Pbar, TempK)


def KSifac(TempK, Pbar):
    """Calculate pressure correction factor for KSi."""
    # === CO2SYS.m comments: =======
    # The only mention of this is Millero, 1995 where it is stated that the
    # values have been estimated from the values of boric acid. HOWEVER,
    # there is no listing of the values in the table.
    # Here we use the values for boric acid.
    TempC = convert.TempK2C(TempK)
    deltaV = -29.48 + 0.1622 * TempC - 0.002608 * TempC ** 2
    Kappa = -2.84 / 1000
    return Kfac(deltaV, Kappa, Pbar, TempK)


def KH2Sfac(TempK, Pbar):
    """Calculate pressure correction factor for KH2S."""
    # === CO2SYS.m comments: =======
    # Millero 1995 gives values for deltaV in fresh water instead of SW.
    # Millero 1995 gives -b0 as -2.89 instead of 2.89.
    # Millero 1983 is correct for both.
    TempC = convert.TempK2C(TempK)
    deltaV = -11.07 - 0.009 * TempC - 0.000942 * TempC ** 2
    Kappa = (-2.89 + 0.054 * TempC) / 1000
    return Kfac(deltaV, Kappa, Pbar, TempK)


def KNH3fac(TempK, Pbar):
    """Calculate pressure correction factor for KNH3."""
    # === CO2SYS.m comments: =======
    # The corrections are from Millero, 1995, which are the same as Millero, 1983.
    TempC = convert.TempK2C(TempK)
    deltaV = -26.43 + 0.0889 * TempC - 0.000905 * TempC ** 2
    Kappa = (-5.03 + 0.0814 * TempC) / 1000
    return Kfac(deltaV, Kappa, Pbar, TempK)


def K1fac(TempK, Pbar, WhichKs):
    """Calculate pressure correction factor for K1."""
    TempC = convert.TempK2C(TempK)
    K1fac = full(size(TempK), nan)  # because GEOSECS doesn't use _pcxKfac p1atm.
    F = WhichKs == 8  # freshwater
    if any(F):
        # Pressure effects on K1 in freshwater: this is from Millero, 1983.
        deltaV = -30.54 + 0.1849 * TempC - 0.0023366 * TempC ** 2
        Kappa = (-6.22 + 0.1368 * TempC - 0.001233 * TempC ** 2) / 1000
        K1fac = where(F, Kfac(deltaV, Kappa, Pbar, TempK), K1fac)
    F = (WhichKs == 6) | (WhichKs == 7)
    if any(F):
        # GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
        # Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
        # Culberson and Pytkowicz, L and O 13:403-417, 1968:
        # but the fits are the same as those in
        # Edmond and Gieskes, GCA, 34:1261-1291, 1970
        # who in turn quote Li, personal communication
        K1fac = where(
            F, exp((24.2 - 0.085 * TempC) * Pbar / (RGasConstant * TempK)), K1fac
        )
        # This one is handled differently because the equation doesn't fit the
        # standard deltaV & Kappa form of _pcxKfac.
    F = (WhichKs != 6) & (WhichKs != 7) & (WhichKs != 8)
    if any(F):
        # These are from Millero, 1995.
        # They are the same as Millero, 1979 and Millero, 1992.
        # They are from data of Culberson and Pytkowicz, 1968.
        deltaV = -25.5 + 0.1271 * TempC
        # deltaV = deltaV - .151*(Sali - 34.8) # Millero, 1979
        Kappa = (-3.08 + 0.0877 * TempC) / 1000
        # Kappa = Kappa - .578*(Sali - 34.8)/1000 # Millero, 1979
        # The fits given in Millero, 1983 are somewhat different.
        K1fac = where(F, Kfac(deltaV, Kappa, Pbar, TempK), K1fac)
    return K1fac


def K2fac(TempK, Pbar, WhichKs):
    """Calculate pressure correction factor for K2."""
    TempC = convert.TempK2C(TempK)
    K2fac = full(size(TempK), nan)  # because GEOSECS doesn't use _pcxKfac p1atm.
    F = WhichKs == 8  # freshwater
    if any(F):
        # Pressure effects on K2 in freshwater: this is from Millero, 1983.
        deltaV = -29.81 + 0.115 * TempC - 0.001816 * TempC ** 2
        Kappa = (-5.74 + 0.093 * TempC - 0.001896 * TempC ** 2) / 1000
        K2fac = where(F, Kfac(deltaV, Kappa, Pbar, TempK), K2fac)
    F = (WhichKs == 6) | (WhichKs == 7)
    if any(F):
        # GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
        # Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
        # Culberson and Pytkowicz, L and O 13:403-417, 1968:
        # but the fits are the same as those in
        # Edmond and Gieskes, GCA, 34:1261-1291, 1970
        # who in turn quote Li, personal communication
        K2fac = where(
            F, exp((16.4 - 0.04 * TempC) * Pbar / (RGasConstant * TempK)), K2fac
        )
        # Takahashi et al had 26.4, but 16.4 is from Edmond and Gieskes
        # and matches the GEOSECS results
        # This one is handled differently because the equation doesn't fit the
        # standard deltaV & Kappa form of _pcxKfac.
    F = (WhichKs != 6) & (WhichKs != 7) & (WhichKs != 8)
    if any(F):
        # These are from Millero, 1995.
        # They are the same as Millero, 1979 and Millero, 1992.
        # They are from data of Culberson and Pytkowicz, 1968.
        deltaV = -15.82 - 0.0219 * TempC
        # deltaV = deltaV + .321*(Sali - 34.8) # Millero, 1979
        Kappa = (1.13 - 0.1475 * TempC) / 1000
        # Kappa = Kappa - .314*(Sali - 34.8)/1000 # Millero, 1979
        # The fit given in Millero, 1983 is different.
        # Not by a lot for deltaV, but by much for Kappa.
        K2fac = where(F, Kfac(deltaV, Kappa, Pbar, TempK), K2fac)
    return K2fac
