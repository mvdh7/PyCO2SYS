from numpy import exp, log, sqrt
from . import conc

def k0_W74(tempK, sal):
    """K0 (Henry's constant for CO2) in mol/kg-sw/atm following W74."""
    # === CO2SYS.m comments: =======
    # Weiss, R. F., Marine Chemistry 2:203-215, 1974.
    # This is in mol/kg-SW/atm.
    TempK100 = tempK/100
    lnK0 = (-60.2409 + 93.4517/TempK100 + 23.3585*log(TempK100) +
            sal*(0.023517 - 0.023656*TempK100 + 0.0047036*TempK100 **2))
    return exp(lnK0)
    
def kS_D90(tempK, sal):
    """Bisulfate dissociation constant following D90."""
    # === CO2SYS.m comments: =======
    # Dickson, A. G., J. Chemical Thermodynamics, 22:113-127, 1990
    # The goodness of fit is .021.
    # It was given in mol/kg-H2O. I convert it to mol/kg-SW.
    # TYPO on p. 121: the constant e9 should be e8.
    # Output KS is on the free pH scale in mol/kg-sw.
    # This is from eqs 22 and 23 on p. 123, and Table 4 on p 121:
    logTempK = log(tempK)
    IonS = conc.ionstr_DOE(sal)
    lnKS = (-4276.1/tempK + 141.328 - 23.093*logTempK +
      (-13856/tempK + 324.57 - 47.986*logTempK)*sqrt(IonS) +
      (35474/tempK - 771.54 + 114.723*logTempK)*IonS +
      (-2698/tempK)*sqrt(IonS)*IonS + (1776/tempK)*IonS**2)
    return exp(lnKS)*(1 - 0.001005*sal)

def kS_K77(TempK, Sal):
    """Bisulfate dissociation constant following K77."""
    # === CO2SYS.m comments: =======
    # Khoo, et al, Analytical Chemistry, 49(1):29-34, 1977
    # KS was found by titrations with a hydrogen electrode
    # of artificial seawater containing sulfate (but without F)
    # at 3 salinities from 20 to 45 and artificial seawater NOT
    # containing sulfate (nor F) at 16 salinities from 15 to 45,
    # both at temperatures from 5 to 40 deg C.
    # KS is on the Free pH scale (inherently so).
    # It was given in mol/kg-H2O. I convert it to mol/kg-SW.
    # He finds log(beta) which = my pKS;
    # his beta is an association constant.
    # The rms error is .0021 in pKS, or about .5% in KS.
    # This is equation 20 on p. 33:
    # Output KS is on the free pH scale in mol/kg-sw.
    IonS = conc.ionstr_DOE(Sal)
    pKS = 647.59/TempK - 6.3451 + 0.019085*TempK - 0.5208*sqrt(IonS)
    return 10.0**-pKS*(1 - 0.001005*Sal)

def kF_DR79(TempK, Sal):
    """Hydrogen fluoride dissociation constant following DR79."""
    # === CO2SYS.m comments: =======
    # Dickson, A. G. and Riley, J. P., Marine Chemistry 7:89-99, 1979:
    # this is on the free pH scale in mol/kg-sw
    IonS = conc.ionstr_DOE(Sal)
    lnKF = 1590.2/TempK - 12.641 + 1.525*IonS**0.5
    return exp(lnKF)*(1 - 0.001005*Sal)
    
def kF_PF87(TempK, Sal):
    """Hydrogen fluoride dissociation constant following PF87."""
    # Note that this is not currently used or an option in CO2SYS,
    # despite the equations below appearing in CO2SYS.m (commented out).
    # === CO2SYS.m comments: =======
    # Another expression exists for KF: Perez and Fraga 1987. Not used here
    # since ill defined for low salinity. (to be used for S: 10-40, T: 9-33)
    # Nonetheless, P&F87 might actually be better than the fit of D&R79 above,
    # which is based on only three salinities: [0 26.7 34.6]
    # Output is on the free pH scale in mol/kg-SW.
    lnKF = 874/TempK - 9.68 + 0.111*Sal**0.5
    return exp(lnKF)
