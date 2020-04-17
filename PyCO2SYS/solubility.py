# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"Calculate saturation states of soluble solids."

from . import assemble
from autograd.numpy import exp, full, log10, nan, size, sqrt

def CaSolubility(Sal, TempC, Pdbar, TC, pH, WhichKs, K1, K2):
    """Calculate calcite and aragonite solubility.

    This calculates omega, the solubility ratio, for calcite and aragonite.
    This is defined by: Omega = [CO3--]*[Ca++]/Ksp,
          where Ksp is the solubility product (either KCa or KAr).
    These are from: M83, I75, M79, ICHP73, B76, TWB82 and CP68.

    Based on CaSolubility, version 01.05, 05-23-97, written by Ernie Lewis.
    """
    TempK, Pbar, RT = assemble.units(TempC, Pdbar)
    Ca = full(size(Sal), nan)
    KCa = full(size(Sal), nan)
    KAr = full(size(Sal), nan)
    # Calculate Ca [RT67]:
    #       this is .010285*Sali/35
    Ca = 0.02128/40.087*(Sal/1.80655)# in mol/kg-SW
    # CalciteSolubility [M83]:
    logKCa = -171.9065 - 0.077993*TempK + 2839.319/TempK
    logKCa = logKCa + 71.595*log10(TempK)
    logKCa = logKCa + (-0.77712 + 0.0028426*TempK + 178.34/TempK)*sqrt(Sal)
    logKCa = logKCa - 0.07711*Sal + 0.0041249*sqrt(Sal)*Sal
    #       sd fit = .01 (for Sal part, not part independent of Sal)
    KCa = 10.0**logKCa # this is in (mol/kg-SW)^2
    # Aragonite Solubility [M83]:
    logKAr = -171.945 - 0.077993*TempK + 2903.293/TempK
    logKAr = logKAr + 71.595*log10(TempK)
    logKAr = logKAr + (-0.068393 + 0.0017276*TempK + 88.135/TempK)*sqrt(Sal)
    logKAr = logKAr - 0.10018*Sal + 0.0059415*sqrt(Sal)*Sal
    #       sd fit = .009 (for Sal part, not part independent of Sal)
    KAr    = 10.0**logKAr # this is in (mol/kg-SW)^2
    # Pressure correction for calcite [I75, M79]:
    #       note that Millero, GCA 1995 has typos
    #       (-.5304, -.3692, and 10^3 for Kappa factor)
    deltaVKCa = -48.76 + 0.5304*TempC
    KappaKCa = (-11.76 + 0.3692*TempC)/1000
    lnKCafac = (-deltaVKCa + 0.5*KappaKCa*Pbar)*Pbar/RT
    KCa = KCa*exp(lnKCafac)
    # Pressure correction for aragonite [M79]:
    #       same as Millero, GCA 1995 except for typos (-.5304, -.3692,
    #       and 10^3 for Kappa factor)
    deltaVKAr = deltaVKCa + 2.8
    KappaKAr = KappaKCa.copy()
    lnKArfac = (-deltaVKAr + 0.5*KappaKAr*Pbar)*Pbar/RT
    KAr = KAr*exp(lnKArfac)
    # Now overwrite GEOSECS values:
    F = (WhichKs==6) | (WhichKs==7)
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
            110.21*log10(Sal[F]) - 0.0000075752*TempK[F]**2)
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
    OmegaCa = CO3*Ca/KCa
    OmegaAr = CO3*Ca/KAr
    return OmegaCa, OmegaAr
