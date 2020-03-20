# NEXT STEPS (in no particular order):
# - Extract pressure correction equations from assemble.equilibria() and add
#   to relevant equilibria module functions (maybe?)
# - Add pressure correction references to docs
# - Extract subfunctions from _CaSolubility() into relevant modules.
# - Add references from _CaSolubility() to docs.
# - Calculate Egleston et al. buffer factors.
# - Calculate isocapnic quotient.
# - Implement generalised buffer factor equations of Middelburg/Hagens.
# - Calculate Revelle factor directly, not by differences.
# - Account for all species in pH solver loops.
# - Calculate high-Mg calcite solubility.
# - Move these steps as issues in the Github repo instead of a list here...!

from . import (
    assemble,
    concentrations,
    convert,
    equilibria,
    meta,
    original,
    solve,
)
__all__ = [
    'assemble',
    'concentrations',
    'convert',
    'equilibria',
    'meta',
    'original',
    'solve',
]

__author__ = 'Matthew P. Humphreys'
__version__ = meta.version

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

from numpy import (array, exp, full, full_like, log, log10, logical_or, nan,
                   shape, sqrt, zeros)
from numpy import min as np_min
from numpy import max as np_max

def _Fugacity(TempC, Sal, WhichKs):
    # CalculateFugacityConstants:
    # This assumes that the pressure is at one atmosphere, or close to it.
    # Otherwise, the Pres term in the exponent affects the results.
    #       Weiss, R. F., Marine Chemistry 2:203-215, 1974.
    #       Delta and B in cm3/mol
    TempK, _, RT = assemble.units(TempC, 0.0)
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

def _RevelleFactor(TA, TC, K0,
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
    dTC = 1e-6 # 1 umol/kg-SW
    # Find fCO2 at TA, TC+dTC
    TC_plus = TC + dTC
    pH_plus = solve.pHfromTATC(TA, TC_plus, *Ks, *Ts)
    fCO2_plus = solve.fCO2fromTCpH(TC_plus, pH_plus, K0, K1, K2)
    # Find fCO2 at TA, TC-dTC
    TC_minus = TC - dTC
    pH_minus = solve.pHfromTATC(TA, TC_minus, *Ks, *Ts)
    fCO2_minus = solve.fCO2fromTCpH(TC_minus, pH_minus, K0, K1, K2)
    # Calculate Revelle Factor
    Revelle = ((fCO2_plus - fCO2_minus)/dTC /
               ((fCO2_plus + fCO2_minus)/TC_minus))
    return Revelle

def _CaSolubility(Sal, TempC, Pdbar, TC, pH, WhichKs, K1, K2):
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
    TempK, Pbar, RT = assemble.units(TempC, Pdbar)
    Ca = full_like(Sal, nan)
    KCa = full_like(Sal, nan)
    KAr = full_like(Sal, nan)
    # CalculateCa:
    #       Riley, J. P. and Tongudai, M., Chemical Geology 2:263-269, 1967:
    #       this is .010285*Sali/35
    Ca = 0.02128/40.087*(Sal/1.80655)# ' in mol/kg-SW
    # CalciteSolubility:
    #       Mucci, Alphonso, Amer. J. of Science 283:781-799, 1983.
    logKCa = -171.9065 - 0.077993*TempK + 2839.319/TempK
    logKCa = logKCa + 71.595*log10(TempK)
    logKCa = logKCa + (-0.77712 + 0.0028426*TempK + 178.34/TempK)*sqrt(Sal)
    logKCa = logKCa - 0.07711*Sal + 0.0041249*sqrt(Sal)*Sal
    #       sd fit = .01 (for Sal part, not part independent of Sal)
    KCa = 10.0**(logKCa)# ' this is in (mol/kg-SW)^2
    # AragoniteSolubility:
    #       Mucci, Alphonso, Amer. J. of Science 283:781-799, 1983.
    logKAr = -171.945 - 0.077993*TempK + 2903.293/TempK
    logKAr = logKAr + 71.595*log10(TempK)
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
    factor[F] = log10(SWStoTOT[F])
    F = pHScale==3 # Free scale
    factor[F] = log10(FREEtoTOT[F])
    F = pHScale==4 # NBS scale
    factor[F] = log10(SWStoTOT[F]) - log10(fH[F])
    pHtot = pH - factor # pH comes into this sub on the given scale
    pHNBS  = pHtot + log10(SWStoTOT) - log10(fH)
    pHfree = pHtot + log10(FREEtoTOT)
    pHsws  = pHtot + log10(SWStoTOT)
    return pHtot, pHsws, pHfree, pHNBS

def _CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN,
        PRESOUT, SI, PO4, NH3, H2S, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANT,
        KFCONSTANT, BORON, KSO4CONSTANTS=0):

    # Condition inputs and assign input to the 'historical' variable names.
    args, ntps = assemble.inputs(locals())
    PAR1 = args['PAR1']
    PAR2 = args['PAR2']
    p1 = args['PAR1TYPE']
    p2 = args['PAR2TYPE']
    Sal = args['SAL'].copy()
    TempCi = args['TEMPIN']
    TempCo = args['TEMPOUT']
    Pdbari = args['PRESIN']
    Pdbaro = args['PRESOUT']
    TSi = args['SI'].copy()
    TP = args['PO4'].copy()
    TNH3 = args['NH3'].copy()
    TH2S = args['H2S'].copy()
    pHScale = args['pHSCALEIN']
    WhichKs = args['K1K2CONSTANTS']
    WhoseKSO4 = args['KSO4CONSTANT']
    WhoseKF = args['KFCONSTANT']
    WhoseTB = args['BORON']

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
    TB, TF, TS = assemble.concentrations(Sal, WhichKs, WhoseTB)
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
    ConstPuts = (pHScale, WhichKs, WhoseKSO4, WhoseKF, TP, TSi, Sal, TF, TS)
    (K0i, K1i, K2i, KWi, KBi, KFi, KSi, KP1i, KP2i, KP3i, KSii, KNH3i, KH2Si,
        fHi) = assemble.equilibria(TempCi, Pdbari, *ConstPuts)
    Kis = [K1i, K2i, KWi, KBi, KFi, KSi, KP1i, KP2i, KP3i, KSii, KNH3i, KH2Si]
    FugFaci, VPFaci = _Fugacity(TempCi, Sal, WhichKs)

    # Make sure fCO2 is available for each sample that has pCO2.
    F = logical_or(p1==4, p2==4)
    FC[F] = PC[F]*FugFaci[F]

    # Generate vector for results, and copy the raw input values into them. This
    # copies ~60% NaNs, which will be replaced for calculated values later on.
    TAc = TA.copy()
    TCc = TC.copy()
    PHic = PH.copy()
    PCic = PC.copy()
    FCic = FC.copy()
    CARBic = CARB.copy()

    # Generate vector describing the combination of input parameters
    # So, the valid ones are: 12,13,14,15,16,23,24,25,26,34,35,36,46,56
    Icase = (10*np_min(array([p1, p2]), axis=0) +
        np_max(array([p1, p2]), axis=0))

    # Calculate missing values for AT, CT, PH, FC:
    # pCO2 will be calculated later on, routines work with fCO2.
    F = Icase==12 # input TA, TC
    if any(F):
        KiFs, TFs = [[X[F] for X in Xs] for Xs in [Kis, Ts]]
        PHic[F] = solve.pHfromTATC(TAc[F]-PengCorrection[F], TCc[F],
                                   *KiFs, *TFs)
        # ^pH is returned on the scale requested in "pHscale" (see 'constants')
        FCic[F] = solve.fCO2fromTCpH(TCc[F], PHic[F], K0i[F], K1i[F], K2i[F])
        CARBic[F] = solve.CarbfromTCpH(TCc[F], PHic[F], K1i[F], K2i[F])
    F = Icase==13 # input TA, pH
    if any(F):
        KiFs, TFs = [[X[F] for X in Xs] for Xs in [Kis, Ts]]
        TCc[F] = solve.TCfromTApH(TAc[F]-PengCorrection[F], PHic[F],
                                  *KiFs, *TFs)
        FCic[F] = solve.fCO2fromTCpH(TCc[F], PHic[F],
                                     K0i[F], K1i[F], K2i[F])
        CARBic[F] = solve.CarbfromTCpH(TCc[F], PHic[F], K1i[F], K2i[F])
    F = logical_or(Icase==14, Icase==15) # input TA, (pCO2 or fCO2)
    if any(F):
        KiFs, TFs = [[X[F] for X in Xs] for Xs in [Kis, Ts]]
        PHic[F] = solve.pHfromTAfCO2(TAc[F]-PengCorrection[F], FCic[F],
                                     K0i[F], *KiFs, *TFs)
        TCc[F] = solve.TCfromTApH(TAc[F]-PengCorrection[F], PHic[F],
                                  *KiFs, *TFs)
        CARBic[F] = solve.CarbfromTCpH(TCc[F], PHic[F], K1i[F], K2i[F])
    F = Icase==16 # input TA, CARB
    if any(F):
        KiFs, TFs = [[X[F] for X in Xs] for Xs in [Kis, Ts]]
        PHic[F] = solve.pHfromTACarb(TAc[F]-PengCorrection[F], CARBic[F],
                                     *KiFs, *TFs)
        TCc[F] = solve.TCfromTApH(TAc[F]-PengCorrection[F], PHic[F],
                                      *KiFs, *TFs)
        FCic[F] = solve.fCO2fromTCpH(TCc[F], PHic[F], K0i[F], K1i[F], K2i[F])
    F = Icase==23 # input TC, pH
    if any(F):
        KiFs, TFs = [[X[F] for X in Xs] for Xs in [Kis, Ts]]
        TAc[F] = (solve.TAfromTCpH(TCc[F], PHic[F], *KiFs, *TFs) +
                  PengCorrection[F])
        FCic[F] = solve.fCO2fromTCpH(TCc[F], PHic[F],
                                     K0i[F], K1i[F], K2i[F])
        CARBic[F] = solve.CarbfromTCpH(TCc[F], PHic[F], K1i[F], K2i[F])
    F = logical_or(Icase==24, Icase==25) # input TC, (pCO2 or fCO2)
    if any(F):
        KiFs, TFs = [[X[F] for X in Xs] for Xs in [Kis, Ts]]
        PHic[F] = solve.pHfromTCfCO2(TCc[F], FCic[F], K0i[F], K1i[F], K2i[F])
        TAc[F] = (solve.TAfromTCpH(TCc[F], PHic[F], *KiFs, *TFs) +
                  PengCorrection[F])
        CARBic[F] = solve.CarbfromTCpH(TCc[F], PHic[F], K1i[F], K2i[F])
    F = Icase==26 # input TC, CARB
    if any(F):
        KiFs, TFs = [[X[F] for X in Xs] for Xs in [Kis, Ts]]
        PHic[F] = solve.pHfromTCCarb(TCc[F], CARBic[F], K1i[F], K2i[F])
        FCic[F] = solve.fCO2fromTCpH(TCc[F], PHic[F],
                                     K0i[F], K1i[F], K2i[F])
        TAc[F] = (solve.TAfromTCpH(TCc[F], PHic[F], *KiFs, *TFs) +
                  PengCorrection[F])
    F = logical_or(Icase==34, Icase==35) # input pH, (pCO2 or fCO2)
    if any(F):
        KiFs, TFs = [[X[F] for X in Xs] for Xs in [Kis, Ts]]
        TCc[F] = solve.TCfrompHfCO2(PHic[F], FCic[F], K0i[F], K1i[F], K2i[F])
        TAc[F] = (solve.TAfromTCpH(TCc[F], PHic[F], *KiFs, *TFs) +
                  PengCorrection[F])
        CARBic[F] = solve.CarbfromTCpH(TCc[F], PHic[F], K1i[F], K2i[F])
    F = Icase==36 # input pH, CARB
    if any(F):
        KiFs, TFs = [[X[F] for X in Xs] for Xs in [Kis, Ts]]
        FCic[F] = solve.fCO2frompHCarb(PHic[F], CARBic[F], K0i[F], K1i[F],
                                       K2i[F])
        TCc[F] = solve.TCfrompHfCO2(PHic[F], FCic[F], K0i[F], K1i[F], K2i[F])
        TAc[F] = (solve.TAfromTCpH(TCc[F], PHic[F], *KiFs, *TFs) +
                  PengCorrection[F])
    F = logical_or(Icase==46, Icase==56) # input (pCO2 or fCO2), CARB
    if any(F):
        KiFs, TFs = [[X[F] for X in Xs] for Xs in [Kis, Ts]]
        PHic[F] = solve.pHfromfCO2Carb(FCic[F], CARBic[F],
                                       K0i[F], K1i[F], K2i[F])
        TCc[F] = solve.TCfrompHfCO2(PHic[F], FCic[F], K0i[F], K1i[F], K2i[F])
        TAc[F] = (solve.TAfromTCpH(TCc[F], PHic[F], *KiFs, *TFs) +
                  PengCorrection[F])
    # By now, an fCO2 value is available for each sample.
    # Generate the associated pCO2 value:
    PCic = FCic/FugFaci

    # Calculate the pKs at input
    pK1i = -log10(K1i)
    pK2i = -log10(K2i)

    # CalculateOtherParamsAtInputConditions:
    (HCO3inp, CO3inp, BAlkinp, OHinp, PAlkinp, SiAlkinp, NH3Alkinp, H2SAlkinp,
        Hfreeinp, HSO4inp, HFinp) = solve.AlkParts(PHic, TCc, *Kis, *Ts)
    PAlkinp += PengCorrection
    CO2inp = TCc - CO3inp - HCO3inp
    Revelleinp = _RevelleFactor(TAc-PengCorrection, TCc, K0i, *Kis, *Ts)
    OmegaCainp, OmegaArinp = _CaSolubility(Sal, TempCi, Pdbari, TCc, PHic,
                                           WhichKs, K1i, K2i)
    xCO2dryinp = PCic/VPFaci # this assumes pTot = 1 atm

    # Just for reference, convert pH at input conditions to the other scales, too.
    pHicT, pHicS, pHicF, pHicN = _FindpHOnAllScales(PHic, pHScale, KSi, KFi,
                                                    TS, TF, fHi)

    # Calculate the constants for all samples at output conditions
    (K0o, K1o, K2o, KWo, KBo, KFo, KSo, KP1o, KP2o, KP3o, KSio, KNH3o, KH2So,
        fHo) = assemble.equilibria(TempCo, Pdbaro, *ConstPuts)
    Kos = [K1o, K2o, KWo, KBo, KFo, KSo, KP1o, KP2o, KP3o, KSio, KNH3o, KH2So]
    FugFaco, VPFaco = _Fugacity(TempCo, Sal, WhichKs)

    # Calculate, for output conditions, using conservative TA and TC, pH, fCO2 and pCO2
    PHoc = solve.pHfromTATC(TAc-PengCorrection, TCc, *Kos, *Ts)
    # ^pH is returned on the scale requested in "pHscale" (see 'constants')
    FCoc = solve.fCO2fromTCpH(TCc, PHoc, K0o, K1o, K2o)
    CARBoc = solve.CarbfromTCpH(TCc, PHoc, K1o, K2o)
    PCoc = FCoc/FugFaco

    # Calculate Other Stuff At Output Conditions:
    (HCO3out, CO3out, BAlkout, OHout, PAlkout, SiAlkout, NH3Alkout, H2SAlkout,
        Hfreeout, HSO4out, HFout) = solve.AlkParts(PHoc, TCc, *Kos, *Ts)
    PAlkout += PengCorrection
    CO2out = TCc - CO3out - HCO3out
    Revelleout = _RevelleFactor(TAc, TCc, K0o, *Kos, *Ts)
    OmegaCaout, OmegaArout = _CaSolubility(Sal, TempCo, Pdbaro, TCc, PHoc,
                                           WhichKs, K1o, K2o)
    xCO2dryout = PCoc/VPFaco # this assumes pTot = 1 atm

    # Just for reference, convert pH at output conditions to the other scales, too.
    pHocT, pHocS, pHocF, pHocN = _FindpHOnAllScales(PHoc, pHScale, KSo, KFo,
                                                    TS, TF, fHo)

    # Calculate the pKs at output
    pK1o = -log10(K1o)
    pK2o = -log10(K2o)

    # Save data directly as a dict to avoid ordering issues
    CO2dict = {
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
        'TEMPIN': args['TEMPIN'],
        'TEMPOUT': args['TEMPOUT'],
        'PRESIN': args['PRESIN'],
        'PRESOUT': args['PRESOUT'],
        'PAR1TYPE': args['PAR1TYPE'],
        'PAR2TYPE': args['PAR2TYPE'],
        'K1K2CONSTANTS': args['K1K2CONSTANTS'],
        'KSO4CONSTANTS': args['KSO4CONSTANTS'],
        'KSO4CONSTANT': args['KSO4CONSTANT'],
        'KFCONSTANT': args['KFCONSTANT'],
        'BORON': args['BORON'],
        'pHSCALEIN': args['pHSCALEIN'],
        'SAL': args['SAL'],
        'PO4': args['PO4'],
        'SI': args['SI'],
        'NH3': args['NH3'],
        'H2S': args['H2S'],
        'K0input': K0i,
        'K1input': K1i,
        'K2input': K2i,
        'pK1input': pK1i,
        'pK2input': pK2i,
        'KWinput': KWi,
        'KBinput': KBi,
        'KFinput': KFi,
        'KSinput': KSi,
        'KP1input': KP1i,
        'KP2input': KP2i,
        'KP3input': KP3i,
        'KSiinput': KSii,
        'KNH3input': KNH3i,
        'KH2Sinput': KH2Si,
        'K0output': K0o,
        'K1output': K1o,
        'K2output': K2o,
        'pK1output': pK1o,
        'pK2output': pK2o,
        'KWoutput': KWo,
        'KBoutput': KBo,
        'KFoutput': KFo,
        'KSoutput': KSo,
        'KP1output': KP1o,
        'KP2output': KP2o,
        'KP3output': KP3o,
        'KSioutput': KSio,
        'KNH3output': KNH3o,
        'KH2Soutput': KH2So,
        'TB': TB*1e6,
        'TF': TF*1e6,
        'TS': TS*1e6,
    }
    return CO2dict

def CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN,
        PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS,
        NH3=0.0, H2S=0.0, KFCONSTANT=1):
    """Solve the carbonate system using the input parameters.

    Based on CO2SYS v1.21 and v2.0.5, both for MATLAB, built over many years
    based on an original program by Ernie Lewis and Doug Wallace, with later
    contributions from S. van Heuven, J.W.B. Rae, J.C. Orr, J.-M. Epitalon,
    A.G. Dickson, J.-P. Gattuso, and D. Pierrot.

    Most recently converted for Python by Matthew Humphreys, NIOZ Royal
    Netherlands Institute for Sea Research, Texel, the Netherlands.
    """
    # Convert traditional inputs to new format before running CO2SYS
    if shape(KSO4CONSTANTS) == ():
        KSO4CONSTANTS = array([KSO4CONSTANTS])
    only2KSO4  = {1: 1, 2: 2, 3: 1, 4: 2,}
    only2BORON = {1: 1, 2: 1, 3: 2, 4: 2,}
    KSO4CONSTANT = array([only2KSO4[K] for K in KSO4CONSTANTS.ravel()])
    BORON = array([only2BORON[K] for K in KSO4CONSTANTS.ravel()])
    return _CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT,
        PRESIN, PRESOUT, SI, PO4, NH3, H2S, pHSCALEIN, K1K2CONSTANTS,
        KSO4CONSTANT, KFCONSTANT, BORON, KSO4CONSTANTS=KSO4CONSTANTS)
