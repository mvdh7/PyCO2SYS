# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from . import (
    assemble,
    buffers,
    concentrations,
    constants,
    convert,
    equilibria,
    meta,
    original,
    solubility,
    solve,
)
__all__ = [
    'assemble',
    'buffers',
    'concentrations',
    'constants',
    'convert',
    'equilibria',
    'meta',
    'original',
    'solubility',
    'solve',
]

__author__ = meta.authors
__version__ = meta.version

from autograd.numpy import (array, exp, full, full_like, log, log10,
                            nan, shape, sqrt, where, zeros)
from autograd.numpy import min as np_min
from autograd.numpy import max as np_max

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
    F = (WhichKs==6) | (WhichKs==7)
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
    F = (WhichKs==6) | (WhichKs==8)
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
    PengCorrection = where(WhichKs==7, TP, 0.0)

    # Calculate the constants for all samples at input conditions
    # The constants calculated for each sample will be on the appropriate pH
    # scale!
    ConstPuts = (pHScale, WhichKs, WhoseKSO4, WhoseKF, TP, TSi, Sal, TF, TS)
    (K0i, K1i, K2i, KWi, KBi, KFi, KSi, KP1i, KP2i, KP3i, KSii, KNH3i, KH2Si,
        fHi) = assemble.equilibria(TempCi, Pdbari, *ConstPuts)
    Kis = [K1i, K2i, KWi, KBi, KFi, KSi, KP1i, KP2i, KP3i, KSii, KNH3i, KH2Si]
    FugFaci, VPFaci = _Fugacity(TempCi, Sal, WhichKs)

    # Make sure fCO2 is available for each sample that has pCO2.
    F = (p1==4) | (p2==4)
    FC[F] = PC[F]*FugFaci[F]

    # Generate vector for results, and copy the raw input values into them. This
    # copies ~60% NaNs, which will be replaced for calculated values later on.
    TAc = TA.copy()
    TCc = TC.copy()
    PHic = PH.copy()
    PCic = PC.copy()
    FCic = FC.copy()
    CARBic = CARB.copy()

    TAc, TCc, PHic, PCic, FCic, CARBic = solve.from2to6(p1, p2, K0i, Kis, Ts,
        TAc, TCc, PH, PC, FC, CARB, PengCorrection, FugFaci)

    # Calculate the pKs at input
    pK1i = -log10(K1i)
    pK2i = -log10(K2i)

    # CalculateOtherParamsAtInputConditions:
    (HCO3inp, CO3inp, BAlkinp, OHinp, PAlkinp, SiAlkinp, NH3Alkinp, H2SAlkinp,
        Hfreeinp, HSO4inp, HFinp) = solve.AlkParts(PHic, TCc, *Kis, *Ts)
    PAlkinp += PengCorrection
    CO2inp = TCc - CO3inp - HCO3inp
    Revelleinp = buffers.RevelleFactor(TAc-PengCorrection, TCc, K0i, *Kis, *Ts)
    OmegaCainp, OmegaArinp = solubility.CaCO3(Sal, TempCi, Pdbari, TCc, PHic,
                                              WhichKs, K1i, K2i)
    xCO2dryinp = PCic/VPFaci # this assumes pTot = 1 atm

    # Just for reference, convert pH at input conditions to the other scales
    pHicT, pHicS, pHicF, pHicN = _FindpHOnAllScales(PHic, pHScale, KSi, KFi,
                                                    TS, TF, fHi)

    # Calculate the constants for all samples at output conditions
    (K0o, K1o, K2o, KWo, KBo, KFo, KSo, KP1o, KP2o, KP3o, KSio, KNH3o, KH2So,
        fHo) = assemble.equilibria(TempCo, Pdbaro, *ConstPuts)
    Kos = [K1o, K2o, KWo, KBo, KFo, KSo, KP1o, KP2o, KP3o, KSio, KNH3o, KH2So]
    FugFaco, VPFaco = _Fugacity(TempCo, Sal, WhichKs)

    # Calculate, for output conditions, using conservative TA and TC, pH, fCO2
    # and pCO2
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
    Revelleout = buffers.RevelleFactor(TAc, TCc, K0o, *Kos, *Ts)
    OmegaCaout, OmegaArout = solubility.CaCO3(Sal, TempCo, Pdbaro, TCc, PHoc,
                                              WhichKs, K1o, K2o)
    xCO2dryout = PCoc/VPFaco # this assumes pTot = 1 atm

    # Just for reference, convert pH at output conditions to the other scales
    pHocT, pHocS, pHocF, pHocN = _FindpHOnAllScales(PHoc, pHScale, KSo, KFo,
                                                    TS, TF, fHo)

    # Calculate the pKs at output
    pK1o = -log10(K1o)
    pK2o = -log10(K2o)

    # Evaluate ESM10 buffer factors (corrected following RAH18) [added v1.2.0]
    gammaTCi, betaTCi, omegaTCi, gammaTAi, betaTAi, omegaTAi = \
        buffers.buffers_ESM10(TCc, TAc, CO2inp, HCO3inp, CARBic, PHic, OHinp,
                              BAlkinp, KBi)
    gammaTCo, betaTCo, omegaTCo, gammaTAo, betaTAo, omegaTAo = \
        buffers.buffers_ESM10(TCc, TAc, CO2out, HCO3out, CARBoc, PHoc, OHout,
                              BAlkout, KBo)

    # Evaluate (approximate) isocapnic quotient [HDW18] and psi [FCG94]
    # [added v1.2.0]
    isoQi = buffers.bgc_isocap(CO2inp, PHic, K1i, K2i, KBi, KWi, TB)
    isoQxi = buffers.bgc_isocap_approx(TCc, PCic, K0i, K1i, K2i)
    psii = buffers.psi(CO2inp, PHic, K1i, K2i, KBi, KWi, TB)
    isoQo = buffers.bgc_isocap(CO2out, PHoc, K1o, K2o, KBo, KWo, TB)
    isoQxo = buffers.bgc_isocap_approx(TCc, PCoc, K0o, K1o, K2o)
    psio = buffers.psi(CO2out, PHoc, K1o, K2o, KBo, KWo, TB)

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
        # Added in v1.2.0:
        'gammaTCin': gammaTCi,
        'betaTCin': betaTCi,
        'omegaTCin': omegaTCi,
        'gammaTAin': gammaTAi,
        'betaTAin': betaTAi,
        'omegaTAin': omegaTAi,
        'gammaTCout': gammaTCo,
        'betaTCout': betaTCo,
        'omegaTCout': omegaTCo,
        'gammaTAout': gammaTAo,
        'betaTAout': betaTAo,
        'omegaTAout': omegaTAo,
        'isoQin': isoQi,
        'isoQout': isoQo,
        'isoQapprox_in': isoQxi,
        'isoQapprox_out': isoQxo,
        'psi_in': psii,
        'psi_out': psio,
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
