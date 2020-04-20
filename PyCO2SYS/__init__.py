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
    gas,
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
    'gas',
    'meta',
    'original',
    'solubility',
    'solve',
]

__author__ = meta.authors
__version__ = meta.version

from autograd.numpy import (array, exp, full, log, log10, nan, shape, size,
                            sqrt, where, zeros)
from autograd.numpy import min as np_min
from autograd.numpy import max as np_max

def _CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN,
        PRESOUT, SI, PO4, NH3, H2S, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANT,
        KFCONSTANT, BORON, KSO4CONSTANTS=0):
    # Condition inputs and assign input to the 'historical' variable names.
    args, ntps = assemble.inputs(locals())
    PAR1 = args['PAR1']
    PAR2 = args['PAR2']
    p1 = args['PAR1TYPE']
    p2 = args['PAR2TYPE']
    Sal = args['SAL']*1
    TempCi = args['TEMPIN']
    TempCo = args['TEMPOUT']
    Pdbari = args['PRESIN']
    Pdbaro = args['PRESOUT']
    TSi = args['SI']*1
    TP = args['PO4']*1
    TNH3 = args['NH3']*1
    TH2S = args['H2S']*1
    pHScale = args['pHSCALEIN']
    WhichKs = args['K1K2CONSTANTS']
    WhoseKSO4 = args['KSO4CONSTANT']
    WhoseKF = args['KFCONSTANT']
    WhoseTB = args['BORON']

    # Generate the columns holding Si, Phos and Sal.
    # Pure Water case:
    Sal = where(WhichKs==8, 0.0, Sal)
    # GEOSECS and Pure Water:
    F = (WhichKs==6) | (WhichKs==8)
    TP = where(F, 0.0, TP)
    TSi = where(F, 0.0, TSi)
    TNH3 = where(F, 0.0, TNH3)
    TH2S = where(F, 0.0, TH2S)
    # Convert micromol to mol
    TP = TP*1e-6
    TSi = TSi*1e-6
    TNH3 = TNH3*1e-6
    TH2S = TH2S*1e-6
    totals = assemble.concentrations(Sal, WhichKs, WhoseTB)
    # Add user inputs to `totals` dict
    totals['TPO4'] = TP
    totals['TSi'] = TSi
    totals['TNH3'] = TNH3
    totals['TH2S'] = TH2S
    # The vector `PengCorrection` is used to modify the value of TA, for those
    # cases where WhichKs==7, since PAlk(Peng) = PAlk(Dickson) + TP.
    # Thus, PengCorrection is 0 for all cases where WhichKs is not 7.
    PengCorrection = where(WhichKs==7, totals['TPO4'], 0.0)

    # Generate empty vectors for...
    TA = full(ntps, nan) # Talk
    TC = full(ntps, nan) # DIC
    PH = full(ntps, nan) # pH
    PC = full(ntps, nan) # pCO2
    FC = full(ntps, nan) # fCO2
    CARB = full(ntps, nan) # CO3 ions
    # Assign values to empty vectors and convert micro[mol|atm] to [mol|atm]
    TA = where(p1==1, PAR1*1e-6, TA)
    TC = where(p1==2, PAR1*1e-6, TC)
    PH = where(p1==3, PAR1, PH)
    PC = where(p1==4, PAR1*1e-6, PC)
    FC = where(p1==5, PAR1*1e-6, FC)
    CARB = where(p1==6, PAR1*1e-6, CARB)
    TA = where(p2==1, PAR2*1e-6, TA)
    TC = where(p2==2, PAR2*1e-6, TC)
    PH = where(p2==3, PAR2, PH)
    PC = where(p2==4, PAR2*1e-6, PC)
    FC = where(p2==5, PAR2*1e-6, FC)
    CARB = where(p2==6, PAR2*1e-6, CARB)

    # Calculate the constants for all samples at input conditions.
    # The constants calculated for each sample will be on the appropriate pH
    # scale!
    ConstPuts = (Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF)
    K0i, fHi, Kis = assemble.equilibria(TempCi, Pdbari, *ConstPuts)
    FugFaci = gas.fugacityfactor(TempCi, WhichKs)
    VPFaci = gas.vpfactor(TempCi, Sal)

    # Make sure fCO2 is available for each sample that has pCO2.
    FC = where((p1==4) | (p2==4), PC*FugFaci, FC)

    # Generate vector for results, and copy the raw input values into them. This
    # copies ~60% NaNs, which will be replaced for calculated values later on.
    TAc = TA*1
    TCc = TC*1
    PHic = PH*1
    PCic = PC*1
    FCic = FC*1
    CARBic = CARB*1

    TAc, TCc, PHic, PCic, FCic, CARBic = solve.from2to6(p1, p2, K0i,
        TAc, TCc, PH, PC, FC, CARB, PengCorrection, FugFaci, Kis, totals)

    # Calculate the pKs at input
    pK1i = -log10(Kis['K1'])
    pK2i = -log10(Kis['K2'])

    # CalculateOtherParamsAtInputConditions:
    (HCO3inp, CO3inp, BAlkinp, OHinp, PAlkinp, SiAlkinp, NH3Alkinp, H2SAlkinp,
        Hfreeinp, HSO4inp, HFinp) = solve.AlkParts(PHic, TCc, **Kis, **totals)
    PAlkinp += PengCorrection
    CO2inp = TCc - CO3inp - HCO3inp
    Revelleinp = buffers.RevelleFactor(TAc-PengCorrection, TCc, K0i, Kis,
                                       totals)
    OmegaCainp, OmegaArinp = solubility.CaCO3(Sal, TempCi, Pdbari, TCc, PHic,
                                              WhichKs, Kis['K1'], Kis['K2'])
    xCO2dryinp = PCic/VPFaci # this assumes pTot = 1 atm

    # Just for reference, convert pH at input conditions to the other scales
    pHicT, pHicS, pHicF, pHicN = convert.pH2allscales(PHic, pHScale,
        Kis['KSO4'], Kis['KF'], totals['TSO4'], totals['TF'], fHi)

    # Calculate the constants for all samples at output conditions
    K0o, fHo, Kos = assemble.equilibria(TempCo, Pdbaro, *ConstPuts)
    FugFaco = gas.fugacityfactor(TempCo, WhichKs)
    VPFaco = gas.vpfactor(TempCo, Sal)

    # Calculate, for output conditions, using conservative TA and TC, pH, fCO2
    # and pCO2
    PHoc = solve.pHfromTATC(TAc-PengCorrection, TCc, **Kos, **totals)
    # ^pH is returned on the scale requested in "pHscale" (see 'constants')
    FCoc = solve.fCO2fromTCpH(TCc, PHoc, K0o, Kos['K1'], Kos['K2'])
    CARBoc = solve.CarbfromTCpH(TCc, PHoc, Kos['K1'], Kos['K2'])
    PCoc = FCoc/FugFaco

    # Calculate other stuff at output conditions:
    (HCO3out, CO3out, BAlkout, OHout, PAlkout, SiAlkout, NH3Alkout, H2SAlkout,
        Hfreeout, HSO4out, HFout) = solve.AlkParts(PHoc, TCc, **Kos, **totals)
    PAlkout += PengCorrection
    CO2out = TCc - CO3out - HCO3out
    Revelleout = buffers.RevelleFactor(TAc-PengCorrection, TCc, K0o, Kos,
                                       totals)
    OmegaCaout, OmegaArout = solubility.CaCO3(Sal, TempCo, Pdbaro, TCc, PHoc,
                                              WhichKs, Kos['K1'], Kos['K2'])
    xCO2dryout = PCoc/VPFaco # this assumes pTot = 1 atm

    # Just for reference, convert pH at output conditions to the other scales
    pHocT, pHocS, pHocF, pHocN = convert.pH2allscales(PHoc, pHScale,
        Kos['KSO4'], Kos['KF'], totals['TSO4'], totals['TF'], fHo)

    # Calculate the pKs at output
    pK1o = -log10(Kos['K1'])
    pK2o = -log10(Kos['K2'])

    # Evaluate ESM10 buffer factors (corrected following RAH18) [added v1.2.0]
    gammaTCi, betaTCi, omegaTCi, gammaTAi, betaTAi, omegaTAi = \
        buffers.buffers_ESM10(TCc, TAc, CO2inp, HCO3inp, CARBic, PHic, OHinp,
                              BAlkinp, Kis['KB'])
    gammaTCo, betaTCo, omegaTCo, gammaTAo, betaTAo, omegaTAo = \
        buffers.buffers_ESM10(TCc, TAc, CO2out, HCO3out, CARBoc, PHoc, OHout,
                              BAlkout, Kos['KB'])

    # Evaluate (approximate) isocapnic quotient [HDW18] and psi [FCG94]
    # [added v1.2.0]
    isoQi = buffers.bgc_isocap(CO2inp, PHic, Kis['K1'], Kis['K2'], Kis['KB'],
        Kis['KW'], totals['TB'])
    isoQxi = buffers.bgc_isocap_approx(TCc, PCic, K0i, Kis['K1'], Kis['K2'])
    psii = buffers.psi(CO2inp, PHic, Kis['K1'], Kis['K2'], Kis['KB'], Kis['KW'],
        totals['TB'])
    isoQo = buffers.bgc_isocap(CO2out, PHoc, Kos['K1'], Kos['K2'], Kos['KB'],
        Kos['KW'], totals['TB'])
    isoQxo = buffers.bgc_isocap_approx(TCc, PCoc, K0o, Kos['K1'], Kos['K2'])
    psio = buffers.psi(CO2out, PHoc, Kos['K1'], Kos['K2'], Kos['KB'], Kos['KW'],
        totals['TB'])

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
        'K1input': Kis['K1'],
        'K2input': Kis['K2'],
        'pK1input': pK1i,
        'pK2input': pK2i,
        'KWinput': Kis['KW'],
        'KBinput': Kis['KB'],
        'KFinput': Kis['KF'],
        'KSinput': Kis['KSO4'],
        'KP1input': Kis['KP1'],
        'KP2input': Kis['KP2'],
        'KP3input': Kis['KP3'],
        'KSiinput': Kis['KSi'],
        'KNH3input': Kis['KNH3'],
        'KH2Sinput': Kis['KH2S'],
        'K0output': K0o,
        'K1output': Kos['K1'],
        'K2output': Kos['K2'],
        'pK1output': pK1o,
        'pK2output': pK2o,
        'KWoutput': Kos['KW'],
        'KBoutput': Kos['KB'],
        'KFoutput': Kos['KF'],
        'KSoutput': Kos['KSO4'],
        'KP1output': Kos['KP1'],
        'KP2output': Kos['KP2'],
        'KP3output': Kos['KP3'],
        'KSioutput': Kos['KSi'],
        'KNH3output': Kos['KNH3'],
        'KH2Soutput': Kos['KH2S'],
        'TB': totals['TB']*1e6,
        'TF': totals['TF']*1e6,
        'TS': totals['TSO4']*1e6,
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
