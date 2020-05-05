import PyCO2SYS as pyco2
from autograd import elementwise_grad as egrad
from autograd import numpy as np


def CO2SYS(
    PAR1,
    PAR2,
    PAR1TYPE,
    PAR2TYPE,
    SAL,
    TEMPIN,
    TEMPOUT,
    PRESIN,
    PRESOUT,
    SI,
    PO4,
    pHSCALEIN,
    K1K2CONSTANTS,
    KSO4CONSTANTS,
    NH3=0.0,
    H2S=0.0,
    KFCONSTANT=1,
    buffers_mode="auto",
    getgrads=[],
):
    # Convert traditional inputs to new format before running CO2SYS
    KSO4CONSTANT, BORON = pyco2.engine._convertoptions(KSO4CONSTANTS)
    # Make list of non-gradable CO2SYS inputs for convenience
    ngargs = [
        pHSCALEIN,
        K1K2CONSTANTS,
        KSO4CONSTANT,
        KFCONSTANT,
        BORON,
        buffers_mode,
        KSO4CONSTANTS,
    ]
    # Solve the marine carbonate system as normal
    outputdict, gradables = pyco2.engine._CO2SYS(
        PAR1,
        PAR2,
        PAR1TYPE,
        PAR2TYPE,
        SAL,
        TEMPIN,
        TEMPOUT,
        PRESIN,
        PRESOUT,
        SI,
        PO4,
        NH3,
        H2S,
        *ngargs
    )
    # Define input variables to get derivatives with respect to
    gradvars = [
        "PAR1",
        "PAR2",
        "SAL",
        "TEMPIN",
        "TEMPOUT",
        "PRESIN",
        "PRESOUT",
        "SI",
        "PO4",
        "NH3",
        "H2S",
    ]
    PARs = np.vstack([outputdict[var] for var in gradvars])
    # Work out which outputs we want to calculate derivatives for based on user input
    allgrads = {}
    if type(getgrads) == str:
        if getgrads == "all":
            getgrads = gradables
        elif getgrads == "none":
            getgrads = []
        else:
            getgrads = [getgrads]
    # Loop through all outputs and calculate derivatives w.r.t. all inputs
    for gradable in gradables:
        if gradable in getgrads:
            print("Getting derivatives of {}...".format(gradable))
            grad_PARs = egrad(
                lambda PARs: pyco2.engine._CO2SYS(
                    PARs[0],
                    PARs[1],
                    PAR1TYPE,
                    PAR2TYPE,
                    PARs[2],
                    PARs[3],
                    PARs[4],
                    PARs[5],
                    PARs[6],
                    PARs[7],
                    PARs[8],
                    PARs[9],
                    PARs[10],
                    *ngargs
                )[0][gradable]
            )
            allgrads_gradable = grad_PARs(PARs)
            allgrads[gradable] = {
                var: allgrads_gradable[i] for i, var in enumerate(gradvars)
            }
    return outputdict, allgrads


# Set input conditions and run analysis
PAR1 = np.array([8.1   , 2150.0])
PAR2 = np.array([2300.0, 2300])
PAR1TYPE = np.array([3, 2])
PAR2TYPE = np.array([1, 1])
SAL = 35.0
TEMPIN = 25.0
TEMPOUT = 10.0
PRESIN = 0.0
PRESOUT = 1000
SI = 3
PO4 = 2
pHSCALEIN = 1
K1K2CONSTANTS = 10
KSO4CONSTANTS = 3
co2args = (
    PAR1,
    PAR2,
    PAR1TYPE,
    PAR2TYPE,
    SAL,
    TEMPIN,
    TEMPOUT,
    PRESIN,
    PRESOUT,
    SI,
    PO4,
    pHSCALEIN,
    K1K2CONSTANTS,
    KSO4CONSTANTS,
)
co2py, allgrads = CO2SYS(*co2args, getgrads="OmegaARin")

# Figure out NaNs
def _CO2SYSprep(
    PAR1,
    PAR2,
    PAR1TYPE,
    PAR2TYPE,
    SAL,
    TEMPIN,
    TEMPOUT,
    PRESIN,
    PRESOUT,
    SI,
    PO4,
    NH3,
    H2S,
    pHSCALEIN,
    K1K2CONSTANTS,
    KSO4CONSTANT,
    KFCONSTANT,
    BORON,
    buffers_mode,
    KSO4CONSTANTS=0,
):
    # Condition inputs and assign input values to the 'historical' variable names
    args, ntps = pyco2.engine.inputs(locals())
    PAR1 = args["PAR1"]
    PAR2 = args["PAR2"]
    p1 = args["PAR1TYPE"]
    p2 = args["PAR2TYPE"]
    Sal = args["SAL"]
    TempCi = args["TEMPIN"]
    TempCo = args["TEMPOUT"]
    Pdbari = args["PRESIN"]
    Pdbaro = args["PRESOUT"]
    TSi = args["SI"]
    TP = args["PO4"]
    TNH3 = args["NH3"]
    TH2S = args["H2S"]
    pHScale = args["pHSCALEIN"]
    WhichKs = args["K1K2CONSTANTS"]
    WhoseKSO4 = args["KSO4CONSTANT"]
    WhoseKF = args["KFCONSTANT"]
    WhoseTB = args["BORON"]
    buffers_mode = args["buffers_mode"]
    # Prepare to solve the core marine carbonate system at input conditions
    totals = pyco2.salts.assemble(Sal, TSi, TP, TNH3, TH2S, WhichKs, WhoseTB)
    Sal = totals["Sal"]
    Kis = pyco2.equilibria.assemble(
        TempCi, Pdbari, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
    )
    # Calculate fugacity factor
    FugFaci = pyco2.gas.fugacityfactor(TempCi, WhichKs)
    # Expand inputs `par1` and `par2` into one array per core MCS variable
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = pyco2.engine.pair2core(
        PAR1, PAR2, PAR1TYPE, PAR2TYPE, True
    )
    # Generate vector describing the combination(s) of input parameters
    Icase = pyco2.engine.getIcase(PAR1TYPE, PAR2TYPE)
    return Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, FugFaci, Kis, totals

Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, FugFaci, Kis, totals = _CO2SYSprep(
    PAR1,
    PAR2,
    PAR1TYPE,
    PAR2TYPE,
    SAL,
    TEMPIN,
    TEMPOUT,
    PRESIN,
    PRESOUT,
    SI,
    PO4,
    0,
    0,
    pHSCALEIN,
    K1K2CONSTANTS,
    1,
    1,
    1,
    "auto",
    KSO4CONSTANTS=0,
)


# Problem is to do with ordering - first `if any()` does egrad, second doesn't
def solvetest(Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, FugFaci, Kis, totals):
    K0 = Kis['K0']
    K1 = Kis['K1']
    K2 = Kis['K2']
    
    F = Icase == 12  # input TA, TC
    if any(F):
        # PH = np.where(F, pyco2.solve.get.pHfromTATC(TA, TC, Kis, totals), PH)
        PH = np.where(F, TA, PH)
        # ^pH is returned on the same scale as `Ks`
        # FC = np.where(F, pyco2.solve.get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        FC = np.where(F, TA, FC)
        # CARB = np.where(F, pyco2.solve.get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        # HCO3 = np.where(F, pyco2.solve.get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
        
    F = Icase == 13  # input TA, pH
    if any(F):
        TC = np.where(F, pyco2.solve.get.TCfromTApH(TA, PH, Kis, totals), TC)
        # FC = np.where(F, pyco2.solve.get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        FC = np.where(F, TA, FC)
        # CARB = np.where(F, pyco2.solve.get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        # HCO3 = np.where(F, pyco2.solve.get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    
    return TC

testout = solvetest(Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, FugFaci, Kis, totals)

testgrad = egrad(lambda TA: solvetest(
    Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, FugFaci, Kis, totals))(TA)

print(Icase)
print(testout)
print(testgrad)


phtest = egrad(lambda TA: pyco2.solve.get.pHfromTATC(TA, TC, Kis, totals))(TA)

