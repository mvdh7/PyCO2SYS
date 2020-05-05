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
PAR1 = np.array([2300.0, 2150.0, 8.1   , 8.1   , 400 , 2150])
PAR2 = np.array([2150.0, 2300.0, 2300.0, 2150.0, 2150, 400 ])
PAR1TYPE = np.array([1, 2, 3, 3, 4, 1])
PAR2TYPE = np.array([2, 1, 1, 2, 1, 4])
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
