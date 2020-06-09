# These tests compare efficiently-computed derivatives with the uncertainty module
# against direct autograds over solve.fill.

from time import time
from autograd import numpy as np
from autograd import elementwise_grad as egrad
import PyCO2SYS as pyco2


def test_uncert_is_function():
    assert type(pyco2.uncertainty.automatic.dcore_dparX__parY).__name__ == "function"


def close_enough(a, b, tol=1e-10):
    """Assess whether `a` and `b` are similar enough to be acceptable."""
    return np.all(np.abs(a - b) < tol)


def prep(parX, parY, parXtype, parYtype):
    npts = np.size(parY)
    # Set and get total molinities
    Sal = np.full(npts, 31.0)
    TSi = np.full(npts, 12.0)
    TPO4 = np.full(npts, 1.5)
    TNH3 = np.full(npts, 2.0)
    TH2S = np.full(npts, 0.5)
    WhichKs = np.full(npts, 10)
    WhoseTB = np.full(npts, 2)
    totals = pyco2.salts.assemble(Sal, TSi, TPO4, TNH3, TH2S, WhichKs, WhoseTB)
    # Set and get equilibrium constants
    TempC = np.full(npts, 22.3)
    Pdbar = np.full(npts, 100.0)
    pHScale = np.full(npts, 1)
    WhoseKSO4 = np.full(npts, 1)
    WhoseKF = np.full(npts, 1)
    Ks = pyco2.equilibria.assemble(
        TempC, Pdbar, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
    )
    # Expand and solve MCS parameters
    Icase = pyco2.solve.getIcase(parXtype, parYtype, checks=True)
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = pyco2.solve.pair2core(
        parX, parY, parXtype, parYtype, convert_units=False, checks=True
    )
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = pyco2.solve.fill(
        Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks
    )
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks


def test_parX_TA():
    """Compare derivatives from uncertainties module vs direct grads of solve.fill."""
    # Define MCS parameters, expand and solve
    parY = np.array([2100e-6, 8.1, 400e-6, 400e-6, 350e-6, 1800e-6, 10e-6])
    parYtype = np.array([2, 3, 4, 5, 6, 7, 8])
    npts = np.size(parY)
    parX = np.full(npts, 2250e-6)
    parXtype = np.full(npts, 1)
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks = prep(
        parX, parY, parXtype, parYtype
    )
    # Get uncertainty derivatives with the uncertainty module
    go = time()
    uncert = pyco2.uncertainty.automatic.dcore_dparX__parY(
        parXtype, parYtype, TA, TC, PH, FC, CARB, HCO3, totals, Ks
    )
    print("  Uncertainty module runtime = {:.5f} s".format(time() - go))
    dTA_uncert = uncert["TAlk"]
    dTC_uncert = uncert["TCO2"]
    dPH_uncert = uncert["pH"]
    dPC_uncert = uncert["pCO2"]
    dFC_uncert = uncert["fCO2"]
    dCARB_uncert = uncert["CO3"]
    dHCO3_uncert = uncert["HCO3"]
    dCO2_uncert = uncert["CO2"]
    # Get corresponding uncertainty derivatives directly
    Icase = pyco2.solve.getIcase(parXtype, parYtype)

    def _dTA_direct(i):
        return egrad(
            lambda TA: pyco2.solve.fill(
                Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks
            )[i]
        )(TA)

    go = time()
    dTA_direct = _dTA_direct(0)
    dTC_direct = _dTA_direct(1)
    dPH_direct = _dTA_direct(2)
    dPC_direct = _dTA_direct(3)
    dFC_direct = _dTA_direct(4)
    dCARB_direct = _dTA_direct(5)
    dHCO3_direct = _dTA_direct(6)
    dCO2_direct = _dTA_direct(7)
    print("Direct uncertainties runtime = {:.5f} s".format(time() - go))
    # Make sure they agree
    # print(np.array([dTA_uncert, dTA_direct]))
    # print(np.array([dTC_uncert, dTC_direct]))
    # print(np.array([dPH_uncert, dPH_direct]))
    # print(np.array([dPC_uncert, dPC_direct]))
    # print(np.array([dFC_uncert, dFC_direct]))
    # print(np.array([dCARB_uncert, dCARB_direct]))
    # print(np.array([dHCO3_uncert, dHCO3_direct]))
    # print(np.array([dCO2_uncert, dCO2_direct]))
    assert close_enough(dTA_uncert, dTA_direct)
    assert close_enough(dTC_uncert, dTC_direct)
    assert close_enough(dPH_uncert, dPH_direct)
    assert close_enough(dPC_uncert, dPC_direct)
    assert close_enough(dFC_uncert, dFC_direct)
    assert close_enough(dCARB_uncert, dCARB_direct)
    assert close_enough(dHCO3_uncert, dHCO3_direct)
    assert close_enough(dCO2_uncert, dCO2_direct)


def test_parX_TC():
    """Compare derivatives from uncertainties module vs direct grads of solve.fill."""
    # Define MCS parameters
    parY = np.array([2300e-6, 8.1, 400e-6, 400e-6, 350e-6, 1800e-6, 10e-6])
    parYtype = np.array([1, 3, 4, 5, 6, 7, 8])
    npts = np.size(parY)
    parX = np.full(npts, 2100e-6)
    parXtype = np.full(npts, 2)
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks = prep(
        parX, parY, parXtype, parYtype
    )
    # Get uncertainty derivatives with the uncertainty module
    go = time()
    uncert = pyco2.uncertainty.automatic.dcore_dparX__parY(
        parXtype, parYtype, TA, TC, PH, FC, CARB, HCO3, totals, Ks
    )
    print("  Uncertainty module runtime = {:.5f} s".format(time() - go))
    dTA_uncert = uncert["TAlk"]
    dTC_uncert = uncert["TCO2"]
    dPH_uncert = uncert["pH"]
    dPC_uncert = uncert["pCO2"]
    dFC_uncert = uncert["fCO2"]
    dCARB_uncert = uncert["CO3"]
    dHCO3_uncert = uncert["HCO3"]
    dCO2_uncert = uncert["CO2"]
    # Get corresponding uncertainty derivatives directly
    Icase = pyco2.solve.getIcase(parXtype, parYtype)

    def _dTC_direct(i):
        return egrad(
            lambda TC: pyco2.solve.fill(
                Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks
            )[i]
        )(TC)

    go = time()
    dTA_direct = _dTC_direct(0)
    dTC_direct = _dTC_direct(1)
    dPH_direct = _dTC_direct(2)
    dPC_direct = _dTC_direct(3)
    dFC_direct = _dTC_direct(4)
    dCARB_direct = _dTC_direct(5)
    dHCO3_direct = _dTC_direct(6)
    dCO2_direct = _dTC_direct(7)
    print("Direct uncertainties runtime = {:.5f} s".format(time() - go))
    # Make sure they agree
    # print(np.array([dTA_uncert, dTA_direct]))
    # print(np.array([dTC_uncert, dTC_direct]))
    # print(np.array([dPH_uncert, dPH_direct]))
    # print(np.array([dPC_uncert, dPC_direct]))
    # print(np.array([dFC_uncert, dFC_direct]))
    # print(np.array([dCARB_uncert, dCARB_direct]))
    # print(np.array([dHCO3_uncert, dHCO3_direct]))
    # print(np.array([dCO2_uncert, dCO2_direct]))
    assert close_enough(dTA_uncert, dTA_direct)
    assert close_enough(dTC_uncert, dTC_direct)
    assert close_enough(dPH_uncert, dPH_direct)
    assert close_enough(dPC_uncert, dPC_direct)
    assert close_enough(dFC_uncert, dFC_direct)
    assert close_enough(dCARB_uncert, dCARB_direct)
    assert close_enough(dHCO3_uncert, dHCO3_direct)
    assert close_enough(dCO2_uncert, dCO2_direct)


def test_parX_PH():
    """Compare derivatives from uncertainties module vs direct grads of solve.fill."""
    # Define MCS parameters
    parY = np.array([2300e-6, 2100e-6, 400e-6, 400e-6, 350e-6, 1800e-6, 10e-6])
    parYtype = np.array([1, 2, 4, 5, 6, 7, 8])
    npts = np.size(parY)
    parX = np.full(npts, 8.1)
    parXtype = np.full(npts, 3)
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks = prep(
        parX, parY, parXtype, parYtype
    )
    # Get uncertainty derivatives with the uncertainty module
    go = time()
    uncert = pyco2.uncertainty.automatic.dcore_dparX__parY(
        parXtype, parYtype, TA, TC, PH, FC, CARB, HCO3, totals, Ks
    )
    print("  Uncertainty module runtime = {:.5f} s".format(time() - go))
    dTA_uncert = uncert["TAlk"]
    dTC_uncert = uncert["TCO2"]
    dPH_uncert = uncert["pH"]
    dPC_uncert = uncert["pCO2"]
    dFC_uncert = uncert["fCO2"]
    dCARB_uncert = uncert["CO3"]
    dHCO3_uncert = uncert["HCO3"]
    dCO2_uncert = uncert["CO2"]
    # Get corresponding uncertainty derivatives directly
    Icase = pyco2.solve.getIcase(parXtype, parYtype)

    def _dPH_direct(i):
        return egrad(
            lambda PH: pyco2.solve.fill(
                Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks
            )[i]
        )(PH)

    go = time()
    dTA_direct = _dPH_direct(0)
    dTC_direct = _dPH_direct(1)
    dPH_direct = _dPH_direct(2)
    dPC_direct = _dPH_direct(3)
    dFC_direct = _dPH_direct(4)
    dCARB_direct = _dPH_direct(5)
    dHCO3_direct = _dPH_direct(6)
    dCO2_direct = _dPH_direct(7)
    print("Direct uncertainties runtime = {:.5f} s".format(time() - go))
    # Make sure they agree
    # print(np.array([dTA_uncert, dTA_direct]))
    # print(np.array([dTC_uncert, dTC_direct]))
    # print(np.array([dPH_uncert, dPH_direct]))
    # print(np.array([dPC_uncert, dPC_direct]))
    # print(np.array([dFC_uncert, dFC_direct]))
    # print(np.array([dCARB_uncert, dCARB_direct]))
    # print(np.array([dHCO3_uncert, dHCO3_direct]))
    # print(np.array([dCO2_uncert, dCO2_direct]))
    assert close_enough(dTA_uncert, dTA_direct)
    assert close_enough(dTC_uncert, dTC_direct)
    assert close_enough(dPH_uncert, dPH_direct)
    assert close_enough(dPC_uncert, dPC_direct)
    assert close_enough(dFC_uncert, dFC_direct)
    assert close_enough(dCARB_uncert, dCARB_direct)
    assert close_enough(dHCO3_uncert, dHCO3_direct)
    assert close_enough(dCO2_uncert, dCO2_direct)


def test_parX_PC():
    """Compare derivatives from uncertainties module vs direct grads of solve.fill."""
    # Define MCS parameters
    parY = np.array([2300e-6, 2150e-6, 8.1, 350e-6, 1800e-6])
    parYtype = np.array([1, 2, 3, 6, 7])
    npts = np.size(parY)
    parX = np.full(npts, 400e-6)
    parXtype = np.full(npts, 4)
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks = prep(
        parX, parY, parXtype, parYtype
    )
    # Get uncertainty derivatives with the uncertainty module
    go = time()
    uncert = pyco2.uncertainty.automatic.dcore_dparX__parY(
        parXtype, parYtype, TA, TC, PH, FC, CARB, HCO3, totals, Ks
    )
    print("  Uncertainty module runtime = {:.5f} s".format(time() - go))
    dTA_uncert = uncert["TAlk"]
    dTC_uncert = uncert["TCO2"]
    dPH_uncert = uncert["pH"]
    dPC_uncert = uncert["pCO2"]
    dFC_uncert = uncert["fCO2"]
    dCARB_uncert = uncert["CO3"]
    dHCO3_uncert = uncert["HCO3"]
    dCO2_uncert = uncert["CO2"]
    # Get corresponding uncertainty derivatives directly
    Icase = pyco2.solve.getIcase(parXtype, parYtype)

    def _dPC_direct(i):
        return egrad(
            lambda PC: pyco2.solve.fill(
                Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks
            )[i]
        )(PC)

    go = time()
    dTA_direct = _dPC_direct(0)
    dTC_direct = _dPC_direct(1)
    dPH_direct = _dPC_direct(2)
    dPC_direct = _dPC_direct(3)
    dFC_direct = _dPC_direct(4)
    dCARB_direct = _dPC_direct(5)
    dHCO3_direct = _dPC_direct(6)
    dCO2_direct = _dPC_direct(7)
    print("Direct uncertainties runtime = {:.5f} s".format(time() - go))
    # Make sure they agree
    # print(np.array([dTA_uncert, dTA_direct]))
    # print(np.array([dTC_uncert, dTC_direct]))
    # print(np.array([dPH_uncert, dPH_direct]))
    # print(np.array([dPC_uncert, dPC_direct]))
    # print(np.array([dFC_uncert, dFC_direct]))
    # print(np.array([dCARB_uncert, dCARB_direct]))
    # print(np.array([dHCO3_uncert, dHCO3_direct]))
    # print(np.array([dCO2_uncert, dCO2_direct]))
    assert close_enough(dTA_uncert, dTA_direct)
    assert close_enough(dTC_uncert, dTC_direct)
    assert close_enough(dPH_uncert, dPH_direct)
    assert close_enough(dPC_uncert, dPC_direct)
    assert close_enough(dFC_uncert, dFC_direct)
    assert close_enough(dCARB_uncert, dCARB_direct)
    assert close_enough(dHCO3_uncert, dHCO3_direct)
    assert close_enough(dCO2_uncert, dCO2_direct)


def test_parX_CARB():
    """Compare derivatives from uncertainties module vs direct grads of solve.fill."""
    # Define MCS parameters
    parY = np.array([2300e-6, 2150e-6, 8.1, 400e-6, 400e-6, 1800e-6, 10e-6])
    parYtype = np.array([1, 2, 3, 4, 5, 7, 8])
    npts = np.size(parY)
    parX = np.full(npts, 350e-6)
    parXtype = np.full(npts, 6)
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks = prep(
        parX, parY, parXtype, parYtype
    )
    # Get uncertainty derivatives with the uncertainty module
    go = time()
    uncert = pyco2.uncertainty.automatic.dcore_dparX__parY(
        parXtype, parYtype, TA, TC, PH, FC, CARB, HCO3, totals, Ks
    )
    print("  Uncertainty module runtime = {:.5f} s".format(time() - go))
    dTA_uncert = uncert["TAlk"]
    dTC_uncert = uncert["TCO2"]
    dPH_uncert = uncert["pH"]
    dPC_uncert = uncert["pCO2"]
    dFC_uncert = uncert["fCO2"]
    dCARB_uncert = uncert["CO3"]
    dHCO3_uncert = uncert["HCO3"]
    dCO2_uncert = uncert["CO2"]
    # Get corresponding uncertainty derivatives directly
    Icase = pyco2.solve.getIcase(parXtype, parYtype)

    def _dCARB_direct(i):
        return egrad(
            lambda CARB: pyco2.solve.fill(
                Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks
            )[i]
        )(CARB)

    go = time()
    dTA_direct = _dCARB_direct(0)
    dTC_direct = _dCARB_direct(1)
    dPH_direct = _dCARB_direct(2)
    dPC_direct = _dCARB_direct(3)
    dFC_direct = _dCARB_direct(4)
    dCARB_direct = _dCARB_direct(5)
    dHCO3_direct = _dCARB_direct(6)
    dCO2_direct = _dCARB_direct(7)
    print("Direct uncertainties runtime = {:.5f} s".format(time() - go))
    # Make sure they agree
    # print(np.array([dTA_uncert, dTA_direct]))
    # print(np.array([dTC_uncert, dTC_direct]))
    # print(np.array([dPH_uncert, dPH_direct]))
    # print(np.array([dPC_uncert, dPC_direct]))
    # print(np.array([dFC_uncert, dFC_direct]))
    # print(np.array([dCARB_uncert, dCARB_direct]))
    # print(np.array([dHCO3_uncert, dHCO3_direct]))
    # print(np.array([dCO2_uncert, dCO2_direct]))
    assert close_enough(dTA_uncert, dTA_direct)
    assert close_enough(dTC_uncert, dTC_direct)
    assert close_enough(dPH_uncert, dPH_direct)
    assert close_enough(dPC_uncert, dPC_direct)
    assert close_enough(dFC_uncert, dFC_direct)
    assert close_enough(dCARB_uncert, dCARB_direct)
    assert close_enough(dHCO3_uncert, dHCO3_direct)
    assert close_enough(dCO2_uncert, dCO2_direct)


def test_parX_HCO3():
    """Compare derivatives from uncertainties module vs direct grads of solve.fill."""
    # Define MCS parameters
    parY = np.array([2300e-6, 2100e-6, 8.1, 400e-6, 400e-6, 350e-6, 10e-6])
    parYtype = np.array([1, 2, 3, 4, 5, 6, 8])
    npts = np.size(parY)
    parX = np.full(npts, 1800e-6)
    parXtype = np.full(npts, 7)
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks = prep(
        parX, parY, parXtype, parYtype
    )
    # Get uncertainty derivatives with the uncertainty module
    go = time()
    uncert = pyco2.uncertainty.automatic.dcore_dparX__parY(
        parXtype, parYtype, TA, TC, PH, FC, CARB, HCO3, totals, Ks
    )
    print("  Uncertainty module runtime = {:.5f} s".format(time() - go))
    dTA_uncert = uncert["TAlk"]
    dTC_uncert = uncert["TCO2"]
    dPH_uncert = uncert["pH"]
    dPC_uncert = uncert["pCO2"]
    dFC_uncert = uncert["fCO2"]
    dCARB_uncert = uncert["CO3"]
    dHCO3_uncert = uncert["HCO3"]
    dCO2_uncert = uncert["CO2"]
    # Get corresponding uncertainty derivatives directly
    Icase = pyco2.solve.getIcase(parXtype, parYtype)

    def _dHCO3_direct(i):
        return egrad(
            lambda HCO3: pyco2.solve.fill(
                Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks
            )[i]
        )(HCO3)

    go = time()
    dTA_direct = _dHCO3_direct(0)
    dTC_direct = _dHCO3_direct(1)
    dPH_direct = _dHCO3_direct(2)
    dPC_direct = _dHCO3_direct(3)
    dFC_direct = _dHCO3_direct(4)
    dCARB_direct = _dHCO3_direct(5)
    dHCO3_direct = _dHCO3_direct(6)
    dCO2_direct = _dHCO3_direct(7)
    print("Direct uncertainties runtime = {:.5f} s".format(time() - go))
    # Make sure they agree
    # print(np.array([dTA_uncert, dTA_direct]))
    # print(np.array([dTC_uncert, dTC_direct]))
    # print(np.array([dPH_uncert, dPH_direct]))
    # print(np.array([dPC_uncert, dPC_direct]))
    # print(np.array([dFC_uncert, dFC_direct]))
    # print(np.array([dCARB_uncert, dCARB_direct]))
    # print(np.array([dHCO3_uncert, dHCO3_direct]))
    # print(np.array([dCO2_uncert, dCO2_direct]))
    assert close_enough(dTA_uncert, dTA_direct)
    assert close_enough(dTC_uncert, dTC_direct)
    assert close_enough(dPH_uncert, dPH_direct)
    assert close_enough(dPC_uncert, dPC_direct)
    assert close_enough(dFC_uncert, dFC_direct)
    assert close_enough(dCARB_uncert, dCARB_direct)
    assert close_enough(dHCO3_uncert, dHCO3_direct)
    assert close_enough(dCO2_uncert, dCO2_direct)
