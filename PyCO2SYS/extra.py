def buffers_ESM10(TC, TA, CO2, HCO3, CO3, pH, OH, BAlk, KB):
    """Buffer factors from ESM10 with corrections for typographical errors
    described in the supp. info. to RAH18.
    """
    H = 10.0**-pH
    # Evaluate ESM10 subfunctions (from their Table 1)
    S = HCO3 + 4*CO3 + H*BAlk/(KB + H) + H - OH
    P = 2*CO2 + HCO3
    Q = HCO3 - H*BAlk/(KB + H) - H - OH # see RAH18
    AC = HCO3 + 2*CO3
    # Calculate buffer factors
    gammaTC = TC - AC**2/S
    betaTC = (TC*S - AC**2)/AC
    omegaTC = TC - AC*P/Q # corrected, see RAH18 supp. info.
    ## omegaTC = TC - AC*P/HCO3 # original ESM10 equation, WRONG
    gammaTA = (AC**2 - TC*S)/AC
    betaTA = AC**2/TC - S
    omegaTA = AC - TC*Q/P # corrected as for omegaTC (RAH18), HCO3 => Q
    ## omegaTA = AC - TC*HCO3/P # original ESM10 equation, WRONG
    return gammaTC, betaTC, omegaTC, gammaTA, betaTA, omegaTA

def bgc_isocap(CO2, pH, K1, K2, KB, KW, TB):
    """Isocapnic quotient of HDW18, Eq. 8."""
    h = 10.0**-pH
    return ((K1*CO2*h + 4*K1*K2*CO2 + KW*h + h**3)*(KB + h)**2 +
        KB*TB*h**3)/(K1*CO2*(2*K2 + h)*(KB + h)**2)

def bgc_isocap_approx(TC, pCO2, K0, K1, K2):
    """Approximate isocapnic quotient of HDW18, Eq. 7."""
    return 1 + 2*(K2/(K0*K1))*TC/pCO2

def psi(CO2, pH, K1, K2, KB, KW, TB):
    """Psi of FCG94."""
    Q = bgc_isocap(CO2, pH, K1, K2, KB, KW, TB)
    return -1 + 2/Q
