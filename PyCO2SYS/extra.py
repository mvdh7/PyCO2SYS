def buffers_ESM10(TC, TA, CO2, HCO3, CO3, H, OH, BAlk, KB):
    """Buffer factors from ESM10 with corrections for typographical errors
    described in the supp. info. to RAH18.
    """
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
