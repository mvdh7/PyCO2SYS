def sws2tot(TS, KS, TF, KF):
    """Seawater to Total pH scale conversion factor."""
    return (1.0 + TS/KS)/(1.0 + TS/KS + TF/KF)

def free2tot(TS, KS):
    """Free to Total pH scale conversion factor."""
    return 1.0 + TS/KS

def fH_PTBO87(TempK, Sal):
    """fH following PTBO87."""
    # === CO2SYS.m comments: =======
    # Peng et al, Tellus 39B:439-458, 1987:
    # They reference the GEOSECS report, but round the value
    # given there off so that it is about .008 (1#) lower. It
    # doesn't agree with the check value they give on p. 456.
    return 1.29 - 0.00204*TempK + (0.00046 - 0.00000148*TempK)*Sal**2
        
def fH_TWB82(TempK, Sal):
    """fH following TWB82."""
    # === CO2SYS.m comments: =======
    # Takahashi et al, Chapter 3 in GEOSECS Pacific Expedition,
    # v. 3, 1982 (p. 80).
    return 1.2948 - 0.002036*TempK + (0.0004607 - 0.000001475*TempK)*Sal**2
