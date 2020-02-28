def ionstr_DOE94(Sal):
    """Ionic strength following DOE94."""
    # === CO2SYS.m comments: =======
    # This is from the DOE handbook, Chapter 5, p. 13/22, eq. 7.2.4.
    return 19.924*Sal/(1000 - 1.005*Sal)

def borate_C65(Sal):
    """Total borate in mol/kg-sw following C65."""
    # === CO2SYS.m comments: =======
    # This is .00001173*Sali, about 1% lower than Uppstrom's value
    # Culkin, F., in Chemical Oceanography, ed. Riley and Skirrow, 1965:
    # GEOSECS references this, but this value is not explicitly given here
    # Output in mol/kg-SW
    return 0.0004106*Sal/35

def borate_U74(Sal):
    """Total borate in mol/kg-sw following U74."""
    # === CO2SYS.m comments: =======
    # Uppstrom, L., Deep-Sea Research 21:161-162, 1974:
    # this is .000416*Sali/35. = .0000119*Sali
    # TB[FF] = (0.000232/10.811)*(Sal[FF]/1.80655); in mol/kg-SW.
    return 0.0004157*Sal/35

def borate_LKB10(Sal):
    """Total borate in mol/kg-sw following LKB10."""
    # === CO2SYS.m comments: =======
    # Lee, Kim, Byrne, Millero, Feely, Yong-Ming Liu. 2010.
    # Geochimica Et Cosmochimica Acta 74 (6): 1801-1811.
    # Output in mol/kg-SW.
    return 0.0004326*Sal/35

def fluoride_R65(Sal):
    """Total fluoride in mol/kg-sw following R65."""
    # === CO2SYS.m comments: =======
    # Riley, J. P., Deep-Sea Research 12:219-220, 1965:
    # this is .000068*Sali/35. = .00000195*Sali; in mol/kg-SW.
    return (0.000067/18.998)*Sal/1.80655

def sulfate_MR66(Sal):
    """Total sulfate in mol/kg-sw following MR66."""
    # === CO2SYS.m comments: =======
    # Morris, A. W., and Riley, J. P., Deep-Sea Research 13:699-705, 1966:
    # this is .02824*Sali/35. = .0008067*Sali; in mol/kg-SW.
    return (0.14/96.062)*Sal/1.80655
