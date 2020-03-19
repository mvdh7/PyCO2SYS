function [DATA,HEADERS,NICEHEADERS]=CO2SYSv1_21(PAR1,PAR2,PAR1TYPE,PAR2TYPE,SAL,TEMPIN,TEMPOUT,PRESIN,PRESOUT,SI,PO4,NH3,H2S,pHSCALEIN,K1K2CONSTANTS,KSO4CONSTANT,KFCONSTANT,BORON)
%**************************************************************************
%
% This is CO2SYS version 1.21 (SEPT-2019)
%
% CO2SYS is a MATLAB-version of the original CO2SYS for DOS. 
% CO2SYS calculates and returns the state of the carbonate system of 
%    oceanographic water samples, if supplied with enough input.
%
% Please note that his software is intended to be exactly identical to the 
%    DOS and Excel versions that have been released previously, meaning that
%    results obtained should be very nearly indentical for identical input.
% Additionally, several of the dissociation constants K1 and K2 that have 
%    been published since the original DOS version was written are implemented.
%    For a complete list of changes since version 1.0, see below.
%
% For much more info please have a look at:
%    Lewis, E., and D. W. R. Wallace. 1998. Program Developed for
%    CO2 System Calculations. ORNL/CDIAC-105. Carbon Dioxide Information
%    Analysis Center, Oak Ridge National Laboratory, U.S. Department of Energy,
%    Oak Ridge, Tennessee. 
%    http://cdiac.ornl.gov/oceans/co2rprt.html
%
%**************************************************************************
%
%  **** SYNTAX:
%  [RESULT,HEADERS,NICEHEADERS]=CO2SYS_v1_21(PAR1,PAR2,PAR1TYPE,PAR2TYPE,...
%        ...SAL,TEMPIN,TEMPOUT,PRESIN,PRESOUT,SI,PO4,NH3,H2S,pHSCALEIN,...
%        ...K1K2CONSTANTS,KSO4CONSTANT,KFCONSTANT,BORON)
% 
%  **** SYNTAX EXAMPLES:
%  [Result]                     = CO2SYS_v1_21(2400,2200,1,2,35,25,4,0,4200,15,1,0,0,1,4,1,1,1)
%  [Result,Headers]             = CO2SYS_v1_21(2400,   8,1,3,35,25,4,0,4200,15,1,0,0,1,4,1,1,1)
%  [Result,Headers,Niceheaders] = CO2SYS_v1_21( 500,   8,5,3,35,25,4,0,4200,15,1,0,0,1,4,1,1,1)
%  [A]                          = CO2SYS_v1_21(2400,2000:10:2400,1,2,35,25,4,0,4200,15,1,0,0,1,4,1,1,1)
%  [A]                          = CO2SYS_v1_21(2400,2200,1,2,0:1:35,25,4,0,4200,15,1,0,0,1,4,1,1,1)
%  [A]                          = CO2SYS_v1_21(2400,2200,1,2,35,25,4,0,0:100:4200,15,1,0,0,1,4,1,1,1)
%  
%  **** APPLICATION EXAMPLE (copy and paste this into command window):
%  tmps=0:40; sals=0:40; [X,Y]=meshgrid(tmps,sals);
%  A = CO2SYS_v1_21(2300,2100,1,2,Y(:),X(:),nan,0,nan,1,1,0,0,1,9,1,1,1);
%  Z=nan(size(X)); Z(:)=A(:,4); figure; contourf(X,Y,Z,20); caxis([0 1200]); colorbar;
%  ylabel('Salinity [psu]'); xlabel('Temperature [degC]'); title('Dependence of pCO2 [uatm] on T and S')
% 
%**************************************************************************
%
% INPUT:
%
%   PAR1  (some unit) : scalar or vector of size n
%   PAR2  (some unit) : scalar or vector of size n
%   PAR1TYPE       () : scalar or vector of size n (Footnote *1 below)
%   PAR2TYPE       () : scalar or vector of size n (Footnote *1 below)
%   SAL            () : scalar or vector of size n
%   TEMPIN  (degr. C) : scalar or vector of size n 
%   TEMPOUT (degr. C) : scalar or vector of size n 
%   PRESIN     (dbar) : scalar or vector of size n 
%   PRESOUT    (dbar) : scalar or vector of size n
%   SI    (umol/kgSW) : scalar or vector of size n
%   PO4   (umol/kgSW) : scalar or vector of size n
%   NH3   (umol/kgSW) : scalar or vector of size n
%   H2S   (umol/kgSW) : scalar or vector of size n
%   pHSCALEIN         : scalar or vector of size n (Footnote *2 below)
%   K1K2CONSTANTS     : scalar or vector of size n (Footnote *3 below)
%   KSO4CONSTANT      : scalar or vector of size n (Footnote *4 below)
%   KFCONSTANT       : scalar or vector of size n (Footnote *5 below)
%   BORON         : scalar or vector of size n (Footnote *6 below)
%
%  (*1) Each element must be an integer, 
%      indicating that PAR1 (or PAR2) is of type: 
%  1 = Total Alkalinity
%  2 = DIC
%  3 = pH
%  4 = pCO2
%  5 = fCO2
% 
%  (*2) Each element must be an integer, 
%       indicating that the pH-input (PAR1 or PAR2, if any) is at:
%  1 = Total scale
%  2 = Seawater scale
%  3 = Free scale
%  4 = NBS scale
% 
%  (*3) Each element must be an integer, 
%        indicating the K1 K2 dissociation constants that are to be used:
%   1 = Roy, 1993											T:    0-45  S:  5-45. Total scale. Artificial seawater.
%   2 = Goyet & Poisson										T:   -1-40  S: 10-50. Seaw. scale. Artificial seawater.
%   3 = HANSSON              refit BY DICKSON AND MILLERO	T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
%   4 = MEHRBACH             refit BY DICKSON AND MILLERO	T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
%   5 = HANSSON and MEHRBACH refit BY DICKSON AND MILLERO	T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
%   6 = GEOSECS (i.e., original Mehrbach)					T:    2-35  S: 19-43. NBS scale.   Real seawater.
%   7 = Peng	(i.e., originam Mehrbach but without XXX)	T:    2-35  S: 19-43. NBS scale.   Real seawater.
%   8 = Millero, 1979, FOR PURE WATER ONLY (i.e., Sal=0)	T:    0-50  S:     0. 
%   9 = Cai and Wang, 1998									T:    2-35  S:  0-49. NBS scale.   Real and artificial seawater.
%  10 = Lueker et al, 2000									T:    2-35  S: 19-43. Total scale. Real seawater.
%  11 = Mojica Prieto and Millero, 2002.					T:    0-45  S:  5-42. Seaw. scale. Real seawater
%  12 = Millero et al, 2002									T: -1.6-35  S: 34-37. Seaw. scale. Field measurements.
%  13 = Millero et al, 2006									T:    0-50  S:  1-50. Seaw. scale. Real seawater.
%  14 = Millero et al, 2010									T:    0-50  S:  1-50. Seaw. scale. Real seawater.
% 
%  (*4) Each element must be an integer that 
%         indicates the KSO4 dissociation constants that are to be used,
%         in combination with the formulation of the borate-to-salinity ratio to be used.
%         Having both these choices in a single argument is somewhat awkward, 
%         but it maintains syntax compatibility with the previous version.
%  1 = KSO4 of Dickson   (PREFERRED) 
%  2 = KSO4 of Khoo   
%
%  (*5) Each element must be an integer that 
%         indicates the KHF dissociation constants that are to be used,
%  1 = KF of Dickson & Riley 1979  
%  2 = KF of Perez & Fraga, 1987  (PREFERRED)
%
%  (*6) Each element must be an integer that 
%         indicates the the formulation of the borate-to-salinity ratio to be used.
%  1 = TB of Uppstrom 1979  (PREFERRED) 
%  2 = TB of Lee 2010
%
%
%**************************************************************************%
%
% OUTPUT: * an array containing the following parameter values (one row per sample):
%         *  a cell-array containing crudely formatted headers
%         *  a cell-array containing nicely formatted headers
%
%    POS  PARAMETER        UNIT
%
%	01	-	TAlk                 (umol/kgSW)	
%	02	-	TCO2                 (umol/kgSW)	
%	03	-	pHin                 ()	
%	04	-	pCO2 input           (uatm)	
%	05	-	fCO2 input           (uatm)	
%	06	-	HCO3 input           (umol/kgSW)	
%	07	-	CO3 input            (umol/kgSW)	
%	08	-	CO2 input            (umol/kgSW)	
%	09	-	BAlk input           (umol/kgSW)	
%	10	-	OH input             (umol/kgSW)	
%	11	-	PAlk input           (umol/kgSW)	
%	12	-	SiAlk input          (umol/kgSW)	
%	13	-	NH4Alk input         (umol/kgSW)	
%	14	-	H2SAlk input         (umol/kgSW)	
%	15	-	Hfree input          (umol/kgSW)	
%	16	-	RevelleFactor input  ()	
%	17	-	OmegaCa input        ()	
%	18	-	OmegaAr input        ()	
%	19	-	xCO2 input           (ppm)	
%	20	-	pH output            ()	
%	21	-	pCO2 output          (uatm)	
%	22	-	fCO2 output          (uatm)	
%	23	-	HCO3 output          (umol/kgSW)	
%	24	-	CO3 output           (umol/kgSW)	
%	25	-	CO2 output           (umol/kgSW)	
%	26	-	BAlk output          (umol/kgSW)	
%	27	-	OH output            (umol/kgSW)	
%	28	-	PAlk output          (umol/kgSW)	
%	29	-	SiAlk output         (umol/kgSW)	
%	30	-	NH4Alk output        (umol/kgSW)	
%	31	-	H2SAlk output        (umol/kgSW)	
%	32	-	Hfree output         (umol/kgSW)	
%	33	-	RevelleFactor output ()	
%	34	-	OmegaCa output       ()	
%	35	-	OmegaAr output       ()	
%	36	-	xCO2 output          (ppm)	
%	37	-	pH input (Total)     ()	
%	38	-	pH input (SWS)       ()	
%	39	-	pH input (Free)      ()	
%	40	-	pH input (NBS)       ()	
%	41	-	pH output (Total)    ()	
%	42	-	pH output (SWS)      ()	
%	43	-	pH output (Free)     ()	
%	44	-	pH output (NBS)      ()	
%	45	-	TEMP input           (deg C)	***
%	46	-	TEMPOUT              (deg C)	***
%	47	-	PRES input           (dbar or m)	***
%	48	-	PRESOUT              (dbar or m)	***
%	49	-	PAR1TYPE             (integer)	***
%	50	-	PAR2TYPE             (integer)	***
%	51	-	K1K2CONSTANTS        (integer)	***
%	52	-	KSO4CONSTANT         (integer)	***
%	53	-	KFCONSTANT           (integer)	***
%	54	-	BORON            (integer)   ***	
%	55	-	pHSCALE of input     (integer)	***
%	56	-	SAL                  (psu)	***
%	57	-	PO4                  (umol/kgSW)	***
%	58	-	SI                   (umol/kgSW)	***
%	59	-	NH3                  (umol/kgSW)	***
%	60	-	H2S                  (umol/kgSW)	***
%	61	-	K0  input            ()	
%	62	-	K1  input            ()	
%	63	-	K2  input            ()	
%	64	-	pK1 input            ()	
%	65	-	pK2 input            ()	
%	66	-	KW  input            ()	
%	67	-	KB  input            ()	
%	68	-	KF  input            ()	
%	69	-	KS  input            ()	
%	70	-	KP1 input            ()	
%	71	-	KP2 input            ()	
%	72	-	KP3 input            ()	
%	73	-	KSi input            ()	
%	74	-	KNH3 input           ()	
%	75	-	KH2S input           ()	
%	76	-	K0  output           ()	
%	77	-	K1  output           ()	
%	78	-	K2  output           ()	
%	79	-	pK1 output           ()	
%	80	-	pK2 output           ()	
%	81	-	KW  output           ()	
%	82	-	KB  output           ()	
%	83	-	KF  output           ()	
%	84	-	KS  output           ()	
%	85	-	KP1 output           ()	
%	86	-	KP2 output           ()	
%	87	-	KP3 output           ()	
%	88	-	KSi output           ()	
%	89	-	KNH3 output          ()	
%	90	-	KH2S output          ()	
%	91	-	TB                   (umol/kgSW)	
%	92	-	TF                   (umol/kgSW)	
%	93	-	TS                   (umol/kgSW)	
%	94	-	TP                   (umol/kgSW)	
%	95	-	TSi                  (umol/kgSW)	
%	96	-	TNH3                 (umol/kgSW)	
%	97	-	TH2S                 (umol/kgSW)	
%
%    *** SIMPLY RESTATES THE INPUT BY USER 
%
% In all the above, the terms "input" and "output" may be understood
%    to refer to the 2 scenarios for which CO2SYS performs calculations, 
%    each defined by its own combination of temperature and pressure.
%    For instance, one may use CO2SYS to calculate, from measured DIC and TAlk,
%    the pH that that sample will have in the lab (e.g., T=25 degC, P=0 dbar),
%    and what the in situ pH would have been (e.g., at T=1 degC, P=4500).
%    A = CO2SYS_v1_21(2400,2200,1,2,35,25,1,0,4200,1,1,0,0,1,4,1,1,1)
%    pH_lab = A(3);  % 7.84
%    pH_sea = A(18); % 8.05
% 
%**************************************************************************
%
% This is version 1.21 (uploaded to NCEI in Sept 2019):
%
% **** Changes since 1.2 (uploaded to NCEI in Sept , 2019):
% - Changed code to set pH values that don't converge to -999. All
%    subsequent calculated values also set to -999.
% - corrected 2 errors in v1.2
%
% **** Changes since 1.1 (uploaded to CDIAC in Sept , 2011):
% - Added NH3 and H2S inputs with corresponding Ks
% - Added possibility of [CO3] ions as one of carbonate parameters for input
% - Modified input to function to separate KHSO4 and TB. Also added KHF or Perez & Fraga as choice
% - Modified output to reflect all changes mentioned above
%
% **** Changes since 1.01 (uploaded to CDIAC at June 11th, 2009):
% - Function cleans up its global variables when done (if you loose variables, this may be the cause -- see around line 570)
% - Added the outputting of K values
% - Implementation of constants of Cai and Wang, 1998
% - Implementation of constants of Lueker et al., 2000
% - Implementation of constants of Mojica-Prieto and Millero, 2002
% - Implementation of constants of Millero et al., 2002 (only their eqs. 19, 20, no TCO2 dependency)
% - Implementation of constants of Millero et al., 2006
% - Implementation of constants of Millero et al., 2010
% - Properly listed Sal and Temp limits for the available constants
% - added switch for using the new Lee et al., (2010) formulation of Total Borate (see KSO4CONSTANT above)
% - Minor corrections to the GEOSECS constants (gave NaN for some output in earlier version)
% - Fixed decimal point error on [H+] (did not get converted to umol/kgSW from mol/kgSW).
% - Changed 'Hfreein' to 'Hfreeout' in the 'NICEHEADERS'-output (typo)
%
% **** Changes since 1.00 (uploaded to CDIAC at May 29th, 2009):
% - added a note explaining that all known bugs were removed before release of 1.00
%
%**************************************************************************
%
% CO2SYS originally by Lewis and Wallace 1998
% Converted to MATLAB by Denis Pierrot at
% CIMAS, University of Miami, Miami, Florida
% Vectorization, internal refinements and speed improvements by
% Steven van Heuven, University of Groningen, The Netherlands.
% Questions, bug reports et cetera: svheuven@gmail.com
%
%**************************************************************************



%**************************************************************************
% NOTHING BELOW THIS SHOULD REQUIRE EDITING BY USER!
%**************************************************************************



% Declare global variables
global pHScale WhichKs WhoseKSO4 WhoseKF WhoseTB Pbar
global Sal sqrSal TempK logTempK TempCi TempCo Pdbari Pdbaro;
global FugFac VPFac PengCorrection ntps RGasConstant;
global fH RT;
global K0 K1 K2 KW KB KF KS KP1 KP2 KP3 KSi KNH3 KH2S;
global TB TF TS TP TSi TNH3 TH2S F;

% Input conditioning

% Determine lengths of input vectors
veclengths=[length(PAR1) length(PAR2) length(PAR1TYPE)...
            length(PAR2TYPE) length(SAL) length(TEMPIN)...
            length(TEMPOUT) length(PRESIN) length(PRESOUT)...
            length(SI) length(PO4) length(NH3) ...
            length(H2S) length(pHSCALEIN)...
            length(K1K2CONSTANTS) length(KSO4CONSTANT)...
            length(KFCONSTANT) length(BORON)];

if length(unique(veclengths))>2
	disp(' '); disp('*** INPUT ERROR: Input vectors must all be of same length, or of length 1. ***'); disp(' '); return
end

% Make column vectors of all input vectors
PAR1         =PAR1         (:);
PAR2         =PAR2         (:);
PAR1TYPE     =PAR1TYPE     (:);
PAR2TYPE     =PAR2TYPE     (:);
SAL          =SAL          (:);
TEMPIN       =TEMPIN       (:);
TEMPOUT      =TEMPOUT      (:);
PRESIN       =PRESIN       (:);
PRESOUT      =PRESOUT      (:);
SI           =SI           (:);
PO4          =PO4          (:);
NH3          =NH3          (:);
H2S          =H2S          (:);
pHSCALEIN    =pHSCALEIN    (:);
K1K2CONSTANTS=K1K2CONSTANTS(:);
KSO4CONSTANT =KSO4CONSTANT (:);
KFCONSTANT   =KFCONSTANT   (:);
BORON        =BORON        (:);

% Find the longest column vector:
ntps = max(veclengths);

% Populate column vectors
PAR1(1:ntps,1)          = PAR1(:)          ;
PAR2(1:ntps,1)          = PAR2(:)          ;
PAR1TYPE(1:ntps,1)      = PAR1TYPE(:)      ;
PAR2TYPE(1:ntps,1)      = PAR2TYPE(:)      ;
SAL(1:ntps,1)           = SAL(:)           ;
TEMPIN(1:ntps,1)        = TEMPIN(:)        ;
TEMPOUT(1:ntps,1)       = TEMPOUT(:)       ;
PRESIN(1:ntps,1)        = PRESIN(:)        ;
PRESOUT(1:ntps,1)       = PRESOUT(:)       ;
SI(1:ntps,1)            = SI(:)            ;
PO4(1:ntps,1)           = PO4(:)           ;
NH3(1:ntps,1)           = NH3(:)           ;
H2S(1:ntps,1)           = H2S(:)           ;
pHSCALEIN(1:ntps,1)     = pHSCALEIN(:)     ;
K1K2CONSTANTS(1:ntps,1) = K1K2CONSTANTS(:) ;
KSO4CONSTANT(1:ntps,1)  = KSO4CONSTANT(:)  ;
KFCONSTANT(1:ntps,1)    = KFCONSTANT(:)    ;
BORON(1:ntps,1)         = BORON(:)         ;

% Assign input to the 'historical' variable names.
pHScale      = pHSCALEIN;
WhichKs      = K1K2CONSTANTS;
WhoseKSO4    = KSO4CONSTANT;
WhoseKF      = KFCONSTANT;
WhoseTB      = BORON;
p1           = PAR1TYPE;
p2           = PAR2TYPE;
TempCi       = TEMPIN;
TempCo       = TEMPOUT;
Pdbari       = PRESIN;
Pdbaro       = PRESOUT;
Sal          = SAL;
sqrSal       = sqrt(SAL);
TP           = PO4;
TSi          = SI;
TNH3          = NH3;
TH2S          = H2S;
RGasConstant = 83.1451;  % ml bar-1 K-1 mol-1, DOEv2
%RGasConstant = 83.14472; % ml bar-1 K-1 mol-1, DOEv3

% Generate empty vectors for...
TA  = nan(ntps,1); % Talk
TC  = nan(ntps,1); % DIC
PH  = nan(ntps,1); % pH
PC  = nan(ntps,1); % pCO2
FC  = nan(ntps,1); % fCO2
CARB  = nan(ntps,1); % CO3 ions

% Assign values to empty vectors.
F=(p1==1); TA(F)=PAR1(F)/1e6; % Convert from micromol/kg to mol/kg
F=(p1==2); TC(F)=PAR1(F)/1e6; % Convert from micromol/kg to mol/kg
F=(p1==3); PH(F)=PAR1(F);
F=(p1==4); PC(F)=PAR1(F)/1e6; % Convert from microatm. to atm.
F=(p1==5); FC(F)=PAR1(F)/1e6; % Convert from microatm. to atm.
F=(p1==6); CARB(F)=PAR1(F)/1e6; % Convert from micromol/kg to mol/kg
F=(p2==1); TA(F)=PAR2(F)/1e6; % Convert from micromol/kg to mol/kg
F=(p2==2); TC(F)=PAR2(F)/1e6; % Convert from micromol/kg to mol/kg
F=(p2==3); PH(F)=PAR2(F);
F=(p2==4); PC(F)=PAR2(F)/1e6; % Convert from microatm. to atm.
F=(p2==5); FC(F)=PAR2(F)/1e6; % Convert from microatm. to atm.
F=(p2==6); CARB(F)=PAR2(F)/1e6; % Convert from micromol/kg to mol/kg

% Generate the columns holding Si, Phos, Amm, H2S and Sal.
% Pure Water case:
F=(WhichKs==8);
Sal(F) = 0;
% GEOSECS and Pure Water:
F=(WhichKs==8 | WhichKs==6);  
TP(F)  = 0;TSi(F) = 0;TNH3(F) = 0;TH2S(F) = 0;

% All other cases
F=~F;                         
TP(F)  = TP(F)./1e6;
TSi(F) = TSi(F)./1e6;
TNH3(F)  = TNH3(F)./1e6;
TH2S(F) = TH2S(F)./1e6;
% The vector 'PengCorrection' is used to modify the value of TA, for those
% cases where WhichKs==7, since PAlk(Peng) = PAlk(Dickson) + TP.
% Thus, PengCorrection is 0 for all cases where WhichKs is not 7
PengCorrection=zeros(ntps,1); F=WhichKs==7; PengCorrection(F)=TP(F);

% Calculate the constants for all samples at input conditions
% The constants calculated for each sample will be on the appropriate pH scale!
Constants(TempCi,Pdbari);

% Make sure fCO2 is available for each sample that has pCO2.
F=find(p1==4 | p2==4); FC(F) = PC(F).*FugFac(F);

% Generate vector for results, and copy the raw input values into them. This
% copies ~60% NaNs, which will be replaced for calculated values later on.
TAc  = TA;
TCc  = TC;
PHic = PH;
PCic = PC;
FCic = FC;
CARBic = CARB;
% Generate vector describing the combination of input parameters
% So, the valid ones are: 12,13,14,15,16,23,24,25,26,34,35,36
Icase = 10*min(p1,p2) + max(p1,p2);

% Calculate missing values for AT,CT,PH,FC:
% pCO2 will be calculated later on, routines work with fCO2.
F=Icase==12; % input TA, TC
if any(F)
%    [PHic(F), FCic(F)] = CalculatepHfCO2fromTATC(TAc(F)-PengCorrection(F), TCc(F));
    PHic(F)   = CalculatepHfromTATC(TAc(F)-PengCorrection(F), TCc(F)); % pH is returned on the scale requested in "pHscale" (see 'constants'...)
    CARBic(PHic==-999 & F)=-999; FCic(PHic==-999 & F)=-999;
    F=(PHic~=-999 & F);
    if any(F)
        FCic(F) = CalculatefCO2fromTCpH(TCc(F), PHic(F));
        CARBic(F)=CalculateCarbfromTCpH(TCc(F), PHic(F));
    end
end
F=Icase==13; % input TA, pH
if any(F)
    TCc(F)  = CalculateTCfromTApH(TAc(F)-PengCorrection(F), PHic(F));
    FCic(F) = CalculatefCO2fromTCpH(TCc(F), PHic(F));
    CARBic(F)=CalculateCarbfromTCpH(TCc(F),PHic(F));
end
F=Icase==14 | Icase==15; % input TA, (pCO2 or fCO2)
if any(F)
    PHic(F) = CalculatepHfromTAfCO2(TAc(F)-PengCorrection(F), FCic(F));
    TCc(PHic==-999 & F)=-999;CARBic(PHic==-999 & F)=-999;
    F=(PHic~=-999 & F);
    if any(F)
        TCc(F) = CalculateTCfromTApH(TAc(F)-PengCorrection(F), PHic(F));
        CARBic(F)=CalculateCarbfromTCpH(TCc(F), PHic(F));
    end
end
F=Icase==16; % input TA, CARB
if any(F)
    PHic(F)  = CalculatepHfromTACarb(TAc(F)-PengCorrection(F), CARBic(F));
    TCc(PHic==-999 & F)=-999;FCic(PHic==-999 & F)=-999;
    F=(PHic~=-999 & F);
    if any(F)
        TCc(F) = CalculateTCfromTApH(TAc(F)-PengCorrection(F), PHic(F));
        FCic(F) = CalculatefCO2fromTCpH(TCc(F), PHic(F));
    end
end
F=Icase==23; % input TC, pHs
if any(F)
    TAc(F)  = CalculateTAfromTCpH  (TCc(F), PHic(F)) + PengCorrection(F);
    FCic(F) = CalculatefCO2fromTCpH(TCc(F), PHic(F));
    CARBic(F)=CalculateCarbfromTCpH(TCc(F),PHic(F));
end
F=Icase==24 | Icase==25;  % input TC, (pCO2 or fCO2)
if any(F)
    PHic(F) = CalculatepHfromTCfCO2(TCc(F), FCic(F));
    TAc(F)  = CalculateTAfromTCpH  (TCc(F), PHic(F)) + PengCorrection(F);
    CARBic(F)=CalculateCarbfromTCpH(TCc(F),PHic(F));
end
F=Icase==26; % input TC, CARB
if any(F)
    PHic(F)  = CalculatepHfromTCCarb(TCc(F), CARBic(F));
    FCic(F) = CalculatefCO2fromTCpH(TCc(F), PHic(F));
    TAc(F)  = CalculateTAfromTCpH  (TCc(F), PHic(F)) + PengCorrection(F);
end
F=Icase==34 | Icase==35; % input pH, (pCO2 or fCO2)
if any(F)
    TCc(F)  = CalculateTCfrompHfCO2(PHic(F), FCic(F));
    TAc(F)  = CalculateTAfromTCpH  (TCc(F),  PHic(F)) + PengCorrection(F);
    CARBic(F)=CalculateCarbfromTCpH(TCc(F),PHic(F));
end
F=Icase==36; % input pH, CARB
if any(F)
    FCic(F) = CalculatefCO2frompHCarb(PHic(F), CARBic(F));
    TCc(F)  = CalculateTCfrompHfCO2(PHic(F), FCic(F));
    TAc(F)  = CalculateTAfromTCpH  (TCc(F),  PHic(F)) + PengCorrection(F);
end
F=Icase==46 | Icase==56; % input (pCO2 or fCO2), CARB
if any(F)
    PHic(F)  = CalculatepHfromfCO2Carb(FCic(F), CARBic(F));
    TCc(F)  = CalculateTCfrompHfCO2(PHic(F), FCic(F));
    TAc(F)  = CalculateTAfromTCpH  (TCc(F),  PHic(F)) + PengCorrection(F);
end
% By now, an fCO2 value is available for each sample.
% Generate the associated pCO2 value:
PCic = FCic./FugFac;
PCic(FCic==-999)=-999;

% CalculateOtherParamsAtInputConditions:
nF=(PHic~=-999); %if PHic = -999, pH calculation  did not converge
[HCO3inp(~nF,1), CO3inp(~nF,1), BAlkinp(~nF,1),...
    OHinp(~nF,1), PAlkinp(~nF,1),...
    SiAlkinp(~nF,1), NH3Alkinp(~nF,1), H2SAlkinp(~nF,1), Hfreeinp(~nF,1),...
    HSO4inp(~nF,1), HFinp(~nF,1),PAlkinp(~nF,1),CO2inp(~nF,1),Revelleinp(~nF,1),OmegaCainp(~nF,1),OmegaArinp(~nF,1)]    = deal(-999);
[HCO3inp(nF), CO3inp(nF), BAlkinp(nF),...
    OHinp(nF), PAlkinp(nF),...
    SiAlkinp(nF), NH3Alkinp(nF), H2SAlkinp(nF), Hfreeinp(nF),...
    HSO4inp(nF), HFinp(nF)]      = CalculateAlkParts(nF,PHic(nF), TCc(nF));
PAlkinp(nF)                = PAlkinp(nF)+PengCorrection(nF);
%By now, CARBic has been calculated if it wasn't an input, so...use of CARBic instead of CO3inp.
CO2inp(nF)                  = TCc(nF) - CARBic(nF) - HCO3inp(nF);
F=nF;           % i.e., do for all samples:
Revelleinp(nF)              = RevelleFactor(TAc(nF)-PengCorrection(nF), TCc(nF));
[OmegaCainp(nF), OmegaArinp(nF)] = CaSolubility(nF,Sal(nF), TempCi(nF), Pdbari(nF), TCc(nF), PHic(nF));
xCO2dryinp(PCic==-999,1)=-999;
xCO2dryinp(PCic~=-999,1)    = PCic(PCic~=-999)./VPFac(PCic~=-999); % ' this assumes pTot = 1 atm

% Just for reference, convert pH at input conditions to the other scales, too. 
[pHicT(~nF,1), pHicS(~nF,1), pHicF(~nF,1), pHicN(~nF,1)]=deal(-999);
[pHicT(nF), pHicS(nF), pHicF(nF), pHicN(nF)]=FindpHOnAllScales(nF,PHic(nF));

% Merge the Ks at input into an array. Ks at output will be glued to this later.
KIVEC=[K0 K1 K2 -log10(K1) -log10(K2) KW KB KF KS KP1 KP2 KP3 KSi KNH3 KH2S];

% Calculate the constants for all samples at output conditions
Constants(TempCo,Pdbaro);

% Calculate, for output conditions, using conservative TA and TC, pH, fCO2 and pCO2, CARB
F=(TAc~=-999 & TCc~=-999); % i.e., do for all samples that are not = -999:
PHoc=nan(ntps,1);PHoc(~F)=-999;[CARBoc FCoc] = deal(PHoc);
PHoc(F)   = CalculatepHfromTATC(TAc(F)-PengCorrection(F), TCc(F)); % pH is returned on the scale requested in "pHscale" (see 'constants'...)
nF=(PHoc~=-999); %if PHoc = -999, pH calculation  might not have converged
CARBoc(~nF)=-999; FCoc(~nF)=-999;
if any(nF)
    FCoc(nF) = CalculatefCO2fromTCpH(TCc(nF), PHoc(nF));
    CARBoc(nF)=CalculateCarbfromTCpH(TCc(nF), PHoc(nF));
end

%CARBoc=CalculateCarbfromTCpH(TCc,PHoc);
PCoc = FCoc./FugFac;
PCoc(FCoc==-999)=-999;

% Calculate Other Stuff At Output Conditions:
[HCO3out(~nF,1), CO3out(~nF,1), BAlkout(~nF,1),...
    OHout(~nF,1), PAlkout(~nF,1),...
    SiAlkout(~nF,1), NH3Alkout(~nF,1), H2SAlkout(~nF,1), Hfreeout(~nF,1),...
    HSO4out(~nF,1), HFout(~nF,1),PAlkout(~nF,1),CO2out(~nF,1),Revelleout(~nF,1),OmegaCaout(~nF,1),OmegaArout(~nF,1)]    = deal(-999);
[HCO3out(nF), CO3out(nF), BAlkout(nF),...
    OHout(nF), PAlkout(nF),...
    SiAlkout(nF), NH3Alkout(nF), H2SAlkout(nF), Hfreeout(nF),...
    HSO4out(nF), HFout(nF)]      = CalculateAlkParts(nF,PHoc(nF), TCc(nF));
PAlkout(nF)                 = PAlkout(nF)+PengCorrection(nF);
CO2out(nF)                  = TCc(nF) - CARBoc(nF) - HCO3out(nF);
Revelleout(nF)              = RevelleFactor(TAc(nF), TCc(nF));
[OmegaCaout(nF) OmegaArout(nF)] = CaSolubility(nF,Sal(nF), TempCo(nF), Pdbaro(nF), TCc(nF), PHoc(nF));
xCO2dryout(PCoc==-999,1)=-999;  
xCO2dryout(PCoc~=-999,1)    = PCoc(PCoc~=-999)./VPFac(PCoc~=-999); % ' this assumes pTot = 1 atm

% Just for reference, convert pH at output conditions to the other scales, too. 
[pHocT(~nF,1), pHocS(~nF,1), pHocF(~nF,1), pHocN(~nF,1)]=deal(-999);
[pHocT(nF) pHocS(nF) pHocF(nF) pHocN(nF)]=FindpHOnAllScales(nF,PHoc(nF));

KOVEC=[K0 K1 K2 -log10(K1) -log10(K2) KW KB KF KS KP1 KP2 KP3 KSi KNH3 KH2S];
TVEC =[TB TF TS];

% Saving data in array, 81 columns, as many rows as samples input
DATA=[TAc*1e6        TCc*1e6         PHic           PCic*1e6        FCic*1e6...
      HCO3inp*1e6    CARBic*1e6      CO2inp*1e6     BAlkinp*1e6     OHinp*1e6...
      PAlkinp*1e6    SiAlkinp*1e6    NH3Alkinp*1e6  H2SAlkinp*1e6   Hfreeinp*1e6   Revelleinp      OmegaCainp... %%% Multiplied Hfreeinp *1e6, svh20100827
      OmegaArinp     xCO2dryinp*1e6  PHoc           PCoc*1e6        FCoc*1e6...
      HCO3out*1e6    CARBoc*1e6      CO2out*1e6     BAlkout*1e6     OHout*1e6...
      PAlkout*1e6    SiAlkout*1e6    NH3Alkout*1e6  H2SAlkout*1e6   Hfreeout*1e6   Revelleout      OmegaCaout... %%% Multiplied Hfreeout *1e6, svh20100827
      OmegaArout     xCO2dryout*1e6  pHicT          pHicS           pHicF...
      pHicN          pHocT           pHocS          pHocF           pHocN...
      TEMPIN         TEMPOUT         PRESIN         PRESOUT         PAR1TYPE...
      PAR2TYPE       K1K2CONSTANTS   KSO4CONSTANT   KFCONSTANT      BORON          pHSCALEIN       SAL...
      PO4            SI              NH3            H2S             KIVEC          KOVEC           TVEC*1e6];
DATA(DATA==-999*1e6)=-999;

HEADERS={'TAlk';'TCO2';'pHin';'pCO2in';'fCO2in';'HCO3in';'CO3in';...
    'CO2in';'BAlkin';'OHin';'PAlkin';'SiAlkin';'NH3Alkin';'H2SAlkin';'Hfreein';'RFin';...
    'OmegaCAin';'OmegaARin';'xCO2in';'pHout';'pCO2out';'fCO2out';...
    'HCO3out';'CO3out';'CO2out';'BAlkout';'OHout';'PAlkout';...
    'SiAlkout';'NH3Alkout';'H2SAlkout';'Hfreeout';'RFout';'OmegaCAout';'OmegaARout';'xCO2out';
    'pHinTOTAL';'pHinSWS';'pHinFREE';'pHinNBS';...
    'pHoutTOTAL';'pHoutSWS';'pHoutFREE';'pHoutNBS';'TEMPIN';'TEMPOUT';...
    'PRESIN';'PRESOUT';'PAR1TYPE';'PAR2TYPE';'K1K2CONSTANTS';'KSO4CONSTANT';'KFCONSTANT';'BORON';...
    'pHSCALEIN';'SAL';'PO4';'SI';'NH3';'H2S';'K0input';'K1input';'K2input';'pK1input';...
    'pK2input';'KWinput';'KBinput';'KFinput';'KSinput';'KP1input';'KP2input';...
    'KP3input';'KSiinput';'KNH3input';'KH2Sinput';'K0output';'K1output';'K2output';'pK1output';...
    'pK2output';'KWoutput';'KBoutput';'KFoutput';'KSoutput';'KP1output';...
    'KP2output';'KP3output';'KSioutput';'KNH3output';'KH2Soutput';'TB';'TF';'TS'};

NICEHEADERS={...
'	01	-	TAlk	(umol/kgSW)';
'	02	-	TCO2	(umol/kgSW)';
'	03	-	pHin ()';
'	04	-	pCO2in	(uatm)';
'	05	-	fCO2in	(uatm)';
'	06	-	HCO3in	(umol/kgSW)';
'	07	-	CO3in	(umol/kgSW)';
'	08	-	CO2in	(umol/kgSW)';
'	09	-	BAlkin	(umol/kgSW)';
'	10	-	OHin	(umol/kgSW)';
'	11	-	PAlkin	(umol/kgSW)';
'	12	-	SiAlkin	(umol/kgSW)';
'	13	-	NH3Alkin	(umol/kgSW)';
'	14	-	H2SAlkin	(umol/kgSW)';
'	15	-	Hfreein	(umol/kgSW)';
'	16	-	RevelleFactorin ()';
'	17	-	OmegaCain ()';
'	18	-	OmegaArin ()';
'	19	-	xCO2in	(ppm)';
'	20	-	pHout ()';
'	21	-	pCO2out	(uatm)';
'	22	-	fCO2out	(uatm)';
'	23	-	HCO3out	(umol/kgSW)';
'	24	-	CO3out	(umol/kgSW)';
'	25	-	CO2out	(umol/kgSW)';
'	26	-	BAlkout	(umol/kgSW)';
'	27	-	OHout	(umol/kgSW)';
'	28	-	PAlkout	(umol/kgSW)';
'	29	-	SiAlkout (umol/kgSW) ';
'	30	-	NH3Alkout (umol/kgSW) ';
'	31	-	H2SAlkout (umol/kgSW) ';
'	32	-	Hfreeout (umol/kgSW)'; %%% Changed 'Hfreein' to 'Hfreeout', svh20100827
'	33	-	RevelleFactorout ()';
'	34	-	OmegaCaout ()';
'	35	-	OmegaArout ()';
'	36	-	xCO2out	(ppm)';
'	37	-	pHin (Total) ()';
'	38	-	pHin (SWS) ()';
'	39	-	pHin (Free) ()';
'	40	-	pHin (NBS ) ()';
'	41	-	pHout(Total) ()';
'	42	-	pHout(SWS) ()';
'	43	-	pHout(Free) ()';
'	44	-	pHout(NBS ) ()';
'	45	-	TEMPIN	(Deg C)';
'	46	-	TEMPOUT	(Deg C)';
'	47	-	PRESIN	(dbar)';
'	48	-	PRESOUT	(dbar)';
'	49	-	PAR1TYPE ()';
'	50	-	PAR2TYPE ()';
'	51	-	K1K2CONSTANTS () ';
'	52	-	KSO4CONSTANT ()';
'	53	-	KFCONSTANT ()'	;
'	54	-	BORON    ()'	;
'	55	-	pHSCALEIN ()';
'	56	-	SAL	(umol/kgSW)';
'	57	-	PO4	(umol/kgSW)';
'	58	-	SI	(umol/kgSW)';
'	59	-	NH3	(umol/kgSW)';
'	60	-	H2S	(umol/kgSW)';
'	61	-	K0input ()';
'	62	-	K1input ()';
'	63	-	K2input ()';
'	64	-	pK1input ()';
'	65	-	pK2input ()';
'	66	-	KWinput ()';
'	67	-	KBinput ()';
'	68	-	KFinput ()';
'	69	-	KSinput ()';
'	70	-	KP1input ()';
'	71	-	KP2input ()';
'	72	-	KP3input ()';
'	73	-	KSiinput ()';
'	74	-	KNH3input ()';
'	75	-	KH2Sinput ()';
'	76	-	K0output ()';
'	77	-	K1output ()';
'	78	-	K2output ()';
'	79	-	pK1output ()';
'	80	-	pK2output ()';
'	81	-	KWoutput ()';
'	82	-	KBoutput ()';
'	83	-	KFoutput ()';
'	84	-	KSoutput ()';
'	85	-	KP1output ()';
'	86	-	KP2output ()';
'	87	-	KP3output ()';
'	88	-	KSioutput ()';
'	89	-	KNH3output ()';
'	90	-	KH2Soutput ()';
'	91	-	TB	(umol/kgSW)';
'	92	-	TF	(umol/kgSW)';
'	93	-	TS	(umol/kgSW)'};

clear global F K2 KP3 Pdbari Sal TS VPFac ntps 
clear global FugFac KB KS Pdbaro T TSi BORON WhichKs pHScale 
clear global K KF KSi KNH3 KH2S PengCorrection TB TempCi WhoseKSO4 WhoseKF WhoseTB sqrSal 
clear global K0 KP1 KW RGasConstant TF TempCo fH 
clear global K1 KP2 Pbar RT TP TempK logTempK
	
end % end main function




%**************************************************************************
% Subroutines:
%**************************************************************************




function Constants(TempC,Pdbar)
global pHScale WhichKs WhoseKSO4 WhoseKF WhoseTB sqrSal Pbar RT;
global K0 fH FugFac VPFac ntps TempK logTempK;
global K1 K2 KW KB KF KS KP1 KP2 KP3 KSi KNH3 KH2S;
global TB TF TS TP TSi RGasConstant Sal;

% SUB Constants, version 04.01, 10-13-97, written by Ernie Lewis.
% Inputs: pHScale%, WhichKs%, WhoseKSO4%, Sali, TempCi, Pdbar
% Outputs: K0, K(), T(), fH, FugFac, VPFac
% This finds the Constants of the CO2 system in seawater or freshwater,
% corrects them for pressure, and reports them on the chosen pH scale.
% The process is as follows: the Constants (except KS, KF which stay on the
% free scale - these are only corrected for pressure) are
%       1) evaluated as they are given in the literature
%       2) converted to the SWS scale in mol/kg-SW or to the NBS scale
%       3) corrected for pressure
%       4) converted to the SWS pH scale in mol/kg-SW
%       5) converted to the chosen pH scale
%
%       PROGRAMMER'S NOTE: all logs are log base e
%       PROGRAMMER'S NOTE: all Constants are converted to the pH scale
%               pHScale% (the chosen one) in units of mol/kg-SW
%               except KS and KF are on the free scale
%               and KW is in units of (mol/kg-SW)^2
TempK    = TempC + 273.15;
RT       = RGasConstant.*TempK;
logTempK = log(TempK);
Pbar     = Pdbar ./ 10;

% Generate empty vectors for holding results
TB = nan(ntps,1);
TF = nan(ntps,1);
TS = nan(ntps,1);

% CalculateTB - Total Borate:
F=(WhichKs==8); % Pure water case.
if any(F)
    TB(F) = 0;
end
F=(WhichKs==6 | WhichKs==7);
if any(F)
    TB(F) = 0.0004106.*Sal(F)./35; % in mol/kg-SW
    % this is .00001173.*Sali
    % this is about 1% lower than Uppstrom's value
    % Culkin, F., in Chemical Oceanography,
    % ed. Riley and Skirrow, 1965:
    % GEOSECS references this, but this value is not explicitly
    % given here
end
F=(WhichKs~=6 & WhichKs~=7 & WhichKs~=8); % All other cases
if any(F)
	FF=F&(WhoseTB==1); % If user opted for Uppstrom's values:
	if any(FF)
	    % Uppstrom, L., Deep-Sea Research 21:161-162, 1974:
	    % this is .000416.*Sali./35. = .0000119.*Sali
		% TB(FF) = (0.000232./10.811).*(Sal(FF)./1.80655); % in mol/kg-SW
	    TB(FF) =  0.0004157.*Sal(FF)./35; % in mol/kg-SW
	end
	FF=F&(WhoseTB==2); % If user opted for the new Lee values:
	if any(FF)
		% Lee, Kim, Byrne, Millero, Feely, Yong-Ming Liu. 2010.	
	 	% Geochimica Et Cosmochimica Acta 74 (6): 1801-1811.
		TB(FF) =  0.0004326.*Sal(FF)./35; % in mol/kg-SW
	end
end

% CalculateTF;
% Riley, J. P., Deep-Sea Research 12:219-220, 1965:
% this is .000068.*Sali./35. = .00000195.*Sali
TF = (0.000067./18.998).*(Sal./1.80655); % in mol/kg-SW

% CalculateTS ;
% Morris, A. W., and Riley, J. P., Deep-Sea Research 13:699-705, 1966:
% this is .02824.*Sali./35. = .0008067.*Sali
TS = (0.14./96.062).*(Sal./1.80655); % in mol/kg-SW

% CalculateK0:
% Weiss, R. F., Marine Chemistry 2:203-215, 1974.
TempK100  = TempK./100;
lnK0 = -60.2409 + 93.4517 ./ TempK100 + 23.3585 .* log(TempK100) + Sal .*...
    (0.023517 - 0.023656 .* TempK100 + 0.0047036 .* TempK100 .^2);
K0   = exp(lnK0);                  % this is in mol/kg-SW/atm

% CalculateIonS:
% This is from the DOE handbook, Chapter 5, p. 13/22, eq. 7.2.4:
IonS         = 19.924 .* Sal ./ (1000 - 1.005   .* Sal);

% CalculateKS:
lnKS = nan(ntps,1); pKS  = nan(ntps,1); KS   = nan(ntps,1);
F=(WhoseKSO4==1);
if any(F)
    % Dickson, A. G., J. Chemical Thermodynamics, 22:113-127, 1990
    % The goodness of fit is .021.
    % It was given in mol/kg-H2O. I convert it to mol/kg-SW.
    % TYPO on p. 121: the constant e9 should be e8.
    % This is from eqs 22 and 23 on p. 123, and Table 4 on p 121:
  lnKS(F) = -4276.1./TempK(F) + 141.328 - 23.093.*logTempK(F) +...             
      (-13856./TempK(F) + 324.57 - 47.986.*logTempK(F)).*sqrt(IonS(F)) +...     
      (35474./TempK(F) - 771.54 + 114.723.*logTempK(F)).*IonS(F) +...           
      (-2698./TempK(F)).*sqrt(IonS(F)).*IonS(F) + (1776./TempK(F)).*IonS(F).^2; 
	KS(F) = exp(lnKS(F))...            % this is on the free pH scale in mol/kg-H2O
        .* (1 - 0.001005 .* Sal(F));   % convert to mol/kg-SW
end
F=(WhoseKSO4==2);
if any(F)
    % Khoo, et al, Analytical Chemistry, 49(1):29-34, 1977
    % KS was found by titrations with a hydrogen electrode
    % of artificial seawater containing sulfate (but without F)
    % at 3 salinities from 20 to 45 and artificial seawater NOT
    % containing sulfate (nor F) at 16 salinities from 15 to 45,
    % both at temperatures from 5 to 40 deg C.
    % KS is on the Free pH scale (inherently so).
    % It was given in mol/kg-H2O. I convert it to mol/kg-SW.
    % He finds log(beta) which = my pKS;
    % his beta is an association constant.
    % The rms error is .0021 in pKS, or about .5% in KS.
    % This is equation 20 on p. 33:
    pKS(F) = 647.59 ./ TempK(F) - 6.3451 + 0.019085.*TempK(F) - 0.5208.*sqrt(IonS(F));
    KS(F) = 10.^(-pKS(F))...          % this is on the free pH scale in mol/kg-H2O
        .* (1 - 0.001005.*Sal(F));    % convert to mol/kg-SW
end

% CalculateKF:
KF = NaN(ntps, 1);
F=(WhoseKF==1);
if any(F)
    % Dickson, A. G. and Riley, J. P., Marine Chemistry 7:89-99, 1979:
    lnKF_F = 1590.2./TempK(F) - 12.641 + 1.525.*IonS(F).^0.5;
    KF(F)   = exp(lnKF_F)...                 % this is on the free pH scale in mol/kg-H2O
        .*(1 - 0.001005.*Sal(F));          % convert to mol/kg-SW
end
F=(WhoseKF==2);
if any(F)
    % Another expression exists for KF: Perez and Fraga 1987.  (to be used for S: 10-40, T: 9-33)
    %  P&F87 might actually be better than the fit of D&R79 above, which is based on only three salinities: [0 26.7 34.6]
    lnKF_F = 874./TempK(F) - 9.68 + 0.111.*Sal(F).^0.5;
    KF(F)   = exp(lnKF_F);          % this is on the free pH scale in mol/kg-SW
end
% Another expression exists for KF: Perez and Fraga 1987.  (to be used for S: 10-40, T: 9-33)
%  P&F87 might actually be better than the fit of D&R79 above, which is based on only three salinities: [0 26.7 34.6]
% lnKF = 874./TempK - 9.68 + 0.111.*Sal.^0.5; 
% KF   = exp(lnKF);                   % this is on the free pH scale in mol/kg-SW

% CalculatepHScaleConversionFactors:
%       These are NOT pressure-corrected.
SWStoTOT  = (1 + TS./KS)./(1 + TS./KS + TF./KF);
FREEtoTOT =  1 + TS./KS;

% CalculatefH
fH = nan(ntps,1);
% Use GEOSECS's value for cases 1,2,3,4,5 (and 6) to convert pH scales.
F=(WhichKs==8);
if any(F)
    fH(F) = 1; % this shouldn't occur in the program for this case
end
F=(WhichKs==7);
if any(F)
    fH(F) = 1.29 - 0.00204.*  TempK(F) + (0.00046 -...
        0.00000148.*TempK(F)).*Sal(F).*Sal(F);
    % Peng et al, Tellus 39B:439-458, 1987:
    % They reference the GEOSECS report, but round the value
    % given there off so that it is about .008 (1%) lower. It
    % doesn't agree with the check value they give on p. 456.
end
F=(WhichKs~=7 & WhichKs~=8);
if any(F)
    fH(F) = 1.2948 - 0.002036.*TempK(F) + (0.0004607 -...
        0.000001475.*TempK(F)).*Sal(F).^2;
    % Takahashi et al, Chapter 3 in GEOSECS Pacific Expedition,
    % v. 3, 1982 (p. 80);
end

% CalculateKB:
KB      = nan(ntps,1); logKB   = nan(ntps,1);
lnKBtop = nan(ntps,1); lnKB    = nan(ntps,1);
F=(WhichKs==8); % Pure water case
if any(F)
    KB(F) = 0;
end
F=(WhichKs==6 | WhichKs==7);
if any(F)
    % This is for GEOSECS and Peng et al.
    % Lyman, John, UCLA Thesis, 1957
    % fit by Li et al, JGR 74:5507-5525, 1969:
    logKB(F) = -9.26 + 0.00886.*Sal(F) + 0.01.*TempC(F);
    KB(F) = 10.^(logKB(F))...  % this is on the NBS scale
        ./fH(F);               % convert to the SWS scale
end
F=(WhichKs~=6 & WhichKs~=7 & WhichKs~=8);
if any(F)
    % Dickson, A. G., Deep-Sea Research 37:755-766, 1990:
    lnKBtop(F) = -8966.9 - 2890.53.*sqrSal(F) - 77.942.*Sal(F) +...
        1.728.*sqrSal(F).*Sal(F) - 0.0996.*Sal(F).^2;
    lnKB(F) = lnKBtop(F)./TempK(F) + 148.0248 + 137.1942.*sqrSal(F) +...
        1.62142.*Sal(F) + (-24.4344 - 25.085.*sqrSal(F) - 0.2474.*...
        Sal(F)).*logTempK(F) + 0.053105.*sqrSal(F).*TempK(F);
    KB(F) = exp(lnKB(F))...    % this is on the total pH scale in mol/kg-SW
        ./SWStoTOT(F);         % convert to SWS pH scale
end

% CalculateKW:
lnKW = nan(ntps,1); KW = nan(ntps,1);
F=(WhichKs==7);
if any(F)
    % Millero, Geochemica et Cosmochemica Acta 43:1651-1661, 1979
    lnKW(F) = 148.9802 - 13847.26./TempK(F) - 23.6521.*logTempK(F) +...
        (-79.2447 + 3298.72./TempK(F) + 12.0408.*logTempK(F)).*...
        sqrSal(F) - 0.019813.*Sal(F);
end
F=(WhichKs==8);
if any(F)
    % Millero, Geochemica et Cosmochemica Acta 43:1651-1661, 1979
    % refit data of Harned and Owen, The Physical Chemistry of
    % Electrolyte Solutions, 1958
    lnKW(F) = 148.9802 - 13847.26./TempK(F) - 23.6521.*logTempK(F);
end
F=(WhichKs~=6 & WhichKs~=7 & WhichKs~=8);
if any(F)
    % Millero, Geochemica et Cosmochemica Acta 59:661-677, 1995.
    % his check value of 1.6 umol/kg-SW should be 6.2
    lnKW(F) = 148.9802 - 13847.26./TempK(F) - 23.6521.*logTempK(F) +...
        (-5.977 + 118.67./TempK(F) + 1.0495.*logTempK(F)).*...
        sqrSal(F) - 0.01615.*Sal(F);
end
KW = exp(lnKW); % this is on the SWS pH scale in (mol/kg-SW)^2
F=(WhichKs==6);
if any(F)
    KW(F) = 0; % GEOSECS doesn't include OH effects
end

% CalculateKP1KP2KP3KSi:
KP1      = nan(ntps,1); KP2      = nan(ntps,1);
KP3      = nan(ntps,1); KSi      = nan(ntps,1);
lnKP1    = nan(ntps,1); lnKP2    = nan(ntps,1);
lnKP3    = nan(ntps,1); lnKSi    = nan(ntps,1);
F=(WhichKs==7);
if any(F)
    KP1(F) = 0.02;
    % Peng et al don't include the contribution from this term,
    % but it is so small it doesn't contribute. It needs to be
    % kept so that the routines work ok.
    % KP2, KP3 from Kester, D. R., and Pytkowicz, R. M.,
    % Limnology and Oceanography 12:243-252, 1967:
    % these are only for sals 33 to 36 and are on the NBS scale
    KP2(F) = exp(-9.039 - 1450./TempK(F))... % this is on the NBS scale
        ./fH(F);                          % convert to SWS scale
    KP3(F) = exp(4.466 - 7276./TempK(F))...  % this is on the NBS scale
        ./fH(F);                          % convert to SWS scale
    % Sillen, Martell, and Bjerrum,  Stability Constants of metal-ion complexes,
    % The Chemical Society (London), Special Publ. 17:751, 1964:
    KSi(F) = 0.0000000004...              % this is on the NBS scale
        ./fH(F);                          % convert to SWS scale
end
F=(WhichKs==6 | WhichKs==8);
if any(F)
    KP1(F) = 0; KP2(F) = 0; KP3(F) = 0; KSi(F) = 0;
    % Neither the GEOSECS choice nor the freshwater choice
    % include contributions from phosphate or silicate.
end
F=(WhichKs~=6 & WhichKs~=7 & WhichKs~=8);
if any(F)
    % Yao and Millero, Aquatic Geochemistry 1:53-88, 1995
    % KP1, KP2, KP3 are on the SWS pH scale in mol/kg-SW.
    % KSi was given on the SWS pH scale in molal units.
    lnKP1(F) = -4576.752./TempK(F) + 115.54 - 18.453.*logTempK(F) + (-106.736./TempK(F) +...
        0.69171).*sqrSal(F) + (-0.65643./TempK(F) - 0.01844).*Sal(F);
    KP1(F) = exp(lnKP1(F));
    lnKP2(F) = -8814.715./TempK(F) + 172.1033 - 27.927.*logTempK(F) + (-160.34./TempK(F) +...
        1.3566).*sqrSal(F) + (0.37335./TempK(F) - 0.05778).*Sal(F);
    KP2(F) = exp(lnKP2(F));
    lnKP3(F) = -3070.75./TempK(F) - 18.126 + (17.27039./TempK(F) + 2.81197).*sqrSal(F) +...
        (-44.99486./TempK(F) - 0.09984).*Sal(F);
    KP3(F) = exp(lnKP3(F));
    lnKSi(F) = -8904.2./TempK(F) + 117.4 - 19.334.*logTempK(F) + (-458.79./TempK(F) +...
        3.5913).*sqrt(IonS(F)) + (188.74./TempK(F) - 1.5998).*IonS(F) +...
        (-12.1652./TempK(F) + 0.07871).*IonS(F).^2;
    KSi(F) = exp(lnKSi(F))...                % this is on the SWS pH scale in mol/kg-H2O
        .*(1 - 0.001005.*Sal(F));        % convert to mol/kg-SW
end

% CalculateKNH3KH2S:
KNH3      = nan(ntps,1); KH2S      = nan(ntps,1);
PKNH3expCW    = nan(ntps,1); lnKH2S    = nan(ntps,1);
F=(WhichKs==6 | WhichKs==7 | WhichKs==8);
if any(F)
    KNH3(F) = 0;KH2S(F)=0;
    % Contributions from Ammonium and H2S not included for these options.
end
F=(WhichKs~=6 & WhichKs~=7 & WhichKs~=8);
if any(F)
    % Constants are on the SWS pH scale in mol/kg-SW.
    % KNH3 (and KH2S) given on the Total scale
    
    %H2S  Millero et. al.( 1988)  Limnol. Oceanogr. 33,269-274.    Yao and Millero, Aquatic Geochemistry 1:53-88, 1995   Total Scale
    %Yao Millero say equations have been refitted to SWS scale but not true as they agree with Millero 1988 which are on Total Scale
    %Also, calculations agree at high H2S with AquaEnv when assuming it is on Total Scale
    lnKH2S(F) = 225.838 - 13275.3 ./ TempK(F)- 34.6435 .* logTempK(F)+ 0.3449 .* sqrSal(F)- 0.0274 .* Sal(F);
    KH2S(F) = exp(lnKH2S(F));
    KH2S(F) = KH2S(F) ./ SWStoTOT(F); % convert to SWS pH scale
    
    %NH4
    %Yao and Millero, Aquatic Geochemistry 1:53-88, 1995   SWS
    %lnKNH3 = -6285.33 / TempK + 0.0001635 .* TempK - 0.25444 + (0.46532 - 123.7184 / TempK) .* sqrSal + (-0.01992 + 3.17556 / TempK) .* Sal
    
    %Clegg Whitfield 1995   Geochimica et Cosmochimica Acta, Vol. 59, No. 12. pp. 2403-2421   eq (18)  Total scale   t=[-2 to 40 oC]  S=[0 to 40 ppt]   pK=+-0.00015
    PKNH3expCW(F)  = 9.244605 - 2729.33 .* (1 / 298.15 - 1 ./ TempK(F));
    PKNH3expCW(F)  = PKNH3expCW(F)  + (0.04203362 - 11.24742 ./ TempK(F)) .* Sal(F).^ 0.25;
    PKNH3expCW(F)  = PKNH3expCW(F)  + (-13.6416 + 1.176949 .* TempK(F).^ 0.5 - 0.02860785 .* TempK(F)+ 545.4834 ./ TempK(F)) .* Sal(F).^ 0.5;
    PKNH3expCW(F)  = PKNH3expCW(F)  + (-0.1462507 + 0.0090226468 .* TempK(F).^ 0.5 - 0.0001471361 .* TempK(F)+ 10.5425 ./ TempK(F)) .* Sal(F).^ 1.5;
    PKNH3expCW(F)  = PKNH3expCW(F)  + (0.004669309 - 0.0001691742 .* TempK(F).^ 0.5 - 0.5677934 ./ TempK(F)) .* Sal(F).^ 2;
    PKNH3expCW(F)  = PKNH3expCW(F)  + (-2.354039e-05 + 0.009698623 ./ TempK(F)) .* Sal(F).^ 2.5;
    KNH3(F) = 10 .^ -PKNH3expCW(F)  ;% this is on the total pH scale in mol/kg-H2O
    KNH3(F) = KNH3(F) .* (1 - 0.001005 .* Sal(F)); % convert to mol/kg-SW
    KNH3(F) = KNH3(F) ./ SWStoTOT(F); % convert to SWS pH scale
        
end

% CalculateK1K2:
lnK1     = nan(ntps,1);pK1      = nan(ntps,1); K1       = nan(ntps,1);
lnK2     = nan(ntps,1);pK2      = nan(ntps,1); K2       = nan(ntps,1);
F=(WhichKs==1);
if any(F)
    % ROY et al, Marine Chemistry, 44:249-267, 1993
    % (see also: Erratum, Marine Chemistry 45:337, 1994
    % and Erratum, Marine Chemistry 52:183, 1996)
    % Typo: in the abstract on p. 249: in the eq. for lnK1* the
    % last term should have S raised to the power 1.5.
    % They claim standard deviations (p. 254) of the fits as
    % .0048 for lnK1 (.5% in K1) and .007 in lnK2 (.7% in K2).
    % They also claim (p. 258) 2s precisions of .004 in pK1 and
    % .006 in pK2. These are consistent, but Andrew Dickson
    % (personal communication) obtained an rms deviation of about
    % .004 in pK1 and .003 in pK2. This would be a 2s precision
    % of about 2% in K1 and 1.5% in K2.
    % T:  0-45  S:  5-45. Total Scale. Artificial sewater.
    % This is eq. 29 on p. 254 and what they use in their abstract:
    lnK1(F) = 2.83655 - 2307.1266./TempK(F) - 1.5529413.*logTempK(F) +...
        (-0.20760841 - 4.0484./TempK(F)).*sqrSal(F) + 0.08468345.*Sal(F) -...
        0.00654208.*sqrSal(F).*Sal(F);
    K1(F) = exp(lnK1(F))...            % this is on the total pH scale in mol/kg-H2O
        .*(1 - 0.001005.*Sal(F))...    % convert to mol/kg-SW
        ./SWStoTOT(F);                 % convert to SWS pH scale
    % This is eq. 30 on p. 254 and what they use in their abstract:
    lnK2(F) = -9.226508 - 3351.6106./TempK(F) - 0.2005743.*logTempK(F) +...
        (-0.106901773 - 23.9722./TempK(F)).*sqrSal(F) + 0.1130822.*Sal(F) -...
        0.00846934.*sqrSal(F).*Sal(F);
    K2(F) = exp(lnK2(F))...            % this is on the total pH scale in mol/kg-H2O
        .*(1 - 0.001005.*Sal(F))...    % convert to mol/kg-SW
        ./SWStoTOT(F);                 % convert to SWS pH scale
end
F=(WhichKs==2);
if any(F)
    % GOYET AND POISSON, Deep-Sea Research, 36(11):1635-1654, 1989
    % The 2s precision in pK1 is .011, or 2.5% in K1.
    % The 2s precision in pK2 is .02, or 4.5% in K2.
    % This is in Table 5 on p. 1652 and what they use in the abstract:
    pK1(F) = 812.27./TempK(F) + 3.356 - 0.00171.*Sal(F).*logTempK(F)...
        + 0.000091.*Sal(F).^2;
    K1(F) = 10.^(-pK1(F)); % this is on the SWS pH scale in mol/kg-SW
    % This is in Table 5 on p. 1652 and what they use in the abstract:
    pK2(F) = 1450.87./TempK(F) + 4.604 - 0.00385.*Sal(F).*logTempK(F)...
        + 0.000182.*Sal(F).^2;
    K2(F) = 10.^(-pK2(F)); % this is on the SWS pH scale in mol/kg-SW
end
F=(WhichKs==3);
if any(F)
    % HANSSON refit BY DICKSON AND MILLERO
    % Dickson and Millero, Deep-Sea Research, 34(10):1733-1743, 1987
    % (see also Corrigenda, Deep-Sea Research, 36:983, 1989)
    % refit data of Hansson, Deep-Sea Research, 20:461-478, 1973
    % and Hansson, Acta Chemica Scandanavia, 27:931-944, 1973.
    % on the SWS pH scale in mol/kg-SW.
    % Hansson gave his results on the Total scale (he called it
    % the seawater scale) and in mol/kg-SW.
    % Typo in DM on p. 1739 in Table 4: the equation for pK2*
    % for Hansson should have a .000132 *S^2
    % instead of a .000116 *S^2.
    % The 2s precision in pK1 is .013, or 3% in K1.
    % The 2s precision in pK2 is .017, or 4.1% in K2.
    % This is from Table 4 on p. 1739.
    pK1(F) = 851.4./TempK(F) + 3.237 - 0.0106.*Sal(F) + 0.000105.*Sal(F).^2;
    K1(F) = 10.^(-pK1(F)); % this is on the SWS pH scale in mol/kg-SW
    % This is from Table 4 on p. 1739.
    pK2(F) = -3885.4./TempK(F) + 125.844 - 18.141.*logTempK(F)...
        - 0.0192.*Sal(F) + 0.000132.*Sal(F).^2;
    K2(F) = 10.^(-pK2(F)); % this is on the SWS pH scale in mol/kg-SW
end
F=(WhichKs==4);
if any(F)
    % MEHRBACH refit BY DICKSON AND MILLERO
    % Dickson and Millero, Deep-Sea Research, 34(10):1733-1743, 1987
    % (see also Corrigenda, Deep-Sea Research, 36:983, 1989)
    % refit data of Mehrbach et al, Limn Oc, 18(6):897-907, 1973
    % on the SWS pH scale in mol/kg-SW.
    % Mehrbach et al gave results on the NBS scale.
    % The 2s precision in pK1 is .011, or 2.6% in K1.
    % The 2s precision in pK2 is .020, or 4.6% in K2.
	% Valid for salinity 20-40.
    % This is in Table 4 on p. 1739.
    pK1(F) = 3670.7./TempK(F) - 62.008 + 9.7944.*logTempK(F)...
             - 0.0118.*Sal(F) + 0.000116.*Sal(F).^2;
    K1(F) = 10.^(-pK1(F)); % this is on the SWS pH scale in mol/kg-SW
    % This is in Table 4 on p. 1739.
    pK2(F) = 1394.7./TempK(F) + 4.777 - 0.0184.*Sal(F) + 0.000118.*Sal(F).^2;
    K2(F) = 10.^(-pK2(F)); % this is on the SWS pH scale in mol/kg-SW
end
F=(WhichKs==5);
if any(F)
    % HANSSON and MEHRBACH refit BY DICKSON AND MILLERO
    % Dickson and Millero, Deep-Sea Research,34(10):1733-1743, 1987
    % (see also Corrigenda, Deep-Sea Research, 36:983, 1989)
    % refit data of Hansson, Deep-Sea Research, 20:461-478, 1973,
    % Hansson, Acta Chemica Scandanavia, 27:931-944, 1973,
    % and Mehrbach et al, Limnol. Oceanogr.,18(6):897-907, 1973
    % on the SWS pH scale in mol/kg-SW.
    % Typo in DM on p. 1740 in Table 5: the second equation
    % should be pK2* =, not pK1* =.
    % The 2s precision in pK1 is .017, or 4% in K1.
    % The 2s precision in pK2 is .026, or 6% in K2.
	% Valid for salinity 20-40.
    % This is in Table 5 on p. 1740.
    pK1(F) = 845./TempK(F) + 3.248 - 0.0098.*Sal(F) + 0.000087.*Sal(F).^2;
    K1(F) = 10.^(-pK1(F)); % this is on the SWS pH scale in mol/kg-SW
    % This is in Table 5 on p. 1740.
    pK2(F) = 1377.3./TempK(F) + 4.824 - 0.0185.*Sal(F) + 0.000122.*Sal(F).^2;
    K2(F) = 10.^(-pK2(F)); % this is on the SWS pH scale in mol/kg-SW
end
F=(WhichKs==6 | WhichKs==7);
if any(F)
    % GEOSECS and Peng et al use K1, K2 from Mehrbach et al,
    % Limnology and Oceanography, 18(6):897-907, 1973.
	% I.e., these are the original Mehrbach dissociation constants.
    % The 2s precision in pK1 is .005, or 1.2% in K1.
    % The 2s precision in pK2 is .008, or 2% in K2.
    pK1(F) = - 13.7201 + 0.031334.*TempK(F) + 3235.76./TempK(F)...
        + 1.3e-5*Sal(F).*TempK(F) - 0.1032.*Sal(F).^0.5;
    K1(F) = 10.^(-pK1(F))...         % this is on the NBS scale
        ./fH(F);                     % convert to SWS scale
    pK2(F) = 5371.9645 + 1.671221.*TempK(F) + 0.22913.*Sal(F) + 18.3802.*log10(Sal(F))...
             - 128375.28./TempK(F) - 2194.3055.*log10(TempK(F)) - 8.0944e-4.*Sal(F).*TempK(F)...
             - 5617.11.*log10(Sal(F))./TempK(F) + 2.136.*Sal(F)./TempK(F); % pK2 is not defined for Sal=0, since log10(0)=-inf
    K2(F) = 10.^(-pK2(F))...         % this is on the NBS scale
        ./fH(F);                     % convert to SWS scale
end
F=(WhichKs==8);
if any(F)	
	% PURE WATER CASE
    % Millero, F. J., Geochemica et Cosmochemica Acta 43:1651-1661, 1979:
    % K1 from refit data from Harned and Davis,
    % J American Chemical Society, 65:2030-2037, 1943.
    % K2 from refit data from Harned and Scholes,
    % J American Chemical Society, 43:1706-1709, 1941.
	% This is only to be used for Sal=0 water (note the absence of S in the below formulations)
    % These are the thermodynamic Constants:
    lnK1(F) = 290.9097 - 14554.21./TempK(F) - 45.0575.*logTempK(F);
    K1(F) = exp(lnK1(F));
    lnK2(F) = 207.6548 - 11843.79./TempK(F) - 33.6485.*logTempK(F);
    K2(F) = exp(lnK2(F));
end
F=(WhichKs==9);
if any(F)
    % From Cai and Wang 1998, for estuarine use.
	% Data used in this work is from:
	% K1: Merhback (1973) for S>15, for S<15: Mook and Keone (1975)
	% K2: Merhback (1973) for S>20, for S<20: Edmond and Gieskes (1970)
	% Sigma of residuals between fits and above data: ±0.015, +0.040 for K1 and K2, respectively.
	% Sal 0-40, Temp 0.2-30
    % Limnol. Oceanogr. 43(4) (1998) 657-668
	% On the NBS scale
	% Their check values for F1 don't work out, not sure if this was correctly published...
	F1 = 200.1./TempK(F) + 0.3220;
	pK1(F) = 3404.71./TempK(F) + 0.032786.*TempK(F) - 14.8435 - 0.071692.*F1.*Sal(F).^0.5 + 0.0021487.*Sal(F);
    K1(F)  = 10.^-pK1(F)...         % this is on the NBS scale
        ./fH(F);                    % convert to SWS scale (uncertain at low Sal due to junction potential);
	F2 = -129.24./TempK(F) + 1.4381;
	pK2(F) = 2902.39./TempK(F) + 0.02379.*TempK(F) - 6.4980 - 0.3191.*F2.*Sal(F).^0.5 + 0.0198.*Sal(F);
    K2(F)  = 10.^-pK2(F)...         % this is on the NBS scale
        ./fH(F);                    % convert to SWS scale (uncertain at low Sal due to junction potential); 
end
F=(WhichKs==10);
if any(F)
    % From Lueker, Dickson, Keeling, 2000
	% This is Mehrbach's data refit after conversion to the total scale, for comparison with their equilibrator work. 
    % Mar. Chem. 70 (2000) 105-119
    % Total scale and kg-sw
    pK1(F) = 3633.86./TempK(F)-61.2172+9.6777.*log(TempK(F))-0.011555.*Sal(F)+0.0001152.*Sal(F).^2;
	K1(F)  = 10.^-pK1(F)...           % this is on the total pH scale in mol/kg-SW
        ./SWStoTOT(F);                % convert to SWS pH scale
    pK2(F) = 471.78./TempK(F)+25.929 -3.16967.*log(TempK(F))-0.01781 .*Sal(F)+0.0001122.*Sal(F).^2;
	K2(F)  = 10.^-pK2(F)...           % this is on the total pH scale in mol/kg-SW
        ./SWStoTOT(F);                % convert to SWS pH scale
end
F=(WhichKs==11);
if any(F)
	% Mojica Prieto and Millero 2002. Geochim. et Cosmochim. Acta. 66(14) 2529-2540.
	% sigma for pK1 is reported to be 0.0056
	% sigma for pK2 is reported to be 0.010
	% This is from the abstract and pages 2536-2537
    pK1 =  -43.6977 - 0.0129037.*Sal(F) + 1.364e-4.*Sal(F).^2 + 2885.378./TempK(F) +  7.045159.*log(TempK(F));
    pK2 = -452.0940 + 13.142162.*Sal(F) - 8.101e-4.*Sal(F).^2 + 21263.61./TempK(F) + 68.483143.*log(TempK(F))...
				+ (-581.4428.*Sal(F) + 0.259601.*Sal(F).^2)./TempK(F) - 1.967035.*Sal(F).*log(TempK(F));
	K1(F) = 10.^-pK1; % this is on the SWS pH scale in mol/kg-SW
	K2(F) = 10.^-pK2; % this is on the SWS pH scale in mol/kg-SW
end
F=(WhichKs==12);
if any(F)
	% Millero et al., 2002. Deep-Sea Res. I (49) 1705-1723.
	% Calculated from overdetermined WOCE-era field measurements 
	% sigma for pK1 is reported to be 0.005
	% sigma for pK2 is reported to be 0.008
	% This is from page 1715
    pK1 =  6.359 - 0.00664.*Sal(F) - 0.01322.*TempC(F) + 4.989e-5.*TempC(F).^2;
    pK2 =  9.867 - 0.01314.*Sal(F) - 0.01904.*TempC(F) + 2.448e-5.*TempC(F).^2;
	K1(F) = 10.^-pK1; % this is on the SWS pH scale in mol/kg-SW
	K2(F) = 10.^-pK2; % this is on the SWS pH scale in mol/kg-SW
end
F=(WhichKs==13);
if any(F)
    % From Millero 2006 work on pK1 and pK2 from titrations
	% Millero, Graham, Huang, Bustos-Serrano, Pierrot. Mar.Chem. 100 (2006) 80-94.
    % S=1 to 50, T=0 to 50. On seawater scale (SWS). From titrations in Gulf Stream seawater.
	pK1_0 = -126.34048 + 6320.813./TempK(F) + 19.568224*log(TempK(F));
	A_1   = 13.4191*Sal(F).^0.5 + 0.0331.*Sal(F) - 5.33e-5.*Sal(F).^2;
	B_1   = -530.123*Sal(F).^0.5 - 6.103.*Sal(F);
	C_1   = -2.06950.*Sal(F).^0.5;
	pK1(F)= A_1 + B_1./TempK(F) + C_1.*log(TempK(F)) + pK1_0; % pK1 sigma = 0.0054
    K1(F) = 10.^-(pK1(F));
	pK2_0= -90.18333 + 5143.692./TempK(F) + 14.613358*log(TempK(F));	
	A_2   = 21.0894*Sal(F).^0.5 + 0.1248.*Sal(F) - 3.687e-4.*Sal(F).^2;
	B_2   = -772.483*Sal(F).^0.5 - 20.051.*Sal(F);
	C_2   = -3.3336.*Sal(F).^0.5;
	pK2(F)= A_2 + B_2./TempK(F) + C_2.*log(TempK(F)) + pK2_0; %pK2 sigma = 0.011
    K2(F) = 10.^-(pK2(F));
end
F=(WhichKs==14);
if any(F)
    % From Millero, 2010, also for estuarine use.
	% Marine and Freshwater Research, v. 61, p. 139–142.
	% Fits through compilation of real seawater titration results:
	% Mehrbach et al. (1973), Mojica-Prieto & Millero (2002), Millero et al. (2006)
	% Constants for K's on the SWS;
	% This is from page 141
	pK10 = -126.34048 + 6320.813./TempK(F) + 19.568224.*log(TempK(F));
	% This is from their table 2, page 140.
	A1 = 13.4038.*Sal(F).^0.5 + 0.03206.*Sal(F) - 5.242e-5.*Sal(F).^2;
	B1 = -530.659.*Sal(F).^0.5 - 5.8210.*Sal(F);
	C1 = -2.0664*Sal(F).^0.5;
	pK1 = pK10 + A1 + B1./TempK(F) + C1.*log(TempK(F));
	K1(F) = 10.^-pK1;
	% This is from page 141
	pK20 =  -90.18333 + 5143.692./TempK(F) + 14.613358.*log(TempK(F));
	% This is from their table 3, page 140.
	A2 = 21.3728.*Sal(F).^0.5 + 0.1218.*Sal(F) - 3.688e-4.*Sal(F).^2;
	B2 = -788.289.*Sal(F).^0.5 - 19.189.*Sal(F);
	C2 = -3.374.*Sal(F).^0.5;
	pK2 = pK20 + A2 + B2./TempK(F) + C2.*log(TempK(F));
	K2(F) = 10.^-pK2;
end

%***************************************************************************
%CorrectKsForPressureNow:
% Currently: For WhichKs% = 1 to 7, all Ks (except KF and KS, which are on
%       the free scale) are on the SWS scale.
%       For WhichKs% = 6, KW set to 0, KP1, KP2, KP3, KSi don't matter.
%       For WhichKs% = 8, K1, K2, and KW are on the "pH" pH scale
%       (the pH scales are the same in this case); the other Ks don't matter.
%
%
% No salinity dependence is given for the pressure coefficients here.
% It is assumed that the salinity is at or very near Sali = 35.
% These are valid for the SWS pH scale, but the difference between this and
% the total only yields a difference of .004 pH units at 1000 bars, much
% less than the uncertainties in the values.
%****************************************************************************
% The sources used are:
% Millero, 1995:
%       Millero, F. J., Thermodynamics of the carbon dioxide system in the
%       oceans, Geochemica et Cosmochemica Acta 59:661-677, 1995.
%       See table 9 and eqs. 90-92, p. 675.
%       TYPO: a factor of 10^3 was left out of the definition of Kappa
%       TYPO: the value of R given is incorrect with the wrong units
%       TYPO: the values of the a's for H2S and H2O are from the 1983
%                values for fresh water
%       TYPO: the value of a1 for B(OH)3 should be +.1622
%        Table 9 on p. 675 has no values for Si.
%       There are a variety of other typos in Table 9 on p. 675.
%       There are other typos in the paper, and most of the check values
%       given don't check.
% Millero, 1992:
%       Millero, Frank J., and Sohn, Mary L., Chemical Oceanography,
%       CRC Press, 1992. See chapter 6.
%       TYPO: this chapter has numerous typos (eqs. 36, 52, 56, 65, 72,
%               79, and 96 have typos).
% Millero, 1983:
%       Millero, Frank J., Influence of pressure on chemical processes in
%       the sea. Chapter 43 in Chemical Oceanography, eds. Riley, J. P. and
%       Chester, R., Academic Press, 1983.
%       TYPO: p. 51, eq. 94: the value -26.69 should be -25.59
%       TYPO: p. 51, eq. 95: the term .1700t should be .0800t
%       these two are necessary to match the values given in Table 43.24
% Millero, 1979:
%       Millero, F. J., The thermodynamics of the carbon dioxide system
%       in seawater, Geochemica et Cosmochemica Acta 43:1651-1661, 1979.
%       See table 5 and eqs. 7, 7a, 7b on pp. 1656-1657.
% Takahashi et al, in GEOSECS Pacific Expedition, v. 3, 1982.
%       TYPO: the pressure dependence of K2 should have a 16.4, not 26.4
%       This matches the GEOSECS results and is in Edmond and Gieskes.
% Culberson, C. H. and Pytkowicz, R. M., Effect of pressure on carbonic acid,
%       boric acid, and the pH of seawater, Limnology and Oceanography
%       13:403-417, 1968.
% Edmond, John M. and Gieskes, J. M. T. M., The calculation of the degree of
%       seawater with respect to calcium carbonate under in situ conditions,
%       Geochemica et Cosmochemica Acta, 34:1261-1291, 1970.
%****************************************************************************
% These references often disagree and give different fits for the same thing.
% They are not always just an update either; that is, Millero, 1995 may agree
%       with Millero, 1979, but differ from Millero, 1983.
% For WhichKs% = 7 (Peng choice) I used the same factors for KW, KP1, KP2,
%       KP3, and KSi as for the other cases. Peng et al didn't consider the
%       case of P different from 0. GEOSECS did consider pressure, but didn't
%       include Phos, Si, or OH, so including the factors here won't matter.
% For WhichKs% = 8 (freshwater) the values are from Millero, 1983 (for K1, K2,
%       and KW). The other aren't used (TB = TS = TF = TP = TSi = 0.), so
%       including the factors won't matter.
%****************************************************************************
%       deltaVs are in cm3/mole
%       Kappas are in cm3/mole/bar
%****************************************************************************

%CorrectK1K2KBForPressure:
deltaV    = nan(ntps,1); Kappa     = nan(ntps,1);
lnK1fac   = nan(ntps,1); lnK2fac   = nan(ntps,1);
lnKBfac   = nan(ntps,1);
F=(WhichKs==8);
if any(F)
    %***PressureEffectsOnK1inFreshWater:
    %               This is from Millero, 1983.
    deltaV(F)  = -30.54 + 0.1849 .*TempC(F) - 0.0023366.*TempC(F).^2;
    Kappa(F)   = (-6.22 + 0.1368 .*TempC(F) - 0.001233 .*TempC(F).^2)./1000;
    lnK1fac(F) = (-deltaV(F) + 0.5.*Kappa(F).*Pbar(F)).*Pbar(F)./RT(F);
    %***PressureEffectsOnK2inFreshWater:
    %               This is from Millero, 1983.
    deltaV(F)  = -29.81 + 0.115.*TempC(F) - 0.001816.*TempC(F).^2;
    Kappa(F)   = (-5.74 + 0.093.*TempC(F) - 0.001896.*TempC(F).^2)./1000;
    lnK2fac(F) = (-deltaV(F) + 0.5.*Kappa(F).*Pbar(F)).*Pbar(F)./RT(F);
    lnKBfac(F) = 0 ;%; this doesn't matter since TB = 0 for this case
end
F=(WhichKs==6 | WhichKs==7);
if any(F)
    %               GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
    %               Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
    %               Culberson and Pytkowicz, L and O 13:403-417, 1968:
    %               but the fits are the same as those in
    %               Edmond and Gieskes, GCA, 34:1261-1291, 1970
    %               who in turn quote Li, personal communication
    lnK1fac(F) = (24.2 - 0.085.*TempC(F)).*Pbar(F)./RT(F);
    lnK2fac(F) = (16.4 - 0.04 .*TempC(F)).*Pbar(F)./RT(F);
    %               Takahashi et al had 26.4, but 16.4 is from Edmond and Gieskes
    %               and matches the GEOSECS results
    lnKBfac(F) = (27.5 - 0.095.*TempC(F)).*Pbar(F)./RT(F);
end
F=(WhichKs~=6 & WhichKs~=7 & WhichKs~=8);
if any(F)
    %***PressureEffectsOnK1:
    %               These are from Millero, 1995.
    %               They are the same as Millero, 1979 and Millero, 1992.
    %               They are from data of Culberson and Pytkowicz, 1968.
    deltaV(F)  = -25.5 + 0.1271.*TempC(F);
    %                 'deltaV = deltaV - .151.*(Sali - 34.8); % Millero, 1979
    Kappa(F)   = (-3.08 + 0.0877.*TempC(F))./1000;
    %                 'Kappa = Kappa  - .578.*(Sali - 34.8)/1000.; % Millero, 1979
 	lnK1fac(F) = (-deltaV(F) + 0.5.*Kappa(F).*Pbar(F)).*Pbar(F)./RT(F);
    %               The fits given in Millero, 1983 are somewhat different.
    
    %***PressureEffectsOnK2:
    %               These are from Millero, 1995.
    %               They are the same as Millero, 1979 and Millero, 1992.
    %               They are from data of Culberson and Pytkowicz, 1968.
    deltaV(F)  = -15.82 - 0.0219.*TempC(F);
    %                  'deltaV = deltaV + .321.*(Sali - 34.8); % Millero, 1979
    Kappa(F)   = (1.13 - 0.1475.*TempC(F))./1000;
    %                 'Kappa = Kappa - .314.*(Sali - 34.8)./1000: % Millero, 1979
	lnK2fac(F) = (-deltaV(F) + 0.5.*Kappa(F).*Pbar(F)).*Pbar(F)./RT(F);
    %               The fit given in Millero, 1983 is different.
    %               Not by a lot for deltaV, but by much for Kappa. %
    
    %***PressureEffectsOnKB:
    %               This is from Millero, 1979.
    %               It is from data of Culberson and Pytkowicz, 1968.
    deltaV(F)  = -29.48 + 0.1622.*TempC(F) - 0.002608.*TempC(F).^2;
    %               Millero, 1983 has:
    %                 'deltaV = -28.56 + .1211.*TempCi - .000321.*TempCi.*TempCi
    %               Millero, 1992 has:
    %                 'deltaV = -29.48 + .1622.*TempCi + .295.*(Sali - 34.8)
    %               Millero, 1995 has:
    %                 'deltaV = -29.48 - .1622.*TempCi - .002608.*TempCi.*TempCi
    %                 'deltaV = deltaV + .295.*(Sali - 34.8); % Millero, 1979
    Kappa(F)   = -2.84./1000; % Millero, 1979
    %               Millero, 1992 and Millero, 1995 also have this.
    %                 'Kappa = Kappa + .354.*(Sali - 34.8)./1000: % Millero,1979
    %               Millero, 1983 has:
    %                 'Kappa = (-3 + .0427.*TempCi)./1000
    lnKBfac(F) = (-deltaV(F) + 0.5.*Kappa(F).*Pbar(F)).*Pbar(F)./RT(F);
end

% CorrectKWForPressure:
lnKWfac   = nan(ntps,1);
F=(WhichKs==8);
if any(F)
    % PressureEffectsOnKWinFreshWater:
    %               This is from Millero, 1983.
    deltaV(F)  =  -25.6 + 0.2324.*TempC(F) - 0.0036246.*TempC(F).^2;
    Kappa(F)   = (-7.33 + 0.1368.*TempC(F) - 0.001233 .*TempC(F).^2)./1000;
 	lnKWfac(F) = (-deltaV(F) + 0.5.*Kappa(F).*Pbar(F)).*Pbar(F)./RT(F);

    %               NOTE the temperature dependence of KappaK1 and KappaKW
    %               for fresh water in Millero, 1983 are the same.
end
F=(WhichKs~=8);
if any(F)
    % GEOSECS doesn't include OH term, so this won't matter.
    % Peng et al didn't include pressure, but here I assume that the KW correction
    %       is the same as for the other seawater cases.
    % PressureEffectsOnKW:
    %               This is from Millero, 1983 and his programs CO2ROY(T).BAS.
    deltaV(F)  = -20.02 + 0.1119.*TempC(F) - 0.001409.*TempC(F).^2;
    %               Millero, 1992 and Millero, 1995 have:
    Kappa(F)   = (-5.13 + 0.0794.*TempC(F))./1000; % Millero, 1983
    %               Millero, 1995 has this too, but Millero, 1992 is different.
	lnKWfac(F) = (-deltaV(F) + 0.5.*Kappa(F).*Pbar(F)).*Pbar(F)./RT(F);
    %               Millero, 1979 does not list values for these.
end

% PressureEffectsOnKF:
%       This is from Millero, 1995, which is the same as Millero, 1983.
%       It is assumed that KF is on the free pH scale.
deltaV = -9.78 - 0.009.*TempC - 0.000942.*TempC.^2;
Kappa = (-3.91 + 0.054.*TempC)./1000;
lnKFfac = (-deltaV + 0.5.*Kappa.*Pbar).*Pbar./RT;
% PressureEffectsOnKS:
%       This is from Millero, 1995, which is the same as Millero, 1983.
%       It is assumed that KS is on the free pH scale.
deltaV = -18.03 + 0.0466.*TempC + 0.000316.*TempC.^2;
Kappa = (-4.53 + 0.09.*TempC)./1000;
lnKSfac = (-deltaV + 0.5.*Kappa.*Pbar).*Pbar./RT;

% CorrectKP1KP2KP3KSiForPressure:
% These corrections don't matter for the GEOSECS choice (WhichKs% = 6) and
%       the freshwater choice (WhichKs% = 8). For the Peng choice I assume
%       that they are the same as for the other choices (WhichKs% = 1 to 5).
% The corrections for KP1, KP2, and KP3 are from Millero, 1995, which are the
%       same as Millero, 1983.
% PressureEffectsOnKP1:
deltaV = -14.51 + 0.1211.*TempC - 0.000321.*TempC.^2;
Kappa  = (-2.67 + 0.0427.*TempC)./1000;
lnKP1fac = (-deltaV + 0.5.*Kappa.*Pbar).*Pbar./RT;
% PressureEffectsOnKP2:
deltaV = -23.12 + 0.1758.*TempC - 0.002647.*TempC.^2;
Kappa  = (-5.15 + 0.09  .*TempC)./1000;
lnKP2fac = (-deltaV + 0.5.*Kappa.*Pbar).*Pbar./RT;
% PressureEffectsOnKP3:
deltaV = -26.57 + 0.202 .*TempC - 0.003042.*TempC.^2;
Kappa  = (-4.08 + 0.0714.*TempC)./1000;
lnKP3fac = (-deltaV + 0.5.*Kappa.*Pbar).*Pbar./RT;
% PressureEffectsOnKSi:
%  The only mention of this is Millero, 1995 where it is stated that the
%    values have been estimated from the values of boric acid. HOWEVER,
%    there is no listing of the values in the table.
%    I used the values for boric acid from above.
deltaV = -29.48 + 0.1622.*TempC - 0.002608.*TempC.^2;
Kappa  = -2.84./1000;
lnKSifac = (-deltaV + 0.5.*Kappa.*Pbar).*Pbar./RT;

% CorrectKNH3KH2SForPressure:
% The corrections are from Millero, 1995, which are the
%       same as Millero, 1983.
% PressureEffectsOnKNH3:
deltaV = -26.43 + 0.0889 .* TempC - 0.000905 .* TempC.^2;
Kappa = (-5.03 + 0.0814 .* TempC) / 1000;
lnKNH3fac = (-deltaV + 0.5 .* Kappa .* Pbar) .* Pbar ./ RT;
% PressureEffectsOnKH2S:
%Millero 1995 gives values for deltaV in fresh water instead of SW.
%Millero 1995 gives -b0 as -2.89 instead of 2.89
%Millero 1983 is correct for both
deltaV = -11.07 - 0.009 .* TempC - 0.000942 .* TempC.^2;
Kappa = (-2.89 + 0.054 .* TempC) ./ 1000;
lnKH2Sfac = (-deltaV + 0.5 .* Kappa .* Pbar) .* Pbar ./ RT;

% CorrectKsForPressureHere:
K1fac  = exp(lnK1fac);  K1  = K1 .*K1fac;
K2fac  = exp(lnK2fac);  K2  = K2 .*K2fac;
KWfac  = exp(lnKWfac);  KW  = KW .*KWfac;
KBfac  = exp(lnKBfac);  KB  = KB .*KBfac;
KFfac  = exp(lnKFfac);  KF  = KF .*KFfac;
KSfac  = exp(lnKSfac);  KS  = KS .*KSfac;
KP1fac = exp(lnKP1fac); KP1 = KP1.*KP1fac;
KP2fac = exp(lnKP2fac); KP2 = KP2.*KP2fac;
KP3fac = exp(lnKP3fac); KP3 = KP3.*KP3fac;
KSifac = exp(lnKSifac); KSi = KSi.*KSifac;
KNH3fac = exp(lnKNH3fac); KNH3 = KNH3.*KNH3fac;
KH2Sfac = exp(lnKH2Sfac); KH2S = KH2S.*KH2Sfac;

% CorrectpHScaleConversionsForPressure:
% fH has been assumed to be independent of pressure.
SWStoTOT  = (1 + TS./KS)./(1 + TS./KS + TF./KF);
FREEtoTOT =  1 + TS./KS;

%  The values KS and KF are already pressure-corrected, so the pH scale
%  conversions are now valid at pressure.

% FindpHScaleConversionFactor:
% this is the scale they will be put on
pHfactor  = nan(ntps,1);
F=(pHScale==1); %Total
pHfactor(F) = SWStoTOT(F);
F=(pHScale==2); %SWS, they are all on this now
pHfactor(F) = 1;
F=(pHScale==3); %pHfree
pHfactor(F) = SWStoTOT(F)./FREEtoTOT(F);
F=(pHScale==4); %pHNBS
pHfactor(F) = fH(F);

% ConvertFromSWSpHScaleToChosenScale:
K1  = K1.* pHfactor; K2  = K2.* pHfactor;
KW  = KW.* pHfactor; KB  = KB.* pHfactor;
KP1 = KP1.*pHfactor; KP2 = KP2.*pHfactor;
KP3 = KP3.*pHfactor; KSi = KSi.*pHfactor;
KNH3 = KNH3.*pHfactor; KH2S = KH2S.*pHfactor;


% CalculateFugacityConstants:
% This assumes that the pressure is at one atmosphere, or close to it.
% Otherwise, the Pres term in the exponent affects the results.
%       Weiss, R. F., Marine Chemistry 2:203-215, 1974.
%       Delta and B in cm3/mol
FugFac=ones(ntps,1);
Delta = (57.7 - 0.118.*TempK);
b = -1636.75 + 12.0408.*TempK - 0.0327957.*TempK.^2 + 3.16528.*0.00001.*TempK.^3;
% For a mixture of CO2 and air at 1 atm (at low CO2 concentrations);
P1atm = 1.01325; % in bar
FugFac = exp((b + 2.*Delta).*P1atm./RT);
F=(WhichKs==6 | WhichKs==7); % GEOSECS and Peng assume pCO2 = fCO2, or FugFac = 1
FugFac(F) = 1;
% CalculateVPFac:
% Weiss, R. F., and Price, B. A., Nitrous oxide solubility in water and
%       seawater, Marine Chemistry 8:347-359, 1980.
% They fit the data of Goff and Gratch (1946) with the vapor pressure
%       lowering by sea salt as given by Robinson (1954).
% This fits the more complicated Goff and Gratch, and Robinson equations
%       from 273 to 313 deg K and 0 to 40 Sali with a standard error
%       of .015%, about 5 uatm over this range.
% This may be on IPTS-29 since they didn't mention the temperature scale,
%       and the data of Goff and Gratch came before IPTS-48.
% The references are:
% Goff, J. A. and Gratch, S., Low pressure properties of water from -160 deg
%       to 212 deg F, Transactions of the American Society of Heating and
%       Ventilating Engineers 52:95-122, 1946.
% Robinson, Journal of the Marine Biological Association of the U. K.
%       33:449-455, 1954.
%       This is eq. 10 on p. 350.
%       This is in atmospheres.
VPWP = exp(24.4543 - 67.4509.*(100./TempK) - 4.8489.*log(TempK./100));
VPCorrWP = exp(-0.000544.*Sal);
VPSWWP = VPWP.*VPCorrWP;
VPFac = 1 - VPSWWP; % this assumes 1 atmosphere
end % end nested function


function varargout=CalculatepHfCO2fromTATC(TAx, TCx)
global FugFac F;
% Outputs pH fCO2, in that order
% SUB FindpHfCO2fromTATC, version 01.02, 10-10-97, written by Ernie Lewis.
% Inputs: pHScale%, WhichKs%, WhoseKSO4%, TA, TC, Sal, K(), T(), TempC, Pdbar
% Outputs: pH, fCO2
% This calculates pH and fCO2 from TA and TC at output conditions.
pHx   = CalculatepHfromTATC(TAx, TCx); % pH is returned on the scale requested in "pHscale" (see 'constants'...)
fCO2x(pHx==-999)=-999;
if any(pHx~=-999)
    fCO2x(pHx~=-999) = CalculatefCO2fromTCpH(TCx(pHx~=-999), pHx(pHx~=-999));
end
varargout{1} = pHx;
varargout{2} = fCO2x;
end % end nested function


function varargout=CalculatepHfromTATC(TAi, TCi)
global  K1 K2 KW KB KF KS KP1 KP2 KP3 KSi KNH3 KH2S;
global TB TF TS TP TSi TNH3 TH2S F
%Outputs pH
% SUB CalculatepHfromTATC, version 04.01, 10-13-96, written by Ernie Lewis.
% Inputs: TA, TC, K(), T()
% Output: pH
% This calculates pH from TA and TC using K1 and K2 by Newton's method.
% It tries to solve for the pH at which Residual = 0.
% The starting guess is pH = 8.
% Though it is coded for H on the total pH scale, for the pH values occuring
% in seawater (pH > 6) it will be equally valid on any pH scale (H terms
% negligible) as long as the K Constants are on that scale.
%
% Made this to accept vectors. It will continue iterating until all
% values in the vector are "abs(deltapH) < pHTol". SVH2007
K1F=K1(F);   K2F=K2(F);   KWF =KW(F);
KP1F=KP1(F); KP2F=KP2(F); KP3F=KP3(F);  TPF=TP(F);
TSiF=TSi(F); KSiF=KSi(F); TBF =TB(F);   KBF=KB(F);
TSF =TS(F);  KSF =KS(F);  TFF =TF(F);   KFF=KF(F);
TNH3F=TNH3(F); KNH3F=KNH3(F); TH2SF=TH2S(F);   KH2SF=KH2S(F);
vl          = sum(F);  % VectorLength
pHGuess     = 8;       % this is the first guess
pHTol       = 0.0001;  % tolerance for iterations end
ln10        = log(10); %
pH(1:vl,1) = pHGuess; % creates a vector holding the first guess for all samples
deltapH(1:vl,1)   = pHTol+1;
loopc=0; nF0=F;csts={'K1';'K2';'KW';'KP1';'KP2';'KP3';'TP';'TSi';'KSi';'TB';'KB';'TS';'KS';'TF';'KF';'TNH3';'KNH3';'TH2S';'KH2S'};
for i=1:size(csts,1), eval([csts{i} 'x=' csts{i} 'F;']); end%e.g evaluates  K2x=K2F;
nF=(abs(deltapH) > pHTol);
while any(nF)    
    if sum(nF0)>sum(nF),for i=1:size(csts,1), eval([csts{i} 'x=' csts{i} 'F(nF);']); end, end %e.g evaluates  K2x=K2F(nF);
    nF0=nF;
    pHx=pH(nF);TCix=TCi(nF);TAix=TAi(nF);
    H         = 10.^(-pHx);
    Denom     = (H.*H + K1x.*H + K1x.*K2x);
    CAlk      = TCix.*K1x.*(H + 2.*K2x)./Denom;
    BAlk      = TBx.*KBx./(KBx + H);
    OH        = KWx./H;
    PhosTop   = KP1x.*KP2x.*H + 2.*KP1x.*KP2x.*KP3x - H.*H.*H;
    PhosBot   = H.*H.*H + KP1x.*H.*H + KP1x.*KP2x.*H + KP1x.*KP2x.*KP3x;
    PAlk      = TPx.*PhosTop./PhosBot;
    SiAlk     = TSix.*KSix./(KSix + H);
    NH3Alk     = TNH3x.*KNH3x./(KNH3x + H);
    H2SAlk     = TH2Sx.*KH2Sx./(KH2Sx + H);
    FREEtoTOT = (1 + TSx./KSx); % pH scale conversion factor
    Hfree     = H./FREEtoTOT; % for H on the total scale
    HSO4      = TSx./(1 + KSx./Hfree); % since KS is on the free scale
    HF        = TFx./(1 + KFx./Hfree); % since KF is on the free scale
    Residual  = TAix - CAlk - BAlk - OH - PAlk - SiAlk  - NH3Alk - H2SAlk + Hfree + HSO4 + HF;
    % find Slope dTA/dpH;
    % (this is not exact, but keeps all important terms);
    Slope     = ln10.*(TCix.*K1x.*H.*(H.*H + K1x.*K2x + 4.*H.*K2x)./Denom./Denom + BAlk.*H./(KBx + H) + OH + H);
    deltapHn   = Residual./Slope; %' this is Newton's method
    % ' to keep the jump from being too big:
    deltapHn(abs(deltapHn) > 1) = 0.9;
    pHx = pHx + deltapHn;
    deltapH(nF)=deltapHn;
    pH(nF) = pHx; 
    loopc=loopc+1;
    nF=(abs(deltapH) > pHTol);
 
    if loopc>10000
        Fr=find(F);
        pH(nF)=-999;  disp(['pH value did not converge for data on row(s): ' num2str((Fr(nF))')]);
        deltapH(nF)=pHTol*0.9;
        nF=(abs(deltapH) > pHTol);
    end
end
varargout{1}=pH;
end % end nested function


function varargout=CalculatefCO2fromTCpH(TCx, pHx)
global K0 K1 K2 F
% ' SUB CalculatefCO2fromTCpH, version 02.02, 12-13-96, written by Ernie Lewis.
% ' Inputs: TC, pH, K0, K1, K2
% ' Output: fCO2
% ' This calculates fCO2 from TC and pH, using K0, K1, and K2.
H            = 10.^(-pHx);
fCO2x        = TCx.*H.*H./(H.*H + K1(F).*H + K1(F).*K2(F))./K0(F);
varargout{1} = fCO2x;
end % end nested function

function varargout=CalculateCarbfromTCpH(TCx, pHx)
global  K1 K2 F
% ' SUB CalculateCarbfromTCpH, version 01.0, 06-12-2019, written by Denis Pierrot.
% ' Inputs: TC, pH, K0, K1, K2
% ' Output: Carbonate
% ' This calculates Carbonate from TC and pH, using K0, K1, and K2.

H            = 10.^(-pHx);
Carbx        = TCx.*K1(F).*K2(F)./(H.*H + K1(F).*H + K1(F).*K2(F));
varargout{1} = Carbx;
end % end nested function


function varargout=CalculatefCO2frompHCarb(pHx,Carbx)
global K0 K1 K2 F
% ' SUB CalculatefCO2frompHCarb, version 01.0, 06-12-2019, written by Denis Pierrot
% ' Inputs: Carb, pH, K0, K1, K2
% ' Output: fCO2
% ' This calculates fCO2 from Carb and pH, using K0, K1, and K2.
H = 10 .^ (-pHx);
fCO2x = Carbx .* H.^2 ./ (K0(F) .* K1(F) .* K2(F));
varargout{1} = fCO2x;
end % end nested function


function varargout=CalculateTCfromTApH(TAx, pHx)
global  K1 K2 KW KB KF KS KP1 KP2 KP3 KSi KNH3 KH2S;
global TB TF TS TP TSi TNH3 TH2S F
K1F=K1(F);   K2F=K2(F);   KWF=KW(F);
KP1F=KP1(F); KP2F=KP2(F); KP3F=KP3(F); TPF=TP(F);
TSiF=TSi(F); KSiF=KSi(F); TBF=TB(F);   KBF=KB(F);
TSF=TS(F);   KSF=KS(F);   TFF=TF(F);   KFF=KF(F);
TNH3F=TNH3(F); KNH3F=KNH3(F); TH2SF=TH2S(F);   KH2SF=KH2S(F);
% ' SUB CalculateTCfromTApH, version 02.03, 10-10-97, written by Ernie Lewis.
% ' Inputs: TA, pH, K(), T()
% ' Output: TC
% ' This calculates TC from TA and pH.
% ' Though it is coded for H on the total pH scale, for the pH values occuring
% ' in seawater (pH > 6) it will be equally valid on any pH scale (H terms
% ' negligible) as long as the K Constants are on that scale.
H         = 10.^(-pHx);
BAlk      = TBF.*KBF./(KBF + H);
OH        = KWF./H;
PhosTop   = KP1F.*KP2F.*H + 2.*KP1F.*KP2F.*KP3F - H.*H.*H;
PhosBot   = H.*H.*H + KP1F.*H.*H + KP1F.*KP2F.*H + KP1F.*KP2F.*KP3F;
PAlk      = TPF.*PhosTop./PhosBot;
SiAlk     = TSiF.*KSiF./(KSiF + H);
NH3Alk     = TNH3F.*KNH3F./(KNH3F + H);
H2SAlk     = TH2SF.*KH2SF./(KH2SF + H);
FREEtoTOT = (1 + TSF./KSF); % pH scale conversion factor
Hfree     = H./FREEtoTOT; %' for H on the total scale
HSO4      = TSF./(1 + KSF./Hfree); %' since KS is on the free scale
HF        = TFF./(1 + KFF./Hfree); %' since KF is on the free scale
CAlk      = TAx - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 + HF;
TCxtemp   = CAlk.*(H.*H + K1F.*H + K1F.*K2F)./(K1F.*(H + 2.*K2F));
varargout{1} = TCxtemp;
end % end nested function


function varargout=CalculatepHfromTAfCO2(TAi, fCO2i)
global K0 K1 K2 KW KB KF KS KP1 KP2 KP3 KSi KNH3 KH2S;
global TB TF TS TP TSi TNH3 TH2S F
% ' SUB CalculatepHfromTAfCO2, version 04.01, 10-13-97, written by Ernie Lewis.
% ' Inputs: TA, fCO2, K0, K(), T()
% ' Output: pH
% ' This calculates pH from TA and fCO2 using K1 and K2 by Newton's method.
% ' It tries to solve for the pH at which Residual = 0.
% ' The starting guess is pH = 8.
% ' Though it is coded for H on the total pH scale, for the pH values occuring
% ' in seawater (pH > 6) it will be equally valid on any pH scale (H terms
% ' negligible) as long as the K Constants are on that scale.
K0F=K0(F);   K1F=K1(F);   K2F=K2(F);   KWF=KW(F);
KP1F=KP1(F); KP2F=KP2(F); KP3F=KP3(F); TPF=TP(F);
TSiF=TSi(F); KSiF=KSi(F); TBF=TB(F);   KBF=KB(F);
TSF=TS(F);   KSF=KS(F);   TFF=TF(F);   KFF=KF(F);
TNH3F=TNH3(F); KNH3F=KNH3(F); TH2SF=TH2S(F);   KH2SF=KH2S(F);
vl         = sum(F); % vectorlength
pHGuess    = 8;      % this is the first guess
pHTol      = 0.0001; % tolerance
ln10       = log(10);
pH(1:vl,1) = pHGuess;
deltapH = pHTol+pH;
loopc=0; nF0=F;csts={'K0';'K1';'K2';'KW';'KP1';'KP2';'KP3';'TP';'TSi';'KSi';'TB';'KB';'TS';'KS';'TF';'KF';'TNH3';'KNH3';'TH2S';'KH2S'};
for i=1:size(csts,1), eval([csts{i} 'x=' csts{i} 'F;']); end%e.g evaluates  K2x=K2F;
nF=(abs(deltapH) > pHTol);
while any(nF)    
    if sum(nF0)>sum(nF),for i=1:size(csts,1), eval([csts{i} 'x=' csts{i} 'F(nF);']); end, end %e.g evaluates  K2x=K2F(nF);
    nF0=nF;
    pHx=pH(nF);fCO2ix=fCO2i(nF);TAix=TAi(nF);
    H         = 10.^(-pHx);
    HCO3      = K0x.*K1x.*fCO2ix./H;
    CO3       = K0x.*K1x.*K2x.*fCO2ix./(H.*H);
    CAlk      = HCO3 + 2.*CO3;
    BAlk      = TBx.*KBx./(KBx + H);
    OH        = KWx./H;
    PhosTop   = KP1x.*KP2x.*H + 2.*KP1x.*KP2x.*KP3x - H.*H.*H;
    PhosBot   = H.*H.*H + KP1x.*H.*H + KP1x.*KP2x.*H + KP1x.*KP2x.*KP3x;
    PAlk      = TPx.*PhosTop./PhosBot;
    SiAlk     = TSix.*KSix./(KSix + H);
    NH3Alk     = TNH3x.*KNH3x./(KNH3x + H);
    H2SAlk     = TH2Sx.*KH2Sx./(KH2Sx + H);
    FREEtoTOT = (1 + TSx./KSx);% ' pH scale conversion factor
    Hfree     = H./FREEtoTOT;%' for H on the total scale
    HSO4      = TSx./(1 + KSx./Hfree); %' since KS is on the free scale
    HF        = TFx./(1 + KFx./Hfree);% ' since KF is on the free scale
    Residual  = TAix - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 + HF;
    % '               find Slope dTA/dpH
    % '               (this is not exact, but keeps all important terms):
    Slope     = ln10.*(HCO3 + 4.*CO3 + BAlk.*H./(KBx + H) + OH + H);
    deltapHn   = Residual./Slope; %' this is Newton's method
    % ' to keep the jump from being too big:
    deltapHn(abs(deltapHn) > 1) = 0.9;
    pHx = pHx + deltapHn;
    deltapH(nF)=deltapHn;
    pH(nF) = pHx; 
    loopc=loopc+1;
    nF=(abs(deltapH) > pHTol);
 
    if loopc>10000
        Fr=find(F);
        pH(nF)=-999;  disp(['pH value did not converge for data on row(s): ' num2str((Fr(nF))')]);
        deltapH(nF)=pHTol*0.9;
        nF=(abs(deltapH) > pHTol);
    end
end
varargout{1}=pH;
end % end nested function

function varargout=CalculatepHfromTACarb(TAi, CARBi)
global K0 K1 K2 KW KB KF KS KP1 KP2 KP3 KSi KNH3 KH2S;
global TB TF TS TP TSi TNH3 TH2S F
% ' SUB CalculatepHfromTACarb, version 01.0, 06-12-2019, written by Denis Pierrot.
% ' Inputs: TA, Carbonate, K(), T()
% ' Output: pH
% ' This calculates pH from TA and Carb using K1 and K2 by Newton's method.
% ' It tries to solve for the pH at which Residual = 0.
% ' The starting guess is pH = 8.
% ' Though it is coded for H on the total pH scale, for the pH values occuring
% ' in seawater (pH > 6) it will be equally valid on any pH scale (H terms
% ' negligible) as long as the K constants are on that scale.
K2F=K2(F);   KWF=KW(F);
KP1F=KP1(F); KP2F=KP2(F); KP3F=KP3(F); TPF=TP(F);
TSiF=TSi(F); KSiF=KSi(F); TBF=TB(F);   KBF=KB(F);
TSF=TS(F);   KSF=KS(F);   TFF=TF(F);   KFF=KF(F);
TNH3F=TNH3(F); KNH3F=KNH3(F); TH2SF=TH2S(F);   KH2SF=KH2S(F);
vl         = sum(F); % vectorlength
pHGuess    = 8;      % this is the first guess
pHTol      = 0.0001; % tolerance
ln10       = log(10);
pH(1:vl,1) = pHGuess;
deltapH = pHTol+pH;
loopc=0; nF0=F;csts={'K2';'KW';'KP1';'KP2';'KP3';'TP';'TSi';'KSi';'TB';'KB';'TS';'KS';'TF';'KF';'TNH3';'KNH3';'TH2S';'KH2S'};
for i=1:size(csts,1), eval([csts{i} 'x=' csts{i} 'F;']); end%e.g evaluates  K2x=K2F;
nF=(abs(deltapH) > pHTol);
while any(nF)    
    if sum(nF0)>sum(nF),for i=1:size(csts,1), eval([csts{i} 'x=' csts{i} 'F(nF);']); end, end %e.g evaluates  K2x=K2F(nF);
    nF0=nF;
    pHx=pH(nF);CARBix=CARBi(nF);TAix=TAi(nF);
    H         = 10.^(-pHx);
    CAlk      = CARBix.*(H+2.*K2x)./K2x;
    BAlk      = TBx.*KBx./(KBx + H);
    OH        = KWx./H;
    PhosTop   = KP1x.*KP2x.*H + 2.*KP1x.*KP2x.*KP3x - H.*H.*H;
    PhosBot   = H.*H.*H + KP1x.*H.*H + KP1x.*KP2x.*H + KP1x.*KP2x.*KP3x;
    PAlk      = TPx.*PhosTop./PhosBot;
    SiAlk     = TSix.*KSix./(KSix + H);
    NH3Alk     = TNH3x.*KNH3x./(KNH3x + H);
    H2SAlk     = TH2Sx.*KH2Sx./(KH2Sx + H);
    FREEtoTOT = (1 + TSx./KSx);% ' pH scale conversion factor
    Hfree     = H./FREEtoTOT;%' for H on the total scale
    HSO4      = TSx./(1 + KSx./Hfree); %' since KS is on the free scale
    HF        = TFx./(1 + KFx./Hfree);% ' since KF is on the free scale
    Residual  = TAix - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 + HF;
    % '               find Slope dTA/dpH
    % '               (this is not exact, but keeps all important terms):
    Slope = ln10 .* (-CARBix .* H ./ K2x + BAlk .* H ./ (KBx + H) + OH + H);
    deltapHn   = Residual./Slope; %' this is Newton's method
    % ' to keep the jump from being too big:
    deltapHn(abs(deltapHn) > 1) = 0.9;
    pHx = pHx + deltapHn;
    deltapH(nF)=deltapHn;
    pH(nF) = pHx; 
    loopc=loopc+1;
    nF=(abs(deltapH) > pHTol);
 
    if loopc>10000
        Fr=find(F);
        pH(nF)=-999;  disp(['pH value did not converge for data on row(s): ' num2str((Fr(nF))')]);
        deltapH(nF)=pHTol*0.9;
        nF=(abs(deltapH) > pHTol);
    end
end
varargout{1}=pH;
end % end nested function


function varargout=CalculateTAfromTCpH(TCi, pHi)
global  K1 K2 KW KB KF KS KP1 KP2 KP3 KSi KNH3 KH2S;
global TB TF TS TP TSi TNH3 TH2S F
% ' SUB CalculateTAfromTCpH, version 02.02, 10-10-97, written by Ernie Lewis.
% ' Inputs: TC, pH, K(), T()
% ' Output: TA
% ' This calculates TA from TC and pH.
% ' Though it is coded for H on the total pH scale, for the pH values occuring
% ' in seawater (pH > 6) it will be equally valid on any pH scale (H terms
% ' negligible) as long as the K Constants are on that scale.
K1F=K1(F);   K2F=K2(F);   KWF=KW(F);
KP1F=KP1(F); KP2F=KP2(F); KP3F=KP3(F); TPF=TP(F);
TSiF=TSi(F); KSiF=KSi(F); TBF=TB(F);   KBF=KB(F);
TSF=TS(F);   KSF=KS(F);   TFF=TF(F);   KFF=KF(F);
TNH3F=TNH3(F);   KNH3F=KNH3(F); TH2SF=TH2S(F);   KH2SF=KH2S(F);

H         = 10.^(-pHi);
CAlk      = TCi.*K1F.*(H + 2.*K2F)./(H.*H + K1F.*H + K1F.*K2F);
BAlk      = TBF.*KBF./(KBF + H);
OH        = KWF./H;
PhosTop   = KP1F.*KP2F.*H + 2.*KP1F.*KP2F.*KP3F - H.*H.*H;
PhosBot   = H.*H.*H + KP1F.*H.*H + KP1F.*KP2F.*H + KP1F.*KP2F.*KP3F;
PAlk      = TPF.*PhosTop./PhosBot;
SiAlk     = TSiF.*KSiF./(KSiF + H);
NH3Alk    = TNH3F .* KNH3F ./ (KNH3F + H);
H2SAlk    = TH2SF .* KH2SF ./ (KH2SF + H);
FREEtoTOT = (1 + TSF./KSF); % ' pH scale conversion factor
Hfree     = H./FREEtoTOT; %' for H on the total scale
HSO4      = TSF./(1 + KSF./Hfree);% ' since KS is on the free scale
HF        = TFF./(1 + KFF./Hfree);% ' since KF is on the free scale
TActemp     = CAlk + BAlk + OH + PAlk + SiAlk + NH3Alk + H2SAlk - Hfree - HSO4 - HF;
varargout{1}=TActemp;
end % end nested function


function varargout=CalculatepHfromTCfCO2(TCi, fCO2i)
global K0 K1 K2 F;
% ' SUB CalculatepHfromTCfCO2, version 02.02, 11-12-96, written by Ernie Lewis.
% ' Inputs: TC, fCO2, K0, K1, K2
% ' Output: pH
% ' This calculates pH from TC and fCO2 using K0, K1, and K2 by solving the
% '       quadratic in H: fCO2.*K0 = TC.*H.*H./(K1.*H + H.*H + K1.*K2).
% ' if there is not a real root, then pH is returned as missingn.
RR = K0(F).*fCO2i./TCi;
%       if RR >= 1
%          varargout{1}= missingn;
%          disp('nein!');return;
%       end
% check after sub to see if pH = missingn.
Discr = (K1(F).*RR).*(K1(F).*RR) + 4.*(1 - RR).*(K1(F).*K2(F).*RR);
H     = 0.5.*(K1(F).*RR + sqrt(Discr))./(1 - RR);
%       if (H <= 0)
%           pHctemp = missingn;
%       else
pHctemp = log(H)./log(0.1);
%       end
varargout{1}=pHctemp;
end % end nested function


function varargout=CalculatepHfromTCCarb(TCi, Carbi)
global  K1 K2 F;
% ' SUB CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, written by Denis Pierrot.
% ' Inputs: Carbonate Ions, TC, K1, K2
% ' Output: pH
% ' This calculates pH from Carbonate and TC using K1, and K2 by solving the
% '       quadratic in H: TC * K1 * K2= Carb * (H * H + K1 * H +  K1 * K2).
% ' If there is not a real root, then pH is returned as -999.

RR = (1-TCi./Carbi);

Discr = K1(F).^2 - 4.*K1(F).*K2(F).*RR;
H     =(-K1(F)+ sqrt(Discr))./2;
pHctemp = log(H)./log(0.1);
varargout{1}=pHctemp;
end % end nested function



function varargout=CalculatepHfromfCO2Carb(fCO2i,Carbi)
global K0 K1 K2 F;
% ' SUB CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, written by Denis Pierrot.
% ' Inputs: Carbonate Ions, fCO2, K0, K1, K2
% ' Output: pH
% ' This calculates pH from Carbonate and fCO2 using K0, K1, and K2 by solving the
% '       equation in H: fCO2 * K0 * K1* K2 = Carb * H * H

RR = (K0(F) .* K1(F) .* K2(F) .* fCO2i) ./ Carbi;
H = sqrt(RR);
pHctemp = log(H) ./ log(0.1);
varargout{1}=pHctemp;
end % end nested function


function varargout=CalculateTCfrompHfCO2(pHi, fCO2i)
global K0 K1 K2 F;
% ' SUB CalculateTCfrompHfCO2, version 01.02, 12-13-96, written by Ernie Lewis.
% ' Inputs: pH, fCO2, K0, K1, K2
% ' Output: TC
% ' This calculates TC from pH and fCO2, using K0, K1, and K2.
H       = 10.^(-pHi);
TCctemp = K0(F).*fCO2i.*(H.*H + K1(F).*H + K1(F).*K2(F))./(H.*H);
varargout{1}=TCctemp;
end % end nested function


function varargout=RevelleFactor(TAi, TCi)
% global WhichKs;
% ' SUB RevelleFactor, version 01.03, 01-07-97, written by Ernie Lewis.
% ' Inputs: WhichKs%, TA, TC, K0, K(), T()
% ' Outputs: Revelle
% ' This calculates the Revelle factor (dfCO2/dTC)|TA/(fCO2/TC).
% ' It only makes sense to talk about it at pTot = 1 atm, but it is computed
% '       here at the given K(), which may be at pressure <> 1 atm. Care must
% '       thus be used to see if there is any validity to the number computed.
TC0 = TCi;
dTC = 0.000001;% ' 1 umol/kg-SW
% ' Find fCO2 at TA, TC + dTC
TCi = TC0 + dTC;
pHc= CalculatepHfromTATC(TAi, TCi);
fCO2c= CalculatefCO2fromTCpH(TCi, pHc);
fCO2plus = fCO2c;
% ' Find fCO2 at TA, TC - dTC
TCi = TC0 - dTC;
pHc= CalculatepHfromTATC(TAi, TCi);
fCO2c= CalculatefCO2fromTCpH(TCi, pHc);
fCO2minus = fCO2c;
% CalculateRevelleFactor:
Revelle = (fCO2plus - fCO2minus)./dTC./((fCO2plus + fCO2minus)./TCi);
varargout{1}=Revelle;
end % end nested function


function varargout=CalculateAlkParts(ind,pHx, TCx)
global K0 K1 K2 KW KB KF KS KP1 KP2 KP3 KSi KNH3 KH2S;
global TB TF TS TP TSi TNH3 TH2S ;
% ' SUB CalculateAlkParts, version 01.03, 10-10-97, written by Ernie Lewis.
% ' Inputs: pH, TC, K(), T()
% ' Outputs: HCO3, CO3, BAlk, OH, PAlk, SiAlk, Hfree, HSO4, HF
% ' This calculates the various contributions to the alkalinity.
% ' Though it is coded for H on the total pH scale, for the pH values occuring
% ' in seawater (pH > 6) it will be equally valid on any pH scale (H terms
% ' negligible) as long as the K Constants are on that scale.
csts={'K1';'K2';'KW';'KP1';'KP2';'KP3';'TP';'TSi';'KSi';'TB';'KB';'TS';'KS';'TF';'KF';'TNH3';'KNH3';'TH2S';'KH2S'};
for i=1:size(csts,1), eval([csts{i} 'x=' csts{i} '(ind);']); end %e.g evaluates  K2x=K2(ind);

H         = 10.^(-pHx);
HCO3      = TCx.*K1x.*H  ./(K1x.*H + H.*H + K1x.*K2x);
CO3       = TCx.*K1x.*K2x./(K1x.*H + H.*H + K1x.*K2x);
BAlk      = TBx.*KBx./(KBx + H);
OH        = KWx./H;
PhosTop   = KP1x.*KP2x.*H + 2.*KP1x.*KP2x.*KP3x - H.*H.*H;
PhosBot   = H.*H.*H + KP1x.*H.*H + KP1x.*KP2x.*H + KP1x.*KP2x.*KP3x;
PAlk      = TPx.*PhosTop./PhosBot;
% this is good to better than .0006*TP:
% PAlk = TP*(-H/(KP1+H) + KP2/(KP2+H) + KP3/(KP3+H))
SiAlk     = TSix.*KSix./(KSix + H);
NH3Alk = TNH3x .* KNH3x ./ (KNH3x + H);
H2SAlk = TH2Sx .* KH2Sx ./ (KH2Sx + H);
FREEtoTOT = (1 + TSx./KSx);        %' pH scale conversion factor
Hfree     = H./FREEtoTOT;          %' for H on the total scale
HSO4      = TSx./(1 + KSx./Hfree); %' since KS is on the free scale
HF        = TFx./(1 + KFx./Hfree); %' since KF is on the free scale

varargout{1} = HCO3; varargout{2} = CO3;   varargout{3} = BAlk;  varargout{4} = OH;
varargout{5} = PAlk; varargout{6} = SiAlk; varargout{7} = NH3Alk; varargout{8} = H2SAlk;
varargout{9} = Hfree; varargout{10} = HSO4;varargout{11} = HF;
end % end nested function


function varargout=CaSolubility(ind, Sal, TempC, Pdbar, TC, pH)
global K1 K2 TempK logTempK sqrSal Pbar RT WhichKs ntps F
% '***********************************************************************
% ' SUB CaSolubility, version 01.05, 05-23-97, written by Ernie Lewis.
% ' Inputs: WhichKs%, Sal, TempCi, Pdbari, TCi, pHi, K1, K2
% ' Outputs: OmegaCa, OmegaAr
% ' This calculates omega, the solubility ratio, for calcite and aragonite.
% ' This is defined by: Omega = [CO3--]*[Ca++]./Ksp,
% '       where Ksp is the solubility product (either KCa or KAr).
% '***********************************************************************
% ' These are from:
% ' Mucci, Alphonso, The solubility of calcite and aragonite in seawater
% '       at various salinities, temperatures, and one atmosphere total
% '       pressure, American Journal of Science 283:781-799, 1983.
% ' Ingle, S. E., Solubility of calcite in the ocean,
% '       Marine Chemistry 3:301-319, 1975,
% ' Millero, Frank, The thermodynamics of the carbonate system in seawater,
% '       Geochemica et Cosmochemica Acta 43:1651-1661, 1979.
% ' Ingle et al, The solubility of calcite in seawater at atmospheric pressure
% '       and 35%o salinity, Marine Chemistry 1:295-307, 1973.
% ' Berner, R. A., The solubility of calcite and aragonite in seawater in
% '       atmospheric pressure and 34.5%o salinity, American Journal of
% '       Science 276:713-730, 1976.
% ' Takahashi et al, in GEOSECS Pacific Expedition, v. 3, 1982.
% ' Culberson, C. H. and Pytkowicz, R. M., Effect of pressure on carbonic acid,
% '       boric acid, and the pHi of seawater, Limnology and Oceanography
% '       13:403-417, 1968.
% '***********************************************************************

Ca=nan(sum(ind),1);
Ar=nan(sum(ind),1);
KCa=nan(sum(ind),1);
KAr=nan(sum(ind),1);
TempKx=TempK(ind);logTempKx=logTempK(ind);sqrSalx=sqrSal(ind);Pbarx=Pbar(ind);RTx=RT(ind);

FF=(WhichKs(ind)~=6 & WhichKs(ind)~=7) ;
if any(FF)
% (below here, F isn't used, since almost always all rows match the above criterium,
%  in all other cases the rows will be overwritten later on).
    % CalculateCa:
    % '       Riley, J. P. and Tongudai, M., Chemical Geology 2:263-269, 1967:
    % '       this is .010285.*Sali./35
    Ca(FF) = 0.02128./40.087.*(Sal(FF)./1.80655);% ' in mol/kg-SW
    % CalciteSolubility:
    % '       Mucci, Alphonso, Amer. J. of Science 283:781-799, 1983.
    logKCa = -171.9065 - 0.077993.*TempKx(FF) + 2839.319./TempKx(FF);
    logKCa = logKCa + 71.595.*logTempKx(FF)./log(10);
    logKCa = logKCa + (-0.77712 + 0.0028426.*TempKx(FF) + 178.34./TempKx(FF)).*sqrSalx(FF);
    logKCa = logKCa - 0.07711.*Sal(FF) + 0.0041249.*sqrSalx(FF).*Sal(FF);
    % '       sd fit = .01 (for Sal part, not part independent of Sal)
    KCa(FF) = 10.^(logKCa);% ' this is in (mol/kg-SW)^2
    % AragoniteSolubility:
    % '       Mucci, Alphonso, Amer. J. of Science 283:781-799, 1983.
    logKAr = -171.945 - 0.077993.*TempKx(FF) + 2903.293./TempKx(FF);
    logKAr = logKAr + 71.595.*logTempKx(FF)./log(10);
    logKAr = logKAr + (-0.068393 + 0.0017276.*TempKx(FF) + 88.135./TempKx(FF)).*sqrSalx(FF);
    logKAr = logKAr - 0.10018.*Sal(FF) + 0.0059415.*sqrSalx(FF).*Sal(FF);
    % '       sd fit = .009 (for Sal part, not part independent of Sal)
    KAr(FF)    = 10.^(logKAr);% ' this is in (mol/kg-SW)^2
    % PressureCorrectionForCalcite:
    % '       Ingle, Marine Chemistry 3:301-319, 1975
    % '       same as in Millero, GCA 43:1651-1661, 1979, but Millero, GCA 1995
    % '       has typos (-.5304, -.3692, and 10^3 for Kappa factor)
    deltaVKCa = -48.76 + 0.5304.*TempC(FF);
    KappaKCa  = (-11.76 + 0.3692.*TempC(FF))./1000;
    lnKCafac  = (-deltaVKCa + 0.5.*KappaKCa.*Pbarx(FF)).*Pbarx(FF)./RTx(FF);
    KCa(FF)       = KCa(FF).*exp(lnKCafac);
    % PressureCorrectionForAragonite:
    % '       Millero, Geochemica et Cosmochemica Acta 43:1651-1661, 1979,
    % '       same as Millero, GCA 1995 except for typos (-.5304, -.3692,
    % '       and 10^3 for Kappa factor)
    deltaVKAr = deltaVKCa + 2.8;
    KappaKAr  = KappaKCa;
    lnKArfac  = (-deltaVKAr + 0.5.*KappaKAr.*Pbarx(FF)).*Pbarx(FF)./RTx(FF);
    KAr(FF)       = KAr(FF).*exp(lnKArfac);
end
FF=(WhichKs==6 | WhichKs==7);
if any(FF)
    %
    % *** CalculateCaforGEOSECS:
    % Culkin, F, in Chemical Oceanography, ed. Riley and Skirrow, 1965:
    % (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982)
    Ca(FF) = 0.01026.*Sal(FF)./35;
    % Culkin gives Ca = (.0213./40.078).*(Sal./1.80655) in mol/kg-SW
    % which corresponds to Ca = .01030.*Sal./35.
    %
    % *** CalculateKCaforGEOSECS:
    % Ingle et al, Marine Chemistry 1:295-307, 1973 is referenced in
    % (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982
    % but the fit is actually from Ingle, Marine Chemistry 3:301-319, 1975)
    KCa(FF) = 0.0000001.*(-34.452 - 39.866.*Sal(FF).^(1./3) +...
        110.21.*log(Sal(FF))./log(10) - 0.0000075752.*TempKx(FF).^2);
    % this is in (mol/kg-SW)^2
    %
    % *** CalculateKArforGEOSECS:
    % Berner, R. A., American Journal of Science 276:713-730, 1976:
    % (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982)
    KAr(FF) = 1.45.*KCa(FF);% ' this is in (mol/kg-SW)^2
    % Berner (p. 722) states that he uses 1.48.
    % It appears that 1.45 was used in the GEOSECS calculations
    %
    % *** CalculatePressureEffectsOnKCaKArGEOSECS:
    % Culberson and Pytkowicz, Limnology and Oceanography 13:403-417, 1968
    % (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982
    % but their paper is not even on this topic).
    % The fits appears to be new in the GEOSECS report.
    % I can't find them anywhere else.
    KCa(FF) = KCa(FF).*exp((36   - 0.2 .*TempC(FF)).*Pbarx(FF)./RTx(FF));
    KAr(FF) = KAr(FF).*exp((33.3 - 0.22.*TempC(FF)).*Pbarx(FF)./RTx(FF));
end
% CalculateOmegasHere:
H = 10.^(-pH);
CO3 = TC.*K1(ind).*K2(ind)./(K1(ind).*H + H.*H + K1(ind).*K2(ind));
varargout{1} = CO3.*Ca./KCa; % OmegaCa, dimensionless
varargout{2} = CO3.*Ca./KAr; % OmegaAr, dimensionless
end % end nested function


function varargout=FindpHOnAllScales(ind,pH)
global pHScale K T TS KS TF KF fH ntps;
% ' SUB FindpHOnAllScales, version 01.02, 01-08-97, written by Ernie Lewis.
% ' Inputs: pHScale%, pH, K(), T(), fH
% ' Outputs: pHNBS, pHfree, pHTot, pHSWS
% ' This takes the pH on the given scale and finds the pH on all scales.
%  TS = T(3); TF = T(2);
%  KS = K(6); KF = K(5);% 'these are at the given T, S, P
TSx=TS(ind); KSx=KS(ind); TFx=TF(ind); KFx=KF(ind);fHx=fH(ind);
FREEtoTOT = (1 + TSx./KSx);% ' pH scale conversion factor
SWStoTOT  = (1 + TSx./KSx)./(1 + TSx./KSx + TFx./KFx);% ' pH scale conversion factor
factor=nan(sum(ind),1);
F=pHScale(ind)==1;  %'"pHtot"
factor(F) = 0;
F=pHScale(ind)==2; % '"pHsws"
factor(F) = -log(SWStoTOT(F))./log(0.1);
F=pHScale(ind)==3; % '"pHfree"
factor(F) = -log(FREEtoTOT(F))./log(0.1);
F=pHScale(ind)==4;  %'"pHNBS"
factor(F) = -log(SWStoTOT(F))./log(0.1) + log(fHx(F))./log(0.1);
pHtot  = pH    - factor;    % ' pH comes into this sub on the given scale
pHNBS  = pHtot - log(SWStoTOT) ./log(0.1) + log(fHx)./log(0.1);
pHfree = pHtot - log(FREEtoTOT)./log(0.1);
pHsws  = pHtot - log(SWStoTOT) ./log(0.1);
varargout{1}=pHtot;
varargout{2}=pHsws;
varargout{3}=pHfree;
varargout{4}=pHNBS;
end % end nested function