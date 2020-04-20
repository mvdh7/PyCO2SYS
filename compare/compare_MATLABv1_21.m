%% Set up input conditions
PARvalues = [2250 2100 8.1 400 405 200];
PARTYPEs = 1:6;
pHSCALEIN_opts = 1:4;
K1K2CONSTANTS_opts = 1:14;
KSO4CONSTANTS_opts = 1:4;
KFCONSTANT_opts = 1:2;
SALvalue = 33.1;
[P1, P2, P1type, P2type, sal, pHscales, K1K2, KSO4_only, KSO4, KF, ...
    BSal] = CO2SYSigen(PARvalues, PARTYPEs, SALvalue, pHSCALEIN_opts, ...
    K1K2CONSTANTS_opts, KSO4CONSTANTS_opts, KFCONSTANT_opts);
tempin = 24;
tempout = 12;
presin = 0;
presout = 1647;
si = 50;
phos = 10;
nh3 = 15;
h2s = 5;

% Run CO2SYS
xrow = 1 + 210; % just do one row, or...
% xrow = 1:numel(P1); % ... do all rows (do this for saving output file)
P1 = P1(xrow);
P2 = P2(xrow);
P1type = P1type(xrow);
P2type = P2type(xrow);
sal = sal(xrow);
pHscales = pHscales(xrow);
K1K2 = K1K2(xrow);
KSO4 = KSO4(xrow);
KF = KF(xrow);
BSal = BSal(xrow);
tic
[DATA, HEADERS] = ...
    CO2SYSv1_21(P1, P2, P1type, P2type, sal, tempin, tempout, presin, ...
    presout, si, phos, nh3, h2s, pHscales, K1K2, KSO4, KF, BSal);
toc

%% Extract and save outputs
clear co2s
for V = 1:numel(HEADERS)
    co2s.(HEADERS{V}) = DATA(:, V);
end % for V
co2s.PAR1 = P1;
co2s.PAR2 = P2;
co2s.PAR1TYPE = P1type;
co2s.PAR2TYPE = P2type;
co2s.KSO4CONSTANTS = KSO4_only;
##save('data/MATLAB_CO2SYSv1_21.mat', 'co2s')
