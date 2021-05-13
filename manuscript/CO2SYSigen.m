function [PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, pHSCALEIN, ...
    K1K2CONSTANTS, KSO4CONSTANTS, KSO4CONSTANT, KFCONSTANT, BORON] = ...
    CO2SYSigen(PARvalues, PARTYPEs, SALvalue, pHSCALEIN_opts, ...
    K1K2CONSTANTS_opts, KSO4CONSTANTS_opts, KFCONSTANT_opts)
%CO2SYSigen Generate all possible combinations of input parameters to test
% CO2SYS functions // Matthew P. Humphreys [2020-03-19]

%% Set example inputs
% PARvalues = fliplr([2300 2000 8.05 400 405]);
% PARTYPEs = 5:-1:1;
% if numel(PARvalues) ~= numel(PARTYPEs)
%     disp('PARvalues and PARTYPEs must be the same size!')
%     return
% end % if
% pHSCALEIN_opts = 1:4;
% K1K2CONSTANTS_opts = 1:14;
% KSO4CONSTANTS_opts = 1:4;
% KFCONSTANT_opts = 1:2;
% SALvalue = 33.1;

%% Get all valid PAR combinations
PAR12ixs_all = combnk(1:numel(PARvalues), 2);
PAR12combos = PARTYPEs(PAR12ixs_all);
validcombos = ~( ...
    (PAR12combos(:, 1) == 4 & PAR12combos(:, 2) == 5) | ...
    (PAR12combos(:, 1) == 5 & PAR12combos(:, 2) == 4) | ...
    (PAR12combos(:, 1) == 4 & PAR12combos(:, 2) == 8) | ...
    (PAR12combos(:, 1) == 8 & PAR12combos(:, 2) == 4) | ...
    (PAR12combos(:, 1) == 5 & PAR12combos(:, 2) == 8) | ...
    (PAR12combos(:, 1) == 8 & PAR12combos(:, 2) == 5));
PAR12ixs = PAR12ixs_all(validcombos, :);
PAR12ixs = [PAR12ixs; fliplr(PAR12ixs)];
ncombos = size(PAR12ixs, 1);

% Count all possible combinations of other inputs
nopts = ...
    numel(pHSCALEIN_opts)* ...
    numel(K1K2CONSTANTS_opts)* ...
    numel(KSO4CONSTANTS_opts)* ...
    numel(KFCONSTANT_opts);

% Get corresponding PAR combinations
PAR12ixs = repmat(PAR12ixs, nopts, 1);
PARsin = PARvalues(PAR12ixs);
PAR1 = PARsin(:, 1);
PAR2 = PARsin(:, 2);
PAR12combos = PARTYPEs(PAR12ixs);
PAR1TYPE = PAR12combos(:, 1);
PAR2TYPE = PAR12combos(:, 2);

% Grid and reshape corresponding other inputs
[pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS, KFCONSTANT] = ...
    ndgrid( ...
        pHSCALEIN_opts, ...
        K1K2CONSTANTS_opts, ...
        KSO4CONSTANTS_opts, ...
        KFCONSTANT_opts);
pHSCALEIN = repmat(pHSCALEIN(:), 1, ncombos)';
pHSCALEIN = pHSCALEIN(:);
K1K2CONSTANTS = repmat(K1K2CONSTANTS(:), 1, ncombos)';
K1K2CONSTANTS = K1K2CONSTANTS(:);
KSO4CONSTANTS = repmat(KSO4CONSTANTS(:), 1, ncombos)';
KSO4CONSTANTS = KSO4CONSTANTS(:);
KFCONSTANT = repmat(KFCONSTANT(:), 1, ncombos)';
KFCONSTANT = KFCONSTANT(:);

% Get new-style KSO4CONSTANT and BORON inputs from KSO4CONSTANTS
both2KSO4 = [1 2 1 2];
KSO4CONSTANT = both2KSO4(KSO4CONSTANTS)';
both2BSal = [1 1 2 2];
BORON = both2BSal(KSO4CONSTANTS)';

% Zero out salinities for freshwater K1K2CONSTANTS option
SAL = SALvalue*ones(size(K1K2CONSTANTS));
SAL(K1K2CONSTANTS == 8) = 0;
