%% Set up input conditions
PARvalues = [2300 2000 8.05 400 405];
PARTYPEs = 1:5;
pHSCALEIN_opts = 1:4;
K1K2CONSTANTS_opts = 1:14;
KSO4CONSTANTS_opts = 1:4;
KFCONSTANT_opts = 1;
SALvalue = 33.1;
[P1, P2, P1type, P2type, sal, pHscales, K1K2, KSO4_only, KSO4, KF, ...
    BSal] = CO2SYSigen(PARvalues, PARTYPEs, SALvalue, pHSCALEIN_opts, ...
    K1K2CONSTANTS_opts, KSO4CONSTANTS_opts, KFCONSTANT_opts);

% Define single value inputs
tempin = 24;
tempout = 12;
presin = 1;
presout = 1647;
si = 10;
phos = 2;
nh3 = 0;
h2s = 0;

% Run CO2SYS
disp('Running CO2SYS v2.0.5:')
tic
[data_v2, headers_v2] = CO2SYSv2_0_5(P1, P2, P1type, P2type, sal, ...
    tempin, tempout, presin, presout, si, phos, pHscales, K1K2, KSO4_only);
toc
disp('Running CO2SYS v1.21:')
tic
[data_v121, headers_v121] = CO2SYSv1_21(P1, P2, P1type, P2type, sal, ...
    tempin, tempout, presin, presout, si, phos, nh3, h2s, pHscales, ...
    K1K2, KSO4, KF, BSal);
toc

% Convert outputs to tables
clear v2 v121
for V = 1:numel(headers_v2)
    v2.(headers_v2{V}) = data_v2(:, V);
end % for V
v2 = struct2table(v2);
for V = 1:numel(headers_v121)
    v121.(headers_v121{V}) = data_v121(:, V);
end % for V
v121 = struct2table(v121);

% Compare!
cvars = v2.Properties.VariableNames;
cvars = cvars(~ismember(cvars, {'KSO4CONSTANTS'}));
clear vdiff
for V = 1:numel(cvars)
    vdiff.(cvars{V}) = v2.(cvars{V}) - v121.(cvars{V});
end % for V
vdiff = struct2table(vdiff);
vmaxdiff_raw = max(vdiff{:, :});
clear vmaxdiff
for V = 1:numel(cvars)
    vmaxdiff.(cvars{V}) = vmaxdiff_raw(V);
end % for V
vmaxdiff = struct2table(vmaxdiff);
