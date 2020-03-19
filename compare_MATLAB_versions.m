%% Set up input conditions
setPARS = [2300 2000 8.05 400 405];
PAR12combos_raw = combnk(1:numel(setPARS), 2);
pHscales_opts = 1:4;
K1K2_opts = 1:14;
KSO4_only_opts = 1:4;
KF_opts = 1;
nopts = numel(pHscales_opts)*numel(K1K2_opts)*numel(KSO4_only_opts)* ...
    numel(KF_opts);
PAR12combos_raw = PAR12combos_raw( ...
    ~(PAR12combos_raw(:, 1) == 4 & PAR12combos_raw(:, 2) == 5), :);
PAR12combos_raw = [PAR12combos_raw; fliplr(PAR12combos_raw)];
ncombos = size(PAR12combos_raw, 1);
PAR12combos = repmat(PAR12combos_raw, nopts, 1);
PARSin = setPARS(PAR12combos);
[pHscales, K1K2, KSO4_only, KF] = ...
    ndgrid(pHscales_opts, K1K2_opts, KSO4_only_opts, KF_opts);
pHscales = repmat(pHscales(:), 1, ncombos)';
K1K2 = repmat(K1K2(:), 1, ncombos)';
KSO4_only = repmat(KSO4_only(:), 1, ncombos)';
KF = repmat(KF(:), 1, ncombos)';
pHscales = pHscales(:);
K1K2 = K1K2(:);
KSO4_only = KSO4_only(:);
KF = KF(:);
sal = 33.1*ones(size(K1K2));
sal(K1K2 == 8) = 0;

% Define single value inputs
tempin = 24;
tempout = 12;
presin = 1;
presout = 1647;
si = 10;
phos = 2;
nh3 = 0;
h2s = 0;
P1 = PARSin(:, 1);
P2 = PARSin(:, 2);
P1type = PAR12combos(:, 1);
P2type = PAR12combos(:, 2);

% Convert old combined KSO4 & BSal input to separate variables
only2KSO4 = [1 2 1 2];
only2BSal = [1 1 2 2];
KSO4 = only2KSO4(KSO4_only)';
BSal = only2BSal(KSO4_only)';

% Run CO2SYS
disp('Running CO2SYS v2.0.5:')
tic
[data_v2, headers_v2] = CO2SYSv2(P1, P2, P1type, P2type, sal, ...
    tempin, tempout, presin, presout, si, phos, pHscales, K1K2, KSO4_only);
toc
disp('Running CO2SYS v1.21:')
tic
[data_v121, headers_v121] = CO2SYS_v1_21(P1, P2, P1type, P2type, sal, ...
    tempin, tempout, presin, presout, si, phos, nh3, h2s, pHscales, ...
    K1K2, KSO4, KF, BSal);
toc

% Convert output to tables
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
