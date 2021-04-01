% Get CO2SYS.m from https://github.com/jamesorr/CO2SYS-MATLAB/blob/master/src/CO2SYS.m
% and rename to CO2SYSv2_0_5 (both the filename and the main function).

rng(1)
npts = 10000;

P1 = 8.1;
P2 = 2000;
P1type = 3;
P2type = 2;
sal = [rand(npts - 1, 1) * 50; 0];
tempin = [rand(npts - 2, 1) * 50; 0; -1];
tempout = [rand(npts - 2, 1) * 50; -1; 0];
presin = [rand(npts - 1, 1) * 10000; 0];
presout = [0; rand(npts - 1, 1) * 10000];
si = 0;
phos = [0; rand(npts - 1, 1) * 20];
pHscales = randi(4, npts, 1);
K1K2 = randi(15, npts, 1);
KSO4_only = randi(4, npts, 1);

[DATA, HEADERS] = ...
    CO2SYSv2_0_5(P1, P2, P1type, P2type, sal, tempin, tempout, presin, ...
    presout, si, phos, pHscales, K1K2, KSO4_only);

%% Extract and save outputs
clear co2s
for V = 1:numel(HEADERS)
    co2s.(HEADERS{V}) = DATA(:, V);
end % for V
co2s.PAR1 = P1 * ones(npts, 1);
co2s.PAR2 = P2 * ones(npts, 1);
co2s.PAR1TYPE = P1type * ones(npts, 1);
co2s.PAR2TYPE = P2type * ones(npts, 1);
% Easy MATLAB saving...
co2s = struct2table(co2s);
writetable(co2s, 'results/compare_equilibrium_constants_v2_0_5.csv')

%% Get CO2SYS.m from https://github.com/jonathansharp/CO2-System-Extd/blob/master/CO2SYS.m
% and rename to CO2SYSv3_1_1 (both the filename and the main function).

rng(1)
npts = 10000;

P1 = 8.1;
P2 = 2000;
P1type = 3;
P2type = 2;
sal = [rand(npts - 1, 1) * 50; 0];
tempin = [rand(npts - 2, 1) * 50; 0; -1];
tempout = [rand(npts - 2, 1) * 50; -1; 0];
presin = [rand(npts - 1, 1) * 10000; 0];
presout = [0; rand(npts - 1, 1) * 10000];
si = 0;
phos = [0; rand(npts - 1, 1) * 20];
pHscales = randi(4, npts, 1);
K1K2 = randi(17, npts, 1);
KSO4 = randi(3, npts, 1);
KF = randi(2, npts, 1);
TB = randi(2, npts, 1);
NH3 = 0;
H2S = 0;

[DATA, HEADERS] = ...
    CO2SYSv3_1_1(P1, P2, P1type, P2type, sal, tempin, tempout, presin, ...
    presout, si, phos, NH3, H2S, pHscales, K1K2, KSO4, KF, TB);

%% Extract and save outputs
clear co2s
for V = 1:numel(HEADERS)
    co2s.(HEADERS{V}) = DATA(:, V);
end % for V
co2s.PAR1 = P1 * ones(npts, 1);
co2s.PAR2 = P2 * ones(npts, 1);
co2s.PAR1TYPE = P1type * ones(npts, 1);
co2s.PAR2TYPE = P2type * ones(npts, 1);
% Easy MATLAB saving...
co2s = struct2table(co2s);
writetable(co2s, 'results/compare_equilibrium_constants_v3_1_1.csv')
