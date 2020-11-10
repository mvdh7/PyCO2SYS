% Get CO2SYS.m from https://github.com/jamesorr/CO2SYS-MATLAB/blob/master/src/CO2SYS.m
% and rename to CO2SYSv2_0_5 (both the filename and the main function).

%% Set up input conditions
PARvalues = [2250 2100 8.1 400 405];
PARTYPEs = 1:5;
% PARvalues = [2250 400];
% PARTYPEs = [1 4];
pHSCALEIN_opts = 1:4;
K1K2CONSTANTS_opts = 1:15;
KSO4CONSTANTS_opts = 1:4;
KFCONSTANT_opts = 1;
SALvalue = 33.1;
[P1, P2, P1type, P2type, sal, pHscales, K1K2, KSO4_only, KSO4, KF, ...
    BSal] = CO2SYSigen(PARvalues, PARTYPEs, SALvalue, pHSCALEIN_opts, ...
    K1K2CONSTANTS_opts, KSO4CONSTANTS_opts, KFCONSTANT_opts);
tempin = 24;
tempout = 12;
presin = 1;
presout = 1647;
si = 10;
phos = 1;

% Run CO2SYS
% xrow = 1 + 210; % just do one row, or...
xrow = 1:numel(P1); % ... do all rows (do this for saving output file)
P1 = P1(xrow);
P2 = P2(xrow);
P1type = P1type(xrow);
P2type = P2type(xrow);
sal = sal(xrow);
pHscales = pHscales(xrow);
K1K2 = K1K2(xrow);
KSO4_only = KSO4_only(xrow);
tic
[DATA, HEADERS] = ...
    CO2SYSv2_0_5(P1, P2, P1type, P2type, sal, tempin, tempout, ...
    presin, presout, si, phos, pHscales, K1K2, KSO4_only);
toc

% Do it again in a loop row-by-row, because this returns different reults
data_loop = NaN(size(DATA));
for i = 1:size(data_loop, 1)
    disp(i)
    data_loop(i, :) = ...
        CO2SYSv2_0_5(P1(i), P2(i), P1type(i), P2type(i), sal(i), ...
        tempin, tempout, presin, presout, si, phos, ...
        pHscales(i), K1K2(i), KSO4_only(i));
end  % for i

%% Extract and save outputs
clear co2s co2s_loop
for V = 1:numel(HEADERS)
    co2s.(HEADERS{V}) = DATA(:, V);
    co2s_loop.(HEADERS{V}) = data_loop(:, V);
end % for V
co2s.PAR1 = P1;
co2s.PAR2 = P2;
co2s.PAR1TYPE = P1type;
co2s.PAR2TYPE = P2type;
co2s_loop.PAR1 = P1;
co2s_loop.PAR2 = P2;
co2s_loop.PAR1TYPE = P1type;
co2s_loop.PAR2TYPE = P2type;
% Easy MATLAB saving...
co2s = struct2table(co2s);
writetable(co2s, 'results/compare_MATLABv2_0_5.csv')
co2s_loop = struct2table(co2s_loop);
writetable(co2s_loop, 'results/compare_MATLABv2_0_5_loop.csv')
% % ... or, prepare for Octave-compatible saving
% co2fields = fieldnames(co2s);
% for f = 1:(numel(co2fields) - 1)
%   co2fields{f} = [co2fields{f} ','];
% end
% co2fields = [co2fields{:}];
% % Create and save file (Octave version)
% co2file = 'results/compare_MATLABv2_0_5.csv';
% fid = fopen(co2file, 'w');
% fdisp(fid, co2fields);
% fclose(fid);
% co2array = struct2cell(co2s);
% co2array = [co2array{:}];
% csvwrite(co2file, co2array, '-append');
