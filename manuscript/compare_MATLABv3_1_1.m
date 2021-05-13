%% Set up input conditions
PARvalues = [2250 2100 8.1 400 405 200 1800 10];
PARTYPEs = 1:8;
pHSCALEIN_opts = 1:4;
K1K2CONSTANTS_opts = 1:17;
KSO4CONSTANTS_opts = 1:6;
KFCONSTANT_opts = 1:2;
SALvalue = 33.1;
[P1, P2, P1type, P2type, sal, pHscales, K1K2, KSO4_only, KSO4, KF, ...
    BSal] = CO2SYSigen_v3(PARvalues, PARTYPEs, SALvalue, pHSCALEIN_opts, ...
    K1K2CONSTANTS_opts, KSO4CONSTANTS_opts, KFCONSTANT_opts);
tempin = 24;
tempout = 12;
presin = 1;
presout = 1647;
si = 10;
phos = 1;
nh3 = 2;
h2s = 3;

% Switch 6 and 7 for CO2SYS_extd
P1type_ext = P1type;
P1type_ext(P1type == 6) = 7;
P1type_ext(P1type == 7) = 6;
P2type_ext = P2type;
P2type_ext(P2type == 6) = 7;
P2type_ext(P2type == 7) = 6;

%% Run CO2SYS
% xrow = 1; + 210; % just do one row, or...
xrow = 1:numel(P1); % ... do all rows (do this for saving output file)
P1 = P1(xrow);
P2 = P2(xrow);
P1type_ext = P1type_ext(xrow);
P2type_ext = P2type_ext(xrow);
sal = sal(xrow);
pHscales = pHscales(xrow);
K1K2 = K1K2(xrow);
KSO4_only = KSO4_only(xrow);
tic
[DATA, HEADERS] = ...
    CO2SYSv3_1_1(P1, P2, P1type_ext, P2type_ext, sal, tempin, tempout, ...
    presin, presout, si, phos, nh3, h2s, pHscales, K1K2, KSO4, KF, BSal);
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
% Rename for consistency
co2s.H2SAlkin = co2s.HSAlkin;
co2s.H2SAlkout = co2s.HSAlkout;
co2s.NH3Alkin = co2s.AmmAlkin;
co2s.NH3Alkout = co2s.AmmAlkout;
co2s.KNH3input = co2s.KNH4input;
co2s.KNH3output = co2s.KNH4output;
co2s.NH3 = co2s.TNH4;
co2s.H2S = co2s.TH2S;
co2s = rmfield(co2s, {'HSAlkin' 'HSAlkout' 'AmmAlkin' 'AmmAlkout' ...
  'KNH4input' 'KNH4output'});
% Easy MATLAB saving...
co2s = struct2table(co2s);
writetable(co2s, 'results/compare_MATLABv3_1_1.csv')
% % ... or, prepare for Octave-compatible saving
% co2fields = fieldnames(co2s);
% for f = 1:(numel(co2fields) - 1)
%   co2fields{f} = [co2fields{f} ','];
% end
% co2fields = [co2fields{:}];
% % Create and save file (Octave version)
% co2file = 'results/compare_MATLABv3_1_1.csv';
% fid = fopen(co2file, 'w');
% fdisp(fid, co2fields);
% fclose(fid);
% co2array = struct2cell(co2s);
% co2array = [co2array{:}];
% csvwrite(co2file, co2array, '-append');
