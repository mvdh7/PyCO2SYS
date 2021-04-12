% Compares CO2SYS v3 with CO2SYS v2.0.5.
%
% CO2SYS v2.0.5 comes from https://github.com/jamesorr/CO2SYS-MATLAB,
%  but you must first rename the function to CO2SYSv2_0_5 (both inside the
%  file and in the file name).
%
% CO2SYS v3 comes from https://github.com/mvdh7/CO2-System-Extd, which
%  is from https://github.com/jonathansharp/CO2-System-Extd but with some
%  corrections applied.
%
% CO2SYSigen comes from
%  https://github.com/mvdh7/PyCO2SYS/blob/master/validate/CO2SYSigen.m,
%  with some modifications by J. Sharp to generate parameter uncertainties
%
% compare_versions.m from Matthew Humphreys, 4 May 2020
%
% Corrctions for KSO4, KF, and BSal inputs and column
% headers from J. Sharp, 10 June 2020
% 
% Further modified by MP Humphreys to only use CO2SYSv3, 12 April 2021
%
% Differences are expected in outputs between the two versions due to minor
% differences in the way pH values are determined by iterations and the
% difference in the ideal gas constant between v2.0.5 and v3.0

%% Add tools to path (if you need to!)
% addpath('/home/matthew/github/PyCO2SYS/validate')

%% Set up input conditions
PARvalues = [2250 2100 8.1 400 405];
PARTYPEs = 1:5;
pHSCALEIN_opts = 1:4;
K1K2CONSTANTS_opts = 1:15;
KSO4CONSTANTS_opts = 1:4;
KFCONSTANT_opts = 1;
SALvalue = 33.1;
[P1, P2, P1type, P2type, sal, pHscales, K1K2, KSO4_only, KSO4, KF, ...
    BSal, U1, U2] = CO2SYSigen_uncert(PARvalues, PARTYPEs, SALvalue, pHSCALEIN_opts, ...
    K1K2CONSTANTS_opts, KSO4CONSTANTS_opts, KFCONSTANT_opts);
tempin = 24;
tempout = 12;
presin = 1;
presout = 1647;
si = 10;
phos = 1;

%% Determine whether to calculate each input row or not
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

%% Run CO2SYSv2.0.5
disp('Running CO2SYS v2.0.5...')
tic
[DATA_v2, HEADERS_v2] = ...
    CO2SYSv2_0_5(P1, P2, P1type, P2type, sal, tempin, tempout, presin, ...
    presout, si, phos, pHscales, K1K2, KSO4_only);
toc

%% Run CO2SYSv3
disp('Running CO2SYS v3...')
tic
[DATA_v3, HEADERS_v3] = ...
    CO2SYSv3_2_0(P1, P2, P1type, P2type, sal, tempin, tempout, presin, ...
    presout, si, phos, 0, 0, pHscales, K1K2, KSO4, KF, BSal);
toc

%% Put results in tables
clear co2s_v2
for V = 1:numel(HEADERS_v2)
    co2s_v2.(HEADERS_v2{V}) = DATA_v2(:, V);
end
co2s_v2 = struct2table(co2s_v2);
clear co2s_v3
for V = 1:numel(HEADERS_v3)
    co2s_v3.(HEADERS_v3{V}) = DATA_v3(:, V);
end
co2s_v3 = struct2table(co2s_v3);

%% Calculate differences
clear co2s_diff
H=1;
for V = 1:numel(HEADERS_v3)
    if H < numel(HEADERS_v2)
    if isequal(HEADERS_v3{V},HEADERS_v2{H})
        co2s_diff.(HEADERS_v2{H}) = ...
           abs((co2s_v3.(HEADERS_v3{V}) - co2s_v2.(HEADERS_v2{H})) ./ co2s_v2.(HEADERS_v2{H})).*100;
    elseif isequal(HEADERS_v2{H},'KSO4CONSTANTS') && isequal(HEADERS_v3{V},'KSO4CONSTANT')
        co2s_diff.(HEADERS_v2{H}) = ...
           abs((co2s_v3.(HEADERS_v3{V}) - co2s_v2.(HEADERS_v2{H})) ./ co2s_v2.(HEADERS_v2{H})).*100;
    else
        H = H-1;
    end
    end
    H = H+1;
end % for V
co2s_diff = struct2table(co2s_diff);

%% Run uncertainties using CO2SYSv2.0.5 --- SLOW!
% % Define dissociation constant uncertainties
% epK = [0.002, 0.0075, 0.015, 0.01, 0.01, 0.02, 0.02];
% disp('Running errors.m (CO2SYS v2.0.5)...')
% tic
% ERR_v2 = nan(size(P1,1),20);
% ERR_HEADERS_v2 = cell(size(P1,1),20);
% UNITS_v2 = cell(size(P1,1),20);
% for n = 1:size(P1,1)
%     [err, head, units] = ...
%         errorsv2_0_5(P1(n), P2(n), P1type(n), P2type(n), sal(n), tempin, tempout, presin, ...
%         presout, si, phos, U1(n), U2(n), 0.01, 0.02, 0.1, 0.01, epK, 0.02, 0.1, ...
%         pHscales(n), K1K2(n), KSO4_only(n));
%     ERR_v2(n,:) = err;
%     ERR_HEADERS_v2 = head;
%     UNITS_v2 = units;
% end
% toc

%% Run uncertainties using CO2SYSv3
epK = [0.002, 0.0075, 0.015, 0.01, 0.01, 0.02, 0.02];
disp('Running errors.m (CO2SYS v3)...')
tic
[ERR_v3, ERR_HEADERS_v3, UNITS_v3] = ...
    errors_v3_2_0(P1, P2, P1type, P2type, sal, tempin, tempout, presin, ...
    presout, si, phos, 0, 0, U1, U2, 0.01, 0.02, 0.1, 0.01, 0, 0, epK, ...
    0.02, 0.1, pHscales, K1K2, KSO4, KF, BSal, 0);
toc

%% Put results in tables
% clear errs_v2
% ERR_HEADERS_v2 = strrep(ERR_HEADERS_v2,'(','_');
% ERR_HEADERS_v2 = strrep(ERR_HEADERS_v2,')','_');
% for V = 1:numel(ERR_HEADERS_v2)
%     errs_v2.(ERR_HEADERS_v2{V}) = ERR_v2(:, V);
% end
% errs_v2 = struct2table(errs_v2);
clear errs_v3
ERR_HEADERS_v3 = strrep(ERR_HEADERS_v3,'(','_');
ERR_HEADERS_v3 = strrep(ERR_HEADERS_v3,')','_');
for V = 1:numel(ERR_HEADERS_v3)
    errs_v3.(ERR_HEADERS_v3{V}) = ERR_v3(:, V);
end
errs_v3 = struct2table(errs_v3);

%% Calculate differences in errors
% clear errs_diff
% for V = 1:numel(ERR_HEADERS_v3)
%     errs_diff.(ERR_HEADERS_v3{V}) = abs((errs_v3.(ERR_HEADERS_v3{V}) - ...
%         errs_v2.(ERR_HEADERS_v2{V})) ./ errs_v2.(ERR_HEADERS_v2{V})).*100;
% end
% errs_diff = struct2table(errs_diff);

%% Save to file
writetable(errs_v3, 'results/compare_versions_uncert.csv')
