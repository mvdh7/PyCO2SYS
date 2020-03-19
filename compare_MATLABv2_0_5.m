%% Set up input conditions
PAR12combos = repmat(combnk(1:5, 2), 240, 1);
PAR12combos = PAR12combos( ...
    ~(PAR12combos(:, 1) == 4 & PAR12combos(:, 2) == 5), :);
setPARS = [2250 2100 8.1 400 405];
PARSin = setPARS(PAR12combos);
[pHscales, K1K2, KSO4] = ndgrid(1:4, 1:15, 1:4);
pHscales = repmat(pHscales(:), 1, 9)';
K1K2 = repmat(K1K2(:), 1, 9)';
KSO4 = repmat(KSO4(:), 1, 9)';
pHscales = pHscales(:);
K1K2 = K1K2(:);
KSO4 = KSO4(:);
sal = 33.1*ones(size(K1K2));
sal(K1K2 == 8) = 0;
tempin = 24;
tempout = 12;
presin = 1;
presout = 1647;
si = 10;
phos = 1;

%% Run CO2SYS
tic
[DATA, HEADERS] = ...
    CO2SYSv2_0_5(PARSin(:, 1), PARSin(:, 2), PAR12combos(:, 1), ...
    PAR12combos(:, 2), sal, tempin, tempout, presin, presout, ...
    si, phos, pHscales, K1K2, KSO4);
toc

%% Extract and save outputs
for V = 1:numel(HEADERS)
    co2s.(HEADERS{V}) = DATA(:, V);
end % for V
co2s.PARSin = PARSin;
co2s.PAR12combos = PAR12combos;
save('compare/MATLAB_CO2SYSv2_0_5.mat', 'co2s')
