% Just a quick comparison of error propagation calculations, with the same
% conditons as in PyCO2SYS-examples/UncertaintyPropagation.ipynb
% Results are very good - consistent!
errtest = errors_extd ( ...
    [2150.1 2175.3 2199.8 2220.4], ...
    [8.121 8.082 8.041 8.001], ...
    2, ...
    3, ...
    33.1, ...
    25, ...
    25, ...
    0, ...
    0, ...
    5, ...
    10, ...
    0.5, ...
    0, ...
    3, ...
    [0.001 0.005 0.005 0.005], ...
    0.002, ...
    0.05, ...
    0, ...
    0, ...
    0, ...
    0, ...
    [0 0.0075 0.015 0 0 0 0], ...
    0, ...
    0, ...
    1, ...
    12, ...
    1, ...
    1, ...
    1);
                
disp("pCO2in")
disp(errtest(:, 4))
disp("OmegaARin")
disp(errtest(:, 10))
