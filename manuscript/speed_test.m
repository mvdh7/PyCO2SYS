gfile = "C:/Users/mphum/Downloads/GLODAPv2.2021_Merged_Master_File.mat";
load(gfile)
G2tco2(G2tco2 == -9999) = NaN;
G2talk(G2talk == -9999) = NaN;
G2salinity(G2salinity == -9999) = NaN;
G2temperature(G2temperature == -9999) = NaN;
G2pressure(G2pressure == -9999) = NaN;
G2silicate(G2silicate == -9999) = NaN;
G2phosphate(G2phosphate == -9999) = NaN;
run_CO2SYS = @() CO2SYSv3_2_0(G2tco2, G2talk, 2, 1, ...
    G2salinity, G2temperature, G2temperature, G2pressure, G2pressure, ...
    G2silicate, G2phosphate, 0, 0, 1, 10, 1, 1, 1);
timeit(run_CO2SYS)

%%
for i = 1:7
    tic
    run_CO2SYS();
    toc
end

%% MATLAB times
times = [13.41, 13.33, 13.00, 12.53, 13.02, 13.29, 12.86];
time_mean = mean(times);
time_std = std(times);

%% OCTAVE CLI times
otimes = [15.26, 16.22, 16.19, 16.16, 16.27, 17.88, 17.45];
otime_mean = mean(otimes);
otime_std = std(otimes);