% run_calibration.m
% 标定实验：在纯噪声和信号条件下采集 s_ij 分布
%
% 目的：确定 cj_low, cj_high, sigma_thr 的合理取值
%
% 实验设计：
%   A. 纯噪声基线：自由迁移（无旋转扰动），采集 s_ij 分布
%   B. 信号条件：收敛后触发 1 个个体 90° 转向，采集信号 s_ij 分布

clc; clear; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'calibration');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

nRuns = 5;
convergenceSteps = 500;
observationSteps = 300;

%% ====== 实验 A：纯噪声基线 ======
fprintf('====== 实验 A：纯噪声基线 ======\n');
noise_sij_all = [];
noise_maxs_all = [];
noise_sigma2_all = [];

for run = 1:nRuns
    fprintf('--- Noise Run %d/%d ---\n', run, nRuns);
    G = setup_calibration_params();
    G = initSimRobots(G, 'unif', [1434, 1446], 100);
    G.myinfoIDs = [];
    G.rotateStep = 99999;

    totalSteps = convergenceSteps + observationSteps;
    G.maxSimSteps = totalSteps;
    G.maxcj = zeros(totalSteps, 1);
    G.activatedIDs = cell(totalSteps, 1);
    G.activatedCount = zeros(totalSteps, 1);
    G.activatedSrcIDs = cell(totalSteps, 1);
    G.weight_target = ones(totalSteps, 1);
    G.swarmCenterPositions = zeros(totalSteps, 2);
    G.nowInfoIDs = nan(totalSteps, 1);
    G.cj_threshold = 999;

    for t = 1:totalSteps
        G.simStep = t;
        [desTurnAngle, desSpeed, G] = swarmAlg_fixed(G);
        G = parallelSimRobots(G, desTurnAngle, desSpeed);

        if t > convergenceSteps
            for i = G.robotsList
                [neighbors, ~, ~] = get_topology_neighbors(G, i);
                if ~isempty(neighbors)
                    cj = get_candidate_neighbors(i, neighbors, G);
                    noise_sij_all = [noise_sij_all; cj];
                    noise_maxs_all = [noise_maxs_all; max(cj)];
                    noise_sigma2_all = [noise_sigma2_all; var(cj)];
                end
            end
        end
    end
end

%% ====== 实验 B：信号条件 ======
fprintf('\n====== 实验 B：信号条件 ======\n');
signal_sij_all = [];
signal_maxs_all = [];
signal_sigma2_all = [];

for run = 1:nRuns
    fprintf('--- Signal Run %d/%d ---\n', run, nRuns);
    G = setup_calibration_params();
    G = initSimRobots(G, 'unif', [1434, 1446], 100);

    signalID = randsample(G.robotsList, 1);
    G.myinfoIDs = [signalID];
    G.rotateStep = convergenceSteps;

    totalSteps = convergenceSteps + observationSteps;
    G.maxSimSteps = totalSteps;
    G.maxcj = zeros(totalSteps, 1);
    G.activatedIDs = cell(totalSteps, 1);
    G.activatedCount = zeros(totalSteps, 1);
    G.activatedSrcIDs = cell(totalSteps, 1);
    G.weight_target = ones(totalSteps, 1);
    G.swarmCenterPositions = zeros(totalSteps, 2);
    G.nowInfoIDs = nan(totalSteps, 1);
    G.cj_threshold = 999;

    for t = 1:totalSteps
        G.simStep = t;
        [desTurnAngle, desSpeed, G] = swarmAlg_fixed(G);
        G = parallelSimRobots(G, desTurnAngle, desSpeed);

        if t > convergenceSteps
            for i = G.robotsList
                [neighbors, ~, ~] = get_topology_neighbors(G, i);
                if ~isempty(neighbors)
                    cj = get_candidate_neighbors(i, neighbors, G);
                    signal_sij_all = [signal_sij_all; cj];
                    signal_maxs_all = [signal_maxs_all; max(cj)];
                    signal_sigma2_all = [signal_sigma2_all; var(cj)];
                end
            end
        end
    end
end

%% ====== 分析与推荐参数 ======
fprintf('\n====== 标定结果 ======\n');

fprintf('\n--- s_ij 分布 (rad/s) ---\n');
fprintf('噪声: mean=%.5f, std=%.5f, 95th=%.5f, 99th=%.5f, max=%.5f\n', ...
    mean(noise_sij_all), std(noise_sij_all), ...
    prctile(noise_sij_all, 95), prctile(noise_sij_all, 99), max(noise_sij_all));
fprintf('信号: mean=%.5f, std=%.5f, 95th=%.5f, 99th=%.5f, max=%.5f\n', ...
    mean(signal_sij_all), std(signal_sij_all), ...
    prctile(signal_sij_all, 95), prctile(signal_sij_all, 99), max(signal_sij_all));

fprintf('\n--- max(s_ij) 分布 (rad/s) ---\n');
fprintf('噪声: mean=%.5f, 95th=%.5f, 99th=%.5f\n', ...
    mean(noise_maxs_all), prctile(noise_maxs_all, 95), prctile(noise_maxs_all, 99));
fprintf('信号: mean=%.5f, 5th=%.5f, 25th=%.5f\n', ...
    mean(signal_maxs_all), prctile(signal_maxs_all, 5), prctile(signal_maxs_all, 25));

fprintf('\n--- sigma2_M 分布 (rad/s)^2 ---\n');
fprintf('噪声: mean=%.7f, 95th=%.7f, 99th=%.7f\n', ...
    mean(noise_sigma2_all), prctile(noise_sigma2_all, 95), prctile(noise_sigma2_all, 99));
fprintf('信号: mean=%.7f, 5th=%.7f, 25th=%.7f\n', ...
    mean(signal_sigma2_all), prctile(signal_sigma2_all, 5), prctile(signal_sigma2_all, 25));

cj_high_rec = prctile(noise_maxs_all, 95);
cj_low_rec = prctile(noise_maxs_all, 50);
sigma_thr_rec = prctile(noise_sigma2_all, 95);

fprintf('\n====== 推荐参数 ======\n');
fprintf('cj_high  = %.5f rad/s\n', cj_high_rec);
fprintf('cj_low   = %.5f rad/s\n', cj_low_rec);
fprintf('sigma_thr = %.7f (rad/s)^2\n', sigma_thr_rec);

%% ====== 绘图 ======
figure('Position', [50, 50, 1400, 400]);

subplot(1,3,1);
histogram(noise_sij_all, 100, 'Normalization', 'pdf', ...
    'FaceColor', [0.3, 0.3, 0.8], 'FaceAlpha', 0.6);
hold on;
histogram(signal_sij_all, 100, 'Normalization', 'pdf', ...
    'FaceColor', [0.8, 0.3, 0.3], 'FaceAlpha', 0.6);
xlabel('s_{ij} (rad/s)'); ylabel('概率密度');
legend('噪声', '信号'); title('s_{ij} 分布'); grid on;

subplot(1,3,2);
histogram(noise_maxs_all, 80, 'Normalization', 'pdf', ...
    'FaceColor', [0.3, 0.3, 0.8], 'FaceAlpha', 0.6);
hold on;
histogram(signal_maxs_all, 80, 'Normalization', 'pdf', ...
    'FaceColor', [0.8, 0.3, 0.3], 'FaceAlpha', 0.6);
xline(cj_high_rec, 'k--', 'cj_{high}', 'LineWidth', 1.5);
xline(cj_low_rec, 'g--', 'cj_{low}', 'LineWidth', 1.5);
xlabel('max(s_{ij}) (rad/s)'); ylabel('概率密度');
legend('噪声', '信号'); title('max(s_{ij}) 分布'); grid on;

subplot(1,3,3);
histogram(log10(noise_sigma2_all + 1e-10), 80, 'Normalization', 'pdf', ...
    'FaceColor', [0.3, 0.3, 0.8], 'FaceAlpha', 0.6);
hold on;
histogram(log10(signal_sigma2_all + 1e-10), 80, 'Normalization', 'pdf', ...
    'FaceColor', [0.8, 0.3, 0.3], 'FaceAlpha', 0.6);
xline(log10(sigma_thr_rec), 'k--', 'sigma_{thr}', 'LineWidth', 1.5);
xlabel('log_{10}(sigma^2_M)'); ylabel('概率密度');
legend('噪声', '信号'); title('sigma^2_M 分布'); grid on;

sgtitle('SwarmBang 数字孪生标定结果');

%% 保存
calibration = struct();
calibration.noise_sij = noise_sij_all;
calibration.noise_maxs = noise_maxs_all;
calibration.noise_sigma2 = noise_sigma2_all;
calibration.signal_sij = signal_sij_all;
calibration.signal_maxs = signal_maxs_all;
calibration.signal_sigma2 = signal_sigma2_all;
calibration.recommended.cj_high = cj_high_rec;
calibration.recommended.cj_low = cj_low_rec;
calibration.recommended.sigma_thr = sigma_thr_rec;
save(fullfile(resultsDir, 'calibration_results.mat'), 'calibration');
fprintf('\n标定数据已保存到: %s\n', resultsDir);

%% ====== 本地函数 ======
function G = setup_calibration_params()
    G = struct();
    G.simStep = 0;
    G.maxID = 50;
    G.cycTime = 0.2;
    G.v0 = 15;
    G.maxRotRate = 10 * 1.91;
    G.weight_align = 15;
    G.Drep = 400;
    G.Dsen = 1000;
    G.weight_rep = 8;
    G.weight_att = 0;
    G.deac_threshold = 0.2;
    G.noise_mov = 0;
    G.max_neighbors = 7;
    G.n_steps = 5;
    G.targetA = [-1434, -1446];
    G.targetB = [1449, 1406];
    G.target = G.targetB;
    G.targetR = 300;
    G.rotateTime = 100;
    G.rotateCycle = zeros(G.maxID, 1);
end
