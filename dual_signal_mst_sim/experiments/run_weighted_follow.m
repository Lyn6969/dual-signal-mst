%% run_weighted_follow.m
% Binary vs Binary+WeightedFollow 多噪声对比实验
% 测试加权跟随在不同 eta 下对 R-P 的影响

clc; clear; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 参数
fprintf('=================================================\n');
fprintf('  Binary vs Binary+WeightedFollow\n');
fprintf('  多噪声水平对比\n');
fprintf('=================================================\n\n');

base = struct();
base.N = 200;
base.rho = 1;
base.v0 = 1;
base.angleUpdateParameter = 10;
base.dt = 0.1;
base.radius = 5;
base.deac_threshold = 0.1745;
base.cj_threshold = 1.5;
base.fieldSize = 50;
base.initDirection = pi/4;
base.useFixedField = true;

resp_base = base;
resp_base.T_max = 600;
resp_base.stabilization_steps = 200;
resp_base.forced_turn_duration = 400;

pers_base = base;
pers_base.T_max = 600;

eta_values = [0.25, 0.30];
num_runs = 50;
num_angles = 1;
base_seed = 20260331;

pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_diffusion = 1e-4;
pers_cfg.min_fit_points = 40;

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5.0;
adaptive_cfg.saliency_threshold = 0.031;
adaptive_cfg.include_self = false;

%% 2. 并行池
cluster = parcluster('Processes');
desired_workers = detectDesiredWorkers(cluster);
fprintf('自动检测到并行 worker 数: %d\n', desired_workers);
if cluster.NumWorkers < desired_workers
    cluster.NumWorkers = desired_workers;
    saveProfile(cluster);
    cluster = parcluster('Processes');
end
pool = gcp('nocreate');
if ~isempty(pool) && pool.NumWorkers ~= desired_workers
    delete(pool);
    pool = [];
end
if isempty(pool)
    pool = parpool(cluster, desired_workers);
end
fprintf('并行池：%d workers\n\n', pool.NumWorkers);

%% 输出目录
results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_dir = fullfile(results_dir, sprintf('weighted_follow_%s', timestamp));
mkdir(out_dir);

fprintf('eta = %s, num_runs = %d\n', mat2str(eta_values), num_runs);
fprintf('共享种子: seed = base_seed + ri\n\n');

%% 3. 主循环：每个 eta 跑 Binary 和 Weighted
num_eta = numel(eta_values);
R_binary_all = NaN(num_runs, num_eta);
P_binary_all = NaN(num_runs, num_eta);
R_weighted_all = NaN(num_runs, num_eta);
P_weighted_all = NaN(num_runs, num_eta);

for ei = 1:num_eta
    eta = eta_values(ei);
    resp_params = resp_base;
    resp_params.angleNoiseIntensity = eta^2 / 2;
    pers_params = pers_base;
    pers_params.angleNoiseIntensity = eta^2 / 2;
    time_vec = (0:resp_params.T_max)' * resp_params.dt;

    % Binary
    fprintf('[eta=%.2f] Binary %d 次...\n', eta, num_runs);
    R_bin = NaN(num_runs, 1); P_bin = NaN(num_runs, 1);
    parfor ri = 1:num_runs
        seed = base_seed + ri;
        rp = resp_params;
        rp.useAdaptiveThreshold = true;
        rp.adaptiveThresholdConfig = adaptive_cfg;
        rp.adaptiveThresholdMode = 'binary';
        rp.useWeightedFollow = false;
        R_bin(ri) = run_responsiveness(rp, num_angles, time_vec, seed);

        pp = pers_params;
        pp.useAdaptiveThreshold = true;
        pp.adaptiveThresholdConfig = adaptive_cfg;
        pp.adaptiveThresholdMode = 'binary';
        pp.useWeightedFollow = false;
        P_bin(ri) = run_persistence(pp, pers_cfg, seed + 50000);
    end
    R_binary_all(:, ei) = R_bin;
    P_binary_all(:, ei) = P_bin;
    fprintf('  R=%.4f+/-%.4f, P=%.2f+/-%.2f\n', ...
        mean(R_bin,'omitnan'), std(R_bin,'omitnan')/sqrt(num_runs), ...
        mean(P_bin,'omitnan'), std(P_bin,'omitnan')/sqrt(num_runs));

    % Weighted
    fprintf('[eta=%.2f] Weighted %d 次...\n', eta, num_runs);
    R_wt = NaN(num_runs, 1); P_wt = NaN(num_runs, 1);
    parfor ri = 1:num_runs
        seed = base_seed + ri;
        rp = resp_params;
        rp.useAdaptiveThreshold = true;
        rp.adaptiveThresholdConfig = adaptive_cfg;
        rp.adaptiveThresholdMode = 'binary';
        rp.useWeightedFollow = true;
        R_wt(ri) = run_responsiveness(rp, num_angles, time_vec, seed);

        pp = pers_params;
        pp.useAdaptiveThreshold = true;
        pp.adaptiveThresholdConfig = adaptive_cfg;
        pp.adaptiveThresholdMode = 'binary';
        pp.useWeightedFollow = true;
        P_wt(ri) = run_persistence(pp, pers_cfg, seed + 50000);
    end
    R_weighted_all(:, ei) = R_wt;
    P_weighted_all(:, ei) = P_wt;
    fprintf('  R=%.4f+/-%.4f, P=%.2f+/-%.2f\n\n', ...
        mean(R_wt,'omitnan'), std(R_wt,'omitnan')/sqrt(num_runs), ...
        mean(P_wt,'omitnan'), std(P_wt,'omitnan')/sqrt(num_runs));
end

%% 4. 汇总
fprintf('========== 结果汇总 ==========\n');
fprintf('%-6s %-10s %-10s %-10s %-10s %8s %8s\n', 'eta', 'R_bin', 'R_wt', 'P_bin', 'P_wt', 'dR%', 'dP%');
fprintf('%s\n', repmat('-', 1, 70));

for ei = 1:num_eta
    Rb = mean(R_binary_all(:,ei),'omitnan');
    Rw = mean(R_weighted_all(:,ei),'omitnan');
    Pb = mean(P_binary_all(:,ei),'omitnan');
    Pw = mean(P_weighted_all(:,ei),'omitnan');
    dR = (Rw - Rb) / Rb * 100;
    dP = (Pw - Pb) / Pb * 100;
    fprintf('%-6.2f %-10.4f %-10.4f %-10.2f %-10.2f %+7.1f%% %+7.1f%%\n', ...
        eta_values(ei), Rb, Rw, Pb, Pw, dR, dP);
end
fprintf('\n');

%% 5. 图1：R 和 P 对比柱状图
fig1 = figure('Visible', 'off', 'Position', [50 50 1200 500], 'Color', 'w');

subplot(1,2,1);
R_means = [mean(R_binary_all,'omitnan'); mean(R_weighted_all,'omitnan')]';
bar(R_means); hold on;
set(gca, 'XTickLabel', string(eta_values));
xlabel('eta'); ylabel('R');
legend({'Binary', 'Weighted'}, 'Location', 'best');
title('R'); grid on;

subplot(1,2,2);
P_means = [mean(P_binary_all,'omitnan'); mean(P_weighted_all,'omitnan')]';
bar(P_means); hold on;
set(gca, 'XTickLabel', string(eta_values));
xlabel('eta'); ylabel('P');
legend({'Binary', 'Weighted'}, 'Location', 'best');
title('P'); grid on;

sgtitle(sprintf('Binary vs Weighted Follow (%d runs)', num_runs));
saveas(fig1, fullfile(out_dir, 'comparison_bar.png'));
close(fig1);

%% 6. 图2：R-P 散点图（每个 eta 一个子图）
fig2 = figure('Visible', 'off', 'Position', [50 50 500*num_eta 500], 'Color', 'w');
for ei = 1:num_eta
    subplot(1, num_eta, ei);
    hold on;
    scatter(R_binary_all(:,ei), P_binary_all(:,ei), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
    scatter(R_weighted_all(:,ei), P_weighted_all(:,ei), 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
    Rb = mean(R_binary_all(:,ei),'omitnan'); Pb = mean(P_binary_all(:,ei),'omitnan');
    Rw = mean(R_weighted_all(:,ei),'omitnan'); Pw = mean(P_weighted_all(:,ei),'omitnan');
    scatter(Rb, Pb, 200, 'r', 'p', 'filled', 'MarkerEdgeColor', 'k');
    scatter(Rw, Pw, 200, 'b', 'h', 'filled', 'MarkerEdgeColor', 'k');
    xlabel('R'); ylabel('P');
    title(sprintf('eta=%.2f', eta_values(ei)));
    legend({'Binary', 'Weighted'}, 'Location', 'best');
    grid on;
end
saveas(fig2, fullfile(out_dir, 'comparison_rp.png'));
close(fig2);

%% 7. 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');
fprintf(fid, '========================================================================\n');
fprintf(fid, 'Binary vs Binary+WeightedFollow 多噪声对比报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '========================================================================\n\n');
fprintf(fid, '设置: N=%d, num_runs=%d, 共享种子\n', base.N, num_runs);
fprintf(fid, 'adaptive_cfg: cj_low=%.1f, cj_high=%.1f, saliency_thr=%.3f\n', ...
    adaptive_cfg.cj_low, adaptive_cfg.cj_high, adaptive_cfg.saliency_threshold);
fprintf(fid, 'WeightedFollow: 过阈值邻居按 s_ij 线性加权确定跟随方向\n\n');

fprintf(fid, '%-6s %-10s %-10s %-10s %-10s %8s %8s\n', 'eta', 'R_bin', 'R_wt', 'P_bin', 'P_wt', 'dR%', 'dP%');
fprintf(fid, '%s\n', repmat('-', 1, 70));
for ei = 1:num_eta
    Rb = mean(R_binary_all(:,ei),'omitnan');
    Rw = mean(R_weighted_all(:,ei),'omitnan');
    Pb = mean(P_binary_all(:,ei),'omitnan');
    Pw = mean(P_weighted_all(:,ei),'omitnan');
    dR = (Rw - Rb) / Rb * 100;
    dP = (Pw - Pb) / Pb * 100;
    fprintf(fid, '%-6.2f %-10.4f %-10.4f %-10.2f %-10.2f %+7.1f%% %+7.1f%%\n', ...
        eta_values(ei), Rb, Rw, Pb, Pw, dR, dP);
end
fclose(fid);

%% 8. 保存数据
saveResultBundle(out_dir, 'comparison_data', ...
    {'R_binary_all', 'P_binary_all', 'R_weighted_all', 'P_weighted_all', ...
     'eta_values', 'adaptive_cfg', 'base', 'num_runs', 'base_seed'});

fprintf('完成。输出目录: %s\n', out_dir);

%% ========================================================================
function desired_workers = detectDesiredWorkers(cluster)
    desired_workers = NaN;
    env_names = {'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'PBS_NP', 'NSLOTS'};
    for i = 1:numel(env_names)
        v = getenv(env_names{i});
        if ~isempty(v)
            token = regexp(v, '\d+', 'match', 'once');
            if ~isempty(token)
                desired_workers = floor(str2double(token) * 0.8);
                fprintf('检测到调度器 %s, 使用 80%% = %d\n', env_names{i}, desired_workers);
                break;
            end
        end
    end
    if isnan(desired_workers)
        try
            desired_workers = floor(maxNumCompThreads * 2 * 0.8);
            fprintf('检测到核心数 %d, x2x80%% = %d\n', maxNumCompThreads, desired_workers);
        catch
            try
                desired_workers = floor(feature('numcores') * 2 * 0.8);
            catch
                desired_workers = cluster.NumWorkers;
            end
        end
    end
    desired_workers = max(1, floor(desired_workers));
end

function R = run_responsiveness(params, num_angles, time_vec, seed)
    rng(seed);
    sim = ParticleSimulationWithExternalPulse(params);
    sim.external_pulse_count = 1;
    sim.resetCascadeTracking();
    sim.initializeParticles();

    V_history = zeros(params.T_max + 1, 2);
    V_history(1, :) = [mean(params.v0*cos(sim.theta)), mean(params.v0*sin(sim.theta))];
    proj_history = zeros(params.T_max + 1, num_angles);
    triggered = false; n_vecs = []; t_start = NaN;

    for t = 1:params.T_max
        sim.step();
        V_history(t+1, :) = [mean(params.v0*cos(sim.theta)), mean(params.v0*sin(sim.theta))];
        if ~triggered && sim.external_pulse_triggered
            triggered = true; t_start = t;
            lidx = find(sim.isExternallyActivated, 1, 'first');
            if isempty(lidx), lidx = 1; end
            phi = sim.external_target_theta(lidx);
            n_vecs = [cos(phi); sin(phi)];
        end
        if triggered
            proj_history(t+1, :) = V_history(t+1, :) * n_vecs;
        end
    end

    if ~triggered || isnan(t_start), R = NaN; return; end
    t_end = min(t_start + params.forced_turn_duration, params.T_max);
    integral_val = trapz(time_vec(t_start+1:t_end+1), proj_history(t_start+1:t_end+1));
    duration = time_vec(t_end+1) - time_vec(t_start+1);
    if duration > 0, R = integral_val / (params.v0 * duration); else, R = NaN; end
end

function P = run_persistence(params, cfg, seed)
    rng(seed);
    sim = ParticleSimulation(params);
    T = sim.T_max;
    burn_in = max(2, floor((T+1) * cfg.burn_in_ratio));

    init_pos = sim.positions;
    centroid0 = mean(init_pos, 1);
    offsets0 = init_pos - centroid0;
    msd = zeros(T+1, 1);

    for t = 1:T
        sim.step();
        positions = sim.positions;
        centroid = mean(positions, 1);
        rel_disp = (positions - centroid) - offsets0;
        msd(t+1) = mean(sum(rel_disp.^2, 2));
    end

    time_vec = (0:T)' * sim.dt;
    x = time_vec(burn_in:end); y = msd(burn_in:end);

    if numel(x) < max(2, cfg.min_fit_points) || all(abs(y - y(1)) < eps)
        D = NaN;
    else
        x_s = x - x(1); y_s = y - y(1);
        if any(x_s > 0) && any(abs(y_s) > eps)
            sw = max(5, floor(numel(y_s)*0.1));
            if sw > 1, y_s = smoothdata(y_s, 'movmean', sw); end
            slope = lsqnonneg(x_s(:), y_s(:));
            D = max(slope / 4, cfg.min_diffusion);
        else
            D = NaN;
        end
    end

    if isnan(D), P = NaN; else, P = 1 / sqrt(D); end
end
