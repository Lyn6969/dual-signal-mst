%% run_cjlow_sweep.m
% 测试不同 cj_low 对 Binary 模式 R-P 的影响
% 固定 eta=0.30，扫描 cj_low = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

clc; clear; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 参数
fprintf('=================================================\n');
fprintf('  Binary cj_low 扫描实验\n');
fprintf('  eta=0.30, cj_high=5.0, 扫描 cj_low\n');
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

resp_params = base;
resp_params.T_max = 600;
resp_params.stabilization_steps = 200;
resp_params.forced_turn_duration = 400;

pers_params = base;
pers_params.T_max = 600;

eta = 0.30;
resp_params.angleNoiseIntensity = eta^2 / 2;
pers_params.angleNoiseIntensity = eta^2 / 2;

num_runs = 50;
num_angles = 1;
base_seed = 20260331;

pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_diffusion = 1e-4;
pers_cfg.min_fit_points = 40;

% 扫描的 cj_low 值
cj_low_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
n_cjlow = numel(cj_low_values);

time_vec = (0:resp_params.T_max)' * resp_params.dt;

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
out_dir = fullfile(results_dir, sprintf('cjlow_sweep_%s', timestamp));
mkdir(out_dir);

fprintf('eta = %.2f, cj_high = 5.0, num_runs = %d\n', eta, num_runs);
fprintf('cj_low 扫描值: %s\n', mat2str(cj_low_values));
fprintf('共享种子: seed = base_seed + ri\n\n');

%% 3. 展平 parfor 扫描
total_jobs = n_cjlow * num_runs;
job_ci = zeros(total_jobs, 1);
job_ri = zeros(total_jobs, 1);
jid = 0;
for ci = 1:n_cjlow
    for ri = 1:num_runs
        jid = jid + 1;
        job_ci(jid) = ci;
        job_ri(jid) = ri;
    end
end

job_R = NaN(total_jobs, 1);
job_P = NaN(total_jobs, 1);

fprintf('[扫描] %d 个 jobs (parfor %d workers)...\n', total_jobs, pool.NumWorkers);
sweep_timer = tic;

parfor jid = 1:total_jobs
    ci = job_ci(jid);
    ri = job_ri(jid);
    cj_low = cj_low_values(ci);
    seed = base_seed + ri;

    adaptive_cfg = struct();
    adaptive_cfg.cj_low = cj_low;
    adaptive_cfg.cj_high = 5.0;
    adaptive_cfg.saliency_threshold = 0.031;
    adaptive_cfg.include_self = false;

    rp = resp_params;
    rp.useAdaptiveThreshold = true;
    rp.adaptiveThresholdConfig = adaptive_cfg;
    rp.adaptiveThresholdMode = 'binary';
    job_R(jid) = run_responsiveness(rp, num_angles, time_vec, seed);

    pp = pers_params;
    pp.useAdaptiveThreshold = true;
    pp.adaptiveThresholdConfig = adaptive_cfg;
    pp.adaptiveThresholdMode = 'binary';
    job_P(jid) = run_persistence(pp, pers_cfg, seed + 50000);
end

sweep_elapsed = toc(sweep_timer);
fprintf('\n扫描完成，耗时 %.1f 分钟\n\n', sweep_elapsed / 60);

%% 4. 聚合结果
R_means = NaN(n_cjlow, 1);
P_means = NaN(n_cjlow, 1);
R_ses = NaN(n_cjlow, 1);
P_ses = NaN(n_cjlow, 1);
R_all = NaN(num_runs, n_cjlow);
P_all = NaN(num_runs, n_cjlow);

for ci = 1:n_cjlow
    mask = (job_ci == ci);
    R_vals = job_R(mask);
    P_vals = job_P(mask);
    R_means(ci) = mean(R_vals, 'omitnan');
    P_means(ci) = mean(P_vals, 'omitnan');
    R_ses(ci) = std(R_vals, 'omitnan') / sqrt(num_runs);
    P_ses(ci) = std(P_vals, 'omitnan') / sqrt(num_runs);
    R_all(:, ci) = R_vals;
    P_all(:, ci) = P_vals;
end

% 以 cj_low=0.5 为 baseline
delta_R = (R_means - R_means(1)) / R_means(1) * 100;
delta_P = (P_means - P_means(1)) / P_means(1) * 100;

%% 5. 打印汇总
fprintf('========== 结果汇总 (eta=%.2f) ==========\n', eta);
fprintf('%-10s %8s %8s %8s %8s\n', 'cj_low', 'R', 'P', 'dR%', 'dP%');
fprintf('%s\n', repmat('-', 1, 50));
for ci = 1:n_cjlow
    if ci == 1
        fprintf('%-10.1f %8.4f %8.2f %8s %8s\n', cj_low_values(ci), R_means(ci), P_means(ci), '-', '-');
    else
        fprintf('%-10.1f %8.4f %8.2f %+7.1f%% %+7.1f%%\n', ...
            cj_low_values(ci), R_means(ci), P_means(ci), delta_R(ci), delta_P(ci));
    end
end
fprintf('\n');

%% 6. 图1：R 和 P 随 cj_low 变化
fig1 = figure('Visible', 'off', 'Position', [50 50 1200 500], 'Color', 'w');

subplot(1,2,1);
errorbar(cj_low_values, R_means, R_ses, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel('cj_{low}');
ylabel('R');
title(sprintf('R vs cj_{low} (eta=%.2f)', eta));
grid on;

subplot(1,2,2);
errorbar(cj_low_values, P_means, P_ses, 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
xlabel('cj_{low}');
ylabel('P');
title(sprintf('P vs cj_{low} (eta=%.2f)', eta));
grid on;

sgtitle(sprintf('Binary: cj_{low} vs R-P (eta=%.2f, %d runs)', eta, num_runs));
saveas(fig1, fullfile(out_dir, 'fig1_rp_vs_cjlow.png'));
close(fig1);

%% 7. 图2：R-P 散点图
fig2 = figure('Visible', 'off', 'Position', [50 50 800 600], 'Color', 'w');
hold on;
colors = lines(n_cjlow);
for ci = 1:n_cjlow
    scatter(R_all(:, ci), P_all(:, ci), 20, colors(ci, :), 'filled', 'MarkerFaceAlpha', 0.3, 'HandleVisibility', 'off');
    scatter(R_means(ci), P_means(ci), 200, colors(ci, :), 'p', 'filled', ...
        'MarkerEdgeColor', 'k', 'DisplayName', sprintf('cj_{low}=%.1f', cj_low_values(ci)));
end
xlabel('R'); ylabel('P');
title(sprintf('R-P (eta=%.2f)', eta));
legend('Location', 'best'); grid on;
saveas(fig2, fullfile(out_dir, 'fig2_rp_scatter.png'));
close(fig2);

%% 8. 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');
fprintf(fid, '========================================================================\n');
fprintf(fid, 'Binary cj_low 扫描报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '耗时: %.1f 分钟\n', sweep_elapsed / 60);
fprintf(fid, '========================================================================\n\n');
fprintf(fid, '设置: eta=%.2f, N=%d, cj_high=5.0, num_runs=%d, 共享种子\n', eta, base.N, num_runs);
fprintf(fid, 'cj_low 扫描值: %s\n\n', mat2str(cj_low_values));

fprintf(fid, '%-10s %8s %8s %8s %8s\n', 'cj_low', 'R', 'P', 'dR%', 'dP%');
fprintf(fid, '%s\n', repmat('-', 1, 50));
for ci = 1:n_cjlow
    if ci == 1
        fprintf(fid, '%-10.1f %8.4f %8.2f %8s %8s\n', cj_low_values(ci), R_means(ci), P_means(ci), '-', '-');
    else
        fprintf(fid, '%-10.1f %8.4f %8.2f %+7.1f%% %+7.1f%%\n', ...
            cj_low_values(ci), R_means(ci), P_means(ci), delta_R(ci), delta_P(ci));
    end
end
fclose(fid);

%% 9. 保存数据
save(fullfile(out_dir, 'sweep_data.mat'), ...
    'cj_low_values', 'R_means', 'P_means', 'R_ses', 'P_ses', ...
    'R_all', 'P_all', 'delta_R', 'delta_P', ...
    'eta', 'base', 'num_runs', 'base_seed', 'sweep_elapsed');

fprintf('\n完成。输出目录: %s\n', out_dir);

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
                fprintf('检测到调度器 %s，使用 80%% = %d\n', env_names{i}, desired_workers);
                break;
            end
        end
    end
    if isnan(desired_workers)
        try
            desired_workers = floor(maxNumCompThreads * 2 * 0.8);
            fprintf('检测到核心数 %d，x2x80%% = %d\n', maxNumCompThreads, desired_workers);
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
