%% run_eta_sweep.m
% Exp1: Binary vs Weighted 多噪声水平系统性扫描
% η = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
% 对应论文 Section IV-A (Fig. 3)

clc; clear; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 参数
fprintf('=================================================\n');
fprintf('  Exp1: Binary vs Weighted 多噪声扫描\n');
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

eta_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40];
num_eta = numel(eta_values);
num_runs = 50;
num_angles = 1;
base_seed = 20260401;

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
out_dir = fullfile(results_dir, sprintf('eta_sweep_%s', timestamp));
mkdir(out_dir);

fprintf('eta = %s\n', mat2str(eta_values));
fprintf('num_runs = %d, 共享种子\n\n', num_runs);

%% 3. 展平 parfor：(eta, mode, run)
% mode: 1=Binary, 2=Weighted
total_jobs = num_eta * 2 * num_runs;
job_ei = zeros(total_jobs, 1);
job_mi = zeros(total_jobs, 1);
job_ri = zeros(total_jobs, 1);
jid = 0;
for ei = 1:num_eta
    for mi = 1:2
        for ri = 1:num_runs
            jid = jid + 1;
            job_ei(jid) = ei;
            job_mi(jid) = mi;
            job_ri(jid) = ri;
        end
    end
end

job_R = NaN(total_jobs, 1);
job_P = NaN(total_jobs, 1);

fprintf('[扫描] %d 个 jobs ...\n', total_jobs);
sweep_timer = tic;

parfor jid = 1:total_jobs
    ei = job_ei(jid);
    mi = job_mi(jid);
    ri = job_ri(jid);
    eta = eta_values(ei);
    seed = base_seed + ri;
    use_wf = (mi == 2);

    rp = resp_base;
    rp.angleNoiseIntensity = eta^2 / 2;
    rp.useAdaptiveThreshold = true;
    rp.adaptiveThresholdConfig = adaptive_cfg;
    rp.adaptiveThresholdMode = 'binary';
    rp.useWeightedFollow = use_wf;
    time_vec = (0:rp.T_max)' * rp.dt;
    job_R(jid) = run_responsiveness(rp, num_angles, time_vec, seed);

    pp = pers_base;
    pp.angleNoiseIntensity = eta^2 / 2;
    pp.useAdaptiveThreshold = true;
    pp.adaptiveThresholdConfig = adaptive_cfg;
    pp.adaptiveThresholdMode = 'binary';
    pp.useWeightedFollow = use_wf;
    job_P(jid) = run_persistence(pp, pers_cfg, seed + 50000);
end

sweep_elapsed = toc(sweep_timer);
fprintf('扫描完成，耗时 %.1f 分钟\n\n', sweep_elapsed / 60);

%% 4. 聚合
R_binary = NaN(num_runs, num_eta);
R_weighted = NaN(num_runs, num_eta);
P_binary = NaN(num_runs, num_eta);
P_weighted = NaN(num_runs, num_eta);

for jid = 1:total_jobs
    ei = job_ei(jid);
    mi = job_mi(jid);
    ri = job_ri(jid);
    if mi == 1
        R_binary(ri, ei) = job_R(jid);
        P_binary(ri, ei) = job_P(jid);
    else
        R_weighted(ri, ei) = job_R(jid);
        P_weighted(ri, ei) = job_P(jid);
    end
end

Rb = mean(R_binary, 1, 'omitnan');
Rw = mean(R_weighted, 1, 'omitnan');
Pb = mean(P_binary, 1, 'omitnan');
Pw = mean(P_weighted, 1, 'omitnan');
Rb_se = std(R_binary, 0, 1, 'omitnan') / sqrt(num_runs);
Rw_se = std(R_weighted, 0, 1, 'omitnan') / sqrt(num_runs);
Pb_se = std(P_binary, 0, 1, 'omitnan') / sqrt(num_runs);
Pw_se = std(P_weighted, 0, 1, 'omitnan') / sqrt(num_runs);

dR = (Rw - Rb) ./ Rb * 100;
dP = (Pw - Pb) ./ Pb * 100;

%% 5. 打印汇总
fprintf('========== 结果汇总 ==========\n');
fprintf('%-6s %8s %8s %8s %8s %8s %8s\n', 'eta', 'R_bin', 'R_wt', 'P_bin', 'P_wt', 'dR%', 'dP%');
fprintf('%s\n', repmat('-', 1, 60));
for ei = 1:num_eta
    fprintf('%-6.2f %8.4f %8.4f %8.2f %8.2f %+7.1f%% %+7.1f%%\n', ...
        eta_values(ei), Rb(ei), Rw(ei), Pb(ei), Pw(ei), dR(ei), dP(ei));
end
fprintf('\n');

%% 6. 图1: dR% 和 dP% 随 eta 变化
fig1 = figure('Visible', 'off', 'Position', [50 50 1000 450], 'Color', 'w');

subplot(1,2,1);
plot(eta_values, dR, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
yline(0, 'k--');
xlabel('\eta'); ylabel('\DeltaR (%)');
title('响应性变化');
grid on;

subplot(1,2,2);
plot(eta_values, dP, 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
yline(0, 'k--');
xlabel('\eta'); ylabel('\DeltaP (%)');
title('持久性变化');
grid on;

sgtitle(sprintf('Weighted vs Binary: \\DeltaR%% 和 \\DeltaP%% vs \\eta (%d runs)', num_runs));
saveas(fig1, fullfile(out_dir, 'fig1_delta_rp_vs_eta.png'));
close(fig1);

%% 7. 图2: R 和 P 的绝对值随 eta 变化
fig2 = figure('Visible', 'off', 'Position', [50 50 1000 450], 'Color', 'w');

subplot(1,2,1);
errorbar(eta_values, Rb, Rb_se, 'rs-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r'); hold on;
errorbar(eta_values, Rw, Rw_se, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel('\eta'); ylabel('R');
legend({'Binary', 'Weighted'}, 'Location', 'best');
title('响应性 R vs \eta');
grid on;

subplot(1,2,2);
errorbar(eta_values, Pb, Pb_se, 'rs-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r'); hold on;
errorbar(eta_values, Pw, Pw_se, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel('\eta'); ylabel('P');
legend({'Binary', 'Weighted'}, 'Location', 'best');
title('持久性 P vs \eta');
grid on;

sgtitle(sprintf('Binary vs Weighted (%d runs)', num_runs));
saveas(fig2, fullfile(out_dir, 'fig2_rp_vs_eta.png'));
close(fig2);

%% 8. 图3: 每个 eta 的 R-P 散点图
fig3 = figure('Visible', 'off', 'Position', [50 50 250*num_eta 450], 'Color', 'w');
for ei = 1:num_eta
    subplot(1, num_eta, ei); hold on;
    scatter(R_binary(:,ei), P_binary(:,ei), 15, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
    scatter(R_weighted(:,ei), P_weighted(:,ei), 15, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
    scatter(Rb(ei), Pb(ei), 150, 'r', 'p', 'filled', 'MarkerEdgeColor', 'k');
    scatter(Rw(ei), Pw(ei), 150, 'b', 'h', 'filled', 'MarkerEdgeColor', 'k');
    xlabel('R'); ylabel('P');
    title(sprintf('\\eta=%.2f', eta_values(ei)));
    if ei == 1, legend({'Bin', 'Wt'}, 'Location', 'best'); end
    grid on;
end
sgtitle('R-P 散点图');
saveas(fig3, fullfile(out_dir, 'fig3_rp_scatter.png'));
close(fig3);

%% 9. 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');
fprintf(fid, '========================================================================\n');
fprintf(fid, 'Exp1: Binary vs Weighted 多噪声扫描报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '耗时: %.1f 分钟\n', sweep_elapsed / 60);
fprintf(fid, '========================================================================\n\n');
fprintf(fid, '设置: N=%d, num_runs=%d, 共享种子 base_seed=%d\n', base.N, num_runs, base_seed);
fprintf(fid, 'adaptive_cfg: cj_low=%.1f, cj_high=%.1f, saliency_thr=%.3f\n', ...
    adaptive_cfg.cj_low, adaptive_cfg.cj_high, adaptive_cfg.saliency_threshold);
fprintf(fid, 'Weighted: 过阈值邻居按 s_ij 线性加权确定跟随方向\n\n');

fprintf(fid, '%-6s %8s %8s %8s %8s %8s %8s\n', 'eta', 'R_bin', 'R_wt', 'P_bin', 'P_wt', 'dR%', 'dP%');
fprintf(fid, '%s\n', repmat('-', 1, 60));
for ei = 1:num_eta
    fprintf(fid, '%-6.2f %8.4f %8.4f %8.2f %8.2f %+7.1f%% %+7.1f%%\n', ...
        eta_values(ei), Rb(ei), Rw(ei), Pb(ei), Pw(ei), dR(ei), dP(ei));
end
fclose(fid);

%% 10. 保存数据
saveResultBundle(out_dir, 'sweep_data', ...
    {'eta_values', 'R_binary', 'R_weighted', 'P_binary', 'P_weighted', ...
     'Rb', 'Rw', 'Pb', 'Pw', 'dR', 'dP', ...
     'Rb_se', 'Rw_se', 'Pb_se', 'Pw_se', ...
     'adaptive_cfg', 'base', 'num_runs', 'base_seed', 'sweep_elapsed'});

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
