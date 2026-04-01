%% run_weight_mode_comparison.m
% Binary vs Absolute-weighted vs Excess-weighted 对比实验
% 测试不同权重模式在 eta=0.25/0.30 下的 R-P 表现

clc; clear; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 参数
fprintf('=================================================\n');
fprintf('  Binary vs Absolute-weighted vs Excess-weighted\n');
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

% 三种模式的配置
modes = struct();
modes(1).name = 'Binary';
modes(1).useWeightedFollow = false;
modes(1).weightMode = 'absolute';  % 无所谓，不会用到

modes(2).name = 'Absolute';
modes(2).useWeightedFollow = true;
modes(2).weightMode = 'absolute';

modes(3).name = 'Excess';
modes(3).useWeightedFollow = true;
modes(3).weightMode = 'excess';

num_modes = numel(modes);

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
out_dir = fullfile(results_dir, sprintf('weight_mode_%s', timestamp));
mkdir(out_dir);

fprintf('eta = %s, num_runs = %d\n', mat2str(eta_values), num_runs);
fprintf('模式: Binary, Absolute (w=s/Σs), Excess (w=(s-M_T)/Σ(s-M_T))\n\n');

%% 3. 主循环
num_eta = numel(eta_values);
R_all = NaN(num_runs, num_modes, num_eta);
P_all = NaN(num_runs, num_modes, num_eta);

for ei = 1:num_eta
    eta = eta_values(ei);
    resp_params = resp_base;
    resp_params.angleNoiseIntensity = eta^2 / 2;
    pers_params = pers_base;
    pers_params.angleNoiseIntensity = eta^2 / 2;
    time_vec = (0:resp_params.T_max)' * resp_params.dt;

    for mi = 1:num_modes
        m = modes(mi);
        fprintf('[eta=%.2f] %s %d 次...\n', eta, m.name, num_runs);

        R_tmp = NaN(num_runs, 1);
        P_tmp = NaN(num_runs, 1);

        wf = m.useWeightedFollow;
        wm = m.weightMode;

        parfor ri = 1:num_runs
            seed = base_seed + ri;

            rp = resp_params;
            rp.useAdaptiveThreshold = true;
            rp.adaptiveThresholdConfig = adaptive_cfg;
            rp.adaptiveThresholdMode = 'binary';
            rp.useWeightedFollow = wf;
            rp.weightMode = wm;
            R_tmp(ri) = run_responsiveness(rp, num_angles, time_vec, seed);

            pp = pers_params;
            pp.useAdaptiveThreshold = true;
            pp.adaptiveThresholdConfig = adaptive_cfg;
            pp.adaptiveThresholdMode = 'binary';
            pp.useWeightedFollow = wf;
            pp.weightMode = wm;
            P_tmp(ri) = run_persistence(pp, pers_cfg, seed + 50000);
        end

        R_all(:, mi, ei) = R_tmp;
        P_all(:, mi, ei) = P_tmp;

        if wf
            base_R = R_all(:, 1, ei);
            valid_R = ~isnan(R_tmp) & ~isnan(base_R);
            if any(valid_R) && all(abs(R_tmp(valid_R) - base_R(valid_R)) < 1e-12)
                error('eta=%.2f, mode=%s: R 与 Binary 逐项完全相同，说明 weighted follow 在响应性分支未生效。', ...
                    eta, m.name);
            end

            base_P = P_all(:, 1, ei);
            valid_P = ~isnan(P_tmp) & ~isnan(base_P);
            if any(valid_P) && all(abs(P_tmp(valid_P) - base_P(valid_P)) < 1e-12)
                error('eta=%.2f, mode=%s: P 与 Binary 逐项完全相同，说明 weighted follow 在持久性分支未生效。', ...
                    eta, m.name);
            end
        end

        fprintf('  R=%.4f+/-%.4f, P=%.2f+/-%.2f\n', ...
            mean(R_tmp,'omitnan'), std(R_tmp,'omitnan')/sqrt(num_runs), ...
            mean(P_tmp,'omitnan'), std(P_tmp,'omitnan')/sqrt(num_runs));
    end
    fprintf('\n');
end

%% 4. 汇总
fprintf('========== 结果汇总 ==========\n');
for ei = 1:num_eta
    eta = eta_values(ei);
    Rb = mean(R_all(:,1,ei),'omitnan');
    Pb = mean(P_all(:,1,ei),'omitnan');

    fprintf('\neta = %.2f (Binary baseline: R=%.4f, P=%.2f)\n', eta, Rb, Pb);
    fprintf('%-12s %8s %8s %8s %8s\n', '方法', 'R', 'P', 'dR%', 'dP%');
    fprintf('%s\n', repmat('-', 1, 52));
    for mi = 1:num_modes
        Rm = mean(R_all(:,mi,ei),'omitnan');
        Pm = mean(P_all(:,mi,ei),'omitnan');
        if mi == 1
            fprintf('%-12s %8.4f %8.2f %8s %8s\n', modes(mi).name, Rm, Pm, '-', '-');
        else
            dR = (Rm - Rb) / Rb * 100;
            dP = (Pm - Pb) / Pb * 100;
            fprintf('%-12s %8.4f %8.2f %+7.1f%% %+7.1f%%\n', modes(mi).name, Rm, Pm, dR, dP);
        end
    end
end
fprintf('\n');

%% 5. 图1：柱状图
fig1 = figure('Visible', 'off', 'Position', [50 50 1200 500], 'Color', 'w');

subplot(1,2,1);
R_means = squeeze(mean(R_all, 1, 'omitnan'))';  % [num_eta x num_modes]
bar(R_means);
set(gca, 'XTickLabel', string(eta_values));
xlabel('eta'); ylabel('R');
legend({modes.name}, 'Location', 'best');
title('R'); grid on;

subplot(1,2,2);
P_means = squeeze(mean(P_all, 1, 'omitnan'))';
bar(P_means);
set(gca, 'XTickLabel', string(eta_values));
xlabel('eta'); ylabel('P');
legend({modes.name}, 'Location', 'best');
title('P'); grid on;

sgtitle(sprintf('权重模式对比 (%d runs)', num_runs));
saveas(fig1, fullfile(out_dir, 'comparison_bar.png'));
close(fig1);

%% 6. 图2：R-P 散点图
colors = [0.8 0.2 0.2; 0.2 0.4 0.8; 0.1 0.7 0.3];
fig2 = figure('Visible', 'off', 'Position', [50 50 500*num_eta 500], 'Color', 'w');
for ei = 1:num_eta
    subplot(1, num_eta, ei); hold on;
    for mi = 1:num_modes
        scatter(R_all(:,mi,ei), P_all(:,mi,ei), 20, colors(mi,:), 'filled', 'MarkerFaceAlpha', 0.3, 'HandleVisibility', 'off');
        Rm = mean(R_all(:,mi,ei),'omitnan');
        Pm = mean(P_all(:,mi,ei),'omitnan');
        scatter(Rm, Pm, 200, colors(mi,:), 'p', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', modes(mi).name);
    end
    xlabel('R'); ylabel('P');
    title(sprintf('eta=%.2f', eta_values(ei)));
    legend('Location', 'best'); grid on;
end
saveas(fig2, fullfile(out_dir, 'comparison_rp.png'));
close(fig2);

%% 7. 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');
fprintf(fid, '========================================================================\n');
fprintf(fid, '权重模式对比报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '========================================================================\n\n');
fprintf(fid, '设置: N=%d, num_runs=%d, 共享种子\n', base.N, num_runs);
fprintf(fid, 'adaptive_cfg: cj_low=%.1f, cj_high=%.1f, saliency_thr=%.3f\n', ...
    adaptive_cfg.cj_low, adaptive_cfg.cj_high, adaptive_cfg.saliency_threshold);
fprintf(fid, '模式:\n');
fprintf(fid, '  Binary:   原始单源跟随 (5.2)\n');
fprintf(fid, '  Absolute: w_j = s_ij / Sigma(s)\n');
fprintf(fid, '  Excess:   w_j = (s_ij - M_T) / Sigma(s - M_T)\n\n');

for ei = 1:num_eta
    eta = eta_values(ei);
    Rb = mean(R_all(:,1,ei),'omitnan');
    Pb = mean(P_all(:,1,ei),'omitnan');

    fprintf(fid, 'eta = %.2f\n', eta);
    fprintf(fid, '%-12s %8s %8s %8s %8s\n', '方法', 'R', 'P', 'dR%', 'dP%');
    fprintf(fid, '%s\n', repmat('-', 1, 52));
    for mi = 1:num_modes
        Rm = mean(R_all(:,mi,ei),'omitnan');
        Pm = mean(P_all(:,mi,ei),'omitnan');
        if mi == 1
            fprintf(fid, '%-12s %8.4f %8.2f %8s %8s\n', modes(mi).name, Rm, Pm, '-', '-');
        else
            dR = (Rm - Rb) / Rb * 100;
            dP = (Pm - Pb) / Pb * 100;
            fprintf(fid, '%-12s %8.4f %8.2f %+7.1f%% %+7.1f%%\n', modes(mi).name, Rm, Pm, dR, dP);
        end
    end
    fprintf(fid, '\n');
end
fclose(fid);

%% 8. 保存数据
saveResultBundle(out_dir, 'comparison_data', ...
    {'R_all', 'P_all', 'modes', 'eta_values', ...
     'adaptive_cfg', 'base', 'num_runs', 'base_seed'});

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
