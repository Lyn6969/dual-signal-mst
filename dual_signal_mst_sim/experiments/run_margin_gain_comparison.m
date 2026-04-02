%% run_margin_gain_comparison.m
% Binary vs Margin-Only Gain 对比实验
% 验证 margin-only gain 是否能以可接受的 R 代价换取更强的误放大抑制

clc; clear; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 参数
fprintf('=================================================\n');
fprintf('  Binary vs Margin-Only Gain Pilot\n');
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
num_runs = 20;
num_angles = 1;
base_seed = 20260402;

pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_diffusion = 1e-4;
pers_cfg.min_fit_points = 40;

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5.0;
adaptive_cfg.saliency_threshold = 0.031;
adaptive_cfg.include_self = false;

% 固定 5 个模式
modes = struct();
modes(1).name = 'Binary';
modes(1).useResponseGainModulation = false;
modes(1).responseGainMode = 'legacy_q';
modes(1).responseGainApply = 'all_active';
modes(1).responseGainC = NaN;

modes(2).name = 'MarginAll_c2.5';
modes(2).useResponseGainModulation = true;
modes(2).responseGainMode = 'margin_only';
modes(2).responseGainApply = 'all_active';
modes(2).responseGainC = 2.5;

modes(3).name = 'MarginAll_c3.0';
modes(3).useResponseGainModulation = true;
modes(3).responseGainMode = 'margin_only';
modes(3).responseGainApply = 'all_active';
modes(3).responseGainC = 3.0;

modes(4).name = 'MarginFirst_c2.5';
modes(4).useResponseGainModulation = true;
modes(4).responseGainMode = 'margin_only';
modes(4).responseGainApply = 'first_hop';
modes(4).responseGainC = 2.5;

modes(5).name = 'MarginFirst_c3.0';
modes(5).useResponseGainModulation = true;
modes(5).responseGainMode = 'margin_only';
modes(5).responseGainApply = 'first_hop';
modes(5).responseGainC = 3.0;

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
out_dir = fullfile(results_dir, sprintf('margin_gain_%s', timestamp));
mkdir(out_dir);

fprintf('eta = %s, num_runs = %d\n', mat2str(eta_values), num_runs);
fprintf('模式:\n');
for mi = 1:num_modes
    m = modes(mi);
    fprintf('  %-16s useGain=%d, mode=%s, apply=%s, c=%s\n', ...
        m.name, m.useResponseGainModulation, ...
        m.responseGainMode, m.responseGainApply, num2str(m.responseGainC));
end
fprintf('\n');

%% 3. 主循环
num_eta = numel(eta_values);
R_all = NaN(num_runs, num_modes, num_eta);
P_all = NaN(num_runs, num_modes, num_eta);
episode_count_all = NaN(num_runs, num_modes, num_eta);
max_false_episode_peak_all = NaN(num_runs, num_modes, num_eta);
max_false_episode_duration_all = NaN(num_runs, num_modes, num_eta);

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
        episode_tmp = NaN(num_runs, 1);
        peak_tmp = NaN(num_runs, 1);
        duration_tmp = NaN(num_runs, 1);

        use_gain = m.useResponseGainModulation;
        gain_mode = m.responseGainMode;
        gain_apply = m.responseGainApply;
        gain_c = m.responseGainC;

        parfor ri = 1:num_runs
            seed = base_seed + ri;

            rp = resp_params;
            rp.useAdaptiveThreshold = true;
            rp.adaptiveThresholdConfig = adaptive_cfg;
            rp.adaptiveThresholdMode = 'binary';
            rp.useWeightedFollow = false;
            rp.useResponseGainModulation = use_gain;
            rp.responseGainMode = gain_mode;
            rp.responseGainApply = gain_apply;
            if ~isnan(gain_c)
                rp.responseGainC = gain_c;
            end
            R_tmp(ri) = run_responsiveness(rp, num_angles, time_vec, seed);

            pp = pers_params;
            pp.useAdaptiveThreshold = true;
            pp.adaptiveThresholdConfig = adaptive_cfg;
            pp.adaptiveThresholdMode = 'binary';
            pp.useWeightedFollow = false;
            pp.useResponseGainModulation = use_gain;
            pp.responseGainMode = gain_mode;
            pp.responseGainApply = gain_apply;
            if ~isnan(gain_c)
                pp.responseGainC = gain_c;
            end
            pers_res = run_persistence_with_false_episodes(pp, pers_cfg, seed + 50000);
            P_tmp(ri) = pers_res.P;
            episode_tmp(ri) = pers_res.episode_count;
            peak_tmp(ri) = pers_res.max_false_episode_peak;
            duration_tmp(ri) = pers_res.max_false_episode_duration;
        end

        R_all(:, mi, ei) = R_tmp;
        P_all(:, mi, ei) = P_tmp;
        episode_count_all(:, mi, ei) = episode_tmp;
        max_false_episode_peak_all(:, mi, ei) = peak_tmp;
        max_false_episode_duration_all(:, mi, ei) = duration_tmp;

        fprintf('  R=%.4f, P=%.2f, episodes=%.2f, peak=%.2f, duration=%.2f\n', ...
            mean(R_tmp,'omitnan'), mean(P_tmp,'omitnan'), ...
            mean(episode_tmp,'omitnan'), mean(peak_tmp,'omitnan'), ...
            mean(duration_tmp,'omitnan'));
    end
    fprintf('\n');
end

%% 4. 汇总
fprintf('========== 结果汇总 ==========\n');
for ei = 1:num_eta
    eta = eta_values(ei);
    Rb = mean(R_all(:,1,ei),'omitnan');
    Pb = mean(P_all(:,1,ei),'omitnan');
    Eb = mean(episode_count_all(:,1,ei),'omitnan');
    Kb = mean(max_false_episode_peak_all(:,1,ei),'omitnan');
    Db = mean(max_false_episode_duration_all(:,1,ei),'omitnan');

    fprintf('\neta = %.2f\n', eta);
    fprintf('%-16s %8s %8s %8s %8s %8s\n', '方法', 'R', 'P', 'episodes', 'peak', 'duration');
    fprintf('%s\n', repmat('-', 1, 72));
    for mi = 1:num_modes
        Rm = mean(R_all(:,mi,ei),'omitnan');
        Pm = mean(P_all(:,mi,ei),'omitnan');
        Em = mean(episode_count_all(:,mi,ei),'omitnan');
        Km = mean(max_false_episode_peak_all(:,mi,ei),'omitnan');
        Dm = mean(max_false_episode_duration_all(:,mi,ei),'omitnan');
        if mi == 1
            fprintf('%-16s %8.4f %8.2f %8.2f %8.2f %8.2f\n', ...
                modes(mi).name, Rm, Pm, Em, Km, Dm);
        else
            fprintf('%-16s %8.4f %8.2f %8.2f %8.2f %8.2f\n', ...
                modes(mi).name, Rm, Pm, Em, Km, Dm);
        end
    end
end
fprintf('\n');

%% 5. 图1：柱状图（5 指标）
fig1 = figure('Visible', 'off', 'Position', [50 50 1600 900], 'Color', 'w');
metric_titles = {'R', 'P', 'False Episodes', 'Max False Peak', 'Max False Duration'};
metric_cells = {R_all, P_all, episode_count_all, max_false_episode_peak_all, max_false_episode_duration_all};

for k = 1:5
    subplot(2,3,k);
    metric_means = squeeze(mean(metric_cells{k}, 1, 'omitnan'))';
    bar(metric_means);
    set(gca, 'XTickLabel', string(eta_values));
    xlabel('eta');
    ylabel(metric_titles{k});
    title(metric_titles{k});
    if k == 1
        legend({modes.name}, 'Location', 'best');
    end
    grid on;
end

sgtitle(sprintf('Margin-Only Gain Pilot (%d runs)', num_runs));
saveas(fig1, fullfile(out_dir, 'comparison_bar.png'));
close(fig1);

%% 6. 图2：R-P 散点图
colors = lines(num_modes);
fig2 = figure('Visible', 'off', 'Position', [50 50 500*num_eta 500], 'Color', 'w');
for ei = 1:num_eta
    subplot(1, num_eta, ei); hold on;
    for mi = 1:num_modes
        scatter(R_all(:,mi,ei), P_all(:,mi,ei), 20, colors(mi,:), ...
            'filled', 'MarkerFaceAlpha', 0.3, 'HandleVisibility', 'off');
        Rm = mean(R_all(:,mi,ei),'omitnan');
        Pm = mean(P_all(:,mi,ei),'omitnan');
        scatter(Rm, Pm, 200, colors(mi,:), 'p', 'filled', ...
            'MarkerEdgeColor', 'k', 'DisplayName', modes(mi).name);
    end
    xlabel('R'); ylabel('P');
    title(sprintf('eta=%.2f', eta_values(ei)));
    legend('Location', 'best');
    grid on;
end
saveas(fig2, fullfile(out_dir, 'comparison_rp.png'));
close(fig2);

%% 7. 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');
fprintf(fid, '========================================================================\n');
fprintf(fid, 'Margin-Only Gain Pilot 报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '========================================================================\n\n');

fprintf(fid, '设置: N=%d, num_runs=%d, eta=%s, 共享种子\n', ...
    base.N, num_runs, mat2str(eta_values));
fprintf(fid, 'adaptive_cfg: cj_low=%.1f, cj_high=%.1f, saliency_thr=%.3f\n\n', ...
    adaptive_cfg.cj_low, adaptive_cfg.cj_high, adaptive_cfg.saliency_threshold);

fprintf(fid, '模式:\n');
for mi = 1:num_modes
    m = modes(mi);
    fprintf(fid, '  %-16s gain=%d, mode=%s, apply=%s, c=%s\n', ...
        m.name, m.useResponseGainModulation, ...
        m.responseGainMode, m.responseGainApply, num2str(m.responseGainC));
end
fprintf(fid, '\n');

for ei = 1:num_eta
    eta = eta_values(ei);
    Rb = mean(R_all(:,1,ei),'omitnan');
    Pb = mean(P_all(:,1,ei),'omitnan');
    Kb = mean(max_false_episode_peak_all(:,1,ei),'omitnan');
    Db = mean(max_false_episode_duration_all(:,1,ei),'omitnan');

    fprintf(fid, 'eta = %.2f\n', eta);
    fprintf(fid, '%-16s %8s %8s %8s %8s %8s %8s %8s\n', ...
        '方法', 'R', 'P', 'dR%', 'dP%', 'peak', 'dPeak%', 'dDur%');
    fprintf(fid, '%s\n', repmat('-', 1, 90));
    for mi = 1:num_modes
        Rm = mean(R_all(:,mi,ei),'omitnan');
        Pm = mean(P_all(:,mi,ei),'omitnan');
        Km = mean(max_false_episode_peak_all(:,mi,ei),'omitnan');
        Dm = mean(max_false_episode_duration_all(:,mi,ei),'omitnan');

        if mi == 1
            fprintf(fid, '%-16s %8.4f %8.2f %8s %8s %8.2f %8s %8s\n', ...
                modes(mi).name, Rm, Pm, '-', '-', Km, '-', '-');
        else
            dR = 100 * (Rm - Rb) / max(abs(Rb), eps);
            dP = 100 * (Pm - Pb) / max(abs(Pb), eps);
            dPeak = 100 * (Kb - Km) / max(abs(Kb), eps);
            dDur = 100 * (Db - Dm) / max(abs(Db), eps);
            fprintf(fid, '%-16s %8.4f %8.2f %+7.1f%% %+7.1f%% %8.2f %+7.1f%% %+7.1f%%\n', ...
                modes(mi).name, Rm, Pm, dR, dP, Km, dPeak, dDur);
        end
    end
    fprintf(fid, '\n');
end

fprintf(fid, '判据建议:\n');
fprintf(fid, '  - 若任一 MarginFirst_* 模式满足 ΔR >= -8%% 且 dPeak 或 dDur >= +20%%，则值得扩大到 50 runs。\n');
fprintf(fid, '  - 若 MarginAll_* 在两个 eta 下均表现为更差的 R 且误放大指标改善有限，则可停止该分支。\n');
fclose(fid);

%% 8. 保存数据
saveResultBundle(out_dir, 'comparison_data', ...
    {'R_all', 'P_all', 'episode_count_all', 'max_false_episode_peak_all', ...
     'max_false_episode_duration_all', 'modes', 'eta_values', ...
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
    if duration > 0
        R = integral_val / (params.v0 * duration);
    else
        R = NaN;
    end
end

function result = run_persistence_with_false_episodes(params, cfg, seed)
    rng(seed);
    sim = ParticleSimulation(params);
    T = sim.T_max;
    burn_in = max(2, floor((T+1) * cfg.burn_in_ratio));

    init_pos = sim.positions;
    centroid0 = mean(init_pos, 1);
    offsets0 = init_pos - centroid0;
    msd = zeros(T+1, 1);
    active_trace = zeros(T, 1);

    for t = 1:T
        sim.step();
        positions = sim.positions;
        centroid = mean(positions, 1);
        rel_disp = (positions - centroid) - offsets0;
        msd(t+1) = mean(sum(rel_disp.^2, 2));
        active_trace(t) = sum(sim.isActive);
    end

    time_vec = (0:T)' * sim.dt;
    x = time_vec(burn_in:end);
    y = msd(burn_in:end);

    if numel(x) < max(2, cfg.min_fit_points) || all(abs(y - y(1)) < eps)
        D = NaN;
    else
        x_s = x - x(1);
        y_s = y - y(1);
        if any(x_s > 0) && any(abs(y_s) > eps)
            sw = max(5, floor(numel(y_s) * 0.1));
            if sw > 1
                y_s = smoothdata(y_s, 'movmean', sw);
            end
            slope = lsqnonneg(x_s(:), y_s(:));
            D = max(slope / 4, cfg.min_diffusion);
        else
            D = NaN;
        end
    end

    if isnan(D)
        P = NaN;
    else
        P = 1 / sqrt(D);
    end

    [episode_count, max_peak, max_duration] = summarizeFalseEpisodes(active_trace);

    result = struct();
    result.P = P;
    result.episode_count = episode_count;
    result.max_false_episode_peak = max_peak;
    result.max_false_episode_duration = max_duration;
end

function [episode_count, max_peak, max_duration] = summarizeFalseEpisodes(active_trace)
    is_active = active_trace > 0;
    episode_count = 0;
    max_peak = 0;
    max_duration = 0;

    t = 1;
    T = numel(active_trace);
    while t <= T
        if ~is_active(t)
            t = t + 1;
            continue;
        end

        t_start = t;
        while t <= T && is_active(t)
            t = t + 1;
        end
        t_end = t - 1;

        episode_count = episode_count + 1;
        max_peak = max(max_peak, max(active_trace(t_start:t_end)));
        max_duration = max(max_duration, t_end - t_start + 1);
    end
end
