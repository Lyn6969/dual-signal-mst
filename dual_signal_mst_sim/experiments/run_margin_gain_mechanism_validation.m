%% run_margin_gain_mechanism_validation.m
% Margin-Only Gain 机制验证实验
%
% 目标：
%   1. 验证真实信号场景下，src_id 对应显著性在激活后持续保持高值；
%      纯噪声误激活场景下，src_id 显著性在首步后快速回落。
%   2. 验证 margin-only gain 会显著压低噪声场景中新激活粒子的实际转向幅度，
%      同时尽量不改变信号场景下的首步转向分布。
%
% 输出（results/margin_gain_mechanism_<timestamp>/）：
%   - analysis_report.txt
%   - fig1_src_saliency_decay.png
%   - fig2_turn_amplitude_distribution.png
%   - fig3_margin_turn_relation.png
%   - mechanism_data.mat
%   - mechanism_data.json

clear; clc; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 参数
fprintf('=================================================\n');
fprintf('  Margin-Only Gain 机制验证实验\n');
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

eta_values = [0.30];
num_runs = 30;
tau_horizon = 5;
base_seed = 20260402;

pulse_params = base;
pulse_params.T_max = 600;
pulse_params.stabilization_steps = 200;
pulse_params.forced_turn_duration = 400;

noise_params = base;
noise_params.T_max = 600;
noise_burn_in = 200;

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5.0;
adaptive_cfg.saliency_threshold = 0.031;
adaptive_cfg.include_self = false;

modes = struct();
modes(1).field = 'Binary';
modes(1).name = 'Binary';
modes(1).useResponseGainModulation = false;
modes(1).responseGainMode = 'legacy_q';
modes(1).responseGainApply = 'all_active';
modes(1).responseGainC = NaN;

modes(2).field = 'MarginAll_c3_0';
modes(2).name = 'MarginAll_c3.0';
modes(2).useResponseGainModulation = true;
modes(2).responseGainMode = 'margin_only';
modes(2).responseGainApply = 'all_active';
modes(2).responseGainC = 3.0;

scene_names = {'signal', 'noise'};

%% 2. 并行池
use_parallel = false;
try
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool; end
    use_parallel = true;
    fprintf('并行池：%d workers\n\n', pool.NumWorkers);
catch
    fprintf('并行池启动失败，使用串行模式\n\n');
end

%% 3. 输出目录
results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_dir = fullfile(results_dir, sprintf('margin_gain_mechanism_%s', timestamp));
mkdir(out_dir);

fprintf('eta = %s, num_runs = %d, tau_horizon = %d\n', ...
    mat2str(eta_values), num_runs, tau_horizon);
fprintf('模式: %s vs %s\n\n', modes(1).name, modes(2).name);

%% 4. 主循环
num_eta = numel(eta_values);
mechanism_results = cell(num_eta, 1);

for ei = 1:num_eta
    eta = eta_values(ei);
    fprintf('========== eta = %.2f ==========\n', eta);

    eta_result = struct();
    eta_result.eta = eta;
    eta_result.signal = struct();
    eta_result.noise = struct();

    for si = 1:numel(scene_names)
        scene_name = scene_names{si};
        fprintf('[scene=%s]\n', scene_name);

        for mi = 1:numel(modes)
            mode = modes(mi);
            fprintf('  [%s] %d runs ...\n', mode.name, num_runs);

            event_runs = cell(num_runs, 1);
            bg_turn_runs = cell(num_runs, 1);
            active_count_runs = cell(num_runs, 1);

            if use_parallel
                parfor ri = 1:num_runs
                    seed = base_seed + ri;
                    run_out = runSingleMechanismExperiment( ...
                        pulse_params, noise_params, adaptive_cfg, ...
                        eta, scene_name, mode, seed, tau_horizon, noise_burn_in);
                    event_runs{ri} = run_out.events;
                    bg_turn_runs{ri} = run_out.mean_abs_turn;
                    active_count_runs{ri} = run_out.activated_counts;
                end
            else
                for ri = 1:num_runs
                    seed = base_seed + ri;
                    run_out = runSingleMechanismExperiment( ...
                        pulse_params, noise_params, adaptive_cfg, ...
                        eta, scene_name, mode, seed, tau_horizon, noise_burn_in);
                    event_runs{ri} = run_out.events;
                    bg_turn_runs{ri} = run_out.mean_abs_turn;
                    active_count_runs{ri} = run_out.activated_counts;
                    fprintf('    run %d/%d 完成\n', ri, num_runs);
                end
            end

            merged_events = mergeEventRuns(event_runs);
            scene_struct = struct();
            scene_struct.events = merged_events;
            scene_struct.mean_abs_turn_runs = bg_turn_runs;
            scene_struct.activated_counts_runs = active_count_runs;
            scene_struct.summary = summarizeEvents(merged_events, tau_horizon);

            eta_result.(scene_name).(mode.field) = scene_struct;

            fprintf('    事件数 = %d\n', numel(merged_events));
        end
        fprintf('\n');
    end

    mechanism_results{ei} = eta_result;
end

%% 5. 图1：src_id 显著性持续性
fig1 = figure('Visible', 'off', 'Position', [50 50 700*num_eta 500], 'Color', 'w');
tiledlayout(1, num_eta, 'Padding', 'compact', 'TileSpacing', 'compact');

for ei = 1:num_eta
    nexttile; hold on;
    eta_result = mechanism_results{ei};
    plotNormalizedSaliencyCurve(eta_result.signal.Binary.events, 'r-', 'Binary-signal');
    plotNormalizedSaliencyCurve(eta_result.noise.Binary.events, 'r--', 'Binary-noise');
    plotNormalizedSaliencyCurve(eta_result.signal.MarginAll_c3_0.events, 'b-', 'Margin-signal');
    plotNormalizedSaliencyCurve(eta_result.noise.MarginAll_c3_0.events, 'b--', 'Margin-noise');
    xlabel('\tau (激活后步数)');
    ylabel('s_{src}(\tau) / s_{src}(0)');
    title(sprintf('\\eta = %.2f', eta_result.eta));
    legend('Location', 'best');
    grid on;
end

sgtitle('src\_id 显著性衰减曲线');
saveas(fig1, fullfile(out_dir, 'fig1_src_saliency_decay.png'));
close(fig1);

%% 6. 图2：首步转向幅度分布
fig2 = figure('Visible', 'off', 'Position', [50 50 700*num_eta 800], 'Color', 'w');
tiledlayout(2, num_eta, 'Padding', 'compact', 'TileSpacing', 'compact');

for ei = 1:num_eta
    eta_result = mechanism_results{ei};

    nexttile; hold on;
    plotTurnHistogram(eta_result.signal.Binary.events, [0.8 0.2 0.2], 'Binary');
    plotTurnHistogram(eta_result.signal.MarginAll_c3_0.events, [0.2 0.4 0.8], 'Margin');
    xlabel('turn_{\tau=0} (deg)');
    ylabel('概率');
    title(sprintf('Signal, \\eta=%.2f', eta_result.eta));
    legend('Location', 'best');
    grid on;

    nexttile; hold on;
    plotTurnHistogram(eta_result.noise.Binary.events, [0.8 0.2 0.2], 'Binary');
    plotTurnHistogram(eta_result.noise.MarginAll_c3_0.events, [0.2 0.4 0.8], 'Margin');
    xlabel('turn_{\tau=0} (deg)');
    ylabel('概率');
    title(sprintf('Noise, \\eta=%.2f', eta_result.eta));
    legend('Location', 'best');
    grid on;
end

sgtitle('新激活粒子的首步实际转向幅度分布');
saveas(fig2, fullfile(out_dir, 'fig2_turn_amplitude_distribution.png'));
close(fig2);

%% 7. 图3：margin 与首步转向关系
fig3 = figure('Visible', 'off', 'Position', [50 50 700*num_eta 800], 'Color', 'w');
tiledlayout(2, num_eta, 'Padding', 'compact', 'TileSpacing', 'compact');

for ei = 1:num_eta
    eta_result = mechanism_results{ei};

    nexttile; hold on;
    plotMarginTurnRelation(eta_result.signal.Binary.events, [0.8 0.2 0.2], 'Binary');
    plotMarginTurnRelation(eta_result.signal.MarginAll_c3_0.events, [0.2 0.4 0.8], 'Margin');
    xlabel('relative excess margin_{\tau=0} = (s_{src}-c_{low})/c_{low}');
    ylabel('turn_{\tau=0} (deg)');
    title(sprintf('Signal, \\eta=%.2f', eta_result.eta));
    legend('Location', 'best');
    grid on;

    nexttile; hold on;
    plotMarginTurnRelation(eta_result.noise.Binary.events, [0.8 0.2 0.2], 'Binary');
    plotMarginTurnRelation(eta_result.noise.MarginAll_c3_0.events, [0.2 0.4 0.8], 'Margin');
    xlabel('relative excess margin_{\tau=0} = (s_{src}-c_{low})/c_{low}');
    ylabel('turn_{\tau=0} (deg)');
    title(sprintf('Noise, \\eta=%.2f', eta_result.eta));
    legend('Location', 'best');
    grid on;
end

sgtitle('margin 与首步转向幅度关系');
saveas(fig3, fullfile(out_dir, 'fig3_margin_turn_relation.png'));
close(fig3);

%% 8. 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');

fprintf(fid, '========================================================================\n');
fprintf(fid, 'Margin-Only Gain 机制验证报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '========================================================================\n\n');

fprintf(fid, '设置:\n');
fprintf(fid, '  eta = %s, num_runs = %d, tau_horizon = %d\n', ...
    mat2str(eta_values), num_runs, tau_horizon);
fprintf(fid, '  比较对象 = Binary vs MarginAll_c3.0\n');
fprintf(fid, '  Signal 场景 = 外源脉冲触发后首次级联激活\n');
fprintf(fid, '  Noise 场景 = burn-in 后纯噪声首次误激活\n');
fprintf(fid, '  adaptive_cfg: cj_low=%.1f, cj_high=%.1f, saliency_thr=%.3f\n\n', ...
    adaptive_cfg.cj_low, adaptive_cfg.cj_high, adaptive_cfg.saliency_threshold);

for ei = 1:num_eta
    eta_result = mechanism_results{ei};
    fprintf(fid, 'eta = %.2f\n', eta_result.eta);
    fprintf(fid, '%-12s %-8s %8s %8s %8s %8s %8s %8s\n', ...
        'scene', 'mode', 'n_evt', 'rho1_m', 'rho1_md', 'Asrc_m', 'turn_m', 'turn_p95');

    reportSceneMode(fid, 'signal', 'Binary', eta_result.signal.Binary.summary);
    reportSceneMode(fid, 'signal', 'Margin', eta_result.signal.MarginAll_c3_0.summary);
    reportSceneMode(fid, 'noise', 'Binary', eta_result.noise.Binary.summary);
    reportSceneMode(fid, 'noise', 'Margin', eta_result.noise.MarginAll_c3_0.summary);

    fprintf(fid, '\n  机制判读:\n');
    printMechanismInterpretation(fid, eta_result);
    fprintf(fid, '\n');
end

fclose(fid);

%% 9. 保存数据
payload = struct();
payload.mechanism_results = mechanism_results;
payload.eta_values = eta_values;
payload.num_runs = num_runs;
payload.tau_horizon = tau_horizon;
payload.adaptive_cfg = adaptive_cfg;
payload.base = base;
payload.pulse_params = pulse_params;
payload.noise_params = noise_params;
payload.noise_burn_in = noise_burn_in;
saveMechanismBundle(out_dir, payload);

fprintf('完成。输出目录: %s\n', out_dir);

%% ========================================================================
function run_out = runSingleMechanismExperiment(pulse_params, noise_params, adaptive_cfg, eta, scene_name, mode, seed, tau_horizon, noise_burn_in)
    rng(seed);

    if strcmp(scene_name, 'signal')
        params = pulse_params;
        params.angleNoiseIntensity = eta^2 / 2;
        params.useAdaptiveThreshold = true;
        params.adaptiveThresholdConfig = adaptive_cfg;
        params.adaptiveThresholdMode = 'binary';
        params.useWeightedFollow = false;
        params.useResponseGainModulation = mode.useResponseGainModulation;
        params.responseGainMode = mode.responseGainMode;
        params.responseGainApply = mode.responseGainApply;
        if ~isnan(mode.responseGainC)
            params.responseGainC = mode.responseGainC;
        end

        sim = ParticleSimulationWithExternalPulse(params);
        sim.external_pulse_count = 1;
        sim.setLogging(false);
        sim.resetCascadeTracking();
        sim.initializeParticles();
        start_collect_step = params.stabilization_steps + 1;
    else
        params = noise_params;
        params.angleNoiseIntensity = eta^2 / 2;
        params.useAdaptiveThreshold = true;
        params.adaptiveThresholdConfig = adaptive_cfg;
        params.adaptiveThresholdMode = 'binary';
        params.useWeightedFollow = false;
        params.useResponseGainModulation = mode.useResponseGainModulation;
        params.responseGainMode = mode.responseGainMode;
        params.responseGainApply = mode.responseGainApply;
        if ~isnan(mode.responseGainC)
            params.responseGainC = mode.responseGainC;
        end

        sim = ParticleSimulation(params);
        start_collect_step = noise_burn_in + 1;
    end

    N = params.N;
    T = params.T_max;

    theta_prev = sim.theta;
    external_ids = false(N, 1);
    seen_activation = false(N, 1);
    ongoing = struct('particle_id', {}, 't_act', {}, 'scene', {}, 'mode', {}, ...
        'run_id', {}, 'src_id', {}, 's_src', {}, 'margin', {}, 'gain', {}, 'turn', {});
    completed_events = struct('particle_id', {}, 't_act', {}, 'scene', {}, 'mode', {}, ...
        'run_id', {}, 'src_id', {}, 's_src', {}, 'margin', {}, 'gain', {}, 'turn', {});

    mean_abs_turn = NaN(T, 1);
    activated_counts = NaN(T, 1);

    for t = 1:T
        was_active = sim.isActive;
        sim.step();

        if strcmp(scene_name, 'signal')
            external_ids = external_ids | sim.isExternallyActivated;
        end

        theta_now = sim.theta;
        turn_step = abs(wrapToPiLocal(theta_now - theta_prev));
        mean_abs_turn(t) = mean(turn_step);
        activated_counts(t) = sum(sim.isActive);
        neighbor_matrix = sim.findNeighbors();

        % 更新所有正在跟踪的事件
        ongoing_idx_to_keep = true(1, numel(ongoing));
        for oi = 1:numel(ongoing)
            tau = t - ongoing(oi).t_act;
            if tau < 1
                continue;
            end
            if tau > tau_horizon
                completed_events(end+1) = ongoing(oi); %#ok<AGROW>
                ongoing_idx_to_keep(oi) = false;
                continue;
            end

            pid = ongoing(oi).particle_id;
            [s_src, margin_val, gain_val] = computeSourceAlignedStats(sim, pid, neighbor_matrix);
            if isnan(s_src)
                ongoing(oi).s_src(tau+1) = 0;
                ongoing(oi).margin(tau+1) = 0;
                ongoing(oi).gain(tau+1) = 0;
                ongoing(oi).turn(tau+1) = 0;
            else
                ongoing(oi).s_src(tau+1) = s_src;
                ongoing(oi).margin(tau+1) = margin_val;
                ongoing(oi).gain(tau+1) = gain_val;
                ongoing(oi).turn(tau+1) = turn_step(pid);
            end
        end
        ongoing = ongoing(ongoing_idx_to_keep);

        % 记录新激活事件
        newly_activated = sim.isActive & ~was_active;
        if strcmp(scene_name, 'signal')
            newly_activated = newly_activated & ~external_ids;
        end
        if t >= start_collect_step
            record_mask = newly_activated & ~seen_activation;
            seen_activation(record_mask) = true;
            for pid = find(record_mask)'
                [s_src, margin_val, gain_val] = computeSourceAlignedStats(sim, pid, neighbor_matrix);
                if isnan(s_src)
                    continue;
                end
                ev = struct();
                ev.run_id = seed;
                ev.scene = scene_name;
                ev.mode = mode.name;
                ev.particle_id = pid;
                ev.t_act = t;
                ev.src_id = sim.src_ids{pid};
                ev.s_src = NaN(1, tau_horizon + 1);
                ev.margin = NaN(1, tau_horizon + 1);
                ev.gain = NaN(1, tau_horizon + 1);
                ev.turn = NaN(1, tau_horizon + 1);
                ev.s_src(1) = s_src;
                ev.margin(1) = margin_val;
                ev.gain(1) = gain_val;
                ev.turn(1) = turn_step(pid);
                ongoing(end+1) = ev; %#ok<AGROW>
            end
        end

        theta_prev = theta_now;
    end

    % 收尾：把未完成窗口的事件也并入结果
    for oi = 1:numel(ongoing)
        completed_events(end+1) = ongoing(oi); %#ok<AGROW>
    end

    run_out = struct();
    run_out.events = completed_events;
    run_out.mean_abs_turn = mean_abs_turn;
    run_out.activated_counts = activated_counts;
end

function [s_src, margin_val, gain_val] = computeSourceAlignedStats(sim, pid, neighbor_matrix)
    if pid < 1 || pid > sim.N || ~sim.isActive(pid)
        s_src = NaN;
        margin_val = NaN;
        gain_val = NaN;
        return;
    end

    src = sim.src_ids{pid};
    if isempty(src)
        s_src = NaN;
        margin_val = NaN;
        gain_val = NaN;
        return;
    end

    neighbor_idx = find(neighbor_matrix(pid, :));
    if ~ismember(src, neighbor_idx)
        s_src = NaN;
        margin_val = NaN;
        gain_val = NaN;
        return;
    end

    s_values = sim.computeSaliencyValues(pid, neighbor_idx);
    src_pos = find(neighbor_idx == src, 1, 'first');
    s_src = s_values(src_pos);

    threshold_ref = sim.getResponseGainReferenceThreshold();
    margin_val = (s_src - threshold_ref) / max(threshold_ref, eps);
    gain_val = computeGainFromPrecomputedSaliency(sim, pid, neighbor_idx, s_values, src_pos, s_src);
end

function merged = mergeEventRuns(event_runs)
    merged = struct('run_id', {}, 'scene', {}, 'mode', {}, 'particle_id', {}, ...
        't_act', {}, 'src_id', {}, 's_src', {}, 'margin', {}, 'gain', {}, 'turn', {});
    for i = 1:numel(event_runs)
        ev = event_runs{i};
        if isempty(ev)
            continue;
        end
        merged = [merged, ev]; %#ok<AGROW>
    end
end

function summary = summarizeEvents(events, tau_horizon)
    summary = struct();
    summary.n_events = numel(events);

    if isempty(events)
        summary.rho1_mean = NaN;
        summary.rho1_median = NaN;
        summary.Asrc_mean = NaN;
        summary.Asrc_median = NaN;
        summary.turn0_mean = NaN;
        summary.turn0_median = NaN;
        summary.turn0_p95 = NaN;
        return;
    end

    s_mat = vertcat(events.s_src);
    turn_mat = vertcat(events.turn);

    rho1 = s_mat(:,2) ./ s_mat(:,1);
    rho1 = rho1(isfinite(rho1));

    norm_s = s_mat ./ s_mat(:,1);
    Asrc = sum(norm_s, 2, 'omitnan');

    turn0 = rad2deg(turn_mat(:,1));
    turn0 = turn0(isfinite(turn0));

    summary.rho1_mean = safeMean(rho1);
    summary.rho1_median = safeMedian(rho1);
    summary.Asrc_mean = safeMean(Asrc);
    summary.Asrc_median = safeMedian(Asrc);
    summary.turn0_mean = safeMean(turn0);
    summary.turn0_median = safeMedian(turn0);
    summary.turn0_p95 = safePercentile(turn0, 95);
end

function plotNormalizedSaliencyCurve(events, line_spec, display_name)
    if isempty(events)
        return;
    end
    s_mat = vertcat(events.s_src);
    norm_s = s_mat ./ s_mat(:,1);
    x = 0:(size(norm_s,2)-1);
    y = mean(norm_s, 1, 'omitnan');
    n = sum(isfinite(norm_s), 1);
    sem = std(norm_s, 0, 1, 'omitnan') ./ max(sqrt(n), 1);
    plot(x, y, line_spec, 'LineWidth', 1.8, 'DisplayName', display_name);
    fill([x, fliplr(x)], [y-sem, fliplr(y+sem)], ...
        getColorFromSpec(line_spec), 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

function plotTurnHistogram(events, color, display_name)
    if isempty(events)
        return;
    end
    turn0 = rad2deg(vertcat(events.turn));
    turn0 = turn0(:,1);
    turn0 = turn0(isfinite(turn0));
    if isempty(turn0)
        return;
    end
    histogram(turn0, 28, 'Normalization', 'probability', ...
        'FaceColor', color, 'FaceAlpha', 0.45, 'DisplayName', display_name);
    xline(median(turn0), '--', sprintf('%s median', display_name), ...
        'Color', color, 'LineWidth', 1.2, 'HandleVisibility', 'off');
end

function plotMarginTurnRelation(events, color, display_name)
    if isempty(events)
        return;
    end
    margin0 = vertcat(events.margin);
    turn0 = rad2deg(vertcat(events.turn));
    margin0 = margin0(:,1);
    turn0 = turn0(:,1);
    valid = isfinite(margin0) & isfinite(turn0);
    margin0 = margin0(valid);
    turn0 = turn0(valid);
    if isempty(margin0)
        return;
    end
    scatter(margin0, turn0, 10, color, 'filled', 'MarkerFaceAlpha', 0.18, 'HandleVisibility', 'off');

    edges = quantile(margin0, linspace(0, 1, 7));
    edges = unique(edges);
    if numel(edges) < 3
        plot(margin0, turn0, '.', 'Color', color, 'DisplayName', display_name);
        return;
    end

    bin_center = [];
    bin_mean = [];
    for bi = 1:numel(edges)-1
        if bi < numel(edges)-1
            mask = margin0 >= edges(bi) & margin0 < edges(bi+1);
        else
            mask = margin0 >= edges(bi) & margin0 <= edges(bi+1);
        end
        if sum(mask) < 3
            continue;
        end
        bin_center(end+1) = mean(margin0(mask)); %#ok<AGROW>
        bin_mean(end+1) = mean(turn0(mask)); %#ok<AGROW>
    end
    plot(bin_center, bin_mean, 'o-', 'Color', color, 'LineWidth', 1.8, 'DisplayName', display_name);
end

function reportSceneMode(fid, scene_name, mode_name, summary)
    fprintf(fid, '%-12s %-8s %8d %8.3f %8.3f %8.3f %8.2f %8.2f\n', ...
        scene_name, mode_name, summary.n_events, ...
        summary.rho1_mean, summary.rho1_median, ...
        summary.Asrc_mean, summary.turn0_mean, summary.turn0_p95);
end

function printMechanismInterpretation(fid, eta_result)
    sig_bin = eta_result.signal.Binary.summary;
    noi_bin = eta_result.noise.Binary.summary;
    sig_mg = eta_result.signal.MarginAll_c3_0.summary;
    noi_mg = eta_result.noise.MarginAll_c3_0.summary;

    fprintf(fid, '  Binary: signal rho1=%.3f, noise rho1=%.3f; signal turn0=%.2f°, noise turn0=%.2f°\n', ...
        sig_bin.rho1_mean, noi_bin.rho1_mean, sig_bin.turn0_mean, noi_bin.turn0_mean);
    fprintf(fid, '  Margin: signal rho1=%.3f, noise rho1=%.3f; signal turn0=%.2f°, noise turn0=%.2f°\n', ...
        sig_mg.rho1_mean, noi_mg.rho1_mean, sig_mg.turn0_mean, noi_mg.turn0_mean);

    if noi_mg.turn0_mean < noi_bin.turn0_mean
        fprintf(fid, '  观察：Margin 在噪声场景下压低了首步实际转向幅度。\n');
    else
        fprintf(fid, '  观察：Margin 未明显压低噪声场景的首步转向幅度。\n');
    end

    if abs(sig_mg.turn0_mean - sig_bin.turn0_mean) < abs(noi_mg.turn0_mean - noi_bin.turn0_mean)
        fprintf(fid, '  观察：Signal 场景的转向变化小于 Noise 场景，主张2得到支持。\n');
    else
        fprintf(fid, '  观察：Signal/Noise 场景的转向差异不够分离，主张2支持有限。\n');
    end

    if sig_bin.rho1_mean > noi_bin.rho1_mean && sig_mg.rho1_mean > noi_mg.rho1_mean
        fprintf(fid, '  观察：Signal 场景的 src 显著性持续性高于 Noise 场景，主张1得到支持。\n');
    else
        fprintf(fid, '  观察：src 显著性持续性差异不明显，主张1支持有限。\n');
    end
end

function value = safeMean(arr)
    arr = arr(isfinite(arr));
    if isempty(arr)
        value = NaN;
    else
        value = mean(arr);
    end
end

function value = safeMedian(arr)
    arr = arr(isfinite(arr));
    if isempty(arr)
        value = NaN;
    else
        value = median(arr);
    end
end

function value = safePercentile(arr, p)
    arr = arr(isfinite(arr));
    if isempty(arr)
        value = NaN;
    else
        value = prctile(arr, p);
    end
end

function color = getColorFromSpec(line_spec)
    if contains(line_spec, 'r')
        color = [0.85 0.2 0.2];
    elseif contains(line_spec, 'b')
        color = [0.2 0.4 0.8];
    else
        color = [0.5 0.5 0.5];
    end
end

function gain_val = computeGainFromPrecomputedSaliency(sim, pid, neighbor_idx, s_values, src_pos, s_src)
    if ~sim.useResponseGainModulation
        gain_val = 1.0;
        return;
    end

    switch sim.responseGainMode
        case 'margin_only'
            threshold_ref = sim.getResponseGainReferenceThreshold();
            c = max(sim.responseGainC, eps);
            gain_val = min(s_src / (c * threshold_ref), 1);
            gain_val = max(gain_val, 0);
        case 'legacy_q'
            threshold_i = sim.getActivationThreshold(pid);
            above_s = s_values(s_values > threshold_i);
            signal_confidence = sim.normalizeConfidenceFromSaliency(above_s, threshold_i);
            gain_val = sim.computeResponseGain(signal_confidence);
        otherwise
            src_id = neighbor_idx(src_pos);
            gain_val = sim.computeResponseGainValue(pid, neighbor_idx, [], src_id);
    end
end

function wrapped = wrapToPiLocal(angle_values)
    wrapped = mod(angle_values + pi, 2 * pi) - pi;
end

function saveMechanismBundle(out_dir, payload)
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    mat_path = fullfile(out_dir, 'mechanism_data.mat');
    save(mat_path, '-struct', 'payload');

    json_path = fullfile(out_dir, 'mechanism_data.json');
    writeJsonFileLocal(json_path, payload);
end

function writeJsonFileLocal(json_path, payload)
    if ~(exist('jsonencode', 'builtin') || exist('jsonencode', 'file'))
        error('当前 MATLAB 版本不支持 jsonencode，无法导出 JSON: %s', json_path);
    end

    try
        json_text = jsonencode(payload, 'PrettyPrint', true);
    catch
        json_text = jsonencode(payload);
    end

    fid = fopen(json_path, 'w');
    if fid == -1
        error('无法写入 JSON 文件: %s', json_path);
    end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', json_text);
end
