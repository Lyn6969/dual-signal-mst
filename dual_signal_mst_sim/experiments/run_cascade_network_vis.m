%% run_cascade_network_vis.m
% 两个场景的级联传播网络可视化：
%   场景A: 有脉冲（响应性）—— 预期 Binary 和 Weighted 相同
%   场景B: 无脉冲（持久性）—— 预期 Weighted 的虚假级联更弱
% Binary vs Weighted 用相同种子各跑一次

clc; clear; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 参数
fprintf('=================================================\n');
fprintf('  级联传播网络可视化\n');
fprintf('  场景A: 有脉冲  场景B: 无脉冲(纯噪声)\n');
fprintf('  Binary vs Weighted 对比\n');
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

eta = 0.30;
base.angleNoiseIntensity = eta^2 / 2;

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5.0;
adaptive_cfg.saliency_threshold = 0.031;
adaptive_cfg.include_self = false;

seed = 20260331;

%% 输出目录
results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_dir = fullfile(results_dir, sprintf('cascade_network_%s', timestamp));
mkdir(out_dir);

%% ================================================================
%% 场景A: 有脉冲（响应性场景）
%% ================================================================
fprintf('========== 场景A: 有脉冲 ==========\n\n');

pulse_params = base;
pulse_params.T_max = 400;
pulse_params.stabilization_steps = 200;
pulse_params.forced_turn_duration = 200;

mode_names = {'Binary', 'Weighted'};
mode_weighted = [false, true];

for mi = 1:2
    fprintf('[场景A-%s] seed=%d ...\n', mode_names{mi}, seed);

    rng(seed);
    params = pulse_params;
    params.useAdaptiveThreshold = true;
    params.adaptiveThresholdConfig = adaptive_cfg;
    params.adaptiveThresholdMode = 'binary';
    params.useWeightedFollow = mode_weighted(mi);

    sim = ParticleSimulationWithExternalPulse(params);
    sim.external_pulse_count = 1;
    sim.setLogging(false);
    sim.resetCascadeTracking();
    sim.initializeParticles();

    [first_act_step, first_act_src, positions_hist, theta_hist, ...
        isActive_hist, cascade_idx, external_idx, pulse_step] = ...
        run_and_record(sim, params);

    total_activated = sum(first_act_step > 0);
    fprintf('  脉冲t=%d, 级联=%d, 总激活=%d/%d\n', ...
        pulse_step, numel(cascade_idx), total_activated, params.N);

    % 画图
    draw_all_figures(out_dir, sprintf('A_%s', lower(mode_names{mi})), ...
        mode_names{mi}, '有脉冲', params, sim, ...
        first_act_step, first_act_src, positions_hist, theta_hist, ...
        isActive_hist, cascade_idx, external_idx, pulse_step);

    fprintf('  完成\n\n');
end

%% ================================================================
%% 场景B: 无脉冲（纯噪声，持久性场景）
%% ================================================================
fprintf('========== 场景B: 无脉冲(纯噪声) ==========\n\n');

nopulse_params = base;
nopulse_params.T_max = 600;
nopulse_params.useAdaptiveThreshold = true;
nopulse_params.adaptiveThresholdConfig = adaptive_cfg;
nopulse_params.adaptiveThresholdMode = 'binary';

for mi = 1:2
    fprintf('[场景B-%s] seed=%d ...\n', mode_names{mi}, seed);

    rng(seed);
    params = nopulse_params;
    params.useWeightedFollow = mode_weighted(mi);

    % 用基础 ParticleSimulation（无脉冲）
    sim = ParticleSimulation(params);

    N = params.N;
    T = params.T_max;

    % 记录状态
    positions_hist = zeros(N, 2, T+1);
    theta_hist = zeros(N, T+1);
    isActive_hist = false(N, T+1);
    src_ids_hist = cell(N, T+1);

    positions_hist(:,:,1) = sim.positions;
    theta_hist(:,1) = sim.theta;

    % 虚假激活事件记录
    all_false_events = [];  % 每行: [粒子ID, 激活时刻, 源头ID, 转向幅度]

    for t = 1:T
        was_active = sim.isActive;
        sim.step();

        positions_hist(:,:,t+1) = sim.positions;
        theta_hist(:,t+1) = sim.theta;
        isActive_hist(:,t+1) = sim.isActive;

        % 检测新激活（纯噪声下的虚假激活）
        newly_activated = sim.isActive & ~was_active;
        for i = find(newly_activated)'
            src = sim.src_ids{i};
            if isempty(src), src = 0; end
            % 计算转向幅度
            if t >= 2
                delta = abs(wrapToPi(theta_hist(i, t+1) - theta_hist(i, t)));
            else
                delta = 0;
            end
            all_false_events = [all_false_events; i, t, src, delta]; %#ok<AGROW>
        end
    end

    n_events = size(all_false_events, 1);
    total_ever_active = sum(any(isActive_hist, 2));
    activated_per_step = sum(isActive_hist, 1);
    max_simultaneous = max(activated_per_step);

    fprintf('  虚假激活事件: %d 次\n', n_events);
    fprintf('  曾激活粒子数: %d / %d\n', total_ever_active, N);
    fprintf('  最大同时激活: %d\n', max_simultaneous);
    if n_events > 0
        fprintf('  平均转向幅度: %.1f°\n', rad2deg(mean(all_false_events(:,4))));
    end

    % --- 图1: 激活粒子数时间序列 ---
    fig1 = figure('Visible', 'off', 'Position', [50 50 800 400], 'Color', 'w');
    time_axis = (0:T) * base.dt;
    plot(time_axis, activated_per_step, 'b-', 'LineWidth', 1);
    xlabel('时间'); ylabel('激活粒子数');
    title(sprintf('场景B %s: 虚假激活时间序列 (总事件=%d)', mode_names{mi}, n_events));
    grid on;
    saveas(fig1, fullfile(out_dir, sprintf('B_fig1_activation_%s.png', lower(mode_names{mi}))));
    close(fig1);

    % --- 图2: 虚假激活的转向幅度分布 ---
    fig2 = figure('Visible', 'off', 'Position', [50 50 600 400], 'Color', 'w');
    if n_events > 0
        histogram(rad2deg(all_false_events(:,4)), 30, ...
            'FaceColor', [0.3 0.5 0.8], 'FaceAlpha', 0.7, 'Normalization', 'probability');
        xline(rad2deg(mean(all_false_events(:,4))), 'r-', ...
            sprintf('均值=%.1f°', rad2deg(mean(all_false_events(:,4)))), 'LineWidth', 1.5);
        xline(rad2deg(median(all_false_events(:,4))), 'k--', ...
            sprintf('中位数=%.1f°', rad2deg(median(all_false_events(:,4)))), 'LineWidth', 1.2);
    end
    xlabel('转向幅度 (°)'); ylabel('概率');
    title(sprintf('场景B %s: 虚假激活转向幅度 (n=%d)', mode_names{mi}, n_events));
    grid on;
    saveas(fig2, fullfile(out_dir, sprintf('B_fig2_turn_amplitude_%s.png', lower(mode_names{mi}))));
    close(fig2);

    % --- 图3: 虚假级联的网络快照（选最大激活时刻）---
    fig3 = figure('Visible', 'off', 'Position', [50 50 800 800], 'Color', 'w');
    hold on;

    [~, peak_step] = max(activated_per_step);
    pos_snap = positions_hist(:,:,peak_step);

    % 未激活粒子
    inactive_at_peak = ~isActive_hist(:, peak_step);
    scatter(pos_snap(inactive_at_peak,1), pos_snap(inactive_at_peak,2), ...
        15, [0.8 0.8 0.8], 'filled');

    % 激活粒子
    active_at_peak = find(isActive_hist(:, peak_step));
    if ~isempty(active_at_peak)
        scatter(pos_snap(active_at_peak,1), pos_snap(active_at_peak,2), ...
            60, [0.8 0.2 0.2], 'filled');

        % 画跟随边
        for ii = 1:numel(active_at_peak)
            pid = active_at_peak(ii);
            % 找这个粒子在 peak_step 附近的激活事件
            evt_mask = all_false_events(:,1) == pid & ...
                abs(all_false_events(:,2) - peak_step) <= 3;
            if any(evt_mask)
                evt = all_false_events(find(evt_mask, 1, 'last'), :);
                src = evt(3);
                if src > 0 && src <= N
                    x1 = pos_snap(src,1); y1 = pos_snap(src,2);
                    x2 = pos_snap(pid,1); y2 = pos_snap(pid,2);
                    dx = x2-x1; dy = y2-y1;
                    if sqrt(dx^2+dy^2) < 20
                        quiver(x1, y1, dx*0.85, dy*0.85, 0, ...
                            'Color', [0.3 0.3 0.7 0.6], 'LineWidth', 1, ...
                            'MaxHeadSize', 0.5);
                    end
                end
            end
        end
    end

    title(sprintf('场景B %s: 虚假级联网络 (t_{peak}=%d, 同时激活=%d)', ...
        mode_names{mi}, peak_step, numel(active_at_peak)));
    xlabel('x'); ylabel('y');
    axis equal; grid on;
    saveas(fig3, fullfile(out_dir, sprintf('B_fig3_network_%s.png', lower(mode_names{mi}))));
    close(fig3);

    % 保存数据
    save(fullfile(out_dir, sprintf('B_data_%s.mat', lower(mode_names{mi}))), ...
        'all_false_events', 'activated_per_step', 'n_events', ...
        'total_ever_active', 'max_simultaneous');

    fprintf('  完成\n\n');
end

%% 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');
fprintf(fid, '========================================================================\n');
fprintf(fid, '级联传播网络可视化报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '========================================================================\n\n');
fprintf(fid, '设置: eta=%.2f, N=%d, seed=%d\n', eta, base.N, seed);
fprintf(fid, 'adaptive_cfg: cj_low=%.1f, cj_high=%.1f, saliency_thr=%.3f\n\n', ...
    adaptive_cfg.cj_low, adaptive_cfg.cj_high, adaptive_cfg.saliency_threshold);
fprintf(fid, '场景A: 有脉冲（响应性场景）\n');
fprintf(fid, '场景B: 无脉冲（纯噪声，持久性场景）\n\n');
fprintf(fid, '输出文件:\n');
fprintf(fid, '  A_*  - 场景A(有脉冲)的图和数据\n');
fprintf(fid, '  B_*  - 场景B(无脉冲)的图和数据\n');
fclose(fid);

fprintf('完成。输出目录: %s\n', out_dir);

%% ========================================================================
%% 辅助函数
%% ========================================================================

function [first_act_step, first_act_src, positions_hist, theta_hist, ...
        isActive_hist, cascade_idx, external_idx, pulse_step] = ...
        run_and_record(sim, params)
% 运行带脉冲的仿真，记录完整状态

    N = params.N;
    T = params.T_max;

    positions_hist = zeros(N, 2, T+1);
    theta_hist = zeros(N, T+1);
    isActive_hist = false(N, T+1);

    positions_hist(:,:,1) = sim.positions;
    theta_hist(:,1) = sim.theta;

    first_act_step = zeros(N, 1);
    first_act_src = zeros(N, 1);
    pulse_step = NaN;

    for t = 1:T
        was_active = sim.isActive;
        sim.step();

        positions_hist(:,:,t+1) = sim.positions;
        theta_hist(:,t+1) = sim.theta;
        isActive_hist(:,t+1) = sim.isActive;

        if isnan(pulse_step) && sim.external_pulse_triggered
            pulse_step = t;
        end

        newly_activated = sim.isActive & ~was_active;
        for i = find(newly_activated)'
            if first_act_step(i) == 0
                first_act_step(i) = t;
                if ~isempty(sim.src_ids{i})
                    first_act_src(i) = sim.src_ids{i};
                end
            end
        end
    end

    external_idx = find(sim.isExternallyActivated);
    cascade_idx = find(first_act_step > 0 & ~sim.isExternallyActivated);
end

function draw_all_figures(out_dir, prefix, mode_name, scene_name, params, sim, ...
        first_act_step, first_act_src, positions_hist, theta_hist, ...
        isActive_hist, cascade_idx, external_idx, pulse_step)
% 画场景A（有脉冲）的全套图

    N = params.N;
    T = params.T_max;
    activated_counts = sum(isActive_hist, 1);
    total_activated = sum(first_act_step > 0);

    % 图1: 领导-追随有向网络
    fig1 = figure('Visible', 'off', 'Position', [50 50 800 800], 'Color', 'w');
    hold on;

    [~, peak_step] = max(activated_counts);
    pos_snap = positions_hist(:,:,peak_step);

    never_activated = (first_act_step == 0) & ~sim.isExternallyActivated;
    scatter(pos_snap(never_activated,1), pos_snap(never_activated,2), ...
        15, [0.8 0.8 0.8], 'filled');

    if ~isempty(cascade_idx)
        act_times = first_act_step(cascade_idx);
        if max(act_times) > min(act_times)
            act_norm = (act_times - min(act_times)) / (max(act_times) - min(act_times));
        else
            act_norm = zeros(size(act_times));
        end
        scatter(pos_snap(cascade_idx,1), pos_snap(cascade_idx,2), 40, act_norm, 'filled');
        colormap(gca, flipud(cool));
        cb = colorbar; cb.Label.String = '激活顺序';
    end

    scatter(pos_snap(external_idx,1), pos_snap(external_idx,2), ...
        200, 'r', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

    for i = 1:N
        src = first_act_src(i);
        if src > 0 && first_act_step(i) > 0
            x1 = pos_snap(src,1); y1 = pos_snap(src,2);
            x2 = pos_snap(i,1); y2 = pos_snap(i,2);
            dx = x2-x1; dy = y2-y1;
            if sqrt(dx^2+dy^2) < 20
                quiver(x1, y1, dx*0.85, dy*0.85, 0, ...
                    'Color', [0.3 0.3 0.7 0.5], 'LineWidth', 0.8, 'MaxHeadSize', 0.5);
            end
        end
    end

    title(sprintf('%s %s: 级联网络 (激活 %d/%d)', scene_name, mode_name, total_activated, N));
    xlabel('x'); ylabel('y'); axis equal; grid on;
    saveas(fig1, fullfile(out_dir, sprintf('%s_fig1_network.png', prefix)));
    close(fig1);

    % 图2: 激活粒子数时间序列
    fig2 = figure('Visible', 'off', 'Position', [50 50 800 400], 'Color', 'w');
    time_axis = (0:T) * params.dt;
    plot(time_axis, activated_counts, 'b-', 'LineWidth', 1.5);
    hold on;
    if ~isnan(pulse_step)
        xline(pulse_step * params.dt, 'r--', '脉冲', 'LineWidth', 1.5);
    end
    xlabel('时间'); ylabel('激活粒子数');
    title(sprintf('%s %s: 激活粒子数', scene_name, mode_name));
    grid on;
    saveas(fig2, fullfile(out_dir, sprintf('%s_fig2_activation.png', prefix)));
    close(fig2);

    % 图3: 转向幅度分布
    fig3 = figure('Visible', 'off', 'Position', [50 50 600 400], 'Color', 'w');
    turn_amp = [];
    for i = cascade_idx'
        t_act = first_act_step(i);
        if t_act >= 2 && t_act < T
            delta = abs(wrapToPi(theta_hist(i, t_act+1) - theta_hist(i, t_act)));
            turn_amp = [turn_amp; delta]; %#ok<AGROW>
        end
    end
    if ~isempty(turn_amp)
        histogram(rad2deg(turn_amp), 30, 'FaceColor', [0.3 0.5 0.8], ...
            'FaceAlpha', 0.7, 'Normalization', 'probability');
        xline(rad2deg(mean(turn_amp)), 'r-', sprintf('均值=%.1f°', rad2deg(mean(turn_amp))), 'LineWidth', 1.5);
    end
    xlabel('转向幅度 (°)'); ylabel('概率');
    title(sprintf('%s %s: 转向幅度 (n=%d)', scene_name, mode_name, numel(turn_amp)));
    grid on;
    saveas(fig3, fullfile(out_dir, sprintf('%s_fig3_turn_amplitude.png', prefix)));
    close(fig3);
end
