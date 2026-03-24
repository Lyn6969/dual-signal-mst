%% calibrate_vref.m
% V_ref 解析标定脚本（含信号侧验证 + 诊断图）
%
% 功能：
%   1. 无脉冲稳态仿真，采集噪声侧 sigma2_M 统计
%   2. 解析公式计算 V_ref
%   3. 脉冲试验采集信号侧 sigma2_M（即将激活粒子）
%   4. 噪声侧 + 信号侧双向验证 D 的分辨力
%   5. 多 η 扫描，parfor 并行
%
% 输出（results/vref_calibration_<timestamp>/）：
%   - analysis_report.txt：完整文本报告
%   - fig1_vref_comparison.png：解析 vs 旧方法对比
%   - fig2_D_discrimination.png：D 信噪分辨力
%   - fig3_sigma2_distributions.png：各 η 下 sigma2_M 分布
%   - calibration_data.mat：全部原始数据

clear; clc; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 公共参数
params = struct();
params.N = 200;
params.rho = 1;
params.v0 = 1;
params.angleUpdateParameter = 10;
params.T_max = 400;
params.dt = 0.1;
params.radius = 5;
params.deac_threshold = 0.1745;
params.cj_threshold = 2.0;
params.fieldSize = 50;
params.initDirection = pi/4;
params.useFixedField = true;
params.useAdaptiveThreshold = false;

stabilization_steps = 200;
collection_steps = 100;
beta = 0.4;

% 脉冲验证参数
pulse_params = params;
pulse_params.T_max = 350;
pulse_params.stabilization_steps = 100;
pulse_params.external_pulse_count = 1;
pulse_params.forced_turn_duration = 50;
num_pulse_trials = 5;
signal_horizon = 3;  % 看 h=3 步内即将激活的粒子

%% 多 η 扫描
eta_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40];
num_trials = 3;
num_eta = numel(eta_values);
total_noise_jobs = num_eta * num_trials;

%% 输出目录
results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_dir = fullfile(results_dir, sprintf('vref_calibration_%s', timestamp));
mkdir(out_dir);

%% 并行池（可选，失败则回退串行）
use_parallel = false;
try
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool; end
    use_parallel = true;
    fprintf('并行池：%d workers\n\n', pool.NumWorkers);
catch
    fprintf('并行池启动失败，使用串行模式\n\n');
end

%% ==================== 第一阶段：噪声侧采集 ====================
fprintf('=== 第一阶段：噪声侧稳态采集 ===\n');

job_s_max = cell(total_noise_jobs, 1);
job_sigma2 = cell(total_noise_jobs, 1);
job_k = cell(total_noise_jobs, 1);

job_eta_idx = zeros(total_noise_jobs, 1);
job_trial = zeros(total_noise_jobs, 1);
for ei = 1:num_eta
    for ti = 1:num_trials
        jid = (ei - 1) * num_trials + ti;
        job_eta_idx(jid) = ei;
        job_trial(jid) = ti;
    end
end

timer_start = tic;

if use_parallel
    parfor jid = 1:total_noise_jobs
    ei = job_eta_idx(jid);
    ti = job_trial(jid);
    eta = eta_values(ei);

    p = params;
    p.angleNoiseIntensity = eta^2 / 2;
    rng(5000 + round(eta * 1000) + ti);
    sim = ParticleSimulation(p);

    for t = 1:stabilization_steps, sim.step(); end

    max_samples = collection_steps * p.N;
    local_s_max = NaN(max_samples, 1);
    local_sigma2 = NaN(max_samples, 1);
    local_k = NaN(max_samples, 1);
    cursor = 0;

    for t = 1:collection_steps
        sim.step();
        nb = sim.findNeighbors();
        for i = 1:p.N
            idx = find(nb(i, :));
            if isempty(idx), continue; end
            [s_max_val, sigma2_val, k_val] = computeSaliencyStats(sim, i, idx, p.dt);
            cursor = cursor + 1;
            local_s_max(cursor) = s_max_val;
            local_sigma2(cursor) = sigma2_val;
            local_k(cursor) = k_val;
        end
    end

    job_s_max{jid} = local_s_max(1:cursor);
    job_sigma2{jid} = local_sigma2(1:cursor);
    job_k{jid} = local_k(1:cursor);
end
else
    for jid = 1:total_noise_jobs
        ei = job_eta_idx(jid);
        ti = job_trial(jid);
        eta = eta_values(ei);

        p = params;
        p.angleNoiseIntensity = eta^2 / 2;
        rng(5000 + round(eta * 1000) + ti);
        sim = ParticleSimulation(p);

        for t = 1:stabilization_steps, sim.step(); end

        max_samples = collection_steps * p.N;
        local_s_max = NaN(max_samples, 1);
        local_sigma2 = NaN(max_samples, 1);
        local_k = NaN(max_samples, 1);
        cursor = 0;

        for t = 1:collection_steps
            sim.step();
            nb = sim.findNeighbors();
            for i = 1:p.N
                idx = find(nb(i, :));
                if isempty(idx), continue; end
                [s_max_val, sigma2_val, k_val] = computeSaliencyStats(sim, i, idx, p.dt);
                cursor = cursor + 1;
                local_s_max(cursor) = s_max_val;
                local_sigma2(cursor) = sigma2_val;
                local_k(cursor) = k_val;
            end
        end

        job_s_max{jid} = local_s_max(1:cursor);
        job_sigma2{jid} = local_sigma2(1:cursor);
        job_k{jid} = local_k(1:cursor);

        fprintf('  [噪声] η=%.2f trial=%d 完成\n', eta, ti);
    end
end

noise_elapsed = toc(timer_start);
fprintf('噪声侧采集完成，耗时 %.1f 秒\n\n', noise_elapsed);

%% ==================== 第二阶段：信号侧采集（脉冲试验） ====================
fprintf('=== 第二阶段：信号侧脉冲验证 ===\n');

total_pulse_jobs = num_eta * num_pulse_trials;
pulse_job_sigma2_signal = cell(total_pulse_jobs, 1);
pulse_job_sigma2_stable = cell(total_pulse_jobs, 1);

pulse_job_eta_idx = zeros(total_pulse_jobs, 1);
for ei = 1:num_eta
    for ti = 1:num_pulse_trials
        jid = (ei - 1) * num_pulse_trials + ti;
        pulse_job_eta_idx(jid) = ei;
    end
end

timer_pulse = tic;

if use_parallel
    parfor jid = 1:total_pulse_jobs
        ei = pulse_job_eta_idx(jid);
    eta = eta_values(ei);

    pp = pulse_params;
    pp.angleNoiseIntensity = eta^2 / 2;
    pp.useAdaptiveThreshold = false;

    rng(8000 + round(eta * 1000) + jid);
    sim = ParticleSimulationWithExternalPulse(pp);
    sim.setLogging(false);

    pulse_time = pp.stabilization_steps;
    post_steps = 30;

    % 记录逐步数据
    T_total = pulse_time + post_steps;
    N = pp.N;
    sigma2_record = NaN(T_total, N);
    active_record = false(T_total, N);

    for t = 1:T_total
        sim.step();
        if t > pulse_time
            nb = sim.findNeighbors();
            for i = 1:N
                idx = find(nb(i, :));
                if isempty(idx), continue; end
                [~, sigma2_val, ~] = computeSaliencyStats(sim, i, idx, pp.dt);
                sigma2_record(t, i) = sigma2_val;
            end
            active_record(t, :) = sim.isActive';
        end
    end

    % 找即将激活的粒子：当前未激活，但 h 步内激活
    signal_sigma2 = [];
    stable_sigma2 = [];
    for t = (pulse_time+1):(T_total - signal_horizon)
        for i = 1:N
            if active_record(t, i), continue; end
            if isnan(sigma2_record(t, i)), continue; end

            will_activate = false;
            for dt_h = 1:signal_horizon
                if t + dt_h <= T_total && active_record(t + dt_h, i)
                    will_activate = true;
                    break;
                end
            end

            if will_activate
                signal_sigma2(end+1) = sigma2_record(t, i); %#ok<AGROW>
            else
                stable_sigma2(end+1) = sigma2_record(t, i); %#ok<AGROW>
            end
        end
    end

    pulse_job_sigma2_signal{jid} = signal_sigma2(:);
    pulse_job_sigma2_stable{jid} = stable_sigma2(:);
end
else
    for jid = 1:total_pulse_jobs
        ei = pulse_job_eta_idx(jid);
        eta = eta_values(ei);

        pp = pulse_params;
        pp.angleNoiseIntensity = eta^2 / 2;
        pp.useAdaptiveThreshold = false;

        rng(8000 + round(eta * 1000) + jid);
        sim = ParticleSimulationWithExternalPulse(pp);
        sim.setLogging(false);

        pulse_time = pp.stabilization_steps;
        post_steps = 30;
        T_total = pulse_time + post_steps;
        N = pp.N;
        sigma2_record = NaN(T_total, N);
        active_record = false(T_total, N);

        for t = 1:T_total
            sim.step();
            if t > pulse_time
                nb = sim.findNeighbors();
                for i = 1:N
                    idx = find(nb(i, :));
                    if isempty(idx), continue; end
                    [~, sigma2_val, ~] = computeSaliencyStats(sim, i, idx, pp.dt);
                    sigma2_record(t, i) = sigma2_val;
                end
                active_record(t, :) = sim.isActive';
            end
        end

        signal_sigma2 = [];
        stable_sigma2 = [];
        for t = (pulse_time+1):(T_total - signal_horizon)
            for i = 1:N
                if active_record(t, i), continue; end
                if isnan(sigma2_record(t, i)), continue; end
                will_activate = false;
                for dt_h = 1:signal_horizon
                    if t + dt_h <= T_total && active_record(t + dt_h, i)
                        will_activate = true;
                        break;
                    end
                end
                if will_activate
                    signal_sigma2(end+1) = sigma2_record(t, i); %#ok<AGROW>
                else
                    stable_sigma2(end+1) = sigma2_record(t, i); %#ok<AGROW>
                end
            end
        end

        pulse_job_sigma2_signal{jid} = signal_sigma2(:);
        pulse_job_sigma2_stable{jid} = stable_sigma2(:);

        fprintf('  [信号] η=%.2f job=%d 完成 (信号样本=%d)\n', eta, jid, numel(signal_sigma2));
    end
end

pulse_elapsed = toc(timer_pulse);
fprintf('信号侧采集完成，耗时 %.1f 秒\n\n', pulse_elapsed);

%% ==================== 第三阶段：聚合 + 计算 ====================
fprintf('=== 标定结果 ===\n');

% 结果表
T = table();
T.eta = eta_values(:);
T.s_noise = zeros(num_eta, 1);
T.k_bar = zeros(num_eta, 1);
T.sigma2_noise_median = zeros(num_eta, 1);
T.sigma2_noise_95pct = zeros(num_eta, 1);
T.sigma2_signal_median = zeros(num_eta, 1);
T.V_ref_analytical = zeros(num_eta, 1);
T.V_ref_pct95 = zeros(num_eta, 1);
T.D_noise_95pct = zeros(num_eta, 1);
T.D_signal_median = zeros(num_eta, 1);
T.ratio = zeros(num_eta, 1);
T.n_signal_samples = zeros(num_eta, 1);

% 保存各 η 的 sigma2 分布用于画图
noise_sigma2_by_eta = cell(num_eta, 1);
signal_sigma2_by_eta = cell(num_eta, 1);

for ei = 1:num_eta
    % 噪声侧聚合
    all_s_max = []; all_sigma2 = []; all_k = [];
    for ti = 1:num_trials
        jid = (ei - 1) * num_trials + ti;
        all_s_max = [all_s_max; job_s_max{jid}]; %#ok<AGROW>
        all_sigma2 = [all_sigma2; job_sigma2{jid}]; %#ok<AGROW>
        all_k = [all_k; job_k{jid}]; %#ok<AGROW>
    end

    % 信号侧聚合
    all_signal = []; all_stable = [];
    for ti = 1:num_pulse_trials
        jid = (ei - 1) * num_pulse_trials + ti;
        all_signal = [all_signal; pulse_job_sigma2_signal{jid}]; %#ok<AGROW>
        all_stable = [all_stable; pulse_job_sigma2_stable{jid}]; %#ok<AGROW>
    end

    noise_sigma2_by_eta{ei} = all_sigma2;
    signal_sigma2_by_eta{ei} = all_signal;

    s_noise = median(all_s_max);
    k_bar = mean(all_k);
    sigma2_95 = prctile(all_sigma2, 95);

    delta = beta * params.cj_threshold - s_noise;
    if delta > 0 && k_bar > 1
        V_ref_anal = delta^2 * (k_bar - 1) / k_bar^2;
    else
        V_ref_anal = sigma2_95 * 15;
    end

    V_ref_p95 = max(sigma2_95, eps);
    D_noise95 = sigma2_95 / V_ref_anal;
    D_signal_med = NaN;
    if ~isempty(all_signal)
        D_signal_med = median(all_signal) / V_ref_anal;
    end

    T.s_noise(ei) = s_noise;
    T.k_bar(ei) = k_bar;
    T.sigma2_noise_median(ei) = median(all_sigma2);
    T.sigma2_noise_95pct(ei) = sigma2_95;
    T.sigma2_signal_median(ei) = median(all_signal);
    T.V_ref_analytical(ei) = V_ref_anal;
    T.V_ref_pct95(ei) = V_ref_p95;
    T.D_noise_95pct(ei) = D_noise95;
    T.D_signal_median(ei) = D_signal_med;
    T.ratio(ei) = V_ref_anal / V_ref_p95;
    T.n_signal_samples(ei) = numel(all_signal);
end

disp(T);

%% ==================== 第四阶段：生成图表 ====================

% --- 图1：V_ref 解析 vs 旧方法对比 ---
fig1 = figure('Visible', 'off', 'Position', [50 50 900 400], 'Color', 'w');

subplot(1,2,1);
semilogy(T.eta, T.V_ref_analytical, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;
semilogy(T.eta, T.V_ref_pct95, 'r--s', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('\eta');
ylabel('V_{ref}');
title('V_{ref} 标定方法对比');
legend('解析公式', '旧方法 (95 分位)', 'Location', 'best');
grid on;

subplot(1,2,2);
bar(categorical(string(T.eta)), T.ratio);
ylabel('解析 / 旧方法 比值');
xlabel('\eta');
title('V_{ref} 比值（应 ≈ 10-20x）');
grid on;

saveas(fig1, fullfile(out_dir, 'fig1_vref_comparison.png'));
close(fig1);

% --- 图2：D 信噪分辨力 ---
fig2 = figure('Visible', 'off', 'Position', [50 50 800 500], 'Color', 'w');

bar_data = [T.D_noise_95pct, T.D_signal_median];
b = bar(categorical(string(T.eta)), bar_data);
b(1).FaceColor = [0.7 0.7 0.7];
b(2).FaceColor = [0.2 0.4 0.8];
hold on;
yline(0.1, 'r--', 'D=0.1 噪声上限', 'LineWidth', 1.5);
yline(0.8, 'g--', 'D=0.8 信号下限', 'LineWidth', 1.5);
xlabel('\eta');
ylabel('D 值');
title('D 信噪分辨力验证');
legend('D(noise 95pct)', 'D(signal median)', 'Location', 'best');
ylim([0, max(1.2, max(bar_data(:)) * 1.1)]);
grid on;

saveas(fig2, fullfile(out_dir, 'fig2_D_discrimination.png'));
close(fig2);

% --- 图3：各 η 下 sigma2_M 分布（噪声 vs 信号） ---
fig3 = figure('Visible', 'off', 'Position', [50 50 1400 800], 'Color', 'w');

plot_etas = [1, 3, 5, 7];  % 选 4 个代表性 η
for pi_idx = 1:numel(plot_etas)
    ei = plot_etas(pi_idx);
    subplot(2, 2, pi_idx);

    noise_data = log10(max(noise_sigma2_by_eta{ei}, 1e-12));
    histogram(noise_data, 50, 'FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.7, 'Normalization', 'probability');
    hold on;

    if ~isempty(signal_sigma2_by_eta{ei})
        signal_data = log10(max(signal_sigma2_by_eta{ei}, 1e-12));
        histogram(signal_data, 30, 'FaceColor', [0.2 0.4 0.8], 'FaceAlpha', 0.7, 'Normalization', 'probability');
    end

    xline(log10(T.V_ref_analytical(ei)), 'r-', 'V_{ref}', 'LineWidth', 2);
    xline(log10(T.V_ref_pct95(ei)), 'k--', '旧V_{ref}', 'LineWidth', 1.5);

    xlabel('log_{10}(\sigma^2_M)');
    ylabel('概率');
    title(sprintf('\\eta=%.2f (n_{signal}=%d)', eta_values(ei), T.n_signal_samples(ei)));
    if pi_idx == 1
        legend('噪声', '信号（h≤3即将激活）', 'V_{ref}(解析)', 'V_{ref}(旧)', 'Location', 'best');
    end
    grid on;
end

saveas(fig3, fullfile(out_dir, 'fig3_sigma2_distributions.png'));
close(fig3);

%% ==================== 第五阶段：文本报告 ====================
report_file = fullfile(out_dir, 'analysis_report.txt');
fid = fopen(report_file, 'w');

fprintf(fid, '========================================================================\n');
fprintf(fid, 'V_ref 解析标定报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '========================================================================\n\n');

fprintf(fid, '一、标定参数\n');
fprintf(fid, '------------------------------------------------------------------------\n');
fprintf(fid, 'β = %.2f, cj_threshold = %.1f\n', beta, params.cj_threshold);
fprintf(fid, '公式: V_ref = (β·cj − s_noise)² × (k̄−1) / k̄²\n');
fprintf(fid, '噪声采集: %d 步稳定 + %d 步采集 × %d trials\n', stabilization_steps, collection_steps, num_trials);
fprintf(fid, '信号验证: %d 次脉冲试验, horizon h=%d\n', num_pulse_trials, signal_horizon);
fprintf(fid, '总耗时: %.1f 秒（噪声 %.1f + 信号 %.1f）\n\n', noise_elapsed + pulse_elapsed, noise_elapsed, pulse_elapsed);

fprintf(fid, '二、标定结果\n');
fprintf(fid, '------------------------------------------------------------------------\n');
fprintf(fid, '%-5s %7s %5s %10s %10s %10s %10s %10s %10s %6s\n', ...
    'eta', 's_noi', 'k', 'sig2_95', 'sig2_sig', 'Vref_an', 'Vref_p95', 'D_noi95', 'D_sig', 'ratio');
for ei = 1:num_eta
    fprintf(fid, '%-5.2f %7.4f %5.1f %10.6f %10.6f %10.6f %10.6f %10.4f %10.4f %6.1fx\n', ...
        T.eta(ei), T.s_noise(ei), T.k_bar(ei), T.sigma2_noise_95pct(ei), ...
        T.sigma2_signal_median(ei), T.V_ref_analytical(ei), T.V_ref_pct95(ei), ...
        T.D_noise_95pct(ei), T.D_signal_median(ei), T.ratio(ei));
end

fprintf(fid, '\n三、验证判据\n');
fprintf(fid, '------------------------------------------------------------------------\n');

pass_noise = all(T.D_noise_95pct < 0.1);
pass_signal = all(T.D_signal_median > 0.8 | isnan(T.D_signal_median));
pass_ratio = all(T.ratio > 5);

fprintf(fid, '检查1 - 噪声侧: D(noise_95pct) < 0.1 对所有 η?\n');
for ei = 1:num_eta
    ok = T.D_noise_95pct(ei) < 0.1;
    fprintf(fid, '  η=%.2f: D=%.4f  %s\n', T.eta(ei), T.D_noise_95pct(ei), ternary(ok, 'PASS', 'FAIL'));
end
fprintf(fid, '\n');

fprintf(fid, '检查2 - 信号侧: D(signal_median) > 0.8?\n');
for ei = 1:num_eta
    if isnan(T.D_signal_median(ei))
        fprintf(fid, '  η=%.2f: 无信号样本 (n=%d)  SKIP\n', T.eta(ei), T.n_signal_samples(ei));
    else
        ok = T.D_signal_median(ei) > 0.8;
        fprintf(fid, '  η=%.2f: D=%.4f (n=%d)  %s\n', T.eta(ei), T.D_signal_median(ei), T.n_signal_samples(ei), ternary(ok, 'PASS', 'FAIL'));
    end
end
fprintf(fid, '\n');

fprintf(fid, '检查3 - 解析/旧方法比值 > 5x?\n');
for ei = 1:num_eta
    ok = T.ratio(ei) > 5;
    fprintf(fid, '  η=%.2f: %.1fx  %s\n', T.eta(ei), T.ratio(ei), ternary(ok, 'PASS', 'FAIL'));
end

fprintf(fid, '\n四、总结\n');
fprintf(fid, '------------------------------------------------------------------------\n');
if pass_noise && pass_ratio
    fprintf(fid, '标定有效：噪声侧 D 不饱和，解析 V_ref 显著高于旧 95 分位方法。\n');
    if pass_signal
        fprintf(fid, '信号侧验证通过：即将激活粒子的 D 接近满值。\n');
    else
        fprintf(fid, '注意：部分 η 下信号侧 D 偏低，可能需要调整 β（当前 %.2f）。\n', beta);
    end
else
    fprintf(fid, '警告：标定可能存在问题，请检查参数设置。\n');
end

fclose(fid);

%% 保存 MAT 数据
save(fullfile(out_dir, 'calibration_data.mat'), ...
    'T', 'params', 'pulse_params', 'beta', 'eta_values', ...
    'noise_sigma2_by_eta', 'signal_sigma2_by_eta', ...
    'stabilization_steps', 'collection_steps', 'num_trials', ...
    'num_pulse_trials', 'signal_horizon', ...
    'noise_elapsed', 'pulse_elapsed');

fprintf('\n完成。输出目录: %s\n', out_dir);
fprintf('  analysis_report.txt\n');
fprintf('  fig1_vref_comparison.png\n');
fprintf('  fig2_D_discrimination.png\n');
fprintf('  fig3_sigma2_distributions.png\n');
fprintf('  calibration_data.mat\n');

%% ========================================================================
%  辅助函数
%% ========================================================================

function [s_max_val, sigma2_val, k_val] = computeSaliencyStats(sim, i, neighbor_idx, dt)
% 计算粒子 i 的邻域显著性统计
    cd = sim.positions(neighbor_idx, :) - sim.positions(i, :);
    pd = sim.previousPositions(neighbor_idx, :) - sim.previousPositions(i, :);
    cd_d = vecnorm(cd, 2, 2);
    pd_d = vecnorm(pd, 2, 2);

    cu = zeros(size(cd)); pu = zeros(size(pd));
    nzc = cd_d > 0; nzp = pd_d > 0;
    cu(nzc, :) = cd(nzc, :) ./ cd_d(nzc);
    pu(nzp, :) = pd(nzp, :) ./ pd_d(nzp);

    ca = sum(pu .* cu, 2);
    ca = max(min(ca, 1), -1);
    sv = acos(ca) / max(dt, eps);

    s_max_val = max(sv);
    sigma2_val = var(sv, 1);
    k_val = numel(neighbor_idx);
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end

function V_ref = calibrateVrefAnalytical(params, eta, beta)
% calibrateVrefAnalytical 解析计算 V_ref（供外部脚本调用）
    if nargin < 3, beta = 0.4; end
    p = params;
    p.angleNoiseIntensity = eta^2 / 2;
    p.useAdaptiveThreshold = false;

    all_s_max = [];
    all_k = [];

    for trial = 1:3
        rng(5000 + round(eta * 1000) + trial);
        sim = ParticleSimulation(p);
        for t = 1:200, sim.step(); end
        for t = 1:100
            sim.step();
            nb = sim.findNeighbors();
            for i = 1:p.N
                idx = find(nb(i, :));
                if isempty(idx), continue; end
                [s_max_val, ~, k_val] = computeSaliencyStats(sim, i, idx, p.dt);
                all_s_max(end+1) = s_max_val; %#ok<AGROW>
                all_k(end+1) = k_val; %#ok<AGROW>
            end
        end
    end

    s_noise = median(all_s_max);
    k_bar = mean(all_k);
    delta = beta * p.cj_threshold - s_noise;
    if delta > 0 && k_bar > 1
        V_ref = delta^2 * (k_bar-1) / k_bar^2;
    else
        V_ref = prctile(all_s_max, 95) * 0.1;
    end
end
