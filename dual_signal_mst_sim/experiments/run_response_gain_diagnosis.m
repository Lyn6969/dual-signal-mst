%% run_response_gain_diagnosis.m
% 诊断 Response-Gain 方法的前置统计可分性
%
% 在 Binary 自适应阈值基线动力学下，比较：
%   - q_i（当前 response-gain 方案的局部置信度）
%   - margin（单源候选时的超阈值幅度）
%   - C_ratio（多源候选时的主导性）
%   - |A_i|（过阈值候选数量）
%
% 注意：这些统计量不是“激活前预测量”，而是“触发时刻响应量”。
% 因此本脚本对 signal 组取“首次激活时刻”的观测；
% 对 stable 组取“脉冲后从未激活粒子”的背景观测。
%
% 候选统计统一使用 cj_low 作为参考阈值，避免当前动态阈值
% 将稳定组候选集合全部压成空集。
%
% 输出（results/response_gain_diagnosis_<timestamp>/）：
%   - analysis_report.txt
%   - fig1_auc_summary.png
%   - fig2_component_distributions.png
%   - fig3_candidate_regimes.png
%   - diagnosis_data.mat
%   - diagnosis_data.json

clear; clc; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 参数
params = struct();
params.N = 200;
params.rho = 1;
params.v0 = 1;
params.angleUpdateParameter = 10;
params.T_max = 600;
params.dt = 0.1;
params.radius = 5;
params.deac_threshold = 0.1745;
params.cj_threshold = 1.5;
params.fieldSize = 50;
params.initDirection = pi/4;
params.useFixedField = true;
params.useWeightedFollow = false;
params.useResponseGainModulation = false;  % 诊断统计本身，不让新方法反过来改变动力学
params.stabilization_steps = 200;
params.external_pulse_count = 1;
params.forced_turn_duration = 400;

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5.0;
adaptive_cfg.saliency_threshold = 0.031;
adaptive_cfg.include_self = false;

params.useAdaptiveThreshold = true;
params.adaptiveThresholdConfig = adaptive_cfg;
params.adaptiveThresholdMode = 'binary';

% 诊断用的 response-gain 统计参数（与当前方案一致）
params.responseGainMin = 0.4;
params.responseGainBeta = 2.0;
params.responseGainClow = 30;
params.responseGainChigh = 150;
params.responseGainMarginRef = 0.5;

eta_values = [0.25, 0.30, 0.40];
num_trials = 10;
signal_horizon = 3;  % 仅保留在输出中作背景说明，不再用于 q/margin 主诊断分组

%% 输出目录
results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_dir = fullfile(results_dir, sprintf('response_gain_diagnosis_%s', timestamp));
mkdir(out_dir);

%% 并行池（可选）
use_parallel = false;
try
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool; end
    use_parallel = true;
    fprintf('并行池：%d workers\n\n', pool.NumWorkers);
catch
    fprintf('并行池启动失败，使用串行模式\n\n');
end

%% 展平任务
num_eta = numel(eta_values);
total_jobs = num_eta * num_trials;
job_eta_idx = zeros(total_jobs, 1);
job_trial = zeros(total_jobs, 1);
for ei = 1:num_eta
    for ti = 1:num_trials
        jid = (ei - 1) * num_trials + ti;
        job_eta_idx(jid) = ei;
        job_trial(jid) = ti;
    end
end

job_results = cell(total_jobs, 1);

fprintf('开始 %d 个 response-gain 诊断任务（%d η × %d trials）...\n', ...
    total_jobs, num_eta, num_trials);
diag_timer = tic;

if use_parallel
    parfor jid = 1:total_jobs
        job_results{jid} = runSingleResponseGainDiagnosis( ...
            params, eta_values, job_eta_idx(jid), job_trial(jid), signal_horizon);
    end
else
    for jid = 1:total_jobs
        job_results{jid} = runSingleResponseGainDiagnosis( ...
            params, eta_values, job_eta_idx(jid), job_trial(jid), signal_horizon);
        fprintf('  [%d/%d] η=%.2f trial=%d 完成\n', ...
            jid, total_jobs, eta_values(job_eta_idx(jid)), job_trial(jid));
    end
end

diagnosis_elapsed = toc(diag_timer);
fprintf('诊断完成，耗时 %.1f 秒\n\n', diagnosis_elapsed);

%% 聚合
all_results = cell(num_eta, 1);

for ei = 1:num_eta
    signal = emptyStatGroup();
    stable = emptyStatGroup();

    for ti = 1:num_trials
        jid = (ei - 1) * num_trials + ti;
        trial_res = job_results{jid};
        signal = appendStatGroup(signal, trial_res.signal);
        stable = appendStatGroup(stable, trial_res.stable);
    end

    res = struct();
    res.eta = eta_values(ei);
    res.signal = signal;
    res.stable = stable;

    sig_cand = signal.above_count > 0;
    stb_cand = stable.above_count > 0;
    sig_single = signal.above_count == 1;
    stb_single = stable.above_count == 1;
    sig_multi = signal.above_count > 1;
    stb_multi = stable.above_count > 1;

    res.n_signal = numel(signal.q);
    res.n_stable = numel(stable.q);
    res.n_signal_candidate = sum(sig_cand);
    res.n_stable_candidate = sum(stb_cand);
    res.n_signal_single = sum(sig_single);
    res.n_stable_single = sum(stb_single);
    res.n_signal_multi = sum(sig_multi);
    res.n_stable_multi = sum(stb_multi);

    res.signal_candidate_rate = mean(sig_cand);
    res.stable_candidate_rate = mean(stb_cand);
    res.signal_single_share = safeRatio(sum(sig_single), sum(sig_cand));
    res.stable_single_share = safeRatio(sum(stb_single), sum(stb_cand));
    res.signal_multi_share = safeRatio(sum(sig_multi), sum(sig_cand));
    res.stable_multi_share = safeRatio(sum(stb_multi), sum(stb_cand));

    res.auc_sigma2_all = mannWhitneyAUC(signal.sigma2, stable.sigma2);
    res.auc_q_all = mannWhitneyAUC(signal.q, stable.q);
    res.auc_q_candidate = mannWhitneyAUC(signal.q(sig_cand), stable.q(stb_cand));
    res.auc_margin_single = mannWhitneyAUC(signal.margin(sig_single), stable.margin(stb_single));
    res.auc_q_single = mannWhitneyAUC(signal.q(sig_single), stable.q(stb_single));
    res.auc_cratio_multi = mannWhitneyAUC(signal.cratio(sig_multi), stable.cratio(stb_multi));
    res.auc_q_multi = mannWhitneyAUC(signal.q(sig_multi), stable.q(stb_multi));

    res.signal_q_candidate_median = safeMedian(signal.q(sig_cand));
    res.stable_q_candidate_median = safeMedian(stable.q(stb_cand));
    res.signal_margin_single_median = safeMedian(signal.margin(sig_single));
    res.stable_margin_single_median = safeMedian(stable.margin(stb_single));
    res.signal_cratio_multi_median = safeMedian(signal.cratio(sig_multi));
    res.stable_cratio_multi_median = safeMedian(stable.cratio(stb_multi));
    res.signal_gain_candidate_median = safeMedian(signal.gain(sig_cand));
    res.stable_gain_candidate_median = safeMedian(stable.gain(stb_cand));

    all_results{ei} = res;

    fprintf('η=%.2f: q_all AUC=%.3f, q_cand AUC=%.3f, margin_single AUC=%.3f, C_ratio_multi AUC=%.3f\n', ...
        res.eta, res.auc_q_all, res.auc_q_candidate, ...
        res.auc_margin_single, res.auc_cratio_multi);
end

%% 图1：AUC 汇总
fig1 = figure('Visible', 'off', 'Position', [50 50 1500 450], 'Color', 'w');

auc_q_mat = zeros(num_eta, 2);
auc_single_mat = zeros(num_eta, 2);
auc_multi_mat = zeros(num_eta, 2);
for ei = 1:num_eta
    r = all_results{ei};
    auc_q_mat(ei, :) = [r.auc_q_all, r.auc_q_candidate];
    auc_single_mat(ei, :) = [r.auc_margin_single, r.auc_q_single];
    auc_multi_mat(ei, :) = [r.auc_cratio_multi, r.auc_q_multi];
end

subplot(1,3,1);
bar(categorical(string(eta_values)), auc_q_mat);
ylabel('AUC');
title('整体/候选观测的 q_i 区分力');
legend({'q_i (全部观测)', 'q_i (候选观测)'}, 'Location', 'best');
ylim([0.4, 1.05]);
yline(0.5, 'k--', '随机');
grid on;

subplot(1,3,2);
bar(categorical(string(eta_values)), auc_single_mat);
ylabel('AUC');
title('单源候选统计');
legend({'margin', 'q_i'}, 'Location', 'best');
ylim([0.4, 1.05]);
yline(0.5, 'k--', '随机');
grid on;

subplot(1,3,3);
bar(categorical(string(eta_values)), auc_multi_mat);
ylabel('AUC');
title('多源候选统计');
legend({'C_{ratio}', 'q_i'}, 'Location', 'best');
ylim([0.4, 1.05]);
yline(0.5, 'k--', '随机');
grid on;

sgtitle('Response-Gain 前置诊断：AUC 汇总');
saveas(fig1, fullfile(out_dir, 'fig1_auc_summary.png'));
close(fig1);

%% 图2：关键分布
fig2 = figure('Visible', 'off', 'Position', [50 50 1500 950], 'Color', 'w');

for ei = 1:num_eta
    r = all_results{ei};

    sig_cand = r.signal.above_count > 0;
    stb_cand = r.stable.above_count > 0;
    sig_single = r.signal.above_count == 1;
    stb_single = r.stable.above_count == 1;
    sig_multi = r.signal.above_count > 1;
    stb_multi = r.stable.above_count > 1;

    subplot(3, num_eta, ei);
    if any(stb_cand)
        histogram(r.stable.q(stb_cand), 30, 'FaceColor', [0.7 0.7 0.7], ...
            'FaceAlpha', 0.7, 'Normalization', 'probability');
        hold on;
    end
    if any(sig_cand)
        histogram(r.signal.q(sig_cand), 25, 'FaceColor', [0.2 0.4 0.8], ...
            'FaceAlpha', 0.7, 'Normalization', 'probability');
    end
    xlabel('q_i'); ylabel('概率');
    title(sprintf('\\eta=%.2f 候选观测 q_i', r.eta));
    if ei == 1, legend('稳定', '即将激活', 'Location', 'best'); end
    grid on;

    subplot(3, num_eta, num_eta + ei);
    if any(stb_single)
        histogram(r.stable.margin(stb_single), 30, 'FaceColor', [0.7 0.7 0.7], ...
            'FaceAlpha', 0.7, 'Normalization', 'probability');
        hold on;
    end
    if any(sig_single)
        histogram(r.signal.margin(sig_single), 25, 'FaceColor', [0.85 0.3 0.1], ...
            'FaceAlpha', 0.7, 'Normalization', 'probability');
    end
    xlabel('margin'); ylabel('概率');
    title(sprintf('\\eta=%.2f 单源候选', r.eta));
    if ei == 1, legend('稳定', '即将激活', 'Location', 'best'); end
    grid on;

    subplot(3, num_eta, 2 * num_eta + ei);
    if any(stb_multi)
        histogram(log10(max(r.stable.cratio(stb_multi), 1)), 30, ...
            'FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.7, ...
            'Normalization', 'probability');
        hold on;
    end
    if any(sig_multi)
        histogram(log10(max(r.signal.cratio(sig_multi), 1)), 25, ...
            'FaceColor', [0.1 0.7 0.3], 'FaceAlpha', 0.7, ...
            'Normalization', 'probability');
    end
    xlabel('log_{10}(C_{ratio})'); ylabel('概率');
    title(sprintf('\\eta=%.2f 多源候选', r.eta));
    if ei == 1, legend('稳定', '即将激活', 'Location', 'best'); end
    grid on;
end

sgtitle('Response-Gain 关键分布');
saveas(fig2, fullfile(out_dir, 'fig2_component_distributions.png'));
close(fig2);

%% 图3：候选结构占比
fig3 = figure('Visible', 'off', 'Position', [50 50 1400 420], 'Color', 'w');
for ei = 1:num_eta
    r = all_results{ei};

    signal_share = [ ...
        mean(r.signal.above_count == 0), ...
        mean(r.signal.above_count == 1), ...
        mean(r.signal.above_count > 1)];
    stable_share = [ ...
        mean(r.stable.above_count == 0), ...
        mean(r.stable.above_count == 1), ...
        mean(r.stable.above_count > 1)];

    subplot(1, num_eta, ei);
    bar([signal_share; stable_share], 'stacked');
    set(gca, 'XTickLabel', {'signal', 'stable'});
    ylabel('占比');
    xlabel('组别');
    title(sprintf('\\eta=%.2f 候选结构', r.eta));
    legend({'|A_i|=0', '|A_i|=1', '|A_i|>1'}, 'Location', 'best');
    ylim([0, 1]);
    grid on;
end

sgtitle('Response-Gain 候选结构占比');
saveas(fig3, fullfile(out_dir, 'fig3_candidate_regimes.png'));
close(fig3);

%% 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');

fprintf(fid, '========================================================================\n');
fprintf(fid, 'Response-Gain 前置诊断报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '========================================================================\n\n');

fprintf(fid, '实验设置:\n');
fprintf(fid, '  η = %s\n', mat2str(eta_values));
fprintf(fid, '  num_trials = %d, signal_horizon = %d\n', num_trials, signal_horizon);
fprintf(fid, '  Binary 基线动力学：useAdaptiveThreshold=true, adaptiveThresholdMode=binary\n');
fprintf(fid, '  signal 组：首次激活时刻的局部统计量\n');
fprintf(fid, '  stable 组：脉冲后从未激活粒子的背景观测\n');
fprintf(fid, '  q_i / margin / |A_i| 统一使用 cj_low 作为参考阈值，不在动力学中开启 response-gain\n\n');

fprintf(fid, 'Response-Gain 参数:\n');
fprintf(fid, '  g_min = %.2f, beta = %.2f, C_low = %.1f, C_high = %.1f, margin_ref = %.2f\n\n', ...
    params.responseGainMin, params.responseGainBeta, ...
    params.responseGainClow, params.responseGainChigh, params.responseGainMarginRef);

fprintf(fid, '一、AUC 汇总\n');
fprintf(fid, '-----------------------------------------------------------------------------------------------\n');
fprintf(fid, '%-6s %8s %8s %8s %8s %8s %8s\n', ...
    'eta', 'q_all', 'q_cand', 'margin1', 'q_1src', 'Cr_multi', 'q_multi');
for ei = 1:num_eta
    r = all_results{ei};
    fprintf(fid, '%-6.2f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
        r.eta, r.auc_q_all, r.auc_q_candidate, r.auc_margin_single, ...
        r.auc_q_single, r.auc_cratio_multi, r.auc_q_multi);
end

fprintf(fid, '\n二、候选结构占比\n');
fprintf(fid, '-----------------------------------------------------------------------------------------------\n');
fprintf(fid, '%-6s %10s %10s %10s %10s %10s %10s\n', ...
    'eta', 'sig_cand', 'stb_cand', 'sig_|A|=1', 'stb_|A|=1', 'sig_|A|>1', 'stb_|A|>1');
for ei = 1:num_eta
    r = all_results{ei};
    fprintf(fid, '%-6.2f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n', ...
        r.eta, r.signal_candidate_rate, r.stable_candidate_rate, ...
        r.signal_single_share, r.stable_single_share, ...
        r.signal_multi_share, r.stable_multi_share);
end

fprintf(fid, '\n三、关键中位数（候选观测）\n');
fprintf(fid, '-----------------------------------------------------------------------------------------------\n');
fprintf(fid, '%-6s %10s %10s %10s %10s %10s %10s\n', ...
    'eta', 'q_sig', 'q_stb', 'm1_sig', 'm1_stb', 'Cr_sig', 'Cr_stb');
for ei = 1:num_eta
    r = all_results{ei};
    fprintf(fid, '%-6.2f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n', ...
        r.eta, r.signal_q_candidate_median, r.stable_q_candidate_median, ...
        r.signal_margin_single_median, r.stable_margin_single_median, ...
        r.signal_cratio_multi_median, r.stable_cratio_multi_median);
end

fprintf(fid, '\n四、初步结论\n');
fprintf(fid, '-----------------------------------------------------------------------------------------------\n');
for ei = 1:num_eta
    r = all_results{ei};
    fprintf(fid, 'η=%.2f:\n', r.eta);
    if r.auc_q_candidate >= 0.75
        fprintf(fid, '  q_i 在候选观测上有较强区分力 (AUC=%.3f)。\n', r.auc_q_candidate);
    elseif r.auc_q_candidate >= 0.65
        fprintf(fid, '  q_i 在候选观测上有中等区分力 (AUC=%.3f)。\n', r.auc_q_candidate);
    else
        fprintf(fid, '  q_i 在候选观测上区分力偏弱 (AUC=%.3f)。\n', r.auc_q_candidate);
    end

    if r.signal_single_share >= 0.5
        fprintf(fid, '  signal 观测以单源候选为主 (占比 %.1f%%)，margin 分支很关键。\n', ...
            100 * r.signal_single_share);
    else
        fprintf(fid, '  signal 观测中多源候选占比较高 (单源占比 %.1f%%)。\n', ...
            100 * r.signal_single_share);
    end

    if r.auc_margin_single >= 0.70
        fprintf(fid, '  单源 margin 有较好区分力 (AUC=%.3f)。\n', r.auc_margin_single);
    elseif r.auc_margin_single >= 0.60
        fprintf(fid, '  单源 margin 只有边缘区分力 (AUC=%.3f)。\n', r.auc_margin_single);
    else
        fprintf(fid, '  单源 margin 区分力较弱 (AUC=%.3f)，response-gain 单源分支存在风险。\n', r.auc_margin_single);
    end

    if r.auc_cratio_multi >= 0.70
        fprintf(fid, '  多源 C_ratio 有较好区分力 (AUC=%.3f)。\n', r.auc_cratio_multi);
    elseif r.auc_cratio_multi >= 0.60
        fprintf(fid, '  多源 C_ratio 只有边缘区分力 (AUC=%.3f)。\n', r.auc_cratio_multi);
    else
        fprintf(fid, '  多源 C_ratio 区分力较弱 (AUC=%.3f)。\n', r.auc_cratio_multi);
    end
    fprintf(fid, '\n');
end

fclose(fid);

%% 保存数据
saveResultBundle(out_dir, 'diagnosis_data', ...
    {'all_results', 'eta_values', 'params', 'adaptive_cfg', ...
     'num_trials', 'signal_horizon', 'diagnosis_elapsed'});

fprintf('\n完成。输出目录: %s\n', out_dir);

%% ========================================================================
function stats = emptyStatGroup()
    stats = struct();
    stats.sigma2 = [];
    stats.cratio = [];
    stats.smax = [];
    stats.above_count = [];
    stats.margin = [];
    stats.q = [];
    stats.gain = [];
end

function stats = appendStatGroup(stats, incoming)
    fields = fieldnames(stats);
    for fi = 1:numel(fields)
        f = fields{fi};
        stats.(f) = [stats.(f); incoming.(f)]; %#ok<AGROW>
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

function ratio = safeRatio(num, den)
    if den <= 0
        ratio = NaN;
    else
        ratio = num / den;
    end
end

function auc = mannWhitneyAUC(pos, neg)
    pos = pos(isfinite(pos));
    neg = neg(isfinite(neg));
    if isempty(pos) || isempty(neg)
        auc = NaN;
        return;
    end
    neg = sort(neg(:));
    score = 0;
    for i = 1:numel(pos)
        left = sum(neg < pos(i));
        right = sum(neg <= pos(i));
        score = score + left + 0.5 * (right - left);
    end
    auc = score / (numel(pos) * numel(neg));
end

function [sigma2_val, cratio_val, smax_val, above_count, margin_val, q_val, gain_val] = ...
        computeResponseGainStats(sim, i, neighbor_idx, threshold_ref)
% computeResponseGainStats 计算 response-gain 所需的局部统计量

    if isempty(neighbor_idx)
        sigma2_val = 0;
        cratio_val = 1;
        smax_val = 0;
        above_count = 0;
        margin_val = 0;
        q_val = 0;
        gain_val = 0;
        return;
    end

    s_values = sim.computeSaliencyValues(i, neighbor_idx);
    if isempty(s_values)
        sigma2_val = 0;
        cratio_val = 1;
        smax_val = 0;
        above_count = 0;
        margin_val = 0;
        q_val = 0;
        gain_val = 0;
        return;
    end

    smax_val = max(s_values);
    sigma2_val = var(s_values, 1, 'omitnan');
    if numel(s_values) >= 2
        cratio_val = max(s_values) / (median(s_values) + eps);
    else
        cratio_val = 1;
    end

    above_s = s_values(s_values > threshold_ref);
    above_count = numel(above_s);
    if above_count == 0
        margin_val = 0;
        q_val = 0;
        gain_val = 0;
        return;
    end

    margin_val = (max(above_s) - threshold_ref) / max(threshold_ref, eps);
    q_val = sim.normalizeConfidenceFromSaliency(above_s, threshold_ref);
    gain_val = sim.computeResponseGain(q_val);
end

function trial_res = runSingleResponseGainDiagnosis(params, eta_values, eta_idx, trial, horizon)
% runSingleResponseGainDiagnosis 单次脉冲诊断：采集 response-gain 相关统计量

    eta = eta_values(eta_idx);
    rng(trial * 1000 + eta_idx);

    p = params;
    p.angleNoiseIntensity = eta^2 / 2;

    sim = ParticleSimulationWithExternalPulse(p);
    sim.external_pulse_count = 1;
    sim.setLogging(false);
    sim.resetCascadeTracking();
    sim.initializeParticles();

    N = p.N;
    T = p.T_max;
    threshold_ref = p.adaptiveThresholdConfig.cj_low;

    sigma2_history = zeros(N, T);
    cratio_history = zeros(N, T);
    smax_history = zeros(N, T);
    above_count_history = zeros(N, T);
    margin_history = zeros(N, T);
    q_history = zeros(N, T);
    gain_history = zeros(N, T);

    first_activation_step = zeros(N, 1);
    external_ids = false(N, 1);
    pulse_step = NaN;

    for t = 1:T
        was_active = sim.isActive;
        sim.step();

        if sim.external_pulse_triggered
            if isnan(pulse_step)
                pulse_step = t;
            end
            external_ids = external_ids | sim.isExternallyActivated;
        end

        newly_activated = sim.isActive & ~was_active;
        for i = find(newly_activated)'
            if first_activation_step(i) == 0
                first_activation_step(i) = t;
            end
        end

        neighbor_matrix = sim.findNeighbors();
        for i = 1:N
            neighbor_idx = find(neighbor_matrix(i, :));
            [s2, cr, sm, ac, mg, qv, gv] = computeResponseGainStats(sim, i, neighbor_idx, threshold_ref);
            sigma2_history(i, t) = s2;
            cratio_history(i, t) = cr;
            smax_history(i, t) = sm;
            above_count_history(i, t) = ac;
            margin_history(i, t) = mg;
            q_history(i, t) = qv;
            gain_history(i, t) = gv;
        end
    end

    if isnan(pulse_step)
        pulse_step = sim.stabilization_steps + 1;
    end

    signal = emptyStatGroup();
    stable = emptyStatGroup();

    for i = 1:N
        if external_ids(i)
            continue;
        end

        if first_activation_step(i) > 0
            t_act = first_activation_step(i);
            signal.sigma2 = [signal.sigma2; sigma2_history(i, t_act)]; %#ok<AGROW>
            signal.cratio = [signal.cratio; cratio_history(i, t_act)]; %#ok<AGROW>
            signal.smax = [signal.smax; smax_history(i, t_act)]; %#ok<AGROW>
            signal.above_count = [signal.above_count; above_count_history(i, t_act)]; %#ok<AGROW>
            signal.margin = [signal.margin; margin_history(i, t_act)]; %#ok<AGROW>
            signal.q = [signal.q; q_history(i, t_act)]; %#ok<AGROW>
            signal.gain = [signal.gain; gain_history(i, t_act)]; %#ok<AGROW>
        else
            t_start = min(pulse_step + 1, T);
            t_end = T;
            if t_end >= t_start
                idx = t_start:t_end;
                stable.sigma2 = [stable.sigma2; sigma2_history(i, idx)']; %#ok<AGROW>
                stable.cratio = [stable.cratio; cratio_history(i, idx)']; %#ok<AGROW>
                stable.smax = [stable.smax; smax_history(i, idx)']; %#ok<AGROW>
                stable.above_count = [stable.above_count; above_count_history(i, idx)']; %#ok<AGROW>
                stable.margin = [stable.margin; margin_history(i, idx)']; %#ok<AGROW>
                stable.q = [stable.q; q_history(i, idx)']; %#ok<AGROW>
                stable.gain = [stable.gain; gain_history(i, idx)']; %#ok<AGROW>
            end
        end
    end

    trial_res = struct();
    trial_res.signal = signal;
    trial_res.stable = stable;
end
