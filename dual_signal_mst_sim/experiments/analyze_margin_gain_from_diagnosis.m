%% analyze_margin_gain_from_diagnosis.m
% 基于已有 response_gain_diagnosis 结果，离线分析
% gain = min((margin + 1) / c, 1) 这条 margin-only 路线是否有前景
%
% 输出（results/margin_gain_scan_<timestamp>/）：
%   - analysis_report.txt
%   - fig1_auc_vs_c.png
%   - fig2_tradeoff_vs_c.png
%   - fig3_histograms_eta_xx.png（每个 eta 一张）
%   - margin_gain_scan.mat
%   - margin_gain_scan.json

clear; clc; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 输入设置
% 若留空，则自动读取最新的 response_gain_diagnosis 结果目录
source_dir = '';

% 扫描的 c 值
c_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0];

% 经验筛选阈值
signal_gain_high_thr = 0.90;
signal_gain_mid_thr = 0.80;
stable_gain_low_thr = 0.50;
stable_gain_vlow_thr = 0.40;

%% 2. 读取数据
if isempty(source_dir)
    results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
    candidates = dir(fullfile(results_dir, 'response_gain_diagnosis_*'));
    candidates = candidates([candidates.isdir]);
    if isempty(candidates)
        error('未找到 response_gain_diagnosis_* 结果目录。');
    end
    [~, idx] = max([candidates.datenum]);
    source_dir = fullfile(candidates(idx).folder, candidates(idx).name);
end

mat_path = fullfile(source_dir, 'diagnosis_data.mat');
if ~exist(mat_path, 'file')
    error('未找到诊断数据文件: %s', mat_path);
end

S = load(mat_path, 'all_results', 'eta_values', 'params', 'adaptive_cfg', 'num_trials', 'signal_horizon', 'diagnosis_elapsed');

all_results = S.all_results;
eta_values = S.eta_values;
params = S.params;
adaptive_cfg = S.adaptive_cfg;
num_trials = S.num_trials;
signal_horizon = S.signal_horizon;

%% 3. 输出目录
results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_dir = fullfile(results_dir, sprintf('margin_gain_scan_%s', timestamp));
mkdir(out_dir);

fprintf('输入目录: %s\n', source_dir);
fprintf('输出目录: %s\n', out_dir);
fprintf('扫描 c = %s\n\n', mat2str(c_values));

%% 4. 主分析
num_eta = numel(eta_values);
num_c = numel(c_values);

summary = cell(num_eta, 1);

for ei = 1:num_eta
    r = all_results{ei};

    sig = r.signal;
    stb = r.stable;

    sig_cand = sig.above_count > 0;
    stb_cand = stb.above_count > 0;
    sig_single = sig.above_count == 1;
    stb_single = stb.above_count == 1;

    gain_sig_cand = zeros(sum(sig_cand), num_c);
    gain_stb_cand = zeros(sum(stb_cand), num_c);
    gain_sig_single = zeros(sum(sig_single), num_c);
    gain_stb_single = zeros(sum(stb_single), num_c);

    auc_candidate = NaN(1, num_c);
    auc_single = NaN(1, num_c);
    sig_ge_09 = NaN(1, num_c);
    sig_ge_08 = NaN(1, num_c);
    stb_le_05 = NaN(1, num_c);
    stb_le_04 = NaN(1, num_c);
    sig_single_ge_09 = NaN(1, num_c);
    stb_single_le_05 = NaN(1, num_c);
    score = NaN(1, num_c);

    for ci = 1:num_c
        c = c_values(ci);

        gain_sig_cand(:, ci) = computeGainFromMargin(sig.margin(sig_cand), c);
        gain_stb_cand(:, ci) = computeGainFromMargin(stb.margin(stb_cand), c);
        gain_sig_single(:, ci) = computeGainFromMargin(sig.margin(sig_single), c);
        gain_stb_single(:, ci) = computeGainFromMargin(stb.margin(stb_single), c);

        auc_candidate(ci) = mannWhitneyAUC(gain_sig_cand(:, ci), gain_stb_cand(:, ci));
        auc_single(ci) = mannWhitneyAUC(gain_sig_single(:, ci), gain_stb_single(:, ci));

        sig_ge_09(ci) = mean(gain_sig_cand(:, ci) >= signal_gain_high_thr);
        sig_ge_08(ci) = mean(gain_sig_cand(:, ci) >= signal_gain_mid_thr);
        stb_le_05(ci) = mean(gain_stb_cand(:, ci) <= stable_gain_low_thr);
        stb_le_04(ci) = mean(gain_stb_cand(:, ci) <= stable_gain_vlow_thr);

        sig_single_ge_09(ci) = mean(gain_sig_single(:, ci) >= signal_gain_high_thr);
        stb_single_le_05(ci) = mean(gain_stb_single(:, ci) <= stable_gain_low_thr);

        % 简单综合分数：希望 signal 保高增益、stable 压低增益
        score(ci) = sig_ge_09(ci) + stb_le_05(ci);
    end

    res = struct();
    res.eta = r.eta;
    res.n_signal_candidate = sum(sig_cand);
    res.n_stable_candidate = sum(stb_cand);
    res.n_signal_single = sum(sig_single);
    res.n_stable_single = sum(stb_single);
    res.c_values = c_values;
    res.auc_candidate = auc_candidate;
    res.auc_single = auc_single;
    res.sig_ge_09 = sig_ge_09;
    res.sig_ge_08 = sig_ge_08;
    res.stb_le_05 = stb_le_05;
    res.stb_le_04 = stb_le_04;
    res.sig_single_ge_09 = sig_single_ge_09;
    res.stb_single_le_05 = stb_single_le_05;
    res.score = score;
    res.gain_sig_cand = gain_sig_cand;
    res.gain_stb_cand = gain_stb_cand;
    res.gain_sig_single = gain_sig_single;
    res.gain_stb_single = gain_stb_single;

    [~, best_idx] = max(score);
    res.best_c = c_values(best_idx);
    res.best_score = score(best_idx);

    good_idx = find(sig_ge_09 >= 0.95 & stb_le_05 >= 0.80, 1, 'first');
    if ~isempty(good_idx)
        res.feasible_c = c_values(good_idx);
    else
        res.feasible_c = NaN;
    end

    summary{ei} = res;

    fprintf('η=%.2f: best c=%.1f, score=%.3f, feasible c=%s\n', ...
        res.eta, res.best_c, res.best_score, num2str(res.feasible_c));
end
fprintf('\n');

%% 5. 图1：AUC vs c
fig1 = figure('Visible', 'off', 'Position', [50 50 1200 450], 'Color', 'w');

subplot(1,2,1); hold on;
for ei = 1:num_eta
    plot(c_values, summary{ei}.auc_candidate, 'o-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\eta=%.2f', eta_values(ei)));
end
xlabel('c'); ylabel('AUC');
title('候选样本上的 gain AUC');
yline(0.5, 'k--', '随机');
legend('Location', 'best');
grid on;

subplot(1,2,2); hold on;
for ei = 1:num_eta
    plot(c_values, summary{ei}.auc_single, 's-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\eta=%.2f', eta_values(ei)));
end
xlabel('c'); ylabel('AUC');
title('单源样本上的 gain AUC');
yline(0.5, 'k--', '随机');
legend('Location', 'best');
grid on;

sgtitle('margin-only gain 的区分力');
saveas(fig1, fullfile(out_dir, 'fig1_auc_vs_c.png'));
close(fig1);

%% 6. 图2：tradeoff 曲线
fig2 = figure('Visible', 'off', 'Position', [50 50 1500 450], 'Color', 'w');

for ei = 1:num_eta
    res = summary{ei};

    subplot(1, num_eta, ei);
    yyaxis left;
    plot(c_values, res.sig_ge_09, 'o-', 'LineWidth', 1.5, ...
        'Color', [0.2 0.4 0.8], 'DisplayName', 'signal gain>=0.9');
    hold on;
    plot(c_values, res.sig_ge_08, 's--', 'LineWidth', 1.2, ...
        'Color', [0.1 0.6 0.9], 'DisplayName', 'signal gain>=0.8');
    ylabel('signal 保持高增益比例');
    ylim([0, 1]);

    yyaxis right;
    plot(c_values, res.stb_le_05, 'd-', 'LineWidth', 1.5, ...
        'Color', [0.85 0.2 0.2], 'DisplayName', 'stable gain<=0.5');
    plot(c_values, res.stb_le_04, '^-', 'LineWidth', 1.2, ...
        'Color', [0.7 0.1 0.1], 'DisplayName', 'stable gain<=0.4');
    ylabel('stable 被压低比例');
    ylim([0, 1]);

    xlabel('c');
    title(sprintf('\\eta=%.2f', res.eta));
    xline(res.best_c, 'k--', 'best c', 'LineWidth', 1);
    grid on;
end

saveas(fig2, fullfile(out_dir, 'fig2_tradeoff_vs_c.png'));
close(fig2);

%% 7. 图3：每个 eta 的增益分布直方图
for ei = 1:num_eta
    res = summary{ei};
    fig = figure('Visible', 'off', 'Position', [50 50 1600 900], 'Color', 'w');

    for ci = 1:num_c
        subplot(2, ceil(num_c / 2), ci); hold on;
        histogram(res.gain_stb_cand(:, ci), 25, 'Normalization', 'probability', ...
            'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.55);
        histogram(res.gain_sig_cand(:, ci), 25, 'Normalization', 'probability', ...
            'FaceColor', [0.2 0.4 0.8], 'FaceAlpha', 0.55);
        xlabel('gain');
        ylabel('概率');
        title(sprintf('\\eta=%.2f, c=%.1f', res.eta, c_values(ci)));
        if ci == 1
            legend('stable-candidate', 'signal', 'Location', 'best');
        end
        grid on;
    end

    sgtitle(sprintf('margin-only gain 分布 (\\eta=%.2f)', res.eta));
    saveas(fig, fullfile(out_dir, sprintf('fig3_histograms_eta_%.2f.png', res.eta)));
    close(fig);
end

%% 8. 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');

fprintf(fid, '========================================================================\n');
fprintf(fid, 'margin-only gain 离线扫描报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '========================================================================\n\n');

fprintf(fid, '输入诊断目录: %s\n', source_dir);
fprintf(fid, '诊断来源: response_gain_diagnosis 结果中的 margin 分布\n');
fprintf(fid, '扫描公式: gain = min((margin + 1) / c, 1)\n');
fprintf(fid, 'c 扫描: %s\n\n', mat2str(c_values));

fprintf(fid, '判据:\n');
fprintf(fid, '  signal gain >= %.2f / %.2f\n', signal_gain_high_thr, signal_gain_mid_thr);
fprintf(fid, '  stable gain <= %.2f / %.2f\n\n', stable_gain_low_thr, stable_gain_vlow_thr);

for ei = 1:num_eta
    res = summary{ei};
    fprintf(fid, 'η = %.2f\n', res.eta);
    fprintf(fid, '  样本量: signal-candidate=%d, stable-candidate=%d, signal-single=%d, stable-single=%d\n', ...
        res.n_signal_candidate, res.n_stable_candidate, ...
        res.n_signal_single, res.n_stable_single);
    fprintf(fid, '  best c = %.1f (score=%.3f)\n', res.best_c, res.best_score);
    if ~isnan(res.feasible_c)
        fprintf(fid, '  first feasible c = %.1f (signal>=0.9 且 stable<=0.5)\n', res.feasible_c);
    else
        fprintf(fid, '  未找到同时满足 signal>=0.9 且 stable<=0.5 的 c\n');
    end
    fprintf(fid, '\n');

    fprintf(fid, '  %-6s %8s %8s %8s %8s %8s %8s\n', ...
        'c', 'AUC_c', 'AUC_1', 'sig>.9', 'sig>.8', 'stb<.5', 'stb<.4');
    for ci = 1:num_c
        fprintf(fid, '  %-6.1f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
            c_values(ci), res.auc_candidate(ci), res.auc_single(ci), ...
            res.sig_ge_09(ci), res.sig_ge_08(ci), ...
            res.stb_le_05(ci), res.stb_le_04(ci));
    end
    fprintf(fid, '\n');
end

fprintf(fid, '总结建议:\n');
fprintf(fid, '  1. 若某个 c 使 signal gain>=0.9 的比例高，同时 stable gain<=0.5 的比例也高，则值得进入 R-P 仿真。\n');
fprintf(fid, '  2. 若所有 c 都无法兼顾两者，则 margin-only gain 作为 5.3 的前景有限。\n');
fprintf(fid, '  3. AUC_candidate 反映在真正候选事件上的整体分离度，AUC_single 反映单源场景上的分离度上限。\n');
fclose(fid);

%% 9. 保存数据
saveResultBundle(out_dir, 'margin_gain_scan', ...
    {'summary', 'c_values', 'signal_gain_high_thr', 'signal_gain_mid_thr', ...
     'stable_gain_low_thr', 'stable_gain_vlow_thr', ...
     'source_dir', 'eta_values', 'params', 'adaptive_cfg', ...
     'num_trials', 'signal_horizon', 'diagnosis_elapsed'});

fprintf('完成。输出目录: %s\n', out_dir);

%% ========================================================================
function gain = computeGainFromMargin(margin_values, c)
    gain = min((margin_values(:) + 1) ./ c, 1);
    gain = max(gain, 0);
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
