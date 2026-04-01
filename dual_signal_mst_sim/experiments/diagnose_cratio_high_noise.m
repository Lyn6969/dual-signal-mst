%% diagnose_cratio_high_noise.m
% 诊断 C_ratio 在高噪声下的信噪区分力
%
% 在 η=0.25/0.30/0.40 下，对比 sigma2_M 和 C_ratio 对
% "即将激活粒子"(h≤3) vs "稳定粒子" 的区分能力。
%
% 输出（results/cratio_diagnosis_<timestamp>/）：
%   - analysis_report.txt
%   - fig1_auc_comparison.png：sigma2_M vs C_ratio 的 AUC 对比
%   - fig2_distributions.png：各 η 下信号/噪声分布
%   - fig3_joint_scatter.png：二维信号空间散点图
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
params.T_max = 350;
params.dt = 0.1;
params.radius = 5;
params.deac_threshold = 0.1745;
params.cj_threshold = 1.5;
params.fieldSize = 50;
params.initDirection = pi/4;
params.useFixedField = true;
params.useAdaptiveThreshold = false;
params.stabilization_steps = 100;
params.external_pulse_count = 1;
params.forced_turn_duration = 50;

eta_values = [0.25, 0.30, 0.40];
num_trials = 10;
signal_horizon = 3;

%% 输出目录
results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_dir = fullfile(results_dir, sprintf('cratio_diagnosis_%s', timestamp));
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

%% 采集（展平为 (eta, trial) 的 parfor）
num_eta = numel(eta_values);
total_jobs = num_eta * num_trials;

% 展开映射
job_eta_idx = zeros(total_jobs, 1);
job_trial = zeros(total_jobs, 1);
for ei = 1:num_eta
    for ti = 1:num_trials
        jid = (ei - 1) * num_trials + ti;
        job_eta_idx(jid) = ei;
        job_trial(jid) = ti;
    end
end

% 每个 job 返回信号和稳定粒子的统计量
job_signal_sigma2 = cell(total_jobs, 1);
job_signal_cratio = cell(total_jobs, 1);
job_signal_smax = cell(total_jobs, 1);
job_stable_sigma2 = cell(total_jobs, 1);
job_stable_cratio = cell(total_jobs, 1);
job_stable_smax = cell(total_jobs, 1);

fprintf('开始 %d 个诊断任务（%d η × %d trials）...\n', total_jobs, num_eta, num_trials);
timer_start = tic;

if use_parallel
    parfor jid = 1:total_jobs
        [sig_s2, sig_cr, sig_sm, stb_s2, stb_cr, stb_sm] = ...
            runSingleDiagnosis(params, eta_values, job_eta_idx(jid), job_trial(jid), signal_horizon);
        job_signal_sigma2{jid} = sig_s2;
        job_signal_cratio{jid} = sig_cr;
        job_signal_smax{jid} = sig_sm;
        job_stable_sigma2{jid} = stb_s2;
        job_stable_cratio{jid} = stb_cr;
        job_stable_smax{jid} = stb_sm;
    end
else
    for jid = 1:total_jobs
        [sig_s2, sig_cr, sig_sm, stb_s2, stb_cr, stb_sm] = ...
            runSingleDiagnosis(params, eta_values, job_eta_idx(jid), job_trial(jid), signal_horizon);
        job_signal_sigma2{jid} = sig_s2;
        job_signal_cratio{jid} = sig_cr;
        job_signal_smax{jid} = sig_sm;
        job_stable_sigma2{jid} = stb_s2;
        job_stable_cratio{jid} = stb_cr;
        job_stable_smax{jid} = stb_sm;
        fprintf('  [%d/%d] η=%.2f trial=%d 完成\n', jid, total_jobs, eta_values(job_eta_idx(jid)), job_trial(jid));
    end
end

elapsed = toc(timer_start);
fprintf('采集完成，耗时 %.1f 秒\n\n', elapsed);

%% 按 η 聚合
all_results = cell(num_eta, 1);

for ei = 1:num_eta
    signal_sigma2 = []; signal_cratio = []; signal_smax = [];
    stable_sigma2 = []; stable_cratio = []; stable_smax = [];

    for ti = 1:num_trials
        jid = (ei - 1) * num_trials + ti;
        signal_sigma2 = [signal_sigma2; job_signal_sigma2{jid}]; %#ok<AGROW>
        signal_cratio = [signal_cratio; job_signal_cratio{jid}]; %#ok<AGROW>
        signal_smax = [signal_smax; job_signal_smax{jid}]; %#ok<AGROW>
        stable_sigma2 = [stable_sigma2; job_stable_sigma2{jid}]; %#ok<AGROW>
        stable_cratio = [stable_cratio; job_stable_cratio{jid}]; %#ok<AGROW>
        stable_smax = [stable_smax; job_stable_smax{jid}]; %#ok<AGROW>
    end

    % 计算 AUC
    auc_sigma2 = mannWhitneyAUC(signal_sigma2(:), stable_sigma2(:));
    auc_cratio = mannWhitneyAUC(signal_cratio(:), stable_cratio(:));
    auc_smax = mannWhitneyAUC(signal_smax(:), stable_smax(:));

    res = struct();
    res.eta = eta_values(ei);
    res.signal_sigma2 = signal_sigma2(:);
    res.signal_cratio = signal_cratio(:);
    res.signal_smax = signal_smax(:);
    res.stable_sigma2 = stable_sigma2(:);
    res.stable_cratio = stable_cratio(:);
    res.stable_smax = stable_smax(:);
    res.auc_sigma2 = auc_sigma2;
    res.auc_cratio = auc_cratio;
    res.auc_smax = auc_smax;
    res.n_signal = numel(signal_sigma2);
    res.n_stable = numel(stable_sigma2);

    all_results{ei} = res;

    fprintf('  AUC: sigma2_M=%.3f, C_ratio=%.3f, s_max=%.3f (n_sig=%d, n_stab=%d)\n', ...
        auc_sigma2, auc_cratio, auc_smax, res.n_signal, res.n_stable);
end

%% 图1：AUC 对比柱状图
fig1 = figure('Visible', 'off', 'Position', [50 50 800 450], 'Color', 'w');

auc_mat = zeros(num_eta, 3);
for ei = 1:num_eta
    auc_mat(ei, :) = [all_results{ei}.auc_sigma2, all_results{ei}.auc_cratio, all_results{ei}.auc_smax];
end

bar(categorical(string(eta_values)), auc_mat);
ylabel('AUC（即将激活 vs 稳定）');
xlabel('\eta');
title('高噪声下三种信号的区分能力');
legend({'\sigma^2_M', 'C_{ratio}', 's_{max}'}, 'Location', 'best');
ylim([0.4, 1.05]);
yline(0.5, 'k--', '随机水平');
grid on;

saveas(fig1, fullfile(out_dir, 'fig1_auc_comparison.png'));
close(fig1);

%% 图2：各 η 下 sigma2_M 和 C_ratio 的信号/噪声分布
fig2 = figure('Visible', 'off', 'Position', [50 50 1400 800], 'Color', 'w');

for ei = 1:num_eta
    r = all_results{ei};

    % sigma2_M 分布
    subplot(2, num_eta, ei);
    histogram(log10(max(r.stable_sigma2, 1e-12)), 40, ...
        'FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.7, 'Normalization', 'probability');
    hold on;
    histogram(log10(max(r.signal_sigma2, 1e-12)), 30, ...
        'FaceColor', [0.2 0.4 0.8], 'FaceAlpha', 0.7, 'Normalization', 'probability');
    xline(log10(0.05), 'r-', 'V_{ref}', 'LineWidth', 1.5);
    xlabel('log_{10}(\sigma^2_M)');
    ylabel('概率');
    title(sprintf('\\eta=%.2f  \\sigma^2_M  AUC=%.3f', r.eta, r.auc_sigma2));
    if ei == 1, legend('稳定', '即将激活', 'Location', 'best'); end
    grid on;

    % C_ratio 分布
    subplot(2, num_eta, num_eta + ei);
    histogram(log10(max(r.stable_cratio, 1)), 40, ...
        'FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.7, 'Normalization', 'probability');
    hold on;
    histogram(log10(max(r.signal_cratio, 1)), 30, ...
        'FaceColor', [0.85 0.3 0.1], 'FaceAlpha', 0.7, 'Normalization', 'probability');
    xline(log10(30), 'k--', 'C_{low}', 'LineWidth', 1.2);
    xline(log10(150), 'k--', 'C_{high}', 'LineWidth', 1.2);
    xlabel('log_{10}(C_{ratio})');
    ylabel('概率');
    title(sprintf('\\eta=%.2f  C_{ratio}  AUC=%.3f', r.eta, r.auc_cratio));
    if ei == 1, legend('稳定', '即将激活', 'Location', 'best'); end
    grid on;
end

saveas(fig2, fullfile(out_dir, 'fig2_distributions.png'));
close(fig2);

%% 图3：二维散点图（sigma2_M vs C_ratio）
fig3 = figure('Visible', 'off', 'Position', [50 50 1400 450], 'Color', 'w');

for ei = 1:num_eta
    r = all_results{ei};
    subplot(1, num_eta, ei);

    % 稳定粒子（灰色）
    scatter(log10(max(r.stable_sigma2, 1e-12)), log10(max(r.stable_cratio, 1)), ...
        6, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.1);
    hold on;
    % 即将激活粒子（蓝色）
    scatter(log10(max(r.signal_sigma2, 1e-12)), log10(max(r.signal_cratio, 1)), ...
        12, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha', 0.5);

    xline(log10(0.05), 'r-', 'V_{ref}', 'LineWidth', 1.2);
    yline(log10(30), 'k--', 'C_{low}', 'LineWidth', 1);
    yline(log10(150), 'k--', 'C_{high}', 'LineWidth', 1);

    xlabel('log_{10}(\sigma^2_M)');
    ylabel('log_{10}(C_{ratio})');
    title(sprintf('\\eta=%.2f (n_{sig}=%d)', r.eta, r.n_signal));
    if ei == 1, legend('稳定', '即将激活', 'Location', 'best'); end
    grid on;
end

saveas(fig3, fullfile(out_dir, 'fig3_joint_scatter.png'));
close(fig3);

%% 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');

fprintf(fid, '========================================================================\n');
fprintf(fid, 'C_ratio 高噪声信号诊断报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '========================================================================\n\n');

fprintf(fid, '实验设置: %d 次脉冲试验/η, horizon h=%d, V_ref=0.05\n\n', num_trials, signal_horizon);

fprintf(fid, '一、AUC 对比（即将激活 vs 稳定）\n');
fprintf(fid, '------------------------------------------------------------------------\n');
fprintf(fid, '%-6s %10s %10s %10s %8s %8s\n', 'eta', 'sigma2_M', 'C_ratio', 's_max', 'n_sig', 'n_stab');
for ei = 1:num_eta
    r = all_results{ei};
    fprintf(fid, '%-6.2f %10.3f %10.3f %10.3f %8d %8d\n', ...
        r.eta, r.auc_sigma2, r.auc_cratio, r.auc_smax, r.n_signal, r.n_stable);
end

fprintf(fid, '\n二、各信号的中位数与百分位数对比\n');
fprintf(fid, '------------------------------------------------------------------------\n');
for ei = 1:num_eta
    r = all_results{ei};
    fprintf(fid, 'η=%.2f:\n', r.eta);
    fprintf(fid, '  sigma2_M: 信号中位数=%.6f, 稳定中位数=%.6f, 比值=%.1fx\n', ...
        median(r.signal_sigma2), median(r.stable_sigma2), ...
        median(r.signal_sigma2) / max(median(r.stable_sigma2), eps));
    fprintf(fid, '  C_ratio:  信号中位数=%.1f, 稳定中位数=%.1f, 比值=%.1fx\n', ...
        median(r.signal_cratio), median(r.stable_cratio), ...
        median(r.signal_cratio) / max(median(r.stable_cratio), eps));
    fprintf(fid, '  s_max:    信号中位数=%.4f, 稳定中位数=%.4f, 比值=%.1fx\n', ...
        median(r.signal_smax), median(r.stable_smax), ...
        median(r.signal_smax) / max(median(r.stable_smax), eps));
    fprintf(fid, '  --- C_ratio 百分位数（用于标定 C_veto）---\n');
    fprintf(fid, '  稳定 C_ratio: 90th=%.1f, 95th=%.1f, 99th=%.1f\n', ...
        prctile(r.stable_cratio, 90), prctile(r.stable_cratio, 95), prctile(r.stable_cratio, 99));
    fprintf(fid, '  信号 C_ratio: 10th=%.1f, 25th=%.1f, 50th=%.1f\n', ...
        prctile(r.signal_cratio, 10), prctile(r.signal_cratio, 25), prctile(r.signal_cratio, 50));
    fprintf(fid, '  → C_veto 推荐区间: [%.0f, %.0f]\n', ...
        prctile(r.stable_cratio, 95), prctile(r.signal_cratio, 25));
end

fprintf(fid, '\n三、关键问题：C_ratio 在高噪声下是否仍有区分力？\n');
fprintf(fid, '------------------------------------------------------------------------\n');
for ei = 1:num_eta
    r = all_results{ei};
    if r.auc_cratio > 0.7
        verdict = '有效（AUC > 0.7）';
    elseif r.auc_cratio > 0.6
        verdict = '边缘有效（0.6 < AUC < 0.7）';
    else
        verdict = '失效（AUC ≤ 0.6）';
    end
    fprintf(fid, '  η=%.2f: C_ratio AUC=%.3f → %s\n', r.eta, r.auc_cratio, verdict);

    % sigma2_M 对比
    if r.auc_sigma2 < 0.6 && r.auc_cratio > 0.7
        fprintf(fid, '    → sigma2_M 已失效(%.3f), C_ratio 仍有效 → 双信号有明确价值\n', r.auc_sigma2);
    elseif r.auc_sigma2 > 0.7 && r.auc_cratio > 0.7
        fprintf(fid, '    → 两者都有效，C_ratio 可作为互补\n');
    elseif r.auc_sigma2 < 0.6 && r.auc_cratio < 0.6
        fprintf(fid, '    → 两者都失效，该噪声水平下自适应机制可能无法工作\n');
    end
end

fclose(fid);

%% 保存数据
saveResultBundle(out_dir, 'diagnosis_data', ...
    {'all_results', 'eta_values', 'params', 'num_trials', 'signal_horizon'});

fprintf('\n完成。输出目录: %s\n', out_dir);

%% ========================================================================
function [s_max_val, sigma2_val, k_val, c_ratio_val] = computeFullStats(sim, i, neighbor_idx, dt)
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
    c_ratio_val = max(sv) / (median(sv) + eps);
end

function auc = mannWhitneyAUC(pos, neg)
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

function [sig_s2, sig_cr, sig_sm, stb_s2, stb_cr, stb_sm] = ...
        runSingleDiagnosis(params, eta_values, eta_idx, trial, horizon)
% runSingleDiagnosis 单次脉冲诊断：采集信号组和稳定组的统计量
%
% 流程：
%   1. 以指定 η 运行带外源脉冲的仿真
%   2. 每步记录每个粒子的 (σ²_M, C_ratio, s_max)
%   3. 记录每个粒子首次激活的时间步
%   4. 回溯标记"即将激活"粒子（激活前 h 步内的观测）
%   5. 其余为稳定粒子观测

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
    dt = p.dt;

    % 每步每粒子的统计量
    sigma2_history = zeros(N, T);
    cratio_history = zeros(N, T);
    smax_history = zeros(N, T);

    % 记录每个粒子首次激活的时间步（0 = 从未激活）
    first_activation_step = zeros(N, 1);

    for t = 1:T
        % 记录激活前的状态
        was_active = sim.isActive;

        sim.step();

        % 检测新激活
        newly_activated = sim.isActive & ~was_active;
        for i = find(newly_activated)'
            if first_activation_step(i) == 0
                first_activation_step(i) = t;
            end
        end

        % 计算每个粒子的邻域统计量
        neighbor_matrix = sim.findNeighbors();
        for i = 1:N
            neighbor_idx = find(neighbor_matrix(i, :));
            if numel(neighbor_idx) < 2
                sigma2_history(i, t) = 0;
                cratio_history(i, t) = 1;
                smax_history(i, t) = 0;
                continue;
            end
            [sm, s2, ~, cr] = computeFullStats(sim, i, neighbor_idx, dt);
            sigma2_history(i, t) = s2;
            cratio_history(i, t) = cr;
            smax_history(i, t) = sm;
        end
    end

    % 分类：信号组 vs 稳定组
    % 信号组：粒子 i 在时间步 t 满足 first_activation_step(i) > 0
    %         且 t 在 [first_activation_step(i) - horizon, first_activation_step(i) - 1] 内
    %         即激活前 horizon 步内的观测
    % 稳定组：从未激活的粒子，脉冲后的所有时间步
    %         （排除外源激活粒子本身）

    sig_s2 = []; sig_cr = []; sig_sm = [];
    stb_s2 = []; stb_cr = []; stb_sm = [];

    pulse_step = sim.stabilization_steps;  % 脉冲触发的大致时间步

    for i = 1:N
        % 排除外源激活的粒子
        if sim.isExternallyActivated(i)
            continue;
        end

        if first_activation_step(i) > 0
            % 该粒子被级联激活过：提取激活前 horizon 步的数据
            t_act = first_activation_step(i);
            t_start = max(t_act - horizon, 1);
            t_end = t_act - 1;
            if t_end >= t_start
                sig_s2 = [sig_s2; sigma2_history(i, t_start:t_end)']; %#ok<AGROW>
                sig_cr = [sig_cr; cratio_history(i, t_start:t_end)']; %#ok<AGROW>
                sig_sm = [sig_sm; smax_history(i, t_start:t_end)']; %#ok<AGROW>
            end
        else
            % 从未激活的粒子：脉冲后的数据作为稳定组
            t_start = pulse_step + 1;
            t_end = T;
            if t_end >= t_start
                stb_s2 = [stb_s2; sigma2_history(i, t_start:t_end)']; %#ok<AGROW>
                stb_cr = [stb_cr; cratio_history(i, t_start:t_end)']; %#ok<AGROW>
                stb_sm = [stb_sm; smax_history(i, t_start:t_end)']; %#ok<AGROW>
            end
        end
    end
end
