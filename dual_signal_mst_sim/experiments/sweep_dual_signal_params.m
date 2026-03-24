%% sweep_dual_signal_params.m
% 双信号参数扫描：λ_base × α_min 网格搜索
%
% 复用已有的 Fixed 和 Binary baseline，只扫描 Dual-signal-gate 参数。
% 在 η=0.25 下运行，输出 R 和 P 的热力图，找最优参数组合。
%
% 输出（results/param_sweep_<timestamp>/）：
%   - fig1_R_heatmap.png：响应性热力图
%   - fig2_P_heatmap.png：持久性热力图
%   - fig3_dominance.png：严格优于 Binary 的区域
%   - analysis_report.txt
%   - sweep_data.mat

clc; clear; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 参数设置
fprintf('=================================================\n');
fprintf('  双信号参数扫描: λ_base × α_min\n');
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

eta = 0.25;
resp_params.angleNoiseIntensity = eta^2 / 2;
pers_params.angleNoiseIntensity = eta^2 / 2;

num_runs = 50;
num_angles = 1;
base_seed = 20260324;

pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_diffusion = 1e-4;
pers_cfg.min_fit_points = 40;

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5.0;
adaptive_cfg.saliency_threshold = 0.031;
adaptive_cfg.include_self = false;

V_ref = 0.05;
lambda_up = 0.5;
gamma = 1.0;
C_low = 30;
C_high = 150;

% 扫描网格
lambda_base_values = 0.1:0.1:1.0;     % 10 个
alpha_min_values = 0.0:0.1:1.0;       % 11 个
n_lambda = numel(lambda_base_values);
n_alpha = numel(alpha_min_values);
total_combos = n_lambda * n_alpha;     % 110

%% 输出目录
results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_dir = fullfile(results_dir, sprintf('param_sweep_%s', timestamp));
mkdir(out_dir);

time_vec = (0:resp_params.T_max)' * resp_params.dt;

fprintf('η = %.2f, V_ref = %.4f\n', eta, V_ref);
fprintf('λ_base: %s (%d 个)\n', mat2str(lambda_base_values, 2), n_lambda);
fprintf('α_min:  %s (%d 个)\n', mat2str(alpha_min_values, 2), n_alpha);
fprintf('共 %d 组合 × %d 次重复 = %d 次仿真\n\n', total_combos, num_runs, total_combos * num_runs * 2);

%% 2. 先跑 Binary baseline（复用同一组种子）
fprintf('[Binary baseline] %d 次重复...\n', num_runs);
R_binary = NaN(num_runs, 1);
P_binary = NaN(num_runs, 1);

parfor ri = 1:num_runs
    seed = base_seed + 100000 + ri;
    rp = resp_params;
    rp.useAdaptiveThreshold = true;
    rp.adaptiveThresholdConfig = adaptive_cfg;
    rp.adaptiveThresholdMode = 'binary';
    R_binary(ri) = run_responsiveness(rp, num_angles, time_vec, seed);

    pp = pers_params;
    pp.useAdaptiveThreshold = true;
    pp.adaptiveThresholdConfig = adaptive_cfg;
    pp.adaptiveThresholdMode = 'binary';
    P_binary(ri) = run_persistence(pp, pers_cfg, seed + 50000);
end

R_bin_mean = mean(R_binary, 'omitnan');
P_bin_mean = mean(P_binary, 'omitnan');
fprintf('  Binary: R=%.4f, P=%.2f\n\n', R_bin_mean, P_bin_mean);

%% 3. 展平网格，parfor 扫描 Dual
% 展平 (lambda_idx, alpha_idx, run) 为单层 parfor
% 每个 job = 一个 (combo, run) 对
total_jobs = total_combos * num_runs;

% 映射表
job_li = zeros(total_jobs, 1);   % lambda index
job_ai = zeros(total_jobs, 1);   % alpha index
job_ri = zeros(total_jobs, 1);   % run index
jid = 0;
for li = 1:n_lambda
    for ai = 1:n_alpha
        for ri = 1:num_runs
            jid = jid + 1;
            job_li(jid) = li;
            job_ai(jid) = ai;
            job_ri(jid) = ri;
        end
    end
end

% 结果数组
job_R = NaN(total_jobs, 1);
job_P = NaN(total_jobs, 1);

fprintf('[Dual 扫描] %d 个 jobs (parfor)...\n', total_jobs);
sweep_timer = tic;

parfor jid = 1:total_jobs
    li = job_li(jid);
    ai = job_ai(jid);
    ri = job_ri(jid);

    lb = lambda_base_values(li);
    am = alpha_min_values(ai);
    seed = base_seed + li*10000 + ai*1000 + ri;

    rp = resp_params;
    rp.useAdaptiveThreshold = true;
    rp.adaptiveThresholdConfig = adaptive_cfg;
    rp.adaptiveThresholdMode = 'dual_signal_gate';
    rp.dualSignalVarianceRef = V_ref;
    rp.dualSignalGamma = gamma;
    rp.dualSignalSmoothingLambda = lb;
    rp.dualSignalAlphaMin = am;
    rp.dualSignalLambdaUp = lambda_up;
    rp.dualSignalClow = C_low;
    rp.dualSignalChigh = C_high;
    job_R(jid) = run_responsiveness(rp, num_angles, time_vec, seed);

    pp = pers_params;
    pp.useAdaptiveThreshold = true;
    pp.adaptiveThresholdConfig = adaptive_cfg;
    pp.adaptiveThresholdMode = 'dual_signal_gate';
    pp.dualSignalVarianceRef = V_ref;
    pp.dualSignalGamma = gamma;
    pp.dualSignalSmoothingLambda = lb;
    pp.dualSignalAlphaMin = am;
    pp.dualSignalLambdaUp = lambda_up;
    pp.dualSignalClow = C_low;
    pp.dualSignalChigh = C_high;
    job_P(jid) = run_persistence(pp, pers_cfg, seed + 50000);
end

sweep_elapsed = toc(sweep_timer);
fprintf('扫描完成，耗时 %.1f 分钟\n\n', sweep_elapsed / 60);

%% 4. 聚合为热力图矩阵
R_grid = NaN(n_lambda, n_alpha);
P_grid = NaN(n_lambda, n_alpha);
R_se_grid = NaN(n_lambda, n_alpha);
P_se_grid = NaN(n_lambda, n_alpha);

for li = 1:n_lambda
    for ai = 1:n_alpha
        mask = (job_li == li) & (job_ai == ai);
        R_vals = job_R(mask);
        P_vals = job_P(mask);
        R_grid(li, ai) = mean(R_vals, 'omitnan');
        P_grid(li, ai) = mean(P_vals, 'omitnan');
        R_se_grid(li, ai) = std(R_vals, 'omitnan') / sqrt(num_runs);
        P_se_grid(li, ai) = std(P_vals, 'omitnan') / sqrt(num_runs);
    end
end

%% 5. 图1：R 热力图
fig1 = figure('Visible', 'off', 'Position', [50 50 800 600], 'Color', 'w');
imagesc(alpha_min_values, lambda_base_values, R_grid);
set(gca, 'YDir', 'normal');
colormap('turbo'); cb = colorbar; cb.Label.String = 'R (响应性)';
hold on;
contour(alpha_min_values, lambda_base_values, R_grid, [R_bin_mean R_bin_mean], 'k--', 'LineWidth', 2);
xlabel('\alpha_{min}'); ylabel('\lambda_{base}');
title(sprintf('响应性 R (Binary baseline=%.3f, 黑虚线)', R_bin_mean));
grid on;
saveas(fig1, fullfile(out_dir, 'fig1_R_heatmap.png'));
close(fig1);

%% 6. 图2：P 热力图
fig2 = figure('Visible', 'off', 'Position', [50 50 800 600], 'Color', 'w');
imagesc(alpha_min_values, lambda_base_values, P_grid);
set(gca, 'YDir', 'normal');
colormap('turbo'); cb = colorbar; cb.Label.String = 'P (持久性)';
hold on;
contour(alpha_min_values, lambda_base_values, P_grid, [P_bin_mean P_bin_mean], 'k--', 'LineWidth', 2);
xlabel('\alpha_{min}'); ylabel('\lambda_{base}');
title(sprintf('持久性 P (Binary baseline=%.1f, 黑虚线)', P_bin_mean));
grid on;
saveas(fig2, fullfile(out_dir, 'fig2_P_heatmap.png'));
close(fig2);

%% 7. 图3：严格优于 Binary 的区域
dominance = (R_grid >= R_bin_mean) & (P_grid >= P_bin_mean);
delta_R = (R_grid - R_bin_mean) / R_bin_mean * 100;
delta_P = (P_grid - P_bin_mean) / P_bin_mean * 100;

fig3 = figure('Visible', 'off', 'Position', [50 50 1200 500], 'Color', 'w');

subplot(1,2,1);
imagesc(alpha_min_values, lambda_base_values, double(dominance));
set(gca, 'YDir', 'normal');
colormap(gca, [0.9 0.9 0.9; 0.2 0.7 0.3]);
xlabel('\alpha_{min}'); ylabel('\lambda_{base}');
title('严格优于 Binary 的区域（绿色）');
grid on;

subplot(1,2,2);
% 综合得分：ΔR% + ΔP%（只看两者都 ≥ 0 的区域）
score = delta_R + delta_P;
score(~dominance) = NaN;
imagesc(alpha_min_values, lambda_base_values, score);
set(gca, 'YDir', 'normal');
colormap(gca, 'turbo'); cb = colorbar; cb.Label.String = 'ΔR% + ΔP%';
xlabel('\alpha_{min}'); ylabel('\lambda_{base}');
title('优势区域综合得分');
grid on;

saveas(fig3, fullfile(out_dir, 'fig3_dominance.png'));
close(fig3);

%% 8. 找最优参数
[best_score, best_idx] = max(score(:));
if isnan(best_score)
    fprintf('警告：没有找到严格优于 Binary 的参数组合！\n');
    % 找 ΔR + ΔP 最大的点（即使不严格优于）
    score_all = delta_R + delta_P;
    [best_score, best_idx] = max(score_all(:));
    [best_li, best_ai] = ind2sub([n_lambda, n_alpha], best_idx);
    fprintf('最接近的组合: λ_base=%.1f, α_min=%.1f → R=%.4f (ΔR=%+.1f%%), P=%.2f (ΔP=%+.1f%%)\n', ...
        lambda_base_values(best_li), alpha_min_values(best_ai), ...
        R_grid(best_li, best_ai), delta_R(best_li, best_ai), ...
        P_grid(best_li, best_ai), delta_P(best_li, best_ai));
else
    [best_li, best_ai] = ind2sub([n_lambda, n_alpha], best_idx);
    fprintf('最优参数: λ_base=%.1f, α_min=%.1f → R=%.4f (ΔR=%+.1f%%), P=%.2f (ΔP=%+.1f%%)\n', ...
        lambda_base_values(best_li), alpha_min_values(best_ai), ...
        R_grid(best_li, best_ai), delta_R(best_li, best_ai), ...
        P_grid(best_li, best_ai), delta_P(best_li, best_ai));
end

%% 9. 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');
fprintf(fid, '========================================================================\n');
fprintf(fid, '双信号参数扫描报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '耗时: %.1f 分钟\n', sweep_elapsed / 60);
fprintf(fid, '========================================================================\n\n');

fprintf(fid, '设置: η=%.2f, V_ref=%.4f, λ_up=%.1f, γ=%.1f, C_low=%d, C_high=%d\n', ...
    eta, V_ref, lambda_up, gamma, C_low, C_high);
fprintf(fid, '网格: λ_base %d个 × α_min %d个 = %d 组合, 每组 %d 次重复\n\n', ...
    n_lambda, n_alpha, total_combos, num_runs);

fprintf(fid, 'Binary baseline: R=%.4f, P=%.2f\n\n', R_bin_mean, P_bin_mean);

fprintf(fid, '一、完整 R 矩阵 (行=λ_base, 列=α_min)\n');
fprintf(fid, '------------------------------------------------------------------------\n');
fprintf(fid, '%8s', 'λ\\α');
for ai = 1:n_alpha
    fprintf(fid, ' %6.1f', alpha_min_values(ai));
end
fprintf(fid, '\n');
for li = 1:n_lambda
    fprintf(fid, '%8.1f', lambda_base_values(li));
    for ai = 1:n_alpha
        fprintf(fid, ' %6.3f', R_grid(li, ai));
    end
    fprintf(fid, '\n');
end

fprintf(fid, '\n二、完整 P 矩阵 (行=λ_base, 列=α_min)\n');
fprintf(fid, '------------------------------------------------------------------------\n');
fprintf(fid, '%8s', 'λ\\α');
for ai = 1:n_alpha
    fprintf(fid, ' %6.1f', alpha_min_values(ai));
end
fprintf(fid, '\n');
for li = 1:n_lambda
    fprintf(fid, '%8.1f', lambda_base_values(li));
    for ai = 1:n_alpha
        fprintf(fid, ' %6.1f', P_grid(li, ai));
    end
    fprintf(fid, '\n');
end

fprintf(fid, '\n三、严格优于 Binary 的参数组合\n');
fprintf(fid, '------------------------------------------------------------------------\n');
n_dominant = sum(dominance(:));
fprintf(fid, '共 %d / %d 组合严格优于 Binary\n\n', n_dominant, total_combos);

if n_dominant > 0
    fprintf(fid, '%-8s %-8s %8s %8s %8s %8s\n', 'λ_base', 'α_min', 'R', 'P', 'ΔR%', 'ΔP%');
    [dom_li, dom_ai] = find(dominance);
    dom_scores = score(dominance);
    [~, sort_idx] = sort(dom_scores, 'descend');
    for k = 1:numel(sort_idx)
        li = dom_li(sort_idx(k));
        ai = dom_ai(sort_idx(k));
        fprintf(fid, '%-8.1f %-8.1f %8.4f %8.2f %+7.1f%% %+7.1f%%\n', ...
            lambda_base_values(li), alpha_min_values(ai), ...
            R_grid(li, ai), P_grid(li, ai), delta_R(li, ai), delta_P(li, ai));
    end
else
    fprintf(fid, '未找到严格优于 Binary 的组合。\n');
    fprintf(fid, '最接近的组合:\n');
    fprintf(fid, '  λ_base=%.1f, α_min=%.1f → R=%.4f (ΔR=%+.1f%%), P=%.2f (ΔP=%+.1f%%)\n', ...
        lambda_base_values(best_li), alpha_min_values(best_ai), ...
        R_grid(best_li, best_ai), delta_R(best_li, best_ai), ...
        P_grid(best_li, best_ai), delta_P(best_li, best_ai));
end

fclose(fid);

%% 10. 保存数据
save(fullfile(out_dir, 'sweep_data.mat'), ...
    'lambda_base_values', 'alpha_min_values', 'R_grid', 'P_grid', ...
    'R_se_grid', 'P_se_grid', 'delta_R', 'delta_P', 'dominance', ...
    'R_bin_mean', 'P_bin_mean', 'R_binary', 'P_binary', ...
    'eta', 'V_ref', 'adaptive_cfg', 'base', 'num_runs', 'sweep_elapsed');

fprintf('\n完成。输出目录: %s\n', out_dir);

%% ========================================================================
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
