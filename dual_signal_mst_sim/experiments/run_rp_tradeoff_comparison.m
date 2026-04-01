%% run_rp_tradeoff_comparison.m
% R-P 权衡对比：Fixed vs Binary vs Dual-Signal-Gate
% 多噪声水平扫描版本
%
% 输出（results/rp_tradeoff_<timestamp>/）：
%   - 每个 η 的 R-P 对比图
%   - 汇总图（所有 η 的 Binary vs Dual 对比）
%   - 文本报告 + MAT 数据

clc; clear; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 公共参数
fprintf('=================================================\n');
fprintf('  Fixed vs Binary vs Dual-Signal-Gate: R-P 权衡\n');
fprintf('  多噪声水平扫描\n');
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

num_runs = 50;
num_angles = 1;
base_seed = 20260324;

pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_diffusion = 1e-4;
pers_cfg.min_fit_points = 40;

cj_scan = 0.0:0.1:5.0;  % 步长 0.1

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5.0;
adaptive_cfg.saliency_threshold = 0.031;
adaptive_cfg.include_self = false;

V_ref = 0.05;

eta_values = [0.10, 0.15, 0.20, 0.25, 0.30];
num_eta = numel(eta_values);

%% 输出目录
results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_dir = fullfile(results_dir, sprintf('rp_tradeoff_%s', timestamp));
mkdir(out_dir);

fprintf('V_ref = %.4f (固定值)\n', V_ref);
fprintf('η 扫描: %s\n', mat2str(eta_values));
fprintf('固定阈值: %d 个点 (步长 0.1)\n', numel(cj_scan));
fprintf('每组 %d 次重复\n\n', num_runs);

%% 2. 主循环：逐 η 运行
all_eta_results = cell(num_eta, 1);
total_timer = tic;

for ei = 1:num_eta
    eta = eta_values(ei);
    noise_intensity = eta^2 / 2;
    fprintf('========== η=%.2f (%d/%d) ==========\n', eta, ei, num_eta);

    resp_params = resp_base;
    resp_params.angleNoiseIntensity = noise_intensity;
    pers_params = pers_base;
    pers_params.angleNoiseIntensity = noise_intensity;

    time_vec = (0:resp_params.T_max)' * resp_params.dt;

    % --- Fixed 扫描 ---
    fprintf('[Fixed] %d 个阈值 × %d 次\n', numel(cj_scan), num_runs);
    R_fixed = NaN(numel(cj_scan), num_runs);
    P_fixed = NaN(numel(cj_scan), num_runs);

    for ci = 1:numel(cj_scan)
        rp = resp_params; rp.cj_threshold = cj_scan(ci); rp.useAdaptiveThreshold = false;
        pp = pers_params; pp.cj_threshold = cj_scan(ci); pp.useAdaptiveThreshold = false;

        parfor ri = 1:num_runs
            seed = base_seed + ei*1e6 + (ci-1)*num_runs + ri;
            R_fixed(ci, ri) = run_responsiveness(rp, num_angles, time_vec, seed);
            P_fixed(ci, ri) = run_persistence(pp, pers_cfg, seed + 50000);
        end

        if mod(ci, 10) == 0
            fprintf('  cj=%.1f: R=%.3f, P=%.3f\n', cj_scan(ci), ...
                mean(R_fixed(ci,:),'omitnan'), mean(P_fixed(ci,:),'omitnan'));
        end
    end

    % --- Binary ---
    fprintf('[Binary] %d 次\n', num_runs);
    R_binary = NaN(num_runs, 1);
    P_binary = NaN(num_runs, 1);

    parfor ri = 1:num_runs
        seed = base_seed + ei*1e6 + 100000 + ri;
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
    fprintf('  R=%.3f±%.3f, P=%.3f±%.3f\n', ...
        mean(R_binary,'omitnan'), std(R_binary,'omitnan')/sqrt(num_runs), ...
        mean(P_binary,'omitnan'), std(P_binary,'omitnan')/sqrt(num_runs));

    % --- Dual-Signal-Gate ---
    fprintf('[Dual] %d 次\n', num_runs);
    R_dual = NaN(num_runs, 1);
    P_dual = NaN(num_runs, 1);

    parfor ri = 1:num_runs
        seed = base_seed + ei*1e6 + 200000 + ri;
        rp = resp_params;
        rp.useAdaptiveThreshold = true;
        rp.adaptiveThresholdConfig = adaptive_cfg;
        rp.adaptiveThresholdMode = 'dual_signal_gate';
        rp.dualSignalVarianceRef = V_ref;
        rp.dualSignalGamma = 1.0;
        rp.dualSignalSmoothingLambda = 0.3;
        rp.dualSignalAlphaMin = 0.3;
        rp.dualSignalLambdaUp = 0.5;
        rp.dualSignalClow = 30;
        rp.dualSignalChigh = 150;
        R_dual(ri) = run_responsiveness(rp, num_angles, time_vec, seed);

        pp = pers_params;
        pp.useAdaptiveThreshold = true;
        pp.adaptiveThresholdConfig = adaptive_cfg;
        pp.adaptiveThresholdMode = 'dual_signal_gate';
        pp.dualSignalVarianceRef = V_ref;
        pp.dualSignalGamma = 1.0;
        pp.dualSignalSmoothingLambda = 0.3;
        pp.dualSignalAlphaMin = 0.3;
        pp.dualSignalLambdaUp = 0.5;
        pp.dualSignalClow = 30;
        pp.dualSignalChigh = 150;
        P_dual(ri) = run_persistence(pp, pers_cfg, seed + 50000);
    end
    fprintf('  R=%.3f±%.3f, P=%.3f±%.3f\n\n', ...
        mean(R_dual,'omitnan'), std(R_dual,'omitnan')/sqrt(num_runs), ...
        mean(P_dual,'omitnan'), std(P_dual,'omitnan')/sqrt(num_runs));

    % --- 保存该 η 的结果 ---
    res = struct();
    res.eta = eta;
    res.cj_scan = cj_scan;
    res.R_fixed = R_fixed; res.P_fixed = P_fixed;
    res.R_binary = R_binary; res.P_binary = P_binary;
    res.R_dual = R_dual; res.P_dual = P_dual;
    all_eta_results{ei} = res;

    % --- 该 η 的 R-P 图 ---
    P_all = [P_fixed(:); P_binary; P_dual];
    P_all = P_all(~isnan(P_all));
    Pmin = min(P_all); Pmax = max(P_all); Prng = max(Pmax - Pmin, eps);

    fig = figure('Visible', 'off', 'Position', [50 50 900 600], 'Color', 'w');
    hold on;
    Rm = mean(R_fixed, 2, 'omitnan');
    Pn = (mean(P_fixed, 2, 'omitnan') - Pmin) / Prng;
    scatter(Rm, Pn, 40, cj_scan(:), 'filled', 'DisplayName', '固定阈值');
    plot(Rm, Pn, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
    colormap('turbo'); cb = colorbar; cb.Label.String = 'cj';

    Pbn = (mean(P_binary,'omitnan') - Pmin) / Prng;
    scatter(mean(R_binary,'omitnan'), Pbn, 200, 'r', 'p', 'filled', ...
        'MarkerEdgeColor', [0.5 0 0], 'DisplayName', 'Binary');

    Pdn = (mean(P_dual,'omitnan') - Pmin) / Prng;
    scatter(mean(R_dual,'omitnan'), Pdn, 200, 'b', 'h', 'filled', ...
        'MarkerEdgeColor', [0 0 0.5], 'DisplayName', 'Dual-Signal');

    xlabel('响应性 R'); ylabel('归一化持久性 P');
    title(sprintf('R-P 权衡 (\\eta=%.2f)', eta));
    legend('Location', 'best'); grid on;
    saveas(fig, fullfile(out_dir, sprintf('rp_eta_%.2f.png', eta)));
    close(fig);
end

total_elapsed = toc(total_timer);
fprintf('全部完成，总耗时 %.1f 分钟\n\n', total_elapsed / 60);

%% 3. 汇总图：Binary vs Dual 在不同 η 下的 R 和 P
fig_summary = figure('Visible', 'off', 'Position', [50 50 1200 500], 'Color', 'w');

subplot(1,2,1);
R_bin_all = zeros(num_eta, 1); R_bin_se = zeros(num_eta, 1);
R_dual_all = zeros(num_eta, 1); R_dual_se = zeros(num_eta, 1);
for ei = 1:num_eta
    r = all_eta_results{ei};
    R_bin_all(ei) = mean(r.R_binary, 'omitnan');
    R_bin_se(ei) = std(r.R_binary, 'omitnan') / sqrt(num_runs);
    R_dual_all(ei) = mean(r.R_dual, 'omitnan');
    R_dual_se(ei) = std(r.R_dual, 'omitnan') / sqrt(num_runs);
end
errorbar(eta_values, R_bin_all, R_bin_se, 'r-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
hold on;
errorbar(eta_values, R_dual_all, R_dual_se, 'b-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel('\eta'); ylabel('响应性 R');
title('R vs \eta');
legend('Binary', 'Dual-Signal', 'Location', 'best'); grid on;

subplot(1,2,2);
P_bin_all = zeros(num_eta, 1); P_bin_se = zeros(num_eta, 1);
P_dual_all = zeros(num_eta, 1); P_dual_se = zeros(num_eta, 1);
for ei = 1:num_eta
    r = all_eta_results{ei};
    P_bin_all(ei) = mean(r.P_binary, 'omitnan');
    P_bin_se(ei) = std(r.P_binary, 'omitnan') / sqrt(num_runs);
    P_dual_all(ei) = mean(r.P_dual, 'omitnan');
    P_dual_se(ei) = std(r.P_dual, 'omitnan') / sqrt(num_runs);
end
errorbar(eta_values, P_bin_all, P_bin_se, 'r-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
hold on;
errorbar(eta_values, P_dual_all, P_dual_se, 'b-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel('\eta'); ylabel('持久性 P');
title('P vs \eta');
legend('Binary', 'Dual-Signal', 'Location', 'best'); grid on;

saveas(fig_summary, fullfile(out_dir, 'summary_R_P_vs_eta.png'));
close(fig_summary);

%% 4. 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');
fprintf(fid, '========================================================================\n');
fprintf(fid, 'R-P 权衡多噪声对比报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '总耗时: %.1f 分钟\n', total_elapsed / 60);
fprintf(fid, '========================================================================\n\n');
fprintf(fid, '参数: N=%d, V_ref=%.4f, num_runs=%d, cj步长=0.1\n', base.N, V_ref, num_runs);
fprintf(fid, '双信号: γ=1.0, λ_base=0.3, α_min=0.3, λ_up=0.5, C_low=30, C_high=150\n\n');

fprintf(fid, '%-6s %10s %10s %10s %10s %8s %8s\n', ...
    'eta', 'R_binary', 'R_dual', 'P_binary', 'P_dual', 'ΔR%', 'ΔP%');
fprintf(fid, '%s\n', repmat('-', 1, 66));
for ei = 1:num_eta
    r = all_eta_results{ei};
    Rb = mean(r.R_binary, 'omitnan');
    Rd = mean(r.R_dual, 'omitnan');
    Pb = mean(r.P_binary, 'omitnan');
    Pd = mean(r.P_dual, 'omitnan');
    dR = (Rd - Rb) / Rb * 100;
    dP = (Pd - Pb) / Pb * 100;
    fprintf(fid, '%-6.2f %10.4f %10.4f %10.2f %10.2f %+7.1f%% %+7.1f%%\n', ...
        eta_values(ei), Rb, Rd, Pb, Pd, dR, dP);
end

fprintf(fid, '\nΔR%%: Dual 相对 Binary 的响应性变化（正=更好）\n');
fprintf(fid, 'ΔP%%: Dual 相对 Binary 的持久性变化（正=更好）\n');
fprintf(fid, '\n若 ΔR>0 且 ΔP>0: Dual 严格优于 Binary\n');
fprintf(fid, '若 ΔR<0 且 ΔP>0: Dual 用响应性换持久性（需调参）\n');
fclose(fid);

%% 5. 保存数据
saveResultBundle(out_dir, 'all_data', ...
    {'all_eta_results', 'eta_values', 'cj_scan', 'V_ref', ...
     'adaptive_cfg', 'base', 'num_runs', 'total_elapsed'});

fprintf('输出目录: %s\n', out_dir);

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
    triggered = false;
    n_vecs = [];
    t_start = NaN;

    for t = 1:params.T_max
        sim.step();
        V_history(t+1, :) = [mean(params.v0*cos(sim.theta)), mean(params.v0*sin(sim.theta))];

        if ~triggered && sim.external_pulse_triggered
            triggered = true;
            t_start = t;
            lidx = find(sim.isExternallyActivated, 1, 'first');
            if isempty(lidx), lidx = 1; end
            phi = sim.external_target_theta(lidx);
            n_vecs = [cos(phi); sin(phi)];
        end

        if triggered
            proj_history(t+1, :) = V_history(t+1, :) * n_vecs;
        end
    end

    if ~triggered || isnan(t_start)
        R = NaN;
        return;
    end

    t_end = min(t_start + params.forced_turn_duration, params.T_max);
    integral_val = trapz(time_vec(t_start+1:t_end+1), proj_history(t_start+1:t_end+1));
    duration = time_vec(t_end+1) - time_vec(t_start+1);
    if duration > 0
        R = integral_val / (params.v0 * duration);
    else
        R = NaN;
    end
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
    x = time_vec(burn_in:end);
    y = msd(burn_in:end);

    if numel(x) < max(2, cfg.min_fit_points) || all(abs(y - y(1)) < eps)
        D = NaN;
    else
        x_s = x - x(1);
        y_s = y - y(1);
        if any(x_s > 0) && any(abs(y_s) > eps)
            sw = max(5, floor(numel(y_s)*0.1));
            if sw > 1, y_s = smoothdata(y_s, 'movmean', sw); end
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
end
