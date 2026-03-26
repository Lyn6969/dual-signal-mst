%% run_hybrid_comparison.m
% Binary vs Sigmoid-gate vs Algebraic-fusion 对比实验
% 共享种子，η=0.25

clc; clear; close all;

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% 1. 参数
fprintf('=================================================\n');
fprintf('  Binary vs Sigmoid-gate vs Algebraic-fusion\n');
fprintf('  共享种子对比 (η=0.25)\n');
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

eta = 0.30;
resp_params.angleNoiseIntensity = eta^2 / 2;
pers_params.angleNoiseIntensity = eta^2 / 2;

num_runs = 50;
num_angles = 1;
base_seed = 20260326;

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

time_vec = (0:resp_params.T_max)' * resp_params.dt;

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
out_dir = fullfile(results_dir, sprintf('hybrid_comparison_%s', timestamp));
mkdir(out_dir);

fprintf('η = %.2f, V_ref = %.4f, num_runs = %d\n', eta, V_ref, num_runs);
fprintf('共享种子: seed = base_seed + ri\n\n');

%% 3. Binary baseline
fprintf('[Binary] %d 次重复...\n', num_runs);
R_binary = NaN(num_runs, 1);
P_binary = NaN(num_runs, 1);

parfor ri = 1:num_runs
    seed = base_seed + ri;
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
fprintf('  R=%.4f±%.4f, P=%.2f±%.2f\n', ...
    mean(R_binary,'omitnan'), std(R_binary,'omitnan')/sqrt(num_runs), ...
    mean(P_binary,'omitnan'), std(P_binary,'omitnan')/sqrt(num_runs));

%% 4. Sigmoid-gate
fprintf('[Sigmoid-gate] %d 次重复 (D_mid=0.6, κ=10)...\n', num_runs);
R_sigmoid = NaN(num_runs, 1);
P_sigmoid = NaN(num_runs, 1);

parfor ri = 1:num_runs
    seed = base_seed + ri;
    rp = resp_params;
    rp.useAdaptiveThreshold = true;
    rp.adaptiveThresholdConfig = adaptive_cfg;
    rp.adaptiveThresholdMode = 'sigmoid_gate';
    rp.dualSignalVarianceRef = V_ref;
    rp.dualSignalGamma = 1.0;
    rp.dualSignalSmoothingLambda = 0.3;
    rp.dualSignalAlphaMin = 0.3;
    rp.dualSignalLambdaUp = 0.5;
    rp.dualSignalClow = 30;
    rp.dualSignalChigh = 150;
    rp.dualSignalDmid = 0.6;
    rp.dualSignalKappa = 10;
    R_sigmoid(ri) = run_responsiveness(rp, num_angles, time_vec, seed);

    pp = pers_params;
    pp.useAdaptiveThreshold = true;
    pp.adaptiveThresholdConfig = adaptive_cfg;
    pp.adaptiveThresholdMode = 'sigmoid_gate';
    pp.dualSignalVarianceRef = V_ref;
    pp.dualSignalGamma = 1.0;
    pp.dualSignalSmoothingLambda = 0.3;
    pp.dualSignalAlphaMin = 0.3;
    pp.dualSignalLambdaUp = 0.5;
    pp.dualSignalClow = 30;
    pp.dualSignalChigh = 150;
    pp.dualSignalDmid = 0.6;
    pp.dualSignalKappa = 10;
    P_sigmoid(ri) = run_persistence(pp, pers_cfg, seed + 50000);
end
fprintf('  R=%.4f±%.4f, P=%.2f±%.2f\n', ...
    mean(R_sigmoid,'omitnan'), std(R_sigmoid,'omitnan')/sqrt(num_runs), ...
    mean(P_sigmoid,'omitnan'), std(P_sigmoid,'omitnan')/sqrt(num_runs));

%% 5. Algebraic-fusion
fprintf('[Algebraic-fusion] %d 次重复...\n', num_runs);
R_algebraic = NaN(num_runs, 1);
P_algebraic = NaN(num_runs, 1);

parfor ri = 1:num_runs
    seed = base_seed + ri;
    rp = resp_params;
    rp.useAdaptiveThreshold = true;
    rp.adaptiveThresholdConfig = adaptive_cfg;
    rp.adaptiveThresholdMode = 'algebraic_fusion';
    rp.dualSignalVarianceRef = V_ref;
    rp.dualSignalGamma = 1.0;
    rp.dualSignalSmoothingLambda = 0.3;
    rp.dualSignalAlphaMin = 0.3;
    rp.dualSignalLambdaUp = 0.5;
    rp.dualSignalClow = 30;
    rp.dualSignalChigh = 150;
    R_algebraic(ri) = run_responsiveness(rp, num_angles, time_vec, seed);

    pp = pers_params;
    pp.useAdaptiveThreshold = true;
    pp.adaptiveThresholdConfig = adaptive_cfg;
    pp.adaptiveThresholdMode = 'algebraic_fusion';
    pp.dualSignalVarianceRef = V_ref;
    pp.dualSignalGamma = 1.0;
    pp.dualSignalSmoothingLambda = 0.3;
    pp.dualSignalAlphaMin = 0.3;
    pp.dualSignalLambdaUp = 0.5;
    pp.dualSignalClow = 30;
    pp.dualSignalChigh = 150;
    P_algebraic(ri) = run_persistence(pp, pers_cfg, seed + 50000);
end
fprintf('  R=%.4f±%.4f, P=%.2f±%.2f\n\n', ...
    mean(R_algebraic,'omitnan'), std(R_algebraic,'omitnan')/sqrt(num_runs), ...
    mean(P_algebraic,'omitnan'), std(P_algebraic,'omitnan')/sqrt(num_runs));

%% 6. Two-stage
fprintf('[Two-stage] %d 次重复 (θ_on=0.6, K=2, α=0.6, Θ_conf=1.0)...\n', num_runs);
R_twostage = NaN(num_runs, 1);
P_twostage = NaN(num_runs, 1);

parfor ri = 1:num_runs
    seed = base_seed + ri;
    rp = resp_params;
    rp.useAdaptiveThreshold = true;
    rp.adaptiveThresholdConfig = adaptive_cfg;
    rp.adaptiveThresholdMode = 'two_stage';
    rp.dualSignalVarianceRef = V_ref;
    rp.dualSignalGamma = 1.0;
    rp.dualSignalClow = 30;
    rp.dualSignalChigh = 150;
    rp.twoStageThresholdOn = 0.6;
    rp.twoStageThresholdOff = 0.3;
    rp.twoStageWindowK = 2;
    rp.twoStageAlpha = 0.6;
    rp.twoStageConfThreshold = 1.0;
    R_twostage(ri) = run_responsiveness(rp, num_angles, time_vec, seed);

    pp = pers_params;
    pp.useAdaptiveThreshold = true;
    pp.adaptiveThresholdConfig = adaptive_cfg;
    pp.adaptiveThresholdMode = 'two_stage';
    pp.dualSignalVarianceRef = V_ref;
    pp.dualSignalGamma = 1.0;
    pp.dualSignalClow = 30;
    pp.dualSignalChigh = 150;
    pp.twoStageThresholdOn = 0.6;
    pp.twoStageThresholdOff = 0.3;
    pp.twoStageWindowK = 2;
    pp.twoStageAlpha = 0.6;
    pp.twoStageConfThreshold = 1.0;
    P_twostage(ri) = run_persistence(pp, pers_cfg, seed + 50000);
end
fprintf('  R=%.4f±%.4f, P=%.2f±%.2f\n\n', ...
    mean(R_twostage,'omitnan'), std(R_twostage,'omitnan')/sqrt(num_runs), ...
    mean(P_twostage,'omitnan'), std(P_twostage,'omitnan')/sqrt(num_runs));

%% 7. 汇总
Rb = mean(R_binary,'omitnan');  Pb = mean(P_binary,'omitnan');
Rs = mean(R_sigmoid,'omitnan'); Ps = mean(P_sigmoid,'omitnan');
Ra = mean(R_algebraic,'omitnan'); Pa = mean(P_algebraic,'omitnan');
Rt = mean(R_twostage,'omitnan'); Pt = mean(P_twostage,'omitnan');

dR_sig = (Rs - Rb) / Rb * 100;
dP_sig = (Ps - Pb) / Pb * 100;
dR_alg = (Ra - Rb) / Rb * 100;
dP_alg = (Pa - Pb) / Pb * 100;
dR_two = (Rt - Rb) / Rb * 100;
dP_two = (Pt - Pb) / Pb * 100;

fprintf('========== 结果汇总 (η=%.2f) ==========\n', eta);
fprintf('%-20s %8s %8s %8s %8s\n', '方法', 'R', 'P', 'ΔR%', 'ΔP%');
fprintf('%s\n', repmat('-', 1, 56));
fprintf('%-20s %8.4f %8.2f %8s %8s\n', 'Binary (v5.2)', Rb, Pb, '-', '-');
fprintf('%-20s %8.4f %8.2f %+7.1f%% %+7.1f%%\n', 'Sigmoid-gate', Rs, Ps, dR_sig, dP_sig);
fprintf('%-20s %8.4f %8.2f %+7.1f%% %+7.1f%%\n', 'Algebraic-fusion', Ra, Pa, dR_alg, dP_alg);
fprintf('%-20s %8.4f %8.2f %+7.1f%% %+7.1f%%\n', 'Two-stage', Rt, Pt, dR_two, dP_two);
fprintf('\n');

methods_names = {'Sigmoid-gate', 'Algebraic-fusion', 'Two-stage'};
dR_all = [dR_sig, dR_alg, dR_two];
dP_all = [dP_sig, dP_alg, dP_two];
for mi = 1:3
    if dR_all(mi) > 0 && dP_all(mi) > 0
        fprintf('%s 严格优于 Binary!\n', methods_names{mi});
    elseif dR_all(mi) < 0 && dP_all(mi) > 0
        fprintf('%s: R换P (响应性下降, 持久性提升)\n', methods_names{mi});
    else
        fprintf('%s: R=%+.1f%%, P=%+.1f%%\n', methods_names{mi}, dR_all(mi), dP_all(mi));
    end
end

%% 8. 对比图
fig1 = figure('Visible', 'off', 'Position', [50 50 1200 500], 'Color', 'w');

% R 对比
subplot(1,2,1);
R_means = [Rb, Rs, Ra, Rt];
R_ses = [std(R_binary,'omitnan'), std(R_sigmoid,'omitnan'), std(R_algebraic,'omitnan'), std(R_twostage,'omitnan')] / sqrt(num_runs);
bar(R_means); hold on;
errorbar(1:4, R_means, R_ses, 'k.', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'Binary', 'Sigmoid', 'Algebraic', 'Two-stage'});
ylabel('响应性 R');
title(sprintf('R 对比 (\\eta=%.2f)', eta));
grid on;

% P 对比
subplot(1,2,2);
P_means = [Pb, Ps, Pa, Pt];
P_ses = [std(P_binary,'omitnan'), std(P_sigmoid,'omitnan'), std(P_algebraic,'omitnan'), std(P_twostage,'omitnan')] / sqrt(num_runs);
bar(P_means); hold on;
errorbar(1:4, P_means, P_ses, 'k.', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'Binary', 'Sigmoid', 'Algebraic', 'Two-stage'});
ylabel('持久性 P');
title(sprintf('P 对比 (\\eta=%.2f)', eta));
grid on;

sgtitle(sprintf('四种方法对比 (\\eta=%.2f, %d runs, 共享种子)', eta, num_runs));
saveas(fig1, fullfile(out_dir, 'comparison_bar.png'));
close(fig1);

% R-P 散点图
fig2 = figure('Visible', 'off', 'Position', [50 50 800 600], 'Color', 'w');
hold on;
scatter(Rb, Pb, 200, 'r', 'p', 'filled', 'MarkerEdgeColor', [0.5 0 0], 'DisplayName', 'Binary');
scatter(Rs, Ps, 200, 'b', 'h', 'filled', 'MarkerEdgeColor', [0 0 0.5], 'DisplayName', 'Sigmoid-gate');
scatter(Ra, Pa, 200, 'g', 'd', 'filled', 'MarkerEdgeColor', [0 0.4 0], 'DisplayName', 'Algebraic-fusion');
scatter(Rt, Pt, 200, 'm', 's', 'filled', 'MarkerEdgeColor', [0.4 0 0.4], 'DisplayName', 'Two-stage');

% 逐次运行的散点（半透明）
scatter(R_binary, P_binary, 20, 'r', 'filled', 'MarkerFaceAlpha', 0.2, 'HandleVisibility', 'off');
scatter(R_sigmoid, P_sigmoid, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.2, 'HandleVisibility', 'off');
scatter(R_algebraic, P_algebraic, 20, 'g', 'filled', 'MarkerFaceAlpha', 0.2, 'HandleVisibility', 'off');
scatter(R_twostage, P_twostage, 20, 'm', 'filled', 'MarkerFaceAlpha', 0.2, 'HandleVisibility', 'off');

xlabel('响应性 R'); ylabel('持久性 P');
title(sprintf('R-P 对比 (\\eta=%.2f, 共享种子)', eta));
legend('Location', 'best'); grid on;
saveas(fig2, fullfile(out_dir, 'comparison_rp.png'));
close(fig2);

%% 9. 文本报告
fid = fopen(fullfile(out_dir, 'analysis_report.txt'), 'w');
fprintf(fid, '========================================================================\n');
fprintf(fid, '四种方法对比报告\n');
fprintf(fid, '生成时间: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf(fid, '========================================================================\n\n');
fprintf(fid, '设置: η=%.2f, N=%d, V_ref=%.4f, num_runs=%d, 共享种子\n', eta, base.N, V_ref, num_runs);
fprintf(fid, 'Two-stage: θ_on=0.6, θ_off=0.3, K=2, α=0.6, Θ_conf=1.0\n\n');

fprintf(fid, '%-20s %8s %8s %8s %8s\n', '方法', 'R', 'P', 'ΔR%', 'ΔP%');
fprintf(fid, '%s\n', repmat('-', 1, 56));
fprintf(fid, '%-20s %8.4f %8.2f %8s %8s\n', 'Binary (v5.2)', Rb, Pb, '-', '-');
fprintf(fid, '%-20s %8.4f %8.2f %+7.1f%% %+7.1f%%\n', 'Sigmoid-gate', Rs, Ps, dR_sig, dP_sig);
fprintf(fid, '%-20s %8.4f %8.2f %+7.1f%% %+7.1f%%\n', 'Algebraic-fusion', Ra, Pa, dR_alg, dP_alg);
fprintf(fid, '%-20s %8.4f %8.2f %+7.1f%% %+7.1f%%\n', 'Two-stage', Rt, Pt, dR_two, dP_two);
fprintf(fid, '\n');

for mi = 1:3
    if dR_all(mi) > 0 && dP_all(mi) > 0
        fprintf(fid, '%s 严格优于 Binary\n', methods_names{mi});
    end
end
fclose(fid);

%% 10. 保存数据
save(fullfile(out_dir, 'comparison_data.mat'), ...
    'R_binary', 'P_binary', 'R_sigmoid', 'P_sigmoid', ...
    'R_algebraic', 'P_algebraic', 'R_twostage', 'P_twostage', ...
    'eta', 'V_ref', 'adaptive_cfg', 'base', 'num_runs', 'base_seed');

fprintf('\n完成。输出目录: %s\n', out_dir);

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
                fprintf('检测到调度器 %s，使用 80%% = %d\n', env_names{i}, desired_workers);
                break;
            end
        end
    end
    if isnan(desired_workers)
        try
            desired_workers = floor(maxNumCompThreads * 2 * 0.8);
            fprintf('检测到核心数 %d，×2×80%% = %d\n', maxNumCompThreads, desired_workers);
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
