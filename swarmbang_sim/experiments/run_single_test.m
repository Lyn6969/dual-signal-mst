% run_single_test.m
% 单次快速测试脚本：跑一个条件，实时显示动画，验证代码正确性
%
% 使用方法：修改 alg_mode 后直接运行

clc; clear; close all;

%% ====== 配置 ======
alg_mode = 'fixed';   % 'fixed' / 'binary' / 'weighted'
showAnimation = true;  % 是否显示实时动画

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

%% ====== 参数初始化 ======
G = struct();
G.simStep = 0;
G.maxID = 50;
G.cycTime = 0.2;
G.v0 = 15;
G.maxRotRate = 10 * 1.91;

G.weight_align = 15;
G.Drep = 400;
G.Dsen = 1000;
G.weight_rep = 8;
G.weight_att = 0;
G.deac_threshold = 0.2;
G.noise_mov = 0;
G.max_neighbors = 7;
G.n_steps = 5;

G.maxSimSteps = 3000;
G.maxcj = zeros(G.maxSimSteps, 1);

G.targetA = [-1434, -1446];
G.targetB = [1449, 1406];
G.target = G.targetB;
G.weight_target = ones(G.maxSimSteps, 1);
G.targetR = 300;
G.swarmCenterPositions = zeros(G.maxSimSteps, 2);

G.rotateTime = 100;
G.rotateCycle = zeros(G.maxID, 1);
G.rotateStep = 300;
G.nowInfoIDs = nan(G.maxSimSteps, 1);

G.activatedIDs = cell(G.maxSimSteps, 1);
G.activatedCount = zeros(G.maxSimSteps, 1);
G.activatedSrcIDs = cell(G.maxSimSteps, 1);

% 算法特定参数
switch alg_mode
    case 'fixed'
        G.cj_threshold = 0.04;
    case 'binary'
        G.cj_low = 0.01;
        G.cj_high = 0.10;
        G.sigma_thr = 0.0005;
        G.cj_threshold = G.cj_high;
        G.threshold_history = zeros(G.maxID, G.maxSimSteps);
        G.sigma2_history = zeros(G.maxID, G.maxSimSteps);
    case 'weighted'
        G.cj_low = 0.01;
        G.cj_high = 0.10;
        G.sigma_thr = 0.0005;
        G.cj_threshold = G.cj_high;
        G.weightMode = 'absolute';
        G.threshold_history = zeros(G.maxID, G.maxSimSteps);
        G.sigma2_history = zeros(G.maxID, G.maxSimSteps);
end

%% ====== 初始化 ======
G = initSimRobots(G, 'unif', [1434, 1446], 100);
G.myinfoIDs = randsample(G.robotsList, 3);

for i = G.robotsList
    G.actor{i}.posHistory = nan(G.maxSimSteps, 2);
    G.actor{i}.posHistory(1, :) = G.actor{i}.pose;
end

reachedTarget = false;

fprintf('算法: %s | 个体数: %d | 信息个体: [%s]\n', ...
    alg_mode, G.maxID, num2str(G.myinfoIDs));
switch alg_mode
    case 'fixed'
        fprintf('显著性阈值: M_T = %.4f rad/s\n', G.cj_threshold);
    otherwise
        fprintf('显著性阈值: cj_low=%.4f, cj_high=%.4f, sigma_thr=%.6f\n', ...
            G.cj_low, G.cj_high, G.sigma_thr);
end

%% ====== 仿真主循环 ======
tic;
for t = 1:G.maxSimSteps
    G.simStep = t;

    switch alg_mode
        case 'fixed'
            [desTurnAngle, desSpeed, G] = swarmAlg_fixed(G);
        case 'binary'
            [desTurnAngle, desSpeed, G] = swarmAlg_binary(G);
        case 'weighted'
            [desTurnAngle, desSpeed, G] = swarmAlg_weighted(G);
    end

    G = parallelSimRobots(G, desTurnAngle, desSpeed);

    for i = G.robotsList
        G.actor{i}.posHistory(t, :) = G.actor{i}.pose;
    end

    if isfield(G, 'now_infoID')
        G.nowInfoIDs(t) = G.now_infoID;
    end

    center = getSwarmCenter(G);
    G.swarmCenterPositions(t, :) = center;

    if norm(center - G.target) < G.targetR
        fprintf('步数 %d: 集群到达目标区域！\n', t);
        reachedTarget = true;
        break;
    end

    if showAnimation && mod(t, 5) == 0
        drawRobotsMotion_dynamic(G);
        pause(0.01);
    end

    if mod(t, 200) == 0
        fprintf('步数 %d | 激活: %d | max_s: %.4f rad/s\n', ...
            t, G.activatedCount(t), G.maxcj(t));
    end
end
elapsed = toc;

%% ====== 结果输出 ======
G.reachedTarget = reachedTarget;
G.reachedStep = G.simStep;

totalDistance = 0;
for i = 2:G.reachedStep
    totalDistance = totalDistance + norm(...
        G.swarmCenterPositions(i,:) - G.swarmCenterPositions(i-1,:));
end
G.totalDistance = totalDistance;
G.tortuosity = computePathTortuosity(G.swarmCenterPositions, G.reachedStep);

fprintf('\n====== 仿真结束 ======\n');
fprintf('到达目标: %s\n', string(reachedTarget));
fprintf('总步数: %d\n', G.reachedStep);
fprintf('总距离: %.1f mm\n', totalDistance);
fprintf('曲折度: %.3f\n', G.tortuosity);
fprintf('耗时: %.1f s\n', elapsed);

%% ====== 绘图 ======
figure('Position', [50, 50, 1200, 500]);

subplot(1,3,1);
drawRobotsTraj(G);
title(sprintf('%s: 运动轨迹', upper(alg_mode)));

subplot(1,3,2);
plot(1:G.reachedStep, G.activatedCount(1:G.reachedStep), 'b-', 'LineWidth', 1.5);
xlabel('仿真步数'); ylabel('激活个体数');
title('激活个体数');
grid on;
for k = 1:length(G.myinfoIDs)
    rotateAt = k * G.rotateStep;
    if rotateAt <= G.reachedStep
        xline(rotateAt, 'r--', sprintf('ID%d', G.myinfoIDs(k)), 'LineWidth', 1);
    end
end

subplot(1,3,3);
plot(1:G.reachedStep, G.maxcj(1:G.reachedStep), 'r-', 'LineWidth', 1);
xlabel('仿真步数'); ylabel('max(s_{ij}) (rad/s)');
title('最大运动显著性');
grid on;
if strcmp(alg_mode, 'fixed')
    yline(G.cj_threshold, 'k--', 'M_T', 'LineWidth', 1.5);
else
    yline(G.cj_high, 'k--', 'cj_{high}', 'LineWidth', 1);
    yline(G.cj_low, 'g--', 'cj_{low}', 'LineWidth', 1);
end

sgtitle(sprintf('SwarmBang 数字孪生 - %s | 步数: %d | 曲折度: %.3f', ...
    upper(alg_mode), G.reachedStep, G.tortuosity));
