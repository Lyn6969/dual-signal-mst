% run_fp_comparison.m
% SwarmBang 数字孪生仿真：False Positive 场景对比实验
%
% 三种算法条件：Fixed / Binary / Weighted
% 运动显著性 s_ij 单位：rad/s（已去除 weight_cj = 200）
%
% 用法：直接运行即可

clc; clear; close all;

%% ====== 实验配置 ======
alg_modes = {'fixed', 'binary', 'weighted'};
nRuns = 5;

% 添加 core 目录到路径
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));

% 保存目录
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'fp_comparison');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

%% ====== 主实验循环 ======
all_results = struct();

for alg_idx = 1:length(alg_modes)
    alg_mode = alg_modes{alg_idx};
    fprintf('\n====== 算法: %s ======\n', alg_mode);

    for run = 1:nRuns
        fprintf('--- Run %d/%d ---\n', run, nRuns);

        % 初始化公共参数
        G = setup_common_params();

        % 算法特定参数
        switch alg_mode
            case 'fixed'
                G.cj_threshold = 0.04;  % rad/s（等效旧 C=8, weight_cj=200）

            case 'binary'
                G.cj_low = 0.01;        % rad/s, 敏感模式
                G.cj_high = 0.10;       % rad/s, 保守模式
                G.sigma_thr = 0.0005;   % (rad/s)², 切换阈值
                G.cj_threshold = G.cj_high;
                G.threshold_history = zeros(G.maxID, G.maxSimSteps);
                G.sigma2_history = zeros(G.maxID, G.maxSimSteps);

            case 'weighted'
                G.cj_low = 0.01;
                G.cj_high = 0.10;
                G.sigma_thr = 0.0005;
                G.cj_threshold = G.cj_high;
                G.weightMode = 'absolute'; % 'absolute' 或 'excess'
                G.threshold_history = zeros(G.maxID, G.maxSimSteps);
                G.sigma2_history = zeros(G.maxID, G.maxSimSteps);
        end

        % 初始化虚拟个体
        G = initSimRobots(G, 'unif', [1434, 1446], 100);

        % 选择信息个体（触发 false positive 的旋转个体）
        if length(G.robotsList) >= 3
            G.myinfoIDs = randsample(G.robotsList, 3);
        else
            error('机器人数量不足');
        end

        % 记录每个个体的轨迹
        for i = G.robotsList
            G.actor{i}.posHistory = nan(G.maxSimSteps, 2);
            G.actor{i}.posHistory(1, :) = G.actor{i}.pose;
        end

        reachedTarget = false;

        %% 仿真主循环
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

            % 记录轨迹
            for i = G.robotsList
                G.actor{i}.posHistory(t, :) = G.actor{i}.pose;
            end

            if isfield(G, 'now_infoID')
                G.nowInfoIDs(t) = G.now_infoID;
            end

            center = getSwarmCenter(G);
            G.swarmCenterPositions(t, :) = center;

            if norm(center - G.target) < G.targetR
                fprintf('  步数 %d: 集群到达目标区域\n', t);
                reachedTarget = true;
                break;
            end

            if mod(t, 500) == 0
                fprintf('  步数 %d, 激活: %d\n', t, G.activatedCount(t));
            end
        end

        %% 保存结果
        G.reachedTarget = reachedTarget;
        G.reachedStep = G.simStep;

        totalDistance = 0;
        for i = 2:G.reachedStep
            totalDistance = totalDistance + norm(...
                G.swarmCenterPositions(i,:) - G.swarmCenterPositions(i-1,:));
        end
        G.totalDistance = totalDistance;
        G.tortuosity = computePathTortuosity(G.swarmCenterPositions, G.reachedStep);

        fprintf('  到达: %s | 步数: %d | 距离: %.1f | 曲折度: %.3f\n', ...
            string(reachedTarget), G.reachedStep, totalDistance, G.tortuosity);

        timeStamp = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
        fileName = fullfile(resultsDir, ...
            sprintf('%s_run%d_%s.mat', alg_mode, run, string(timeStamp)));
        save(fileName, 'G', '-v7.3');

        all_results.(alg_mode)(run).reachedStep = G.reachedStep;
        all_results.(alg_mode)(run).totalDistance = G.totalDistance;
        all_results.(alg_mode)(run).tortuosity = G.tortuosity;
        all_results.(alg_mode)(run).reachedTarget = reachedTarget;
        all_results.(alg_mode)(run).activatedCount = G.activatedCount(1:G.reachedStep);
    end
end

%% ====== 结果汇总 ======
fprintf('\n====== 结果汇总 ======\n');
fprintf('%-12s %-12s %-12s %-12s %-12s\n', ...
    '算法', '到达率', '平均步数', '平均距离', '平均曲折度');

for alg_idx = 1:length(alg_modes)
    alg = alg_modes{alg_idx};
    data = all_results.(alg);
    steps = [data.reachedStep];
    dists = [data.totalDistance];
    torts = [data.tortuosity];
    reached = [data.reachedTarget];

    fprintf('%-12s %-12s %-12.1f %-12.1f %-12.3f\n', ...
        alg, sprintf('%d/%d', sum(reached), nRuns), ...
        mean(steps), mean(dists), mean(torts));
end

save(fullfile(resultsDir, 'all_results.mat'), 'all_results');
fprintf('\n结果已保存到: %s\n', resultsDir);

%% ====== 本地函数 ======
function G = setup_common_params()
    G = struct();
    G.simStep = 0;

    G.maxID = 50;
    G.cycTime = 0.2;            % s
    G.v0 = 15;                  % mm/s
    G.maxRotRate = 10 * 1.91;   % deg/s

    G.weight_align = 15;
    G.Drep = 400;               % mm
    G.Dsen = 1000;              % mm
    G.weight_rep = 8;
    G.weight_att = 0;
    G.deac_threshold = 0.2;
    G.noise_mov = 0;
    G.max_neighbors = 7;

    % 运动显著性：纯 rad/s，无 weight_cj
    G.n_steps = 5;

    G.maxSimSteps = 5000;
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
end
