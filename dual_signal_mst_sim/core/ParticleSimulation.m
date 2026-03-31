classdef ParticleSimulation < handle
% ParticleSimulation 粒子群运动仿真类
%
% 类描述:
%   该类实现了一个基于自驱动粒子的集体运动仿真系统，支持粒子间的
%   激活传播机制。粒子在二维空间中运动，能够根据邻居的运动状态
%   被激活并跟随特定粒子运动。
%
% 主要功能:
%   - 粒子位置和方向的动态更新
%   - 基于半径或拓扑的邻居查找
%   - 粒子激活/去激活状态管理
%   - 运动显著性检测和激活传播
%   - 系统阶参数计算
%
% 使用示例:
%   params.N = 200;
%   params.cj_threshold = 2.0;
%   sim = ParticleSimulation(params);
%   sim.runSimulation();
%
% 作者：系统生成
% 日期：2025年
% 版本：MATLAB 2025a兼容

    properties
        % 仿真参数
        N = 1000;            % 粒子数量
        rho = 1;             % 粒子密度
        simulationAreaSize; % 模拟区域大小 simulationAreaSize x simulationAreaSize
        v0 = 1;              % 粒子的固定速度大小
        angleUpdateParameter = 5; % 角度更新参数
        angleNoiseIntensity = 5;   % 角度噪声强度
        T_max = 2000;        % 最大仿真时间步数
        dt = 0.1;            % 时间步长
        radius = 5;          % 邻居查找半径
        deac_threshold = 10; % 取消激活的角度阈值（弧度）
        cj_threshold = 50;   % 激活阈值（弧度/时间）
        useAdaptiveThreshold = false; % 是否启用自适应运动显著性阈值
        adaptiveThresholdConfig = struct(); % 自适应阈值配置结构体
        cj_threshold_dynamic; % [N x 1] 实时阈值
        local_order_state; % [N x 1] 记录局部诊断量（binary 模式下为局部序参量，双信号模式下为 C_norm）
        local_saliency_state; % [N x 1] 记录邻域运动显著性统计

        % 双信号门控连续自适应参数
        adaptiveThresholdMode = 'binary';   % 'binary' | 'binary_veto' | 'dual_signal_gate' | 'sigmoid_gate' | 'algebraic_fusion' | 'two_stage'
        dualSignalGamma = 1.0;              % γ: 方差映射指数
        dualSignalVarianceRef = [];         % V_ref: 方差归一化参考尺度（由 calibrate_vref 标定）
        dualSignalSmoothingLambda = 0.3;    % λ_base: 下降分支基础平滑速度
        dualSignalAlphaMin = 0.3;           % α_min: 最低平滑速度保底（防止 C_ratio 完全阻断）
        dualSignalLambdaUp = 0.5;           % λ_up: 上升分支快速恢复速度
        dualSignalClow = 30;                % C_low: C_ratio 归一化 log 空间下界
        dualSignalChigh = 150;              % C_high: C_ratio 归一化 log 空间上界

        % v5.4 sigmoid 门控参数
        dualSignalDmid = 0.6;               % D_mid: sigmoid 门控中心点
        dualSignalKappa = 10;               % κ: sigmoid 陡峭度

        % 二阶段判决参数
        twoStageThresholdOn = 0.6;          % θ_on: 候选触发阈值 (D 空间)
        twoStageThresholdOff = 0.3;         % θ_off: 恢复阈值 (D 空间)
        twoStageWindowK = 2;                % K: 确认窗口长度 (步数)
        twoStageAlpha = 0.6;                % α: 证据函数中 D 的权重
        twoStageConfThreshold = 1.0;        % Θ_conf: 累积证据确认阈值
        twoStageMidThreshold = 3.0;         % M_mid: 候选状态中间阈值

        % 二阶段判决内部状态 (0=H, 1=P, 2=L)
        twoStageState;                      % [N x 1] 粒子判决状态
        twoStageTimer;                      % [N x 1] 确认窗口计数器
        twoStageEvidence;                   % [N x 1] 累积证据

        % 拓扑邻居选择参数
        use_topology = false; % 是否使用拓扑邻居选择（false: 基于半径, true: 基于拓扑）
        k_neighbors = 7;      % 拓扑邻居选择中的最近邻数量

        % 固定场地参数
        fieldSize = 50;              % 固定场地大小
        initDirection = pi/4;        % 统一初始方向 (45度)
        useFixedField = false;       % 固定场地模式开关

        % 粒子状态
        positions;          % [N x 2] 粒子位置
        theta;              % [N x 1] 粒子方向
        previousPositions;  % [N x 2] 上一时刻的位置
        isActive;           % [N x 1] 粒子激活状态
        src_ids;            % {N x 1} 源头 ID 数组

        % 仿真结果
        order_parameter;    % [T_max x 1] 阶参数
        activated_counts;   % [T_max x 1] 激活个体数

        % 当前时间步
        current_step = 0;
    end

    methods
        function obj = ParticleSimulation(params)
        % ParticleSimulation 构造函数
        %
        % 功能描述:
        %   初始化粒子仿真对象，设置仿真参数并创建粒子初始状态
        %
        % 输入参数:
        %   params - 包含仿真参数的结构体，可包含以下字段:
        %     N - 粒子数量
        %     rho - 粒子密度
        %     v0 - 粒子速度
        %     cj_threshold - 激活阈值
        %     useFixedField - 是否使用固定场地
        %     等等...
        %
        % 输出结果:
        %   obj - 初始化完成的粒子仿真对象
        
            % 构造函数，初始化参数和粒子状态
            if nargin > 0
                fields = fieldnames(params);
                for i = 1:length(fields)
                    if isprop(obj, fields{i})
                        obj.(fields{i}) = params.(fields{i});
                    end
                end
            end

            if obj.useAdaptiveThreshold
                obj.applyAdaptiveThresholdDefaults();
            else
                obj.cj_threshold_dynamic = [];
                obj.local_order_state = [];
                obj.local_saliency_state = [];
            end

            % 场地大小设置
            if obj.useFixedField
                obj.simulationAreaSize = obj.fieldSize;
            else
                obj.simulationAreaSize = sqrt(obj.N / obj.rho);
            end

            obj.initializeParticles();
        end

        function initializeParticles(obj)
        % initializeParticles 初始化粒子位置和状态
        %
        % 功能描述:
        %   设置粒子的初始位置、方向和激活状态，使用物理分散方法
        %   避免粒子位置重叠，确保合理的初始分布
        %
        % 算法特点:
        %   - 支持固定场地和全区域两种初始化模式
        %   - 使用物理分散算法避免粒子重叠
        %   - 预分配所有状态数组，提高性能

            % 初始化粒子的位置、方向和状态
            % 使用物理分散方法避免位置重叠

            if obj.useFixedField
                % 固定场地模式：左下角初始化
                [dispersed_positions, ~] = obj.generateCornerDispersedPositions(obj.N);
                obj.positions = dispersed_positions';

                % 统一设置45度方向
                obj.theta = ones(obj.N, 1) * obj.initDirection;
            else
                % 原有模式：全区域随机初始化
                [dispersed_positions, ~] = obj.generateDispersedPositions(obj.N);
                obj.positions = dispersed_positions';
            obj.theta = rand(obj.N, 1) * 2 * pi;      % 随机分布方向
            end

        obj.previousPositions = obj.positions;     % 上一时刻的位置
        obj.isActive = false(obj.N, 1);           % 初始化激活状态
        obj.src_ids = cell(obj.N, 1);             % 初始化源头 ID 数组
        obj.order_parameter = zeros(obj.T_max, 1);
        obj.activated_counts = zeros(obj.T_max, 1);

        if obj.useAdaptiveThreshold
            obj.initializeAdaptiveThresholdState();
        end
        end

        function [final_positions, final_velocities] = generateDispersedPositions(obj, num_agents)
        % generateDispersedPositions 生成不重叠的粒子位置和速度
        %
        % 功能描述:
        %   使用物理分散算法生成不重叠的粒子初始位置，避免粒子
        %   在初始状态时过于集中，影响仿真结果
        %
        % 算法原理:
        %   基于Lennard-Jones势能模型，粒子间同时存在短程排斥力和
        %   长程吸引力，通过向量化计算实现高效的分散过程
        %
        % 输入参数:
        %   num_agents - 需要生成的粒子数量
        %
        % 输出结果:
        %   final_positions - 最终位置矩阵 [2 x num_agents]
        %   final_velocities - 最终速度矩阵 [2 x num_agents]
        %
        % 性能优化:
        %   - 完全向量化实现，避免循环
        %   - 预分配所有矩阵
        %   - 使用MATLAB内置向量运算函数

            % 生成不重叠的个体位置和速度 (向量化优化版本)
            % 通过矩阵运算实现高效的物理分散算法
            %
            % 输入参数:
            %   num_agents - 个体数量
            %
            % 输出参数:
            %   final_positions - 最终位置矩阵 [2 x num_agents]
            %   final_velocities - 最终速度矩阵 [2 x num_agents]

            % === 参数设置 ===
            max_steps = 20;                 % 最大仿真步数
            dt_init = 0.1;                       % 时间步长
            repulsion_range = 3;            % 排斥功作用范围
            attraction_decay = 10;          % 吸引力衰减系数
            max_accel = 5;                  % 最大加速度
            target_speed = 0;               % 目标速度
            speed_relax = 0.1;              % 速度松弛时间
            noise_strength = 0.5;           % 噪声强度

            % === 初始化 ===
            init_scale = min(obj.simulationAreaSize * 0.8, 3);
            pos = rand(2, num_agents) * init_scale;  % [2 x N] 位置矩阵
            vel = ones(2, num_agents) * 0.1;         % [2 x N] 速度矩阵

            % === 主循环 (向量化) ===
            for step = 1:max_steps
                % 计算距离矩阵 [N x N]
                dx_mat = pos(1,:) - pos(1,:)';
                dy_mat = pos(2,:) - pos(2,:)';
                dist_mat = sqrt(dx_mat.^2 + dy_mat.^2);
                dist_mat(dist_mat == 0) = inf;  % 避免自作用

                % 计算力大小矩阵 [N x N]
                force_mag = (1 - (repulsion_range ./ dist_mat).^2) .* exp(-dist_mat / attraction_decay);

                % 计算单位方向向量 [N x N]
                dx_unit = dx_mat ./ dist_mat;
                dy_unit = dy_mat ./ dist_mat;

                % 计算每个粒子受到的总力 [2 x N]
                fx = sum(force_mag .* dx_unit, 2, 'omitnan')';  % [1 x N]
                fy = sum(force_mag .* dy_unit, 2, 'omitnan')';  % [1 x N]
                interaction_force = [fx; fy];  % [2 x N]

                % 计算自驱力 [2 x N]
                speeds = vecnorm(vel, 2, 1);  % [1 x N]
                speed_error = (target_speed - speeds) / speed_relax;  % [1 x N]
                safe_speeds = max(speeds, eps);
                vel_unit = vel ./ safe_speeds;  % [2 x N]
                self_force = vel_unit .* speed_error;  % [2 x N]

                % 总力和加速度限制 [2 x N]
                total_force = self_force + interaction_force;
                force_mags = vecnorm(total_force, 2, 1);  % [1 x N]
                scale_factors = min(max_accel ./ max(force_mags, eps), 1);  % [1 x N]
                limited_force = total_force .* scale_factors;  % [2 x N]

                % 添加噪声和状态更新 [2 x N]
                noise = (rand(2, num_agents) - 0.5) * 2 * noise_strength;
                accel = limited_force + noise;
                vel = vel + accel * dt_init;
                pos = pos + vel * dt_init;
            end

            % === 返回结果 ===
            final_positions = pos;
            final_velocities = vel;
        end

        function init_zone_size = calculateInitZoneSize(obj)
        % calculateInitZoneSize 计算初始化区域大小
        %
        % 功能描述:
        %   根据粒子数量和场地大小计算合适的初始化区域，
        %   确保粒子初始分布既不过于集中也不会超出边界
        %
        % 算法特点:
        %   - 固定场地模式：基于粒子数量的平方根缩放
        %   - 自由场地模式：使用总场地大小的30%
        %   - 设置上限为场地大小的50%，避免超出边界

            % 计算初始化区域大小
            if ~obj.useFixedField
                init_zone_size = obj.simulationAreaSize * 0.3;
                return;
            end

            % 简化公式：初始化区域 = sqrt(粒子数量) * 缩放因子
            scale_factor = 0.3;
            init_zone_size = sqrt(obj.N) * scale_factor;

            % 边界限制：不超过场地的一半
            max_size = obj.fieldSize * 0.5;
            init_zone_size = min(init_zone_size, max_size);
        end

        function [final_positions, final_velocities] = generateCornerDispersedPositions(obj, num_agents)
        % generateCornerDispersedPositions 左下角区域粒子分散初始化
        %
        % 功能描述:
        %   在场地左下角区域生成不重叠的粒子位置，用于固定场地模式
        %   的初始化。使用物理分散算法确保粒子间有合理间距
        %
        % 算法特点:
        %   - 在计算出的初始化区域内随机分布粒子
        %   - 使用与generateDispersedPositions相同的物理分散算法
        %   - 适应固定场地模式的特殊需求

            % 左下角初始化 + randPose物理分散

            % 计算初始化区域大小
            init_zone_size = obj.calculateInitZoneSize();

            % 物理分散参数
            max_steps = 10;
            dt_init = 0.1;
            repulsion_range = 3;
            attraction_decay = 10;
            max_accel = 5;
            target_speed = 0;
            speed_relax = 0.1;
            noise_strength = 0.5;

            % MATLAB向量化初始化
            pos = rand(2, num_agents) * init_zone_size;
            vel = zeros(2, num_agents);

            % 物理分散循环
            for step = 1:max_steps
                % 距离矩阵计算
                dx_mat = pos(1,:) - pos(1,:)';
                dy_mat = pos(2,:) - pos(2,:)';
                dist_mat = sqrt(dx_mat.^2 + dy_mat.^2);
                dist_mat(dist_mat == 0) = inf;

                % 力计算
                force_mag = (1 - (repulsion_range ./ dist_mat).^2) .* exp(-dist_mat / attraction_decay);
                dx_unit = dx_mat ./ dist_mat;
                dy_unit = dy_mat ./ dist_mat;

                fx = sum(force_mag .* dx_unit, 2, 'omitnan')';
                fy = sum(force_mag .* dy_unit, 2, 'omitnan')';
                interaction_force = [fx; fy];

                % 速度调节力
                speeds = vecnorm(vel, 2, 1);
                speed_error = (target_speed - speeds) / speed_relax;
                safe_speeds = max(speeds, eps);
                vel_unit = vel ./ safe_speeds;
                self_force = vel_unit .* speed_error;

                % 状态更新
                total_force = self_force + interaction_force;
                force_mags = vecnorm(total_force, 2, 1);
                scale_factors = min(max_accel ./ max(force_mags, eps), 1);
                limited_force = total_force .* scale_factors;

                noise = (rand(2, num_agents) - 0.5) * 2 * noise_strength;
                accel = limited_force + noise;
                vel = vel + accel * dt_init;
                pos = pos + vel * dt_init;
            end

            final_positions = pos;
            final_velocities = vel;
        end

        function neighbor_matrix = findNeighbors(obj)
        % findNeighbors 查找粒子邻居
        %
        % 功能描述:
        %   根据选择的邻居查找策略(基于半径或基于拓扑)，
        %   找到每个粒子的邻居，返回邻接矩阵
        %
        % 输出结果:
        %   neighbor_matrix - 布尔邻接矩阵 [N x N]，元素(i,j)为true表示j是i的邻居

            % 查找每个粒子的邻居，返回邻接矩阵 [N x N]
            if obj.use_topology
                neighbor_matrix = obj.findTopologyNeighbors();
            else
                neighbor_matrix = obj.findRadiusNeighbors();
            end
        end

        function neighbor_matrix = findRadiusNeighbors(obj)
        % findRadiusNeighbors 基于半径的邻居查找
        %
        % 功能描述:
        %   查找每个粒子固定半径内的所有邻居，这是经典的
        %   基于距离的邻居查找方法
        %
        % 算法特点:
        %   - 使用向量化计算，避免循环
        %   - 基于欧几里得距离计算
        %   - 排除自身(距离为0的情况)
        %
        % 性能考虑:
        %   时间复杂度为O(N^2)，适合中等规模的粒子系统

            % 基于半径的邻居查找（原始方法）
            % 使用向量化方法提高查找效率，使用标准欧几里得距离
            dx = abs(obj.positions(:,1) - obj.positions(:,1)');
            dy = abs(obj.positions(:,2) - obj.positions(:,2)');
            distance_matrix = sqrt(dx.^2 + dy.^2);
            neighbor_matrix = distance_matrix < obj.radius & distance_matrix > 0;
        end

        function neighbor_matrix = findTopologyNeighbors(obj)
        % findTopologyNeighbors 基于拓扑的邻居查找
        %
        % 功能描述:
        %   查找每个粒子的k个最近邻，不考虑实际距离，
        %   只基于拓扑关系确定邻居
        %
        % 算法特点:
        %   - 每个粒子有固定数量的邻居(k_neighbors)
        %   - 使用距离排序找到最近的k个邻居
        %   - 向量化实现提高效率
        %
        % 优势:
        %   保证每个粒子有相同数量的邻居，避免稀疏区域
        %   的粒子连接度过低的问题

            % 基于拓扑的邻居查找（k-nearest neighbors）
            % 返回邻接矩阵 [N x N]
            % 使用向量化方法提高效率

            % 计算所有粒子间的距离矩阵
            dx = obj.positions(:,1) - obj.positions(:,1)';
            dy = obj.positions(:,2) - obj.positions(:,2)';
            distance_matrix = sqrt(dx.^2 + dy.^2);

            % 排除自身距离（设为无穷大）
            distance_matrix(1:obj.N+1:end) = inf;

            % 向量化找到每行的最小k个元素的索引
            [~, sorted_indices] = sort(distance_matrix, 2);

            % 生成邻接矩阵
            neighbor_matrix = false(obj.N, obj.N);
            k_actual = min(obj.k_neighbors, obj.N-1);

            % 向量化设置邻接关系
            row_indices = repmat((1:obj.N)', 1, k_actual);
            col_indices = sorted_indices(:, 1:k_actual);
            linear_indices = sub2ind([obj.N, obj.N], row_indices(:), col_indices(:));
            neighbor_matrix(linear_indices) = true;
        end

        function step(obj)
        % step 执行一个时间步的粒子仿真
        %
        % 功能描述:
        %   这是仿真核心函数，执行一个完整的时间步，包括邻居查找、
        %   激活状态更新、角度和位置更新，以及统计量计算
        %
        % 算法流程:
        %   1. 查找每个粒子的邻居
        %   2. 更新激活状态和期望运动方向
        %   3. 根据期望方向更新粒子角度
        %   4. 根据角度更新粒子位置
        %   5. 计算并存储统计量
        %
        % 激活机制:
        %   - 未激活粒子检测邻居的运动显著性，超过阈值则被激活
        %   - 已激活粒子跟踪源头粒子，角度差过大则取消激活
        %   - 运动显著性通过邻居位置变化的角度速度计算

            % 执行一个时间步的仿真
            if obj.current_step >= obj.T_max
                return;
            end
            obj.current_step = obj.current_step + 1;
            t_step = obj.current_step;

            % 1. 查找邻居
            neighbor_matrix = obj.findNeighbors();

            if obj.useAdaptiveThreshold
                obj.updateAdaptiveThresholds(neighbor_matrix);
            end

            desired_theta = NaN(obj.N, 1);

            % 2. 更新粒子状态和期望方向
            for i = 1:obj.N
                neibor_idx = find(neighbor_matrix(i, :));
                if obj.isActive(i)
                    % 已激活，更新源头 ID
                    if ismember(obj.src_ids{i}, neibor_idx)  % 检查唯一的源头是否还在邻居中
                        % 计算与源头的角度差（弧度）
                        src_direction = obj.theta(obj.src_ids{i});
                        angle_diff_rad = abs(wrapToPi(obj.theta(i) - src_direction));

                        if angle_diff_rad < obj.deac_threshold
                            % 取消激活
                            obj.isActive(i) = false;
                            obj.src_ids{i} = [];
                            if ~isempty(neibor_idx)
                                neibor_directions = obj.theta(neibor_idx);
                                avg_neibor_dir = angle(mean(exp(1j * neibor_directions)));
                                avg_neibor_dir = wrapTo2Pi(avg_neibor_dir);
                                desired_theta(i) = avg_neibor_dir;
                            else
                                desired_theta(i) = obj.theta(i);
                            end
                        else
                            % 保持激活，跟随源头
                            desired_theta(i) = src_direction;
                        end
                    else
                        % 源头不在邻居中，取消激活
                        obj.isActive(i) = false;
                        obj.src_ids{i} = [];
                        if ~isempty(neibor_idx)
                            neibor_directions = obj.theta(neibor_idx);
                            avg_neibor_dir = angle(mean(exp(1j * neibor_directions)));
                            avg_neibor_dir = wrapTo2Pi(avg_neibor_dir);
                            desired_theta(i) = avg_neibor_dir;
                        else
                            desired_theta(i) = obj.theta(i);
                        end
                    end
                else
                    % 未激活，检查候选邻居
                    if isempty(neibor_idx)
                        desired_theta(i) = obj.theta(i);
                    else
                        % 计算相对位置变化
                        current_diff = obj.positions(neibor_idx, :) - obj.positions(i, :);
                        past_diff = obj.previousPositions(neibor_idx, :) - obj.previousPositions(i, :);

                        % 使用标准欧几里得距离计算，不考虑周期性边界条件
                        current_dist = vecnorm(current_diff, 2, 2);
                        past_dist = vecnorm(past_diff, 2, 2);

                        % 计算单位方向向量
                        current_diff_unit = zeros(size(current_diff));
                        past_diff_unit = zeros(size(past_diff));
                        non_zero_current = current_dist > 0;
                        non_zero_past = past_dist > 0;

                        current_diff_unit(non_zero_current, :) = current_diff(non_zero_current, :) ./ current_dist(non_zero_current);
                        past_diff_unit(non_zero_past, :) = past_diff(non_zero_past, :) ./ past_dist(non_zero_past);

                        % 计算角度差（弧度）
                        angle_cos = sum(past_diff_unit .* current_diff_unit, 2);
                        angle_cos = max(min(angle_cos, 1), -1);
                        angle_diff_rad = acos(angle_cos);

                        % 计算显著性值 s
                        s_values = angle_diff_rad / obj.dt;
                        % 找出显著性最高的邻居
                        [max_s, max_s_idx] = max(s_values);

                        threshold_i = obj.getActivationThreshold(i);
                        if max_s > threshold_i
                            % 激活并只跟随最显著的邻居
                            obj.isActive(i) = true;
                            obj.src_ids{i} = neibor_idx(max_s_idx);  % 只存储显著性最高的邻居ID
                            desired_theta(i) = obj.theta(neibor_idx(max_s_idx));  % 直接使用该邻居的方向
                        else
                            % 不激活，设置期望方向为邻居平均方向
                            neighbor_directions = obj.theta(neibor_idx);
                            avg_dir = angle(mean(exp(1j * neighbor_directions)));
                            avg_dir = wrapTo2Pi(avg_dir);
                            desired_theta(i) = avg_dir;
                        end
                    end
                end
            end

            % 3. 更新角度
            delta_theta = desired_theta - obj.theta;
            weighted_sin = sin(delta_theta);
            obj.theta = obj.theta + obj.angleUpdateParameter * weighted_sin * obj.dt + sqrt(2 * obj.angleNoiseIntensity * obj.dt) * randn(obj.N, 1);
            obj.theta = mod(obj.theta, 2 * pi);

            % 4. 更新位置
            obj.previousPositions = obj.positions;
            obj.positions = obj.positions + obj.v0 * [cos(obj.theta), sin(obj.theta)] * obj.dt;
            % 移除周期性边界条件：粒子可以移动到边界之外

            % 5. 计算并存储阶参数和激活个体数
            obj.order_parameter(t_step) = norm(sum(exp(1j * obj.theta))) / obj.N;
            obj.activated_counts(t_step) = sum(obj.isActive);
        end

        function applyAdaptiveThresholdDefaults(obj)
        % applyAdaptiveThresholdDefaults 补全自适应阈值配置
            cfg = obj.adaptiveThresholdConfig;

            if ~isfield(cfg, 'cj_low') || ~isnumeric(cfg.cj_low)
                cfg.cj_low = max(0.1, obj.cj_threshold * 0.5);
            end
            if ~isfield(cfg, 'cj_high') || ~isnumeric(cfg.cj_high)
                cfg.cj_high = max(cfg.cj_low, obj.cj_threshold);
            end
            if cfg.cj_high < cfg.cj_low
                tmp = cfg.cj_low;
                cfg.cj_low = cfg.cj_high;
                cfg.cj_high = tmp;
            end

            if ~isfield(cfg, 'order_threshold') || ~isnumeric(cfg.order_threshold)
                cfg.order_threshold = [];
            end
            if ~isempty(cfg.order_threshold)
                cfg.order_threshold = min(max(cfg.order_threshold, 0), 1);
            end

            if ~isfield(cfg, 'saliency_threshold') || ~isnumeric(cfg.saliency_threshold)
                cfg.saliency_threshold = [];
            end
            if ~isempty(cfg.saliency_threshold)
                cfg.saliency_threshold = max(cfg.saliency_threshold, 0);
            end

            if ~isfield(cfg, 'include_self') || ~islogical(cfg.include_self)
                cfg.include_self = true;
            end

            obj.adaptiveThresholdConfig = cfg;
        end

        function initializeAdaptiveThresholdState(obj)
        % initializeAdaptiveThresholdState 初始化自适应阈值状态
            obj.cj_threshold_dynamic = ones(obj.N, 1) * obj.cj_threshold;
            obj.local_order_state = zeros(obj.N, 1);
            obj.local_saliency_state = zeros(obj.N, 1);
            % 二阶段判决状态初始化
            obj.twoStageState = zeros(obj.N, 1);     % 全部从 H(0) 开始
            obj.twoStageTimer = zeros(obj.N, 1);
            obj.twoStageEvidence = zeros(obj.N, 1);
        end

        function updateAdaptiveThresholds(obj, neighbor_matrix)
        % updateAdaptiveThresholds 根据局部序参量更新实时阈值
            cfg = obj.adaptiveThresholdConfig;
            if isempty(obj.cj_threshold_dynamic) || numel(obj.cj_threshold_dynamic) ~= obj.N
                obj.initializeAdaptiveThresholdState();
            end

            use_order_metric = ~isempty(cfg.order_threshold);
            use_saliency_metric = ~isempty(cfg.saliency_threshold);

            % 双信号模式强制启用显著性统计
            if ismember(obj.adaptiveThresholdMode, {'binary_veto', 'dual_signal_gate', 'sigmoid_gate', 'algebraic_fusion', 'two_stage'})
                use_saliency_metric = true;
            end

            if ~use_order_metric && ~use_saliency_metric
                % 若未配置任何自适应指标，保持静态阈值
                return;
            end

            order_thr = cfg.order_threshold;
            saliency_thr = cfg.saliency_threshold;

            for i = 1:obj.N
                neighbor_idx = find(neighbor_matrix(i, :));
                order_norm = 0;
                saliency_var = 0;
                s_values = [];

                if use_order_metric
                    if cfg.include_self
                        angles = [obj.theta(i); obj.theta(neighbor_idx)];
                    else
                        angles = obj.theta(neighbor_idx);
                    end

                    if isempty(angles)
                        order_norm = 0;
                    else
                        vecs = [cos(angles), sin(angles)];
                        mean_vec = mean(vecs, 1);
                        order_norm = norm(mean_vec);
                    end

                    order_norm = min(max(order_norm, 0), 1);
                    obj.local_order_state(i) = order_norm;
                end

                if use_saliency_metric
                    if isempty(neighbor_idx)
                        saliency_var = 0;
                    else
                        current_diff = obj.positions(neighbor_idx, :) - obj.positions(i, :);
                        past_diff = obj.previousPositions(neighbor_idx, :) - obj.previousPositions(i, :);

                        current_dist = vecnorm(current_diff, 2, 2);
                        past_dist = vecnorm(past_diff, 2, 2);

                        current_diff_unit = zeros(size(current_diff));
                        past_diff_unit = zeros(size(past_diff));

                        non_zero_current = current_dist > 0;
                        non_zero_past = past_dist > 0;

                        current_diff_unit(non_zero_current, :) = current_diff(non_zero_current, :) ./ current_dist(non_zero_current);
                        past_diff_unit(non_zero_past, :) = past_diff(non_zero_past, :) ./ past_dist(non_zero_past);

                        angle_cos = sum(past_diff_unit .* current_diff_unit, 2);
                        angle_cos = max(min(angle_cos, 1), -1);
                        angle_diff_rad = acos(angle_cos);

                        s_values = angle_diff_rad / max(obj.dt, eps);
                        saliency_var = var(s_values, 1, 'omitnan');
                    end

                    obj.local_saliency_state(i) = saliency_var;
                end

                if use_saliency_metric
                    switch obj.adaptiveThresholdMode
                        case 'binary'
                            if saliency_var >= saliency_thr
                                target_cj = cfg.cj_low;
                            else
                                target_cj = cfg.cj_high;
                            end
                            obj.cj_threshold_dynamic(i) = target_cj;

                        case 'binary_veto'
                            % C_ratio 否决门：σ²_M 触发 + C_ratio 确认
                            % 复用 dualSignalClow 作为否决阈值（噪声 95th 上界）
                            if saliency_var >= saliency_thr && numel(s_values) >= 2
                                C_raw = max(s_values) / (median(s_values) + eps);
                                if C_raw >= obj.dualSignalClow
                                    target_cj = cfg.cj_low;
                                else
                                    target_cj = cfg.cj_high;
                                end
                            else
                                target_cj = cfg.cj_high;
                            end
                            obj.cj_threshold_dynamic(i) = target_cj;
                            if numel(s_values) >= 2
                                obj.local_order_state(i) = max(s_values) / (median(s_values) + eps);
                            end

                        case 'dual_signal_gate'
                            [D, C_norm] = obj.computeDualSignalMetrics(saliency_var, s_values);
                            target_cj = cfg.cj_high - (cfg.cj_high - cfg.cj_low) * (D ^ obj.dualSignalGamma);
                            if target_cj < obj.cj_threshold_dynamic(i)
                                % 下降分支：C_ratio 门控速度，α_min 保底
                                lam_eff = obj.dualSignalSmoothingLambda * ...
                                    (obj.dualSignalAlphaMin + (1 - obj.dualSignalAlphaMin) * C_norm);
                            else
                                % 上升分支：快速恢复，不受 C_ratio 限制
                                lam_eff = obj.dualSignalLambdaUp;
                            end
                            lam_eff = min(max(lam_eff, 0), 1);
                            obj.cj_threshold_dynamic(i) = ...
                                (1 - lam_eff) * obj.cj_threshold_dynamic(i) + lam_eff * target_cj;
                            obj.local_order_state(i) = C_norm;

                        case 'sigmoid_gate'
                            [D, C_norm] = obj.computeDualSignalMetrics(saliency_var, s_values);
                            target_cj = cfg.cj_high - (cfg.cj_high - cfg.cj_low) * (D ^ obj.dualSignalGamma);
                            if target_cj < obj.cj_threshold_dynamic(i)
                                w = 1 / (1 + exp(-obj.dualSignalKappa * (D - obj.dualSignalDmid)));
                                confidence = w + (1 - w) * C_norm;
                                lam_floor = obj.dualSignalSmoothingLambda * obj.dualSignalAlphaMin;
                                lam_eff = lam_floor + (1 - lam_floor) * confidence;
                            else
                                lam_eff = obj.dualSignalLambdaUp;
                            end
                            lam_eff = min(max(lam_eff, 0), 1);
                            obj.cj_threshold_dynamic(i) = ...
                                (1 - lam_eff) * obj.cj_threshold_dynamic(i) + lam_eff * target_cj;
                            obj.local_order_state(i) = C_norm;

                        case 'algebraic_fusion'
                            [D, C_norm] = obj.computeDualSignalMetrics(saliency_var, s_values);
                            target_cj = cfg.cj_high - (cfg.cj_high - cfg.cj_low) * (D ^ obj.dualSignalGamma);
                            if target_cj < obj.cj_threshold_dynamic(i)
                                lam_floor = obj.dualSignalSmoothingLambda * obj.dualSignalAlphaMin;
                                lam_raw = D * (D + (1 - D) * C_norm);
                                lam_eff = max(lam_floor, lam_raw);
                            else
                                lam_eff = obj.dualSignalLambdaUp;
                            end
                            lam_eff = min(max(lam_eff, 0), 1);
                            obj.cj_threshold_dynamic(i) = ...
                                (1 - lam_eff) * obj.cj_threshold_dynamic(i) + lam_eff * target_cj;
                            obj.local_order_state(i) = C_norm;

                        case 'two_stage'
                            [D, C_norm] = obj.computeDualSignalMetrics(saliency_var, s_values);
                            E_t = D * (obj.twoStageAlpha + (1 - obj.twoStageAlpha) * C_norm);
                            si = obj.twoStageState(i);

                            if si == 0  % H: 高阈值稳态
                                if D >= obj.twoStageThresholdOn
                                    % 候选触发：降到中间阈值
                                    obj.twoStageState(i) = 1;  % → P
                                    obj.twoStageTimer(i) = 1;
                                    obj.twoStageEvidence(i) = E_t;
                                    obj.cj_threshold_dynamic(i) = obj.twoStageMidThreshold;
                                else
                                    obj.cj_threshold_dynamic(i) = cfg.cj_high;
                                end

                            elseif si == 1  % P: 候选状态，累积证据
                                obj.twoStageTimer(i) = obj.twoStageTimer(i) + 1;
                                obj.twoStageEvidence(i) = obj.twoStageEvidence(i) + E_t;
                                obj.cj_threshold_dynamic(i) = obj.twoStageMidThreshold;  % 窗口内用中间阈值

                                if obj.twoStageTimer(i) >= obj.twoStageWindowK
                                    % 窗口结束，判决
                                    if obj.twoStageEvidence(i) >= obj.twoStageConfThreshold
                                        % 确认成功 → L
                                        obj.twoStageState(i) = 2;
                                        obj.cj_threshold_dynamic(i) = cfg.cj_low;
                                    else
                                        % 确认失败 → 撤回到 H
                                        obj.twoStageState(i) = 0;
                                        obj.twoStageTimer(i) = 0;
                                        obj.twoStageEvidence(i) = 0;
                                        obj.cj_threshold_dynamic(i) = cfg.cj_high;
                                    end
                                end

                            else  % L: 已确认低阈值
                                if D < obj.twoStageThresholdOff
                                    % 信号消退，恢复高阈值
                                    obj.twoStageState(i) = 0;
                                    obj.twoStageTimer(i) = 0;
                                    obj.twoStageEvidence(i) = 0;
                                    obj.cj_threshold_dynamic(i) = cfg.cj_high;
                                else
                                    obj.cj_threshold_dynamic(i) = cfg.cj_low;
                                end
                            end
                            obj.local_order_state(i) = C_norm;

                        otherwise
                            obj.cj_threshold_dynamic(i) = obj.cj_threshold;
                    end
                else
                    if order_norm >= order_thr
                        target_cj = cfg.cj_high;
                    else
                        target_cj = cfg.cj_low;
                    end
                    obj.cj_threshold_dynamic(i) = target_cj;
                end
            end
        end

        function [D, C_norm] = computeDualSignalMetrics(obj, saliency_var, s_values)
        % computeDualSignalMetrics 计算双信号模式的两个归一化输入
        %   D: sigma2_M 归一化值 ∈ [0,1]，驱动目标阈值幅值
        %   C_norm: C_ratio 归一化值 ∈ [0,1]，门控下降速度
            V_ref = obj.dualSignalVarianceRef;
            if isempty(V_ref) || V_ref <= 0
                V_ref = 1.0;
            end
            D = min(saliency_var / max(V_ref, eps), 1);

            C_norm = 0;
            if numel(s_values) < 2, return; end

            C_raw = max(s_values) / (median(s_values) + eps);
            log_C = log10(max(C_raw, 1));
            C_low = max(obj.dualSignalClow, 1 + eps);
            C_high = max(obj.dualSignalChigh, C_low * (1 + 1e-6));
            log_low = log10(C_low);
            log_high = log10(C_high);
            C_norm = max(0, min((log_C - log_low) / (log_high - log_low), 1));
        end

        function threshold = getActivationThreshold(obj, idx)
        % getActivationThreshold 获取粒子的当前阈值
            if obj.useAdaptiveThreshold && ~isempty(obj.cj_threshold_dynamic)
                threshold = obj.cj_threshold_dynamic(idx);
            else
                threshold = obj.cj_threshold;
            end
        end

        function runSimulation(obj)
        % runSimulation 运行完整的仿真过程
        %
        % 功能描述:
        %   执行从开始到结束的完整仿真过程，调用step()函数
        %   直到达到最大时间步数
        %
        % 使用方法:
        %   sim = ParticleSimulation(params);
        %   sim.runSimulation();
        %   results = sim.order_parameter;  % 获取阶参数历史

            % 运行完整的仿真过程
            for t_step = 1:obj.T_max
                obj.step();
            end
        end
    end
end
