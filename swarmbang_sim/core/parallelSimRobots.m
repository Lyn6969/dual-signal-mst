function G = parallelSimRobots(G, desTurnAngle, desSpeed)
% parallelSimRobots 虚拟个体运动学更新（数字孪生核心）
%
% 功能：根据期望转角和速度，使用运动学模型更新虚拟个体的位姿
%       模拟 swarmBang 机器人的 bang-bang 转向控制
%
% 输入:
%   G             - 全局结构体
%   desTurnAngle  - 期望转向角度 [maxID x 1]，单位 deg，右转为正
%   desSpeed      - 期望速度 [maxID x 1]，单位 mm/s

    for i = G.robotsList
        % 当前速度方向角（deg）
        currentAngle = atan2d(G.actor{i}.vel(2), G.actor{i}.vel(1));

        % 限幅转角（模拟 bang-bang 控制器的最大角速度限制）
        turnAngle = desTurnAngle(i);
        maxTurn = G.maxRotRate * G.cycTime; % 单步最大转角 (deg)
        if abs(turnAngle) > maxTurn
            turnAngle = sign(turnAngle) * maxTurn;
        end

        % 更新方向角
        newAngle = currentAngle + turnAngle;

        % 更新速度向量（保持单位向量 × 速度大小）
        speed = desSpeed(i);
        newVel = speed * [cosd(newAngle), sind(newAngle)];

        % 更新位置
        newPose = G.actor{i}.pose + newVel * G.cycTime;

        % 添加运动噪声
        if isfield(G, 'noise_mov') && G.noise_mov > 0
            noiseAngle = G.noise_mov * (rand - 0.5) * 2 * maxTurn;
            newAngle = newAngle + noiseAngle;
            newVel = speed * [cosd(newAngle), sind(newAngle)];
            newPose = G.actor{i}.pose + newVel * G.cycTime;
        end

        % 写回 actor
        G.actor{i}.pose = newPose;
        if speed > 0
            G.actor{i}.vel = [cosd(newAngle), sind(newAngle)]; % 单位速度向量
        end

        % 更新 memory（滑窗，FIFO）
        newRow = [newPose, G.actor{i}.vel];
        G.actor{i}.memory = [G.actor{i}.memory(2:end, :); newRow];
    end
end
