function G = initSimRobots(G, rand_mode, pos_offset, r)
% initSimRobots 初始化虚拟个体（数字孪生/平行仿真）
%
% 输入:
%   G         - 全局结构体
%   rand_mode - 'rand' 或 'unif'
%   pos_offset - 位置偏移 [x, y]（mm）
%   r          - 初始区域半径（mm）
%
% 使用 G.actor{i} cell 数组结构，兼容 swarmBang 平台

    robotsNum = G.maxID;
    G.robotsList = 1:robotsNum;
    G.hawkID = -1; % 无捕食者

    switch rand_mode
        case 'rand'
            positions = r * rand(robotsNum, 2) - r / 2;
            velocities = rand(robotsNum, 2) - 0.5;
        case 'unif'
            [rp, rv] = randpose_local(robotsNum);
            positions = r * rp' - pos_offset;
            velocities = rv';
        otherwise
            error('不支持的 rand_mode: %s，使用 "rand" 或 "unif"', rand_mode);
    end

    % 单位化初始速度
    for i = 1:robotsNum
        vn = norm(velocities(i,:));
        if vn > 0
            velocities(i,:) = velocities(i,:) / vn;
        else
            ang = rand * 2 * pi;
            velocities(i,:) = [cos(ang), sin(ang)];
        end
    end

    % 初始化 G.actor cell 数组
    G.actor = cell(1, robotsNum);
    for i = 1:robotsNum
        agent = struct();
        agent.id = i;
        agent.pose = positions(i, :);
        agent.vel = velocities(i, :);
        agent.is_activated = false;
        agent.src_id = NaN;
        agent.desiredTurnAngle = zeros(G.maxSimSteps, 1);
        agent.desiredSpeed = zeros(G.maxSimSteps, 1);
        % memory: 每行 [x, y, vx, vy]，用于 s_ij 计算
        agent.memory = repmat([positions(i,:), velocities(i,:)], G.n_steps + 1, 1);
        G.actor{i} = agent;
    end
end

%% ---- 内嵌的 randpose 函数 ----
function [pos, vel] = randpose_local(Nc)
% 使用力模型生成分散的初始位姿
    t_max = 10; dt = 0.1;
    la = 3; lc = 10;
    max_force = 5;
    v_0 = 0; relaxTime = 0.1;
    noiseStr = 0.5;

    pos = rand(2, Nc) * 2;
    vel = ones(2, Nc);

    for t = 1:t_max
        speed = sqrt(sum(vel.^2, 1));
        speed(speed == 0) = 1;
        u_auto = ((v_0 - speed) / relaxTime) .* (vel ./ speed);

        pos_t = pos';
        dij = pdist2(pos_t, pos_t, 'euclidean');
        dij(1:Nc+1:end) = Inf;

        nij_x = (pos(1,:) - pos(1,:)') ./ dij;
        nij_y = (pos(2,:) - pos(2,:)') ./ dij;
        ra = (1 - (la ./ dij).^2) .* exp(-dij / lc);
        ra(dij == Inf) = 0;

        fx = ra .* nij_x;
        fy = ra .* nij_y;
        u_grad = [sum(fx, 2)'; sum(fy, 2)'];

        u_sum = u_auto + u_grad;
        force = sqrt(sum(u_sum.^2, 1));
        force(force == 0) = 1;
        ut = min(max_force, force) .* (u_sum ./ force);
        u = ut + (rand(2, Nc) * 2 - 1) .* noiseStr;

        vel = vel + u * dt;
        pos = pos + vel * dt;
    end
end
