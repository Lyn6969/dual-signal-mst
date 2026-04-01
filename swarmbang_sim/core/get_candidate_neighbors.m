function cj = get_candidate_neighbors(focal_agent_id, neighbors, G)
% get_candidate_neighbors 计算邻居的运动显著性 s_ij
%
% 运动显著性定义为相对位置矢量的角变化率（rad/s）:
%   s_ij = Δα_ij / Δt
%   其中 Δt = n_steps × cycTime 为观测时间窗口
%
% 注意：已去除 weight_cj 缩放因子，直接输出物理量 (rad/s)
%
% 输入:
%   focal_agent_id - 焦点个体 ID
%   neighbors      - cell 数组（来自 get_topology_neighbors）
%   G              - 全局结构体
%
% 输出:
%   cj - [num_neighbors x 1]，每个邻居的运动显著性 (rad/s)

    num_neis = numel(neighbors);
    cj = zeros(num_neis, 1);

    focal_agent = G.actor{focal_agent_id};
    my_pos = focal_agent.pose;

    % 检查是否有足够的历史数据
    if ~isfield(focal_agent, 'memory')
        return;
    end

    % 观测时间窗口
    n = G.n_steps;
    delta_t = G.cycTime * n;

    for j = 1:num_neis
        nei = neighbors{j};
        nei_pos = nei.pose;

        % 当前相对位置方向
        current_diff = nei_pos - my_pos;
        current_norm = norm(current_diff);
        if current_norm < 1e-6
            continue;
        end
        current_diff = current_diff / current_norm;

        % 过去的相对位置方向（n 步前）
        history_length = size(nei.memory, 1);
        if history_length >= n + 1
            my_past_pos = focal_agent.memory(end - n, 1:2);
            nei_past_pos = nei.memory(end - n, 1:2);
        else
            my_past_pos = focal_agent.memory(1, 1:2);
            nei_past_pos = nei.memory(1, 1:2);
        end

        past_diff = nei_past_pos - my_past_pos;
        past_norm = norm(past_diff);
        if past_norm < 1e-6
            continue;
        end
        past_diff = past_diff / past_norm;

        % 角变化（弧度）
        angle_cos = max(min(dot(past_diff, current_diff), 1), -1);
        delta_alpha = acos(angle_cos);

        % 运动显著性 = 角变化率 (rad/s)
        cj(j) = delta_alpha / delta_t;
    end
end
