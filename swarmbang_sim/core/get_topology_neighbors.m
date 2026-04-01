function [neighbors, num_neighbors, neighbors_id_list] = get_topology_neighbors(G, focal_agent_id)
% get_topology_neighbors 度量邻居选择
%
% 选择距离 < G.Dsen 内的所有邻居
%
% 输入:
%   G              - 全局结构体（需包含 G.Dsen 感知半径，单位 mm）
%   focal_agent_id - 焦点个体 ID
%
% 输出:
%   neighbors         - cell 数组，每个元素是一个 agent struct
%   num_neighbors     - 邻居数量
%   neighbors_id_list - 邻居 ID 列表

    filteredAgents = {};
    distances = [];
    count = 0;

    focal_pos = G.actor{focal_agent_id}.pose;

    for i = G.robotsList
        if G.actor{i}.id ~= focal_agent_id && G.actor{i}.id ~= G.hawkID
            d = norm(G.actor{i}.pose - focal_pos);
            if d < G.Dsen
                count = count + 1;
                filteredAgents{count} = G.actor{i};
            end
        end
    end

    neighbors = filteredAgents;
    num_neighbors = count;
    neighbors_id_list = zeros(1, num_neighbors);
    for i = 1:num_neighbors
        neighbors_id_list(i) = neighbors{i}.id;
    end
end
