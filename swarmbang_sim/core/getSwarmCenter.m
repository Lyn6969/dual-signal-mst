function center = getSwarmCenter(G)
% getSwarmCenter 计算集群中心位置
    positions = zeros(length(G.robotsList), 2);
    for i = 1:length(G.robotsList)
        positions(i, :) = G.actor{G.robotsList(i)}.pose;
    end
    center = mean(positions, 1);
end
