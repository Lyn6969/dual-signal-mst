function drawRobotsMotion_dynamic(G)
% drawRobotsMotion_dynamic 动态显示机器人运动状态
%
% 在仿真循环中调用，实时更新个体位置、方向和激活状态

    persistent hFig

    if isempty(hFig) || ~isvalid(hFig)
        hFig = figure('Name', '实时运动状态', 'Position', [100, 100, 800, 800]);
    end

    figure(hFig);
    cla;
    hold on;

    arrowLen = 30; % 箭头长度 (mm)

    for i = G.robotsList
        pos = G.actor{i}.pose;
        vel = G.actor{i}.vel;
        if G.actor{i}.is_activated
            plot(pos(1), pos(2), 'r^', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
        else
            plot(pos(1), pos(2), 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b');
        end
        quiver(pos(1), pos(2), vel(1)*arrowLen, vel(2)*arrowLen, 0, 'k', 'LineWidth', 0.5);
    end

    % 目标
    if isfield(G, 'target')
        plot(G.target(1), G.target(2), 'p', 'MarkerSize', 15, ...
            'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.2, 0.8, 0.2]);
    end

    % 集群中心
    center = getSwarmCenter(G);
    plot(center(1), center(2), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'y');

    axis equal;
    xlabel('X (mm)');
    ylabel('Y (mm)');
    title(sprintf('步数: %d | 激活: %d/%d', G.simStep, ...
        G.activatedCount(G.simStep), G.maxID));
    grid on;
    hold off;
    drawnow;
end
