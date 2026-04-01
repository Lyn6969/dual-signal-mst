function drawRobotsTraj(G)
% drawRobotsTraj 绘制所有个体运动轨迹
%
% 从 G.actor{i}.memory 中提取历史位置并绘制

    figure('Name', '个体运动轨迹', 'Position', [100, 100, 800, 800]);
    hold on;

    cmap = lines(G.maxID);

    for i = G.robotsList
        % 从 memory 重建完整轨迹（如果可用）
        % 由于 memory 是滑窗的，这里使用 posHistory
        if isfield(G.actor{i}, 'posHistory')
            traj = G.actor{i}.posHistory;
            valid = ~isnan(traj(:,1));
            plot(traj(valid, 1), traj(valid, 2), '-', 'Color', [cmap(i,:), 0.3], 'LineWidth', 0.5);
        end
    end

    % 绘制当前位置和速度方向
    for i = G.robotsList
        pos = G.actor{i}.pose;
        vel = G.actor{i}.vel;
        % 绘制个体位置
        if G.actor{i}.is_activated
            plot(pos(1), pos(2), 'r^', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
        else
            plot(pos(1), pos(2), 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b');
        end
        % 绘制速度方向箭头
        quiver(pos(1), pos(2), vel(1)*30, vel(2)*30, 0, 'k', 'LineWidth', 0.5);
    end

    % 绘制目标位置
    if isfield(G, 'target')
        plot(G.target(1), G.target(2), 'p', 'MarkerSize', 15, ...
            'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.2, 0.8, 0.2]);
    end

    % 绘制集群中心轨迹
    if isfield(G, 'swarmCenterPositions') && G.reachedStep > 1
        centerTraj = G.swarmCenterPositions(1:G.reachedStep, :);
        plot(centerTraj(:,1), centerTraj(:,2), 'k-', 'LineWidth', 2);
        plot(centerTraj(end,1), centerTraj(end,2), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    end

    axis equal;
    xlabel('X (mm)');
    ylabel('Y (mm)');
    title(sprintf('集群运动轨迹 (步数: %d)', G.reachedStep));
    grid on;
    hold off;
end
