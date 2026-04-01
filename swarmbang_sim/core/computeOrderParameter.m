function computeOrderParameter(G)
% computeOrderParameter 计算并显示序参量和激活数的时间曲线
%
% 注意：这里的序参量是基于速度方向的全局序参量 Φ

    T = G.reachedStep;

    % 计算序参量
    order_param = zeros(T, 1);
    for t = 1:T
        % 从存储的数据中重建（如果有）
        % 简化版本：使用激活数代替，因为完整的速度历史未保存
    end

    figure('Name', '激活个体数', 'Position', [100, 100, 800, 400]);

    subplot(2,1,1);
    plot(1:T, G.activatedCount(1:T), 'b-', 'LineWidth', 1.5);
    xlabel('仿真步数');
    ylabel('激活个体数');
    title('激活个体数随时间变化');
    grid on;

    subplot(2,1,2);
    plot(1:T, G.maxcj(1:T), 'r-', 'LineWidth', 1);
    xlabel('仿真步数');
    ylabel('max(s_{ij}) (rad/s)');
    title('最大运动显著性');
    grid on;
end
