function tortuosity = computePathTortuosity(positions, reachedstep)
% computePathTortuosity 计算路径曲折度
%   tortuosity = 实际路径长度 / 起终点直线距离
    if reachedstep < 2
        tortuosity = 1;
        return;
    end
    pathLength = 0;
    for i = 2:reachedstep
        pathLength = pathLength + norm(positions(i,:) - positions(i-1,:));
    end
    straightLineDistance = norm(positions(reachedstep,:) - positions(1,:));
    if straightLineDistance < 1e-6
        tortuosity = Inf;
    else
        tortuosity = pathLength / straightLineDistance;
    end
end
