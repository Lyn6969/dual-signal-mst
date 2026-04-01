function [desiredTurnAngle, desiredSpeed, G] = swarmAlg_fixed(G)
% swarmAlg_fixed 固定阈值集群算法
%
% 使用固定的 cj_threshold (rad/s) 进行激活判断
% 与 RAS 论文中的 swarmAlg_lyn_fp 逻辑一致，但去除了 weight_cj
%
% 输出:
%   desiredTurnAngle - [maxID x 1] 期望转角 (deg)
%   desiredSpeed     - [maxID x 1] 期望速度 (mm/s)
%   G                - 更新后的全局结构体

    activated_this_step = [];
    activated_src_this_step = [];

    %% 个体自驱动方向
    i_vel = nan(G.maxID, 2);
    for i = G.robotsList
        i_vel(i, :) = unitvel_local(G.actor{i}.vel);
    end

    %% 速度协同
    dir_vel = nan(G.maxID, 2);
    for i = G.robotsList
        focal_agent = G.actor{i};
        [neighbors, ~, neighbors_id_list] = get_topology_neighbors(G, focal_agent.id);
        if ~isempty(neighbors)
            if focal_agent.is_activated
                src_id = focal_agent.src_id;
                src_agent = G.actor{src_id};
                if vecnorm(src_agent.vel - focal_agent.vel) < G.deac_threshold
                    % 取消激活
                    G.actor{i}.is_activated = false;
                    G.actor{i}.src_id = NaN;
                    temp_vel = [0, 0];
                    for j = neighbors_id_list
                        temp_vel = temp_vel + unitvel_local(G.actor{j}.vel);
                    end
                    dir_vel(i, :) = unitvel_local(temp_vel);
                else
                    % 保持激活，跟随源头
                    dir_vel(i, :) = unitvel_local(src_agent.vel);
                end
            else
                cj = get_candidate_neighbors(focal_agent.id, neighbors, G);
                if ~isempty(cj) && max(cj) > G.cj_threshold
                    % 激活
                    G.actor{i}.is_activated = true;
                    G.maxcj(G.simStep) = max(cj);
                    [~, max_idx] = max(cj);
                    src_agent = neighbors{max_idx};
                    G.actor{i}.src_id = src_agent.id;
                    dir_vel(i, :) = unitvel_local(src_agent.vel);
                else
                    % 不激活，平均速度
                    temp_vel = [0, 0];
                    for j = neighbors_id_list
                        temp_vel = temp_vel + unitvel_local(G.actor{j}.vel);
                    end
                    dir_vel(i, :) = unitvel_local(temp_vel);
                end
            end
        end
        if G.actor{i}.is_activated
            activated_this_step(end+1) = i;
            activated_src_this_step(end+1) = G.actor{i}.src_id;
        end
    end

    % 保存激活信息
    G.activatedIDs{G.simStep} = activated_this_step;
    G.activatedCount(G.simStep) = numel(activated_this_step);
    G.activatedSrcIDs{G.simStep} = activated_src_this_step;

    %% 位置协同
    dir_pos = nan(G.maxID, 2);
    for i = G.robotsList
        neig_pos = setdiff(G.robotsList, i);
        if ~isempty(neig_pos)
            temp_pos = [0, 0];
            for j = neig_pos
                rij = G.actor{j}.pose - G.actor{i}.pose;
                dij = norm(rij);
                if dij < 1e-6; continue; end
                nij = rij / dij;
                if dij <= G.Drep
                    ra = G.weight_rep * (dij / G.Drep - 1);
                else
                    ra = G.weight_att * (1 - (G.Dsen - dij) / (G.Dsen - G.Drep));
                end
                temp_pos = temp_pos + ra * nij;
            end
            dir_pos(i, :) = unitvel_local(temp_pos);
        end
    end

    %% 目标吸引力
    dir_target = nan(G.maxID, 2);
    for i = G.robotsList
        dir_target(i, :) = unitvel_local(G.target - G.actor{i}.pose);
    end

    %% 信息个体旋转触发
    if mod(G.simStep, G.rotateStep) == 0 && G.simStep < G.maxSimSteps
        infoIdx = G.simStep / G.rotateStep;
        if infoIdx >= 1 && infoIdx <= length(G.myinfoIDs)
            G.now_infoID = G.myinfoIDs(infoIdx);
            G.rotateCycle(G.now_infoID) = G.rotateTime;
        end
    end

    %% 行为综合
    temp_socialDir = dir_pos + G.weight_align * dir_vel + G.weight_target(G.simStep) * dir_target;
    desDir = zeros(G.maxID, 2);
    for i = G.robotsList
        v = temp_socialDir(i, :);
        vn = norm(v);
        if vn > 0
            desDir(i, :) = v / vn;
        end
    end

    %% 转换为期望转角和速度
    desTurnAngle = zeros(G.maxID, 1);
    desSpeed = zeros(G.maxID, 1);
    for i = G.robotsList
        if G.rotateCycle(i) > 0
            desTurnAngle(i) = 360 / G.rotateTime;
            desSpeed(i) = 10;
            G.rotateCycle(i) = G.rotateCycle(i) - 1;
        else
            if G.simStep <= 50
                if all(G.target == G.targetA)
                    desDir(i, :) = [-1, -1] / sqrt(2);
                else
                    desDir(i, :) = [1, 1] / sqrt(2);
                end
            end
            iDir = unitvel_local(G.actor{i}.vel);
            desTurnAngle(i) = angleOfVectors_local(iDir', desDir(i,:)');
            desSpeed(i) = G.v0;
            G.actor{i}.desiredTurnAngle(G.simStep) = desTurnAngle(i);
            G.actor{i}.desiredSpeed(G.simStep) = desSpeed(i);
        end
    end

    desiredTurnAngle = desTurnAngle;
    desiredSpeed = desSpeed;
end

%% 内嵌工具函数
function uv = unitvel_local(v)
    vn = norm(v);
    if vn > 0
        uv = v / vn;
    else
        uv = [0, 0];
    end
end

function angle = angleOfVectors_local(v1, v2)
% 计算 v1 到 v2 的有符号角度 (deg)，右转为正
    angle = atan2d(v1(1)*v2(2) - v1(2)*v2(1), v1(1)*v2(1) + v1(2)*v2(2));
end
