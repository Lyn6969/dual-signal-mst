function saveResultBundle(out_dir, base_name, variable_names)
% saveResultBundle 同时导出 MAT 和 JSON 结果文件
%
% 输入:
%   out_dir        - 输出目录
%   base_name      - 文件基名（不带扩展名）
%   variable_names - 变量名 cell 数组，将从调用者工作区读取

    if ~(iscell(variable_names) && all(cellfun(@ischar, variable_names)))
        error('variable_names 必须是字符向量 cell 数组');
    end

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    payload = struct();
    for i = 1:numel(variable_names)
        var_name = variable_names{i};
        payload.(var_name) = evalin('caller', var_name);
    end

    mat_path = fullfile(out_dir, [base_name, '.mat']);
    save(mat_path, '-struct', 'payload');

    json_path = fullfile(out_dir, [base_name, '.json']);
    writeJsonFile(json_path, payload);
end

function writeJsonFile(json_path, payload)
    if ~(exist('jsonencode', 'builtin') || exist('jsonencode', 'file'))
        error('当前 MATLAB 版本不支持 jsonencode，无法导出 JSON: %s', json_path);
    end

    try
        json_text = jsonencode(payload, 'PrettyPrint', true);
    catch
        json_text = jsonencode(payload);
    end

    fid = fopen(json_path, 'w');
    if fid == -1
        error('无法写入 JSON 文件: %s', json_path);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s', json_text);
end
