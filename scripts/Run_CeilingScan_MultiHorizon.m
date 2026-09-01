% =========================================================================
% 腳本：Run_CeilingScan_MultiHorizon.m (Workstream A: 多 Horizon 訊號天花板掃描)
% 升級：Phase 15.5 診斷方案 (★ 記憶體極致瘦身 + DataQueue 順序即時輸出 + Task Slicing)
% 職責：系統性掃描橫截面 Beat-the-Median 訊號天花板，定位微觀 18D 與宏觀 10D 資訊含量
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🚀 [Workstream A] 啟動多 Horizon 天花板掃描 (記憶體優化 + 循序輸出版)');
disp('=================================================================');

%% 0. 環境路徑掛載與平行池管理
currentFile = mfilename('fullpath');
if isempty(currentFile)
    currentPath = pwd;
else
    currentPath = fileparts(currentFile);
end

projectRoot = currentPath;
while ~exist(fullfile(projectRoot, 'configs'), 'dir')
    parentDir = fileparts(projectRoot);
    if strcmp(parentDir, projectRoot)
        error('❌ 找不到專案根目錄 (包含 configs/)！');
    end
    projectRoot = parentDir;
end

addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'models')));
addpath(genpath(fullfile(projectRoot, 'agents')));
addpath(genpath(fullfile(projectRoot, 'utils')));
rehash toolboxcache;

configObj = Config();

% 啟動 6 個專屬 Worker (適配 32GB RAM)
targetWorkers = 6;
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp('⚡ 啟動最佳化平行池 (6 Workers)...');
    parpool('Processes', targetWorkers);
elseif poolobj.NumWorkers ~= targetWorkers
    delete(poolobj);
    parpool('Processes', targetWorkers);
end

%% 1. 載入特徵快取與極致資料型態轉換 (Memory Downcasting)
disp('--- 步驟 1：載入全域 3D 特徵面板並執行記憶體型態壓縮 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
if ~exist(cachePath, 'file')
    error('❌ 找不到特徵快取檔案，請先執行 1_Run_Data_and_Features.m！');
end

load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC';

seqLen = configObj.SeqLen;
numDaysRaw = length(Dates_Active);
valid_idx = seqLen : numDaysRaw;

% ★ 資料型態瘦身：double -> single / logical
Prices_Active = single(Prices_Active(valid_idx, :));
Expert_Active = logical(Expert_Active(valid_idx, :));
Dates_Active  = Dates_Active(valid_idx);
X_norm_3D     = single(X_norm_3D(valid_idx, :, :));
numDays       = length(Dates_Active);
numT          = configObj.NumTickers;
totalFeats    = size(X_norm_3D, 2);

fprintf('  📊 面板維度：有效天數 %d 天 | 特徵數 %d 維 | 宇宙規模 %d 檔 (全單精度/邏輯矩陣)\n', ...
    numDays, totalFeats, numT);

%% 2. 構建 24 個獨立任務清單與順序輸出隊列 (DataQueue)
horizons = [1, 3, 5, 10, 15, 20, 30, 60];
subsets = { ...
    struct('name', '18D (Rel+Micro)', 'idx', 1:18), ...
    struct('name', '10D (Macro)',     'idx', 19:totalFeats), ...
    struct('name', '28D (Full)',      'idx', 1:totalFeats) ...
};

numH = length(horizons);
numS = length(subsets);
total_tasks = numH * numS;

task_list = cell(total_tasks, 1);
t_idx = 1;
for h_i = 1:numH
    for s_i = 1:numS
        task_list{t_idx} = struct(...
            'task_id',  t_idx, ...
            'H',        horizons(h_i), ...
            'sub_name', subsets{s_i}.name, ...
            'sub_idx',  subsets{s_i}.idx, ...
            'sub_dim',  length(subsets{s_i}.idx) ...
        );
        t_idx = t_idx + 1;
    end
end

fprintf('  🔥 成功建立 %d 個獨立掃描任務，準備由 6 個 Worker 平行搶佔執行！\n', total_tasks);

% ★ 綁定順序輸出回呼函式
clear handle_ordered_output;
dq = parallel.pool.DataQueue;
afterEach(dq, @(item) handle_ordered_output(item, total_tasks));

%% 3. 執行外層 parfor 平行掃描 (隔離函式防記憶體洩漏)
disp('--- 步驟 2：啟動 6 執行緒外層 parfor 狂飆運算 (按順序即時顯示) ---');

out_H            = zeros(total_tasks, 1);
out_Subset       = strings(total_tasks, 1);
out_AUC          = zeros(total_tasks, 1);
out_CILower      = zeros(total_tasks, 1);
out_CIUpper      = zeros(total_tasks, 1);
out_PosRatio     = zeros(total_tasks, 1);
out_HACSigCount  = zeros(total_tasks, 1);
out_MaxAbsICIR   = zeros(total_tasks, 1);
out_Decision     = strings(total_tasks, 1);

tic;

parfor task_i = 1:total_tasks
    t_info = task_list{task_i};
    
    % ★ 呼叫隔離任務函式：函式執行完畢後內部暫存陣列會立即被回收釋放
    res = execute_single_task(t_info, Prices_Active, Expert_Active, X_norm_3D, configObj, total_tasks);
    
    % 賦值回平行切片變數
    out_H(task_i)           = res.H;
    out_Subset(task_i)      = res.sub_name;
    out_AUC(task_i)         = res.AUC;
    out_CILower(task_i)     = res.CILower;
    out_CIUpper(task_i)     = res.CIUpper;
    out_PosRatio(task_i)    = res.PosRatio;
    out_HACSigCount(task_i) = res.HACSigCount;
    out_MaxAbsICIR(task_i)  = res.MaxAbsICIR;
    out_Decision(task_i)    = res.Decision;
    
    % 發送至主執行緒 DataQueue 佇列
    send(dq, struct('id', task_i, 'msg', res.log_msg));
end

elapsed_time = toc;
fprintf('\n⚡ 24 個天花板掃描任務全平行運算完畢！總耗時: %.2f 分鐘 (%.2f 秒)\n', elapsed_time / 60, elapsed_time);

%% 4. 組裝結構化報告與落地
results_table = table(out_H, out_Subset, out_AUC, out_CILower, out_CIUpper, out_PosRatio, out_HACSigCount, out_MaxAbsICIR, out_Decision, ...
    'VariableNames', {'Horizon', 'Subset', 'OOF_AUC', 'CI_Lower', 'CI_Upper', 'PosRatio', 'HAC_SigCount', 'Max_Abs_ICIR', 'Decision'});

results_table = sortrows(results_table, {'Horizon', 'Subset'}, {'ascend', 'descend'});

disp(' ');
disp('========================================================================================');
disp('📊 【Workstream A: 多 Horizon 訊號天花板掃描綜合報告 (完整排序版)】');
disp('========================================================================================');
fprintf(' Horizon(天) | 特徵子集         | OOF AUC | 95%% CI (Day-Block) | 正類比例 | HAC顯著數 | max|ICIR| | 決策\n');
fprintf('----------------------------------------------------------------------------------------------------\n');
for i = 1:height(results_table)
    fprintf(' %3d        | %-16s | %.4f  | [%.4f, %.4f]   |  %5.2f%%  |   %2d 檔    |  %6.4f   | %s\n', ...
        results_table.Horizon(i), ...
        results_table.Subset(i), ...
        results_table.OOF_AUC(i), ...
        results_table.CI_Lower(i), ...
        results_table.CI_Upper(i), ...
        results_table.PosRatio(i) * 100, ...
        results_table.HAC_SigCount(i), ...
        results_table.Max_Abs_ICIR(i), ...
        results_table.Decision(i));
end
fprintf('========================================================================================\n\n');

if ~exist(configObj.ResultDir, 'dir')
    mkdir(configObj.ResultDir);
end

reportCsvPath = fullfile(configObj.ResultDir, 'WorkstreamA_CeilingScan_Report.csv');
writetable(results_table, reportCsvPath);
fprintf('💾 掃描數據報告已匯出至: %s\n', reportCsvPath);

%% 5. 產出標準學術白底黑字視覺化圖表
disp('--- 步驟 3：產出天花板掃描視覺化圖表 (白底黑字) ---');

fig = figure('Name', 'Workstream A Multi-Horizon Ceiling Scan', ...
    'Color', 'w', 'Position', [100, 100, 1200, 800], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

% 子圖 1：OOF AUC 與 Day-Block 95% 信賴區間
subplot(3, 1, 1);
colors = {'#0072BD', '#D95319', '#7E2F8E'};
markers = {'o', 's', '^'};

for s_i = 1:numS
    sub_mask = results_table.Subset == string(subsets{s_i}.name);
    h_vals   = results_table.Horizon(sub_mask);
    auc_vals = results_table.OOF_AUC(sub_mask);
    ci_l     = results_table.CI_Lower(sub_mask);
    ci_u     = results_table.CI_Upper(sub_mask);
    
    err_neg = auc_vals - ci_l;
    err_pos = ci_u - auc_vals;
    
    errorbar(h_vals, auc_vals, err_neg, err_pos, ['-' markers{s_i}], ...
        'Color', colors{s_i}, 'MarkerFaceColor', colors{s_i}, 'LineWidth', 1.4, ...
        'CapSize', 6, 'DisplayName', subsets{s_i}.name);
    hold on;
end

yline(0.50, '--k', 'Random Guess (0.50)', 'LineWidth', 1.2, 'Color', [0.3, 0.3, 0.3]);
yline(0.52, ':r', 'Signal Threshold (0.52)', 'LineWidth', 1.2);
title('GBDT OOF AUC across Horizons with Day-Level Block Bootstrap 95% CI', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Prediction Horizon (Days)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('OOF AUC', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'XTick', horizons, 'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

% 子圖 2：最大絕對 ICIR 變化趨勢
subplot(3, 1, 2);
for s_i = 1:numS
    sub_mask = results_table.Subset == string(subsets{s_i}.name);
    h_vals   = results_table.Horizon(sub_mask);
    icir_max = results_table.Max_Abs_ICIR(sub_mask);
    
    plot(h_vals, icir_max, '-o', 'Color', colors{s_i}, 'MarkerFaceColor', colors{s_i}, ...
        'LineWidth', 1.4, 'DisplayName', subsets{s_i}.name);
    hold on;
end

title('Maximum Absolute Feature ICIR across Horizons', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Prediction Horizon (Days)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Max |ICIR|', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

% 子圖 3：HAC 顯著特徵數量比例 (%)
subplot(3, 1, 3);
sig_matrix = zeros(numH, numS);
for s_i = 1:numS
    sub_mask = results_table.Subset == string(subsets{s_i}.name);
    sub_dim_val = length(subsets{s_i}.idx);
    sig_matrix(:, s_i) = results_table.HAC_SigCount(sub_mask) / sub_dim_val * 100;
end

b = bar(horizons, sig_matrix, 'grouped');
b(1).FaceColor = [0.0000, 0.4470, 0.7410];
b(2).FaceColor = [0.8500, 0.3250, 0.0980];
b(3).FaceColor = [0.4940, 0.1840, 0.5560];
for s_i = 1:numS
    b(s_i).EdgeColor = 'none';
end

title('Percentage of Statistically Significant Features (Newey-West HAC p < 0.05)', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Prediction Horizon (Days)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Significant Features (%)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend({'18D (Rel+Micro)', '10D (Macro)', '28D (Full)'}, 'Location', 'northeast', ...
    'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'XTick', horizons, 'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

figPath = fullfile(configObj.ResultDir, 'WorkstreamA_CeilingScan.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化報表 (白底黑字) 已儲存至: %s\n', figPath);
close(fig);

disp('=================================================================');
disp('🎯 [Workstream A] 多 Horizon 天花板掃描完成！請進行 Round 6 驗證。');
disp('=================================================================');

%% =====================================================================
% 區域函式：循序輸出回呼函式 (使用 Persistent State 解決 parfor 非同步問題)
% =====================================================================
function handle_ordered_output(item, total_tasks)
    persistent buffer next_to_print;
    if isempty(buffer)
        buffer = cell(total_tasks, 1);
        next_to_print = 1;
    end
    
    buffer{item.id} = item.msg;
    while next_to_print <= total_tasks && ~isempty(buffer{next_to_print})
        fprintf('%s\n', buffer{next_to_print});
        buffer{next_to_print} = [];
        next_to_print = next_to_print + 1;
    end
    
    if next_to_print > total_tasks
        clear buffer next_to_print;
    end
end

%% =====================================================================
% 區域函式：單一任務獨立執行體 (記憶體完全隔離與自動回收)
% =====================================================================
function res = execute_single_task(t_info, Prices_Active, Expert_Active, X_norm_3D, configObj, total_tasks)
    H        = t_info.H;
    sub_name = t_info.sub_name;
    sub_idx  = t_info.sub_idx;
    sub_dim  = t_info.sub_dim;
    task_id  = t_info.task_id;
    
    numDays = size(Prices_Active, 1);
    numT    = size(Prices_Active, 2);
    valid_label_days = numDays - H;
    
    % 1. 向量化前向報酬與 Beat-the-Median 標籤
    R_fwd_H = (Prices_Active(1+H:end, :) - Prices_Active(1:end-H, :)) ./ (Prices_Active(1:end-H, :) + 1e-8);
    R_fwd_H(isnan(R_fwd_H) | isinf(R_fwd_H)) = NaN;
    
    Y_Labels_H = false(valid_label_days, numT);
    for t = 1:valid_label_days
        act_m = Expert_Active(t, :);
        if sum(act_m) > 10
            med_r = median(R_fwd_H(t, act_m), 'omitnan');
            Y_Labels_H(t, act_m) = (R_fwd_H(t, act_m) > med_r);
        end
    end
    
    % 2. 特徵切片與極速橫截面 Spearman IC 計算
    X_sub_3D = X_norm_3D(1:valid_label_days, sub_idx, :);
    Daily_IC = zeros(valid_label_days, sub_dim, 'single');
    
    for t = 1:valid_label_days
        act_m = Expert_Active(t, :);
        if sum(act_m) >= 20
            ret_t = R_fwd_H(t, act_m)';
            for f = 1:sub_dim
                feat_t = squeeze(X_sub_3D(t, f, act_m));
                Daily_IC(t, f) = corr(feat_t, ret_t, 'Type', 'Spearman', 'Rows', 'complete');
            end
        end
    end
    
    icir_vec = mean(Daily_IC, 1, 'omitnan') ./ (std(Daily_IC, 0, 1, 'omitnan') + 1e-8);
    
    p_hac_vec = ones(sub_dim, 1);
    for f = 1:sub_dim
        [~, p_hac_vec(f)] = hac_significance_test(Daily_IC(:, f));
    end
    sig_count = sum(p_hac_vec < 0.05);
    max_abs_icir = max(abs(icir_vec));
    
    clear Daily_IC R_fwd_H;
    
    % 3. 展平特徵矩陣
    active_counts = sum(Expert_Active(1:valid_label_days, :), 'all');
    X_flat  = zeros(active_counts, sub_dim, 'single');
    Y_flat  = false(active_counts, 1);
    day_arr = zeros(active_counts, 1, 'uint16');
    
    row_cur = 1;
    for t = 1:valid_label_days
        act_ids = find(Expert_Active(t, :));
        n_t = length(act_ids);
        if n_t > 0
            x_slice = permute(X_sub_3D(t, :, act_ids), [3, 2, 1]);
            X_flat(row_cur : row_cur + n_t - 1, :) = x_slice;
            Y_flat(row_cur : row_cur + n_t - 1)    = Y_Labels_H(t, act_ids)';
            day_arr(row_cur : row_cur + n_t - 1)   = uint16(t);
            row_cur = row_cur + n_t;
        end
    end
    
    clear X_sub_3D Y_Labels_H;
    
    % 4. 呼叫輕量化 RawBaselineTrainer 進行 5-Fold Purged CV 與 Bootstrap
    local_trainer = RawBaselineTrainer(configObj);
    metrics = local_trainer.train_and_eval(X_flat, Y_flat, day_arr, 1000);
    
    clear X_flat Y_flat day_arr;
    
    % 5. 決策門檻判定
    if metrics.AUC_Point >= 0.5200 && metrics.CI_Lower > 0.5050
        decision_str = "GO (Strong Signal)";
    elseif metrics.AUC_Point >= 0.5050 && metrics.CI_Lower > 0.5000
        decision_str = "MARGINAL (Weak)";
    else
        decision_str = "NO-GO (Noise)";
    end
    
    % 6. 組裝回傳結構體與日誌字串
    res = struct();
    res.H           = H;
    res.sub_name    = string(sub_name);
    res.AUC         = metrics.AUC_Point;
    res.CILower     = metrics.CI_Lower;
    res.CIUpper     = metrics.CI_Upper;
    res.PosRatio    = metrics.PosClassRatio;
    res.HACSigCount = sig_count;
    res.MaxAbsICIR  = max_abs_icir;
    res.Decision    = decision_str;
    
    res.log_msg = sprintf('  ✅ [Worker 完成 %2d/%2d] Horizon: %2d 日 | %-16s | AUC: %.4f [%.4f, %.4f] | HAC: %2d/%2d | %s', ...
        task_id, total_tasks, H, sub_name, metrics.AUC_Point, metrics.CI_Lower, metrics.CI_Upper, sig_count, sub_dim, decision_str);
end