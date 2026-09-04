% =========================================================================
% 腳本：Run_CeilingScan_MultiHorizon.m (Workstream A: 多 Horizon 訊號天花板掃描)
% 升級：Phase 15.5 Task A 方案 (★ mrg32k3a 獨立子串流 Substream=task_id 綁定、
%       Worker 無關確定性重現、day_arr double 型別一致性、HAC 顯式貫穿防偽顯著)
% 職責：系統性掃描橫截面連續與分類訊號天花板，檢驗中長週期是否純屬自相關假象
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🚀 [Workstream A] 啟動多 Horizon 天花板掃描 (mrg32k3a 確定性子串流版)');
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

% 資料型態瘦身：double -> single / logical
Prices_Active = single(Prices_Active(valid_idx, :));
Expert_Active = logical(Expert_Active(valid_idx, :));
Dates_Active  = Dates_Active(valid_idx);
X_norm_3D     = single(X_norm_3D(valid_idx, :, :));
numDays       = length(Dates_Active);
totalFeats    = size(X_norm_3D, 2);

fprintf('  📊 面板維度：有效天數 %d 天 | 特徵數 %d 維 | 宇宙規模 %d 檔 (全單精度/邏輯矩陣)\n', ...
    numDays, totalFeats, size(Prices_Active, 2));

%% 2. 構建 24 個獨立任務清單與順序輸出隊列 (DataQueue)
horizons = [1, 3, 5, 10, 20, 40, 60, 90];
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

fprintf('  🔥 成功建立 %d 個獨立掃描任務 (動態 Embargo=H)，準備由 Worker 平行搶佔！\n', total_tasks);

clear handle_ordered_output;
dq = parallel.pool.DataQueue;
afterEach(dq, @(item) handle_ordered_output(item, total_tasks));

%% 3. 執行外層 parfor 平行掃描 (綁定 mrg32k3a Substream 確保確定性)
disp('--- 步驟 2：啟動平行天花板掃描 (獨立子串流確保跨 Worker 可重現) ---');
out_H           = zeros(total_tasks, 1);
out_Subset      = strings(total_tasks, 1);
out_AUC         = zeros(total_tasks, 1);
out_CILower     = zeros(total_tasks, 1);
out_CIUpper     = zeros(total_tasks, 1);
out_OOF_IC      = zeros(total_tasks, 1);
out_OOF_HAC_p   = zeros(total_tasks, 1);
out_HACSigCount = zeros(total_tasks, 1);
out_MaxAbsICIR  = zeros(total_tasks, 1);
out_Decision    = strings(total_tasks, 1);

rng_seed = configObj.RNG_Seed;
rng_gen  = configObj.RNG_Generator;

tic;
parfor task_i = 1:total_tasks
    t_info = task_list{task_i};
    
    % ★ 核心修復：為每個任務建立專屬 mrg32k3a 串流並綁定 task_id 作為子串流索引
    % 保證無論任務被哪個 Worker 搶佔，其內部抽樣序列完全一致
    task_stream = RandStream(rng_gen, 'Seed', rng_seed);
    task_stream.Substream = task_i;
    
    res = execute_single_task_stage2(t_info, Prices_Active, Expert_Active, X_norm_3D, total_tasks, task_stream);
    
    out_H(task_i)           = res.H;
    out_Subset(task_i)      = res.sub_name;
    out_AUC(task_i)         = res.AUC;
    out_CILower(task_i)     = res.CILower;
    out_CIUpper(task_i)     = res.CIUpper;
    out_OOF_IC(task_i)      = res.OOF_IC;
    out_OOF_HAC_p(task_i)   = res.OOF_HAC_p;
    out_HACSigCount(task_i) = res.HACSigCount;
    out_MaxAbsICIR(task_i)  = res.MaxAbsICIR;
    out_Decision(task_i)    = res.Decision;
    
    send(dq, struct('id', task_i, 'msg', res.log_msg));
end
elapsed_time = toc;
fprintf('\n⚡ 24 個天花板掃描任務全平行運算完畢！總耗時: %.2f 分鐘 (%.2f 秒)\n', elapsed_time / 60, elapsed_time);

%% 4. 組裝結構化報告與落地
results_table = table(out_H, out_Subset, out_AUC, out_CILower, out_CIUpper, out_OOF_IC, out_OOF_HAC_p, out_HACSigCount, out_MaxAbsICIR, out_Decision, ...
    'VariableNames', {'Horizon', 'Subset', 'OOF_AUC', 'CI_Lower', 'CI_Upper', 'OOF_Rank_IC', 'OOF_HAC_p', 'HAC_SigCount', 'Max_Abs_ICIR', 'Decision'});
results_table = sortrows(results_table, {'Horizon', 'Subset'}, {'ascend', 'descend'});

disp(' ');
disp('====================================================================================================================');
disp('📊 【Workstream A: 多 Horizon 訊號天花板掃描綜合報告 (Task A HAC 貫穿修復版)】');
disp('====================================================================================================================');
fprintf(' Horizon(天) | 特徵子集         | OOF AUC | 95%% CI (Day-Block) | OOF Rank IC | OOF HAC p-val | HAC顯著數 | max|ICIR| | 決策\n');
fprintf('--------------------------------------------------------------------------------------------------------------------\n');
for i = 1:height(results_table)
    fprintf(' %3d        | %-16s | %.4f  | [%.4f, %.4f]   |   %+.4f   |    %.4f     |   %2d 檔    |  %6.4f   | %s\n', ...
        results_table.Horizon(i), ...
        results_table.Subset(i), ...
        results_table.OOF_AUC(i), ...
        results_table.CI_Lower(i), ...
        results_table.CI_Upper(i), ...
        results_table.OOF_Rank_IC(i), ...
        results_table.OOF_HAC_p(i), ...
        results_table.HAC_SigCount(i), ...
        results_table.Max_Abs_ICIR(i), ...
        results_table.Decision(i));
end
fprintf('====================================================================================================================\n\n');

if ~exist(configObj.ResultDir, 'dir')
    mkdir(configObj.ResultDir);
end
reportCsvPath = fullfile(configObj.ResultDir, 'WorkstreamA_CeilingScan_Report.csv');
writetable(results_table, reportCsvPath);
fprintf('💾 掃描數據報告已匯出至: %s\n', reportCsvPath);

%% 5. 產出標準學術白底黑字視覺化圖表
disp('--- 步驟 3：產出天花板掃描視覺化圖表 (白底黑字) ---');
fig = figure('Name', 'Workstream A Multi-Horizon Ceiling Scan (Stage 2 Task A)', ...
    'Color', 'w', 'Position', [100, 100, 1200, 850], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

colors = {'#0072BD', '#D95319', '#7E2F8E'};
markers = {'o', 's', '^'};

% 子圖 1：OOF AUC 與 Day-Block 95% 信賴區間
subplot(3, 1, 1);
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
        'CapSize', 5, 'DisplayName', subsets{s_i}.name);
    hold on;
end
yline(0.50, '--k', 'Random Guess (0.50)', 'LineWidth', 1.1, 'Color', [0.3, 0.3, 0.3]);
yline(0.52, ':r', 'Signal Threshold (0.52)', 'LineWidth', 1.1);
title('OOF AUC across Horizons (Purged Embargo = H Days)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('OOF AUC', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'XTick', horizons, 'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

% 子圖 2：折外 OOF Spearman Rank IC 曲線
subplot(3, 1, 2);
for s_i = 1:numS
    sub_mask = results_table.Subset == string(subsets{s_i}.name);
    h_vals   = results_table.Horizon(sub_mask);
    ic_vals  = results_table.OOF_Rank_IC(sub_mask);
    
    plot(h_vals, ic_vals, ['-' markers{s_i}], 'Color', colors{s_i}, 'MarkerFaceColor', colors{s_i}, ...
        'LineWidth', 1.4, 'DisplayName', subsets{s_i}.name);
    hold on;
end
yline(0.00, '--k', 'Zero IC', 'LineWidth', 1.0);
yline(0.02, ':r', 'Viable Alpha Threshold (IC = 0.02)', 'LineWidth', 1.1);
title('Out-of-Fold Continuous Spearman Rank IC (Rigorous Overlap Control)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('OOF Rank IC', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'XTick', horizons, 'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

% 子圖 3：最大絕對特徵 ICIR 與 HAC 顯著比例
subplot(3, 1, 3);
sig_matrix = zeros(numH, numS);
for s_i = 1:numS
    sub_mask = results_table.Subset == string(subsets{s_i}.name);
    dim_len  = length(subsets{s_i}.idx);
    sig_matrix(:, s_i) = results_table.HAC_SigCount(sub_mask) / dim_len * 100;
end
b = bar(horizons, sig_matrix, 'grouped');
b(1).FaceColor = [0.0000, 0.4470, 0.7410];
b(2).FaceColor = [0.8500, 0.3250, 0.0980];
b(3).FaceColor = [0.4940, 0.1840, 0.5560];
for s_i = 1:numS, b(s_i).EdgeColor = 'none'; end
title('Percentage of HAC Significant Features (Lag Adjusted p < 0.05)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Prediction Horizon (Days)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Significant Feats (%)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend({'18D (Rel+Micro)', '10D (Macro)', '28D (Full)'}, 'Location', 'northeast', ...
    'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'XTick', horizons, 'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

figPath = fullfile(configObj.ResultDir, 'WorkstreamA_CeilingScan.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化報表 (白底黑字) 已儲存至: %s\n', figPath);
close(fig);

disp('=================================================================');
disp('🎯 [Workstream A] Stage 2 Task A 天花板掃描完成！');
disp('=================================================================');

%% =====================================================================
% 區域函式：循序輸出回呼函式
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
% 區域函式：單一任務獨立執行體 (注入 task_stream 實現確定性運算)
% =====================================================================
function res = execute_single_task_stage2(t_info, Prices_Active, Expert_Active, X_norm_3D, total_tasks, task_stream)
    % 將 Worker 當前全域串流切換為專屬子串流
    RandStream.setGlobalStream(task_stream);
    
    H        = t_info.H;
    sub_name = t_info.sub_name;
    sub_idx  = t_info.sub_idx;
    sub_dim  = t_info.sub_dim;
    task_id  = t_info.task_id;
    
    numDays = size(Prices_Active, 1);
    numT    = size(Prices_Active, 2);
    valid_label_days = numDays - H;
    
    % 1. 前向連續報酬與二元中位數標籤
    R_fwd_H = (Prices_Active(1+H:end, :) - Prices_Active(1:end-H, :)) ./ (Prices_Active(1:end-H, :) + 1e-8);
    R_fwd_H(isnan(R_fwd_H) | isinf(R_fwd_H)) = NaN;
    
    Y_bin_H  = false(valid_label_days, numT);
    Y_cont_H = NaN(valid_label_days, numT, 'single');
    
    for t = 1:valid_label_days
        act_m = Expert_Active(t, :);
        if sum(act_m) > 10
            r_t = R_fwd_H(t, act_m);
            med_r = median(r_t, 'omitnan');
            Y_bin_H(t, act_m) = (r_t > med_r);
            Y_cont_H(t, act_m) = (r_t - mean(r_t, 'omitnan')) ./ (std(r_t, 0, 'omitnan') + 1e-6);
        end
    end
    
    % 2. 原始特徵截面 Spearman IC 與 HAC 檢定
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
    
    auto_lag_feat = max(1, floor(4 * (valid_label_days / 100)^(2/9)));
    hac_lag_feat  = max(H, auto_lag_feat);
    p_hac_vec = ones(sub_dim, 1);
    for f = 1:sub_dim
        [~, p_hac_vec(f)] = hac_significance_test(Daily_IC(:, f), hac_lag_feat);
    end
    sig_count    = sum(p_hac_vec < 0.05);
    max_abs_icir = max(abs(icir_vec));
    clear Daily_IC R_fwd_H;
    
    % 3. 展平特徵與時序樣本 (使用 double 型別)
    active_counts = sum(Expert_Active(1:valid_label_days, :), 'all');
    X_flat      = zeros(active_counts, sub_dim, 'single');
    Y_bin_flat  = false(active_counts, 1);
    Y_cont_flat = zeros(active_counts, 1, 'single');
    day_arr     = zeros(active_counts, 1, 'double');
    
    row_cur = 1;
    for t = 1:valid_label_days
        act_ids = find(Expert_Active(t, :));
        n_t = length(act_ids);
        if n_t > 0
            x_slice = permute(X_sub_3D(t, :, act_ids), [3, 2, 1]);
            X_flat(row_cur : row_cur + n_t - 1, :)      = x_slice;
            Y_bin_flat(row_cur : row_cur + n_t - 1)     = Y_bin_H(t, act_ids)';
            Y_cont_flat(row_cur : row_cur + n_t - 1)    = Y_cont_H(t, act_ids)';
            day_arr(row_cur : row_cur + n_t - 1)        = double(t);
            row_cur = row_cur + n_t;
        end
    end
    clear X_sub_3D Y_bin_H Y_cont_H;
    
    % 顯式過濾無效樣本 (去除 NaN 與 Inf)
    valid_mask = ~isnan(Y_cont_flat) & ~isinf(Y_cont_flat) & ...
                 ~isnan(Y_bin_flat) & ...
                 all(~isnan(X_flat) & ~isinf(X_flat), 2);
             
    X_flat      = X_flat(valid_mask, :);
    Y_bin_flat  = Y_bin_flat(valid_mask);
    Y_cont_flat = Y_cont_flat(valid_mask);
    day_arr     = day_arr(valid_mask);
    active_counts = length(Y_cont_flat);
    
    % ★ 核心修復：使用 task_stream 執行 randperm，移除 rng('twister') 干擾
    if active_counts > 80000
        sample_idx  = sort(randperm(task_stream, active_counts, 80000));
        X_flat      = X_flat(sample_idx, :);
        Y_bin_flat  = Y_bin_flat(sample_idx);
        Y_cont_flat = Y_cont_flat(sample_idx);
        day_arr     = day_arr(sample_idx);
    end
    
    % 4. 執行 5-Fold Purged CV (動態 Embargo = H 天)
    u_days = unique(day_arr);
    n_days = length(u_days);
    k_folds = 5;
    fold_edges = round(linspace(1, n_days + 1, k_folds + 1));
    
    oof_prob_bin  = zeros(length(day_arr), 1, 'single');
    oof_pred_cont = zeros(length(day_arr), 1, 'single');
    eval_mask     = false(length(day_arr), 1);
    
    for k = 1:k_folds
        test_days_k = u_days(fold_edges(k) : fold_edges(k+1) - 1);
        min_test_d  = min(test_days_k);
        max_test_d  = max(test_days_k);
        
        % 訓練集前後各 Purge 隔離 H 天
        train_mask_k = (day_arr < (min_test_d - H)) | (day_arr > (max_test_d + H));
        test_mask_k  = (day_arr >= min_test_d) & (day_arr <= max_test_d);
        
        if sum(train_mask_k) > 500 && sum(test_mask_k) > 50 && numel(unique(Y_bin_flat(train_mask_k))) > 1
            mdl_cls = fitclinear(X_flat(train_mask_k, :), Y_bin_flat(train_mask_k), 'Learner', 'logistic');
            [~, score] = predict(mdl_cls, X_flat(test_mask_k, :));
            oof_prob_bin(test_mask_k) = single(score(:, 2));
            
            mdl_reg = fitrlinear(X_flat(train_mask_k, :), Y_cont_flat(train_mask_k), ...
                'Learner', 'leastsquares', 'Regularization', 'ridge');
            oof_pred_cont(test_mask_k) = single(predict(mdl_reg, X_flat(test_mask_k, :)));
            
            eval_mask(test_mask_k) = true;
        end
    end
    
    % 5. 統計指標結算
    y_true_eval = Y_bin_flat(eval_mask);
    p_pred_eval = oof_prob_bin(eval_mask);
    [~, ~, ~, oof_auc] = perfcurve(y_true_eval, p_pred_eval, 1);
    
    % ★ 核心修復：使用 task_stream 執行 Day-Block Bootstrap 重抽樣
    u_eval_days = unique(day_arr(eval_mask));
    n_eval_days = length(u_eval_days);
    boot_n = 200;
    boot_aucs = zeros(boot_n, 1);
    
    for b_i = 1:boot_n
        sample_d = randsample(task_stream, u_eval_days, n_eval_days, true);
        b_mask = ismember(day_arr(eval_mask), sample_d(1:min(50, n_eval_days)));
        if sum(y_true_eval(b_mask) == 1) > 5 && sum(y_true_eval(b_mask) == 0) > 5
            [~, ~, ~, boot_aucs(b_i)] = perfcurve(y_true_eval(b_mask), p_pred_eval(b_mask), 1);
        else
            boot_aucs(b_i) = oof_auc;
        end
    end
    ci_l = prctile(boot_aucs, 2.5);
    ci_u = prctile(boot_aucs, 97.5);
    
    % 計算逐日 OOF Spearman Rank IC 與 HAC 檢定
    eval_days_arr = day_arr(eval_mask);
    eval_pred_c   = oof_pred_cont(eval_mask);
    eval_true_c   = Y_cont_flat(eval_mask);
    
    daily_oof_ic = zeros(n_eval_days, 1);
    valid_ic_days = 0;
    for d_i = 1:n_eval_days
        d_val = u_eval_days(d_i);
        d_idx = (eval_days_arr == d_val);
        if sum(d_idx) >= 10
            valid_ic_days = valid_ic_days + 1;
            daily_oof_ic(valid_ic_days) = corr(eval_pred_c(d_idx), eval_true_c(d_idx), 'Type', 'Spearman', 'Rows', 'complete');
        end
    end
    
    daily_oof_ic = daily_oof_ic(1:valid_ic_days);
    oof_rank_ic  = mean(daily_oof_ic, 'omitnan');
    if length(daily_oof_ic) >= 20
        auto_lag_oof = max(1, floor(4 * (length(daily_oof_ic) / 100)^(2/9)));
        hac_lag_oof  = max(H, auto_lag_oof);
        [~, p_hac_oof_ic] = hac_significance_test(daily_oof_ic, hac_lag_oof);
    else
        p_hac_oof_ic = 1.0;
    end
    
    clear X_flat Y_bin_flat Y_cont_flat day_arr;
    
    % 6. Stage 2 決策判定邏輯
    if (oof_auc >= 0.5200 && ci_l > 0.5050) || (oof_rank_ic >= 0.0250 && p_hac_oof_ic < 0.05)
        decision_str = "GO (Strong Signal)";
    elseif (oof_auc >= 0.5050 && ci_l > 0.5000) || (oof_rank_ic >= 0.0150 && p_hac_oof_ic < 0.05)
        decision_str = "MARGINAL (Weak)";
    else
        decision_str = "NO-GO (Noise/Overlap)";
    end
    
    res = struct();
    res.H           = H;
    res.sub_name    = string(sub_name);
    res.AUC         = oof_auc;
    res.CILower     = ci_l;
    res.CIUpper     = ci_u;
    res.OOF_IC      = oof_rank_ic;
    res.OOF_HAC_p   = p_hac_oof_ic;
    res.HACSigCount = sig_count;
    res.MaxAbsICIR  = max_abs_icir;
    res.Decision    = decision_str;
    
    res.log_msg = sprintf('  ✅ [Worker 完成 %2d/%2d] Horizon: %2d 日 | %-16s | AUC: %.4f [%.4f, %.4f] | OOF IC: %+.4f (p=%.4f) | %s', ...
        task_id, total_tasks, H, sub_name, oof_auc, ci_l, ci_u, oof_rank_ic, p_hac_oof_ic, decision_str);
end