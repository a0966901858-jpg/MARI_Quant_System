% =========================================================================
% 腳本：3_Run_GBDT_Guards_Pipeline.m
% 架構：Phase 15.5 生產基準整合版 (Feature & Label ETL + GBDT Guards + SHAP)
% 整合：原 3_Run_Feature_Label_Generation.m 與 3_Run_GBDT_Guards_Training.m
% 亮點：
%   1. 【記憶體直通 ＋ 快取雙軌】：消除重複讀寫瓶頸，同時自動落盤 GBDT_Dataset.mat
%   2. 【純時序剪枝相容】：完整支援 Pure-Time 剪枝，自動跳過無效空間樹訓練
%   3. 【SSOT 參數嚴格對齊】：全域繼承 Config.Horizon / PurgeEmbargo / HACLag
%   4. 【多體制防洩漏切片】：嚴格 IS (2006-2021) 訓練 vs OOS (2022-2026) 盲測解耦
%   5. 【LSBoost 橫截面迴歸】：對齊 Phase 2 Soft-IC 目標，監控折外 OOF Rank IC
%   6. 【Platt 崩盤事後校準】：生成具備真實機率含義之極端下行風控機率
%   7. 【確定性串流鎖定】：注入 mrg32k3a 獨立隨機數串流，保證結果 100% 可重現
% =========================================================================
clear; clc; close all;

% 啟動全域總計時器
t_total_start = tic;
disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 特徵標籤工程 ＋ GBDT 守衛訓練整合管線 (All-in-One)');
disp('=================================================================');

%% 0. 環境路徑掛載與隨機串流鎖定
t_step0 = tic;
disp('--- 步驟 0：環境路徑掛載、Config 載入與隨機串流鎖定 ---');
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
        error('❌ 找不到 MARI_Quant_System 專案根目錄 (包含 configs/)，請確認執行路徑！');
    end
    projectRoot = parentDir;
end

addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'agents')));
addpath(genpath(fullfile(projectRoot, 'models')));
addpath(genpath(fullfile(projectRoot, 'envs')));
addpath(genpath(fullfile(projectRoot, 'utils')));
rehash path;
rehash;

if exist('Config', 'class') ~= 8
    error('❌ 已掛載路徑但仍找不到 Config 類別，請檢查 configs/Config.m 權限或語法錯誤！');
end
configObj = Config();

% 統一生產 mrg32k3a 隨機數引擎，並設為全域主串流 (Substream = 1)
stream = configObj.getRandStream(1);
RandStream.setGlobalStream(stream);
disp('🔒 已成功掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定 GBDT 訓練、抽樣與 SHAP 確定性。');
time_step0 = toc(t_step0);
fprintf('⏱️ [步驟 0 完成] 耗時: %.2f 秒\n\n', time_step0);

%% 1. 載入前置快取 (Phase 1 降噪特徵與 Phase 2 DL 最佳快照表徵)
t_step1 = tic;
disp('--- 步驟 1：載入全域降噪特徵與 Phase 2 最佳快照 DL 表徵 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
modelPath = fullfile(configObj.ModelDir, 'DL_Extractors.mat');

if ~exist(cachePath, 'file') || ~exist(modelPath, 'file')
    error('❌ 找不到前置快取，請先確認 Phase 1 與 Phase 2 腳本均已成功執行！');
end

load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = ''; 
load(modelPath, 'E_time_all', 'E_space_all');

numDaysRaw = length(Dates_Active);
numT = configObj.NumTickers;
seqLen = configObj.SeqLen;

% ★ 動態繼承 Config.m 全域 Horizon (單一事實來源 SSOT)
if isprop(configObj, 'Horizon') && ~isempty(configObj.Horizon)
    horizon_stock = configObj.Horizon;
else
    horizon_stock = 20;
end
horizon_crash = 10;  % 崩盤護欄固定為 10 日短期極端回撤
max_horizon = max(horizon_stock, horizon_crash);
valid_idx = seqLen : (numDaysRaw - max_horizon);

% ★ 空間專家狀態探針：檢查是否啟用空間專家且具備非零特徵
enable_space = isprop(configObj, 'EnableSpaceExpertTraining') && configObj.EnableSpaceExpertTraining && any(E_space_all(:) ~= 0);
if enable_space
    disp('  -> 空間專家狀態: 【已啟用】(DyGAT 雙軌特徵共同參與橫截面選股)');
else
    disp('  -> 空間專家狀態: 【⏩ 已剪枝】(Pure-Time 純時序架構，杜絕圖卷積過度平滑噪聲)');
end

time_step1 = toc(t_step1);
fprintf('⏱️ [步驟 1 完成] 耗時: %.2f 秒\n\n', time_step1);

%% 2. 特徵切片、通道解構與總經品質審計 (原 Generation 模組核心)
t_step2 = tic;
disp('--- 步驟 2：特徵切片、通道解構與總經資料品質審計 ---');
numRel   = 3;
numMicro = configObj.NumMicroFeatures; % 預設 15
numMacro = configObj.NumMacroFeatures; % 預設 10

idx_raw18 = 1 : (numRel + numMicro);
idx_macro = (numRel + numMicro + 1) : (numRel + numMicro + numMacro);

X_norm_18D = X_norm_3D(:, idx_raw18, :); 
Macro_2D   = X_norm_3D(:, idx_macro, 1); 

% ★ 總經特徵即時審計：檢驗高收益債信用利差 (第 8 欄位 HY Spread)
hy_spread_raw = Macro_2D(:, 8);
zero_ratio = sum(hy_spread_raw == 0) / numDaysRaw;
fprintf('  -> 特徵維度配置: 個股微觀 %d 維 | 宏觀總經 %d 維 | DL 表徵 %d 維\n', ...
    length(idx_raw18), length(idx_macro), size(E_time_all, 2));
fprintf('  -> [總經品質審計] 信用利差特徵零值佔比: %.2f%% (標準差: %.4f)\n', zero_ratio * 100, std(hy_spread_raw, 'omitnan'));

if zero_ratio > 0.30
    warning('⚠️ 信用利差存在顯著常數或缺失回填區間！Fold 5 崩盤護欄 AUC 增益需謹慎歸因。');
end
time_step2 = toc(t_step2);
fprintf('⏱️ [步驟 2 完成] 耗時: %.2f 秒\n\n', time_step2);

%% 3. 構建連續超額選股標籤與崩盤護欄標籤 (原 Generation 模組核心)
t_step3 = tic;
fprintf('--- 步驟 3：構建 %d 日連續橫截面超額報酬 Z-Score 與 10 日崩盤標籤 ---\n', horizon_stock);

% 3.1 橫截面遠期連續超額報酬 Z-Score (與 Phase 2 預訓練目標函數嚴格對齊)
R_fwd = NaN(numDaysRaw, numT, 'single');
R_fwd(1:end-horizon_stock, :) = (Prices_Active(1+horizon_stock:end, :) - Prices_Active(1:end-horizon_stock, :)) ...
                                ./ (Prices_Active(1:end-horizon_stock, :) + 1e-8);
R_fwd(isnan(R_fwd) | isinf(R_fwd)) = NaN;

Y_Labels_3D = zeros(numDaysRaw, numT, 'single');
for t = 1 : (numDaysRaw - horizon_stock)
    active_mask = Expert_Active(t, :) & ~isnan(R_fwd(t, :)) & ~isinf(R_fwd(t, :));
    if sum(active_mask) >= 10
        r_t = R_fwd(t, active_mask);
        mu_t  = mean(r_t, 'omitnan');
        std_t = std(r_t, 0, 'omitnan') + 1e-6;
        Y_Labels_3D(t, active_mask) = (r_t - mu_t) ./ std_t;
    end
end
Y_Labels_3D(isnan(Y_Labels_3D) | isinf(Y_Labels_3D)) = 0;

% 3.2 崩盤護欄標籤 (SPY 未來 10 日累積回撤 > 5%)
spy_idx = find(strcmp(configObj.IdxTickers, 'SPY'));
if isempty(spy_idx), spy_idx = 1; end
spy_prices = Prices_Active(:, spy_idx);
fwd_cum_ret = NaN(numDaysRaw, 1, 'single');
for t = 1 : (numDaysRaw - horizon_crash)
    fwd_cum_ret(t) = (spy_prices(t+horizon_crash) - spy_prices(t)) / (spy_prices(t) + 1e-8);
end
Y_Crash_1D = single(fwd_cum_ret < -0.05);
Y_Crash_1D(isnan(Y_Crash_1D)) = 0;

% ★ 智慧快取落地：將生成之標準特徵與標籤陣列保存至 CacheDir，供單獨特徵分析使用
gbdtDatasetPath = fullfile(configObj.CacheDir, 'GBDT_Dataset.mat');
save(gbdtDatasetPath, 'X_norm_18D', 'Macro_2D', 'E_time_all', 'E_space_all', ...
    'Y_Labels_3D', 'Y_Crash_1D', 'Expert_Active', 'valid_idx', '-v7.3');
fprintf('💾 [資料集落盤] GBDT 訓練特徵與標籤資料集已成功快取至: %s\n', gbdtDatasetPath);

time_step3 = toc(t_step3);
fprintf('⏱️ [步驟 3 完成] 耗時: %.2f 秒\n\n', time_step3);

%% 4. 時間軸嚴格解耦切分 (In-Sample 訓練 vs OOS 盲測推論)
t_step4 = tic;
disp('--- 步驟 4：切分樣本時間軸 (Train IS 2006-2021 vs. Blind OOS 2022-2026) ---');
Train_Start_Date = datetime('2006-01-01');
OOS_Start_Date   = datetime('2022-01-01');

idx_IS_raw  = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);
idx_OOS_raw = find(Dates_Active >= OOS_Start_Date);

idx_IS  = intersect(valid_idx, idx_IS_raw);
idx_OOS = intersect(valid_idx, idx_OOS_raw);

% 提取 In-Sample 切片
E_time_IS   = E_time_all(idx_IS, :, :);
E_space_IS  = E_space_all(idx_IS, :, :);
X_18D_IS    = X_norm_18D(idx_IS, :, :);
Macro_IS    = Macro_2D(idx_IS, :);
Y_Labels_IS = Y_Labels_3D(idx_IS, :);
Expert_IS   = Expert_Active(idx_IS, :);
Y_Crash_IS  = Y_Crash_1D(idx_IS);

% 提取 Out-of-Sample 切片
E_time_OOS  = E_time_all(idx_OOS, :, :);
E_space_OOS = E_space_all(idx_OOS, :, :);
Macro_OOS   = Macro_2D(idx_OOS, :);
Expert_OOS  = Expert_Active(idx_OOS, :);

fprintf('✅ 時間軸劃分完畢！IS 訓練區間: %d 天 | OOS 盲測推論區間: %d 天\n', ...
    length(idx_IS), length(idx_OOS));
time_step4 = toc(t_step4);
fprintf('⏱️ [步驟 4 完成] 耗時: %.2f 秒\n\n', time_step4);

%% 5. 啟動 GBDT 連續迴歸專家訓練與 Expanding-Window OOF 交叉驗證
t_step5 = tic;
disp('--- 步驟 5：實例化 GBDT 專家並執行 Expanding-Window 迴歸交叉驗證 ---');
disp(' 💡 [時序交叉驗證機制]：Fold 1 作為初始訓練基準，折外驗證自 Fold 2 依序展開。');

gbdt_agent = GBDTExpertAgent(configObj, stream);

% 優先繼承 Config.m 之交叉驗證隔離島天數與 HAC 滯後階數
if isprop(configObj, 'PurgeEmbargo') && ~isempty(configObj.PurgeEmbargo)
    embargo_days = configObj.PurgeEmbargo;
else
    embargo_days = horizon_stock;
end

if isprop(configObj, 'HACLag') && ~isempty(configObj.HACLag)
    hac_lag = configObj.HACLag;
else
    hac_lag = horizon_stock;
end

if isprop(gbdt_agent, 'TargetHorizon'), gbdt_agent.TargetHorizon = horizon_stock; end
if isprop(gbdt_agent, 'EmbargoDays'),   gbdt_agent.EmbargoDays   = embargo_days; end
if isprop(gbdt_agent, 'HACLag'),        gbdt_agent.HACLag        = hac_lag; end

% 執行 LSBoost 連續迴歸訓練與折外 Rank IC 監控 (內部自動辨識空間剪枝)
[Score_time_oof_IS, Score_space_oof_IS] = gbdt_agent.train_and_predict_oof_cross_sectional(...
    E_time_IS, E_space_IS, X_18D_IS, Macro_IS, Y_Labels_IS, Expert_IS, stream);

% 訓練崩盤護欄與 Platt Scaling 事後校準
P_crash_oof_IS = gbdt_agent.train_and_predict_oof_crash(Macro_IS, Y_Crash_IS, stream);

time_step5 = toc(t_step5);
fprintf('⏱️ [步驟 5 完成] 耗時: %.2f 秒 (%.2f 分鐘)\n\n', time_step5, time_step5 / 60);

%% 6. 執行 OOS 盲測期真實前向推論
t_step6 = tic;
disp('--- 步驟 6：執行 OOS 盲測期無洩漏推論 (LSBoost 排序推論 + Platt 崩盤校準) ---');
[Score_time_oos_sub, Score_space_oos_sub, P_crash_oos_sub] = gbdt_agent.predict_oos(...
    E_time_OOS, E_space_OOS, Macro_OOS, Expert_OOS);
time_step6 = toc(t_step6);
fprintf('⏱️ [步驟 6 完成] 耗時: %.2f 秒\n\n', time_step6);

%% 7. 矩陣全域縫合 (組裝全歷史選股排序得分與崩盤機率面板)
t_step7 = tic;
disp('--- 步驟 7：組裝全歷史選股排序得分與崩盤機率矩陣 ---');
P_time_all  = zeros(numDaysRaw, numT, 'single');
P_space_all = zeros(numDaysRaw, numT, 'single');
P_crash_all = zeros(numDaysRaw, 1, 'single');

% 填入 IS 區間 (嚴格 OOF 百分位排序得分)
P_time_all(idx_IS, :)   = Score_time_oof_IS;
P_space_all(idx_IS, :)  = Score_space_oof_IS;
P_crash_all(idx_IS)     = P_crash_oof_IS;

% 填入 OOS 區間 (無未來函數盲測推論分數)
P_time_all(idx_OOS, :)  = Score_time_oos_sub;
P_space_all(idx_OOS, :) = Score_space_oos_sub;
P_crash_all(idx_OOS)    = P_crash_oos_sub;

% 非活躍標的強制作為 0 分，防止進入選股候選池
P_time_all(~Expert_Active)  = 0.0;
P_space_all(~Expert_Active) = 0.0;

time_step7 = toc(t_step7);
fprintf('⏱️ [步驟 7 完成] 耗時: %.2f 秒\n\n', time_step7);

%% 8. 抽樣 SHAP 解釋性分析 (白底黑字 + 伺服器防禦)
t_step8 = tic;
disp('--- 步驟 8：產出 SHAP 特徵邊際貢獻度視覺化 (確定性無漂移版) ---');
try
    sample_active_idx = find(Expert_IS(end, :), 1, 'first');
    
    if ~isempty(sample_active_idx)
        e_sample_t = permute(E_time_IS(end, :, sample_active_idx), [3, 2, 1]);
        mac_sample = Macro_IS(end, :);
        x_query = [e_sample_t, mac_sample];
        
        bg_samples = min(200, length(idx_IS));
        bg_t = randsample(stream, length(idx_IS), bg_samples);
        x_bg = zeros(bg_samples, size(x_query, 2), 'single');
        for b = 1:bg_samples
            act_cand = find(Expert_IS(bg_t(b), :), 1, 'first');
            if isempty(act_cand), act_cand = 1; end
            e_bg = permute(E_time_IS(bg_t(b), :, act_cand), [3, 2, 1]);
            x_bg(b, :) = [e_bg, Macro_IS(bg_t(b), :)];
        end
        
        RandStream.setGlobalStream(stream);
        gbdt_agent.explain_shapley(x_query, x_bg, 'time', stream);
        
        fig_shap = gcf;
        set(fig_shap, 'Visible', 'off', 'Color', 'w', 'InvertHardcopy', 'off');
        
        all_axes = findall(fig_shap, 'type', 'axes');
        for ax_i = 1:length(all_axes)
            ax = all_axes(ax_i);
            set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
                'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, ...
                'FontName', 'Helvetica', 'FontSize', 10);
            grid(ax, 'on'); 
            box(ax, 'on');
            
            if isprop(ax, 'Title') && ~isempty(ax.Title)
                set(ax.Title, 'Color', 'k', 'FontWeight', 'bold', 'FontSize', 11);
            end
            if isprop(ax, 'XLabel') && ~isempty(ax.XLabel)
                set(ax.XLabel, 'Color', 'k', 'FontWeight', 'bold');
            end
            if isprop(ax, 'YLabel') && ~isempty(ax.YLabel)
                set(ax.YLabel, 'Color', 'k', 'FontWeight', 'bold');
            end
        end
        
        shapFigPath = fullfile(configObj.ModelDir, 'Phase3_SHAP_TimeExpert.png');
        exportgraphics(fig_shap, shapFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
        close(fig_shap);
        fprintf(' 📊 SHAP 特徵歸因圖 (白底黑字) 已儲存至: %s\n', shapFigPath);
    end
catch ME
    warning('⚠️ SHAP 繪圖跳過 (非致命): %s', ME.message);
end
time_step8 = toc(t_step8);
fprintf('⏱️ [步驟 8 完成] 耗時: %.2f 秒\n\n', time_step8);

%% 9. 儲存 GBDT 模型實體與全域得分矩陣
t_step9 = tic;
disp('--- 步驟 9：儲存 GBDT 模型實體與全域選股得分矩陣 ---');
savePath = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');
save(savePath, 'gbdt_agent', 'P_time_all', 'P_space_all', 'P_crash_all', '-v7.3');
fprintf('💾 顯性風險與連續排序得分已安全落地至: %s\n', savePath);
time_step9 = toc(t_step9);
fprintf('⏱️ [步驟 9 完成] 耗時: %.2f 秒\n\n', time_step9);

%% =========================================================================
% 結算全流程執行時長審計報告
% =========================================================================
total_elapsed_sec = toc(t_total_start);
tot_hours = floor(total_elapsed_sec / 3600);
tot_mins  = floor(mod(total_elapsed_sec, 3600) / 60);
tot_secs  = mod(total_elapsed_sec, 60);

fprintf('=================================================================\n');
fprintf('📊 【Phase 3 各階段耗時明細與總時長審計報告】\n');
fprintf('=================================================================\n');
fprintf(' 步驟 0：環境掛載與隨機串流鎖定   : %8.2f 秒 (%5.1f%%)\n', time_step0, (time_step0 / total_elapsed_sec) * 100);
fprintf(' 步驟 1：降噪特徵與 DL 表徵載入   : %8.2f 秒 (%5.1f%%)\n', time_step1, (time_step1 / total_elapsed_sec) * 100);
fprintf(' 步驟 2：動態特徵解構與總經品質健檢: %8.2f 秒 (%5.1f%%)\n', time_step2, (time_step2 / total_elapsed_sec) * 100);
fprintf(' 步驟 3：選股標籤構建與資料集快取 : %8.2f 秒 (%5.1f%%)\n', time_step3, (time_step3 / total_elapsed_sec) * 100);
fprintf(' 步驟 4：時間軸解耦切分 (IS vs OOS): %8.2f 秒 (%5.1f%%)\n', time_step4, (time_step4 / total_elapsed_sec) * 100);
fprintf(' 步驟 5：GBDT 交叉驗證與護欄校準  : %8.2f 秒 (%5.1f%%)\n', time_step5, (time_step5 / total_elapsed_sec) * 100);
fprintf(' 步驟 6：OOS 盲測期真實前向推論   : %8.2f 秒 (%5.1f%%)\n', time_step6, (time_step6 / total_elapsed_sec) * 100);
fprintf(' 步驟 7：全歷史得分與崩盤矩陣縫合 : %8.2f 秒 (%5.1f%%)\n', time_step7, (time_step7 / total_elapsed_sec) * 100);
fprintf(' 步驟 8：SHAP 特徵貢獻歸因分析    : %8.2f 秒 (%5.1f%%)\n', time_step8, (time_step8 / total_elapsed_sec) * 100);
fprintf(' 步驟 9：GBDT 模型與全域得分矩陣存檔: %8.2f 秒 (%5.1f%%)\n', time_step9, (time_step9 / total_elapsed_sec) * 100);
fprintf('-----------------------------------------------------------------\n');
if tot_hours > 0
    fprintf('⏱️ 【總執行時長】: %d 小時 %d 分 %.2f 秒 (共 %.2f 秒)\n', tot_hours, tot_mins, tot_secs, total_elapsed_sec);
else
    fprintf('⏱️ 【總執行時長】: %d 分 %.2f 秒 (共 %.2f 秒)\n', tot_mins, tot_secs, total_elapsed_sec);
end
fprintf('=================================================================\n');
disp('🎯 [Phase 3] 特徵標籤生成、GBDT 訓練與 SHAP 歸因一鍵執行完畢！請推進至 Phase 4。');
disp('=================================================================');