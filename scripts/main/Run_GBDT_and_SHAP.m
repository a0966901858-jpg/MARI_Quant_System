% =========================================================================
% 腳本：3_Run_GBDT_and_SHAP.m (階段 3：顯性風險預測流與 SHAP 解釋器)
% 升級：Phase 15.5 參數驅動版 (★ 自動繼承 Config.m 之 Horizon/PurgeEmbargo/HACLag、
%       嚴格對齊 Phase 2 預訓練特徵表徵、mrg32k3a 獨立子串流注入、
%       背景樣本 randsample 與 K-Means 質心壓縮隨機串流顯式綁定、
%       LSBoost 連續迴歸選股、OOF 橫截面 Rank IC 監控、Platt 事後機率校準、
%       各階段獨立高精度計時與總運行耗時審計)
% 職責：訓練雙軌選股 GBDT 與崩盤護欄，輸出全域無洩漏 OOF/OOS 專家排序得分與崩盤機率矩陣
% =========================================================================
clear; clc; close all;

% 啟動全域總計時器
t_total_start = tic;

disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 GBDT 連續迴歸選股預測流與 SHAP 歸因分析管線 (Config 參數動態對齊版)');
disp('=================================================================');

%% 0. 環境路徑掛載 (規範化階層回溯解析與路徑重新整理)
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

% 由 Config 統一生產 mrg32k3a 隨機數引擎，並設為全域主串流 (Substream = 1)
stream = configObj.getRandStream(1);
RandStream.setGlobalStream(stream);
disp('🔒 已成功掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定 GBDT 訓練、抽樣與 SHAP 歸因確定性。');

time_step0 = toc(t_step0);
fprintf('⏱️ [步驟 0 完成] 耗時: %.2f 秒\n\n', time_step0);

%% 1. 載入特徵快取與 Phase 2 萃取之 Embedding 表徵
t_step1 = tic;
disp('--- 步驟 1：載入全域降噪特徵與 DL 雙軌表徵 ---');
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

% ★ 核心修復 1：優先繼承 Config.m 全域 Horizon，徹底消除硬編碼
if isprop(configObj, 'Horizon') && ~isempty(configObj.Horizon)
    horizon_stock = configObj.Horizon;
else
    horizon_stock = 60;
end

horizon_crash = 10;  % 崩盤護欄維持 10 日短期極端回撤
max_horizon = max(horizon_stock, horizon_crash);
valid_idx = seqLen : (numDaysRaw - max_horizon);

time_step1 = toc(t_step1);
fprintf('⏱️ [步驟 1 完成] 耗時: %.2f 秒\n\n', time_step1);

%% 2. 動態特徵切片 (18D 微觀/相對特徵與 10D 宏觀總經特徵)
t_step2 = tic;
disp('--- 步驟 2：執行特徵矩陣動態通道解構 (防範硬編碼索引缺失) ---');
numRel   = 3;
numMicro = configObj.NumMicroFeatures; % 預設 15
numMacro = configObj.NumMacroFeatures; % 預設 10

idx_raw18 = 1 : (numRel + numMicro);
idx_macro = (numRel + numMicro + 1) : (numRel + numMicro + numMacro);

X_norm_18D = X_norm_3D(:, idx_raw18, :); 
Macro_2D   = X_norm_3D(:, idx_macro, 1); 

fprintf('  -> 原始特徵維度: %d 維 (Rel %d + Micro %d) | 宏觀特徵維度: %d 維\n', ...
    length(idx_raw18), numRel, numMicro, length(idx_macro));

time_step2 = toc(t_step2);
fprintf('⏱️ [步驟 2 完成] 耗時: %.2f 秒\n\n', time_step2);

%% 3. 構建橫截面連續選股標籤與大盤崩盤標籤
t_step3 = tic;
% ★ 核心修復 2：終端提示動態反映當前 Horizon
fprintf('--- 步驟 3：構建 %d 日連續橫截面超額報酬與 10 日累積崩盤護欄標籤 ---\n', horizon_stock);

% 3.1 橫截面遠期連續超額報酬 Z-Score (與 Phase 2 預訓練目標函數嚴格同構)
R_fwd = NaN(numDaysRaw, numT, 'single');
R_fwd(1:end-horizon_stock, :) = (Prices_Active(1+horizon_stock:end, :) - Prices_Active(1:end-horizon_stock, :)) ...
                                ./ (Prices_Active(1:end-horizon_stock, :) + 1e-8);
R_fwd(isnan(R_fwd) | isinf(R_fwd)) = NaN;

Y_Labels_3D = zeros(numDaysRaw, numT, 'single');
for t = 1:numDaysRaw-horizon_stock
    active_mask = Expert_Active(t, :) & ~isnan(R_fwd(t, :)) & ~isinf(R_fwd(t, :));
    if sum(active_mask) >= 10
        r_t = R_fwd(t, active_mask);
        mu_t  = mean(r_t, 'omitnan');
        std_t = std(r_t, 0, 'omitnan') + 1e-6;
        Y_Labels_3D(t, active_mask) = (r_t - mu_t) ./ std_t;
    end
end
Y_Labels_3D(isnan(Y_Labels_3D) | isinf(Y_Labels_3D)) = 0;

% 3.2 崩盤護欄標籤 (未來 10 日累積回撤 > 5%)
spy_idx = find(strcmp(configObj.IdxTickers, 'SPY'));
if isempty(spy_idx), spy_idx = 1; end
spy_prices = Prices_Active(:, spy_idx);

fwd_cum_ret = NaN(numDaysRaw, 1, 'single');
for t = 1:numDaysRaw-horizon_crash
    fwd_cum_ret(t) = (spy_prices(t+horizon_crash) - spy_prices(t)) / (spy_prices(t) + 1e-8);
end
Y_Crash_1D = single(fwd_cum_ret < -0.05);
Y_Crash_1D(isnan(Y_Crash_1D)) = 0;

time_step3 = toc(t_step3);
fprintf('⏱️ [步驟 3 完成] 耗時: %.2f 秒\n\n', time_step3);

%% 4. 切分時間軸 (嚴格 In-Sample 訓練與 OOS 盲測推論解耦)
t_step4 = tic;
disp('--- 步驟 4：切分樣本時間軸 (Train IS vs. Blind OOS) ---');
Train_Start_Date = datetime('2006-01-01');
OOS_Start_Date   = datetime('2022-01-01');

idx_IS_raw  = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);
idx_OOS_raw = find(Dates_Active >= OOS_Start_Date);

idx_IS  = intersect(valid_idx, idx_IS_raw);
idx_OOS = intersect(valid_idx, idx_OOS_raw);

E_time_IS   = E_time_all(idx_IS, :, :);
E_space_IS  = E_space_all(idx_IS, :, :);
X_18D_IS    = X_norm_18D(idx_IS, :, :);
Macro_IS    = Macro_2D(idx_IS, :);
Y_Labels_IS = Y_Labels_3D(idx_IS, :);
Expert_IS   = Expert_Active(idx_IS, :);
Y_Crash_IS  = Y_Crash_1D(idx_IS);

E_time_OOS  = E_time_all(idx_OOS, :, :);
E_space_OOS = E_space_all(idx_OOS, :, :);
Macro_OOS   = Macro_2D(idx_OOS, :);
Expert_OOS  = Expert_Active(idx_OOS, :);

fprintf('✅ 時間軸劃分完畢！IS 訓練區間: %d 天 | OOS 盲測推論區間: %d 天\n', ...
    length(idx_IS), length(idx_OOS));

time_step4 = toc(t_step4);
fprintf('⏱️ [步驟 4 完成] 耗時: %.2f 秒\n\n', time_step4);

%% 5. 啟動 GBDT 連續迴歸專家訓練與 OOF 排序得分生成
t_step5 = tic;
disp('--- 步驟 5：實例化 GBDT 專家並執行 Expanding-Window 迴歸交叉驗證 ---');
gbdt_agent = GBDTExpertAgent(configObj, stream);

% ★ 核心修復 3：優先讀取 Config.m 的時序交叉驗證參數
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

% 執行 LSBoost 連續迴歸訓練與折外 Rank IC 監控 (傳入 stream 鎖定抽樣確定性)
[Score_time_oof_IS, Score_space_oof_IS] = gbdt_agent.train_and_predict_oof_cross_sectional(...
    E_time_IS, E_space_IS, X_18D_IS, Macro_IS, Y_Labels_IS, Expert_IS, stream);

% 訓練崩盤護欄與 Platt Scaling 事後校準 (傳入 stream 鎖定 500 次 Bootstrap 抽樣確定性)
P_crash_oof_IS = gbdt_agent.train_and_predict_oof_crash(Macro_IS, Y_Crash_IS, stream);

time_step5 = toc(t_step5);
fprintf('⏱️ [步驟 5 完成] 耗時: %.2f 秒 (%.2f 分鐘)\n\n', time_step5, time_step5 / 60);

%% 6. 執行 OOS 盲測期真實前向推論
t_step6 = tic;
disp('--- 步驟 6：執行 OOS 盲測期無洩漏推論 (LSBoost 排序推論 + Platt 崩盤校準) ---');
% ★ 核心修復 4：變數規範命名為 _oos_sub，消除 _oof_ 筆誤
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

% 填入 OOS 區間 (前向盲測百分位排序得分，精確對齊步驟 6 的 oos_sub 變數)
P_time_all(idx_OOS, :)  = Score_time_oos_sub;
P_space_all(idx_OOS, :) = Score_space_oos_sub;
P_crash_all(idx_OOS)    = P_crash_oos_sub;

% 非活躍標的強制作為 0 分，防止進入選股候選池
P_time_all(~Expert_Active)  = 0.0;
P_space_all(~Expert_Active) = 0.0;

time_step7 = toc(t_step7);
fprintf('⏱️ [步驟 7 完成] 耗時: %.2f 秒\n\n', time_step7);

%% 8. 快速抽樣 SHAP 解釋性視覺化 (白底黑字 + K-Means 串流顯式約束)
t_step8 = tic;
disp('--- 步驟 8：產出 SHAP 特徵邊際貢獻度視覺化 (白底黑字 - 確定性無漂移版) ---');
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
        set(fig_shap, 'Color', 'w', 'InvertHardcopy', 'off');
        
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

%% 9. 儲存 GBDT 模型與全域得分矩陣
t_step9 = tic;
disp('--- 步驟 9：儲存 GBDT 模型實體與全域選股得分矩陣 ---');
savePath = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');
save(savePath, 'gbdt_agent', 'P_time_all', 'P_space_all', 'P_crash_all', '-v7.3');
fprintf('💾 顯性風險與連續排序得分已安全落地至: %s\n', savePath);

time_step9 = toc(t_step9);
fprintf('⏱️ [步驟 9 完成] 耗時: %.2f 秒\n\n', time_step9);

%% =========================================================================
% 結算全流程執行時長審計
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
fprintf(' 步驟 2：動態特徵切片 (18D + 10D) : %8.2f 秒 (%5.1f%%)\n', time_step2, (time_step2 / total_elapsed_sec) * 100);
fprintf(' 步驟 3：連續選股與崩盤護欄標籤構建: %8.2f 秒 (%5.1f%%)\n', time_step3, (time_step3 / total_elapsed_sec) * 100);
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
disp('🎯 [Phase 3] 連續迴歸選股流與 SHAP 歸因執行完成！');
disp('=================================================================');
