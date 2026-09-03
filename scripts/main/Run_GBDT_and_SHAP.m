% =========================================================================
% 腳本：3_Run_GBDT_and_SHAP.m (階段 3：顯性風險預測流與 SHAP 解釋器)
% 升級：Phase 15.5 Stage 2.5 方案 (★ mrg32k3a 獨立子串流注入、背景樣本 randsample
%       與 K-Means 質心壓縮隨機串流顯式綁定、徹底消除 SHAP 解釋圖漂移、
%       5 日連續橫截面超額報酬 Z-Score 目標、LSBoost 連續迴歸選股、
%       OOF 橫截面 Rank IC 監控、截面百分位排序分數輸出、Platt 事後機率校準)
% 職責：訓練雙軌選股 GBDT 與崩盤護欄，輸出全域無洩漏 OOF/OOS 專家排序得分與崩盤機率矩陣
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 GBDT 連續迴歸選股預測流與 SHAP 歸因分析管線 (mrg32k3a 確定性串流版)');
disp('=================================================================');

%% 0. 環境路徑掛載 (規範化階層回溯解析與路徑重新整理)
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

% ★ 核心修復 1：由 Config 統一生產 mrg32k3a 隨機數引擎，並設為全域主串流 (Substream = 1)
stream = configObj.getRandStream(1);
RandStream.setGlobalStream(stream);
disp('🔒 已成功掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定 GBDT 訓練、抽樣與 SHAP 歸因確定性。');

%% 1. 載入特徵快取與 Phase 2 萃取之 Embedding 表徵
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
horizon_stock = 5;
horizon_crash = 10;
max_horizon = max(horizon_stock, horizon_crash);
valid_idx = seqLen : (numDaysRaw - max_horizon);

%% 2. 動態特徵切片 (18D 微觀/相對特徵與 10D 宏觀總經特徵)
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

%% 3. 構建橫截面連續選股標籤與大盤崩盤標籤
disp('--- 步驟 3：構建 5 日連續橫截面超額報酬與 10 日累積崩盤護欄標籤 ---');

% 3.1 橫截面 5 日遠期連續超額報酬 Z-Score (打破二元硬切資訊天花板)
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

%% 4. 切分時間軸 (嚴格 In-Sample 訓練與 OOS 盲測推論解耦)
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

%% 5. 啟動 GBDT 連續迴歸專家訓練與 OOF 排序得分生成
disp('--- 步驟 5：實例化 GBDT 專家並執行 Expanding-Window 迴歸交叉驗證 ---');
% ★ 核心修復 2：注入 stream 實例化代理人
gbdt_agent = GBDTExpertAgent(configObj, stream);

% 執行 LSBoost 連續迴歸訓練與折外 Rank IC 監控 (傳入 stream 鎖定抽樣確定性)
[Score_time_oof_IS, Score_space_oof_IS] = gbdt_agent.train_and_predict_oof_cross_sectional(...
    E_time_IS, E_space_IS, X_18D_IS, Macro_IS, Y_Labels_IS, Expert_IS, stream);

% 訓練崩盤護欄與 Platt Scaling 事後校準 (傳入 stream 鎖定 500 次 Bootstrap 抽樣確定性)
P_crash_oof_IS = gbdt_agent.train_and_predict_oof_crash(Macro_IS, Y_Crash_IS, stream);

%% 6. 執行 OOS 盲測期真實前向推論
disp('--- 步驟 6：執行 OOS 盲測期無洩漏推論 (LSBoost 排序推論 + Platt 崩盤校準) ---');
[Score_time_oos_sub, Score_space_oos_sub, P_crash_oos_sub] = gbdt_agent.predict_oos(...
    E_time_OOS, E_space_OOS, Macro_OOS, Expert_OOS);

%% 7. 矩陣全域縫合 (組裝全歷史選股排序得分與崩盤機率面板)
disp('--- 步驟 7：組裝全歷史選股排序得分與崩盤機率矩陣 ---');
P_time_all  = zeros(numDaysRaw, numT, 'single');
P_space_all = zeros(numDaysRaw, numT, 'single');
P_crash_all = zeros(numDaysRaw, 1, 'single');

% 填入 IS 區間 (嚴格 OOF 百分位排序得分)
P_time_all(idx_IS, :)   = Score_time_oof_IS;
P_space_all(idx_IS, :)  = Score_space_oof_IS;
P_crash_all(idx_IS)     = P_crash_oof_IS;

% 填入 OOS 區間 (前向盲測百分位排序得分)
P_time_all(idx_OOS, :)  = Score_time_oos_sub;
P_space_all(idx_OOS, :) = Score_space_oos_sub;
P_crash_all(idx_OOS)    = P_crash_oos_sub;

% 非活躍標的強制作為 0 分，防止進入選股候選池
P_time_all(~Expert_Active)  = 0.0;
P_space_all(~Expert_Active) = 0.0;

%% 8. 快速抽樣 SHAP 解釋性視覺化 (白底黑字 + K-Means 串流顯式約束)
disp('--- 步驟 8：產出 SHAP 特徵邊際貢獻度視覺化 (白底黑字 - 確定性無漂移版) ---');
try
    sample_active_idx = find(Expert_IS(end, :), 1, 'first');
    
    if ~isempty(sample_active_idx)
        e_sample_t = permute(E_time_IS(end, :, sample_active_idx), [3, 2, 1]);
        mac_sample = Macro_IS(end, :);
        x_query = [e_sample_t, mac_sample];
        
        % ★ 核心修復 3：背景樣本抽樣嚴格使用 stream 進行確定性抽樣
        bg_samples = min(200, length(idx_IS));
        bg_t = randsample(stream, length(idx_IS), bg_samples);
        x_bg = zeros(bg_samples, size(x_query, 2), 'single');
        for b = 1:bg_samples
            act_cand = find(Expert_IS(bg_t(b), :), 1, 'first');
            if isempty(act_cand), act_cand = 1; end
            e_bg = permute(E_time_IS(bg_t(b), :, act_cand), [3, 2, 1]);
            x_bg(b, :) = [e_bg, Macro_IS(bg_t(b), :)];
        end
        
        % ★ 核心修復 4：顯式傳入 stream 約束 K-Means 質心初始化與 SHAP 蒙地卡羅抽樣
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

%% 9. 儲存 GBDT 模型與全域得分矩陣
disp('--- 步驟 9：儲存 GBDT 模型實體與全域選股得分矩陣 ---');
savePath = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');
save(savePath, 'gbdt_agent', 'P_time_all', 'P_space_all', 'P_crash_all', '-v7.3');
fprintf('💾 顯性風險與連續排序得分已安全落地至: %s\n', savePath);

disp('=================================================================');
disp('🎯 [Phase 3] 連續迴歸選股流與 SHAP 歸因執行完成！');
disp('=================================================================');