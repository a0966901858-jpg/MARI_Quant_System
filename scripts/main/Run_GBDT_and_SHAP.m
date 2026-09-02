% =========================================================================
% 腳本：3_Run_GBDT_and_SHAP.m (階段 3：顯性風險預測流與 SHAP 解釋器)
% 升級：Phase 14.25 (★ 10日累積崩盤標籤、動態10D宏觀特徵切片、原始18D Baseline對齊、Platt校準)
% 職責：訓練雙軌選股 GBDT 與崩盤護欄，輸出全域無洩漏 OOF/OOS 專家機率矩陣
% =========================================================================
clear; clc; close all;
disp('=================================================================');
disp('🚀 [Phase 14.25] 啟動 GBDT 顯性風險預測流與 SHAP 歸因分析管線');
disp('=================================================================');

%% 0. 環境路徑掛載 (規範化階層回溯解析與路徑重新整理)
currentFile = mfilename('fullpath');
if isempty(currentFile)
    currentPath = pwd;
else
    currentPath = fileparts(currentFile);
end

% 循環向上搜尋包含 configs/ 的專案根目錄
projectRoot = currentPath;
while ~exist(fullfile(projectRoot, 'configs'), 'dir')
    parentDir = fileparts(projectRoot);
    if strcmp(parentDir, projectRoot)
        error('❌ 找不到 MARI_Quant_System 專案根目錄 (包含 configs/)，請確認執行路徑！');
    end
    projectRoot = parentDir;
end

% 掛載核心模組目錄
addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'agents')));
addpath(genpath(fullfile(projectRoot, 'models')));
addpath(genpath(fullfile(projectRoot, 'envs')));
addpath(genpath(fullfile(projectRoot, 'utils')));

% 強制刷新 MATLAB 類別與路徑快取
rehash path;
rehash;

% 斷言驗證 Config 類別是否已正確加載
if exist('Config', 'class') ~= 8
    error('❌ 已掛載路徑但仍找不到 Config 類別，請檢查 configs/Config.m 權限或語法錯誤！');
end

configObj = Config();

if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

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

% ★ Phase 14.25 修正：定義預測窗口 (選股 5 日、崩盤護欄 10 日)
horizon_stock = 5;
horizon_crash = 10;
max_horizon = max(horizon_stock, horizon_crash);

valid_idx = seqLen : (numDaysRaw - max_horizon);
num_valid = length(valid_idx);

%% 2. 動態特徵切片 (★ Phase 14.25 核心修復：動態切分 18D 技術指標與 10D 宏觀特徵)
disp('--- 步驟 2：執行特徵矩陣動態通道解構 (防範硬編碼索引缺失) ---');
numRel   = 3;
numMicro = configObj.NumMicroFeatures; % 預設 15
numMacro = configObj.NumMacroFeatures; % 預設 10 (4項SPY衍生 + 6項FRED)

idx_raw18 = 1 : (numRel + numMicro);
idx_macro = (numRel + numMicro + 1) : (numRel + numMicro + numMacro);

% 提取原始 18 維微觀/相對特徵 (供 Sanity Baseline 比對使用)
X_norm_18D = X_norm_3D(:, idx_raw18, :); 

% 提取完整 10 維宏觀總經特徵 (供 GBDT 輔助特徵與崩盤護欄使用)
Macro_2D = X_norm_3D(:, idx_macro, 1); 

fprintf('  -> 原始特徵維度: %d 維 (Rel %d + Micro %d) | 宏觀特徵維度: %d 維\n', ...
    length(idx_raw18), numRel, numMicro, length(idx_macro));

%% 3. 構建橫截面選股標籤與大盤崩盤標籤
disp('--- 步驟 3：構建 5 日遠期選股目標與 10 日累積崩盤護欄標籤 ---');

% 3.1 橫截面 5 日遠期 Beat the Median 標籤
R_fwd = NaN(numDaysRaw, numT, 'single');
R_fwd(1:end-horizon_stock, :) = (Prices_Active(1+horizon_stock:end, :) - Prices_Active(1:end-horizon_stock, :)) ...
                                ./ Prices_Active(1:end-horizon_stock, :);
R_fwd(isinf(R_fwd)) = NaN;

Y_Labels_3D = zeros(numDaysRaw, numT, 'single');
for t = 1:numDaysRaw-horizon_stock
    active_mask = Expert_Active(t, :);
    if sum(active_mask) > 10
        med_ret = median(R_fwd(t, active_mask), 'omitnan');
        Y_Labels_3D(t, active_mask) = single(R_fwd(t, active_mask) > med_ret);
    end
end

% 3.2 ★ Phase 14.25 修正：崩盤護欄標籤改為「未來 10 日累積跌幅超過 5%」
spy_idx = find(strcmp(configObj.IdxTickers, 'SPY'));
if isempty(spy_idx), spy_idx = 1; end
spy_prices = Prices_Active(:, spy_idx);

fwd_cum_ret = NaN(numDaysRaw, 1, 'single');
for t = 1:numDaysRaw-horizon_crash
    fwd_cum_ret(t) = (spy_prices(t+horizon_crash) - spy_prices(t)) / (spy_prices(t) + 1e-8);
end

Y_Crash_1D = single(fwd_cum_ret < -0.05); % 10 天內累積回撤 > 5% 定義為風險體制
Y_Crash_1D(isnan(Y_Crash_1D)) = 0;

%% 4. 切分時間軸 (嚴格 In-Sample 訓練與 OOS 盲測推論解耦)
disp('--- 步驟 4：切分樣本時間軸 (Train IS vs. Blind OOS) ---');
Train_Start_Date = datetime('2006-01-01');
OOS_Start_Date   = datetime('2022-01-01');

idx_IS_raw  = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);
idx_OOS_raw = find(Dates_Active >= OOS_Start_Date);

idx_IS  = intersect(valid_idx, idx_IS_raw);
idx_OOS = intersect(valid_idx, idx_OOS_raw);

% 切分 In-Sample 訓練資料
E_time_IS   = E_time_all(idx_IS, :, :);
E_space_IS  = E_space_all(idx_IS, :, :);
X_18D_IS    = X_norm_18D(idx_IS, :, :);
Macro_IS    = Macro_2D(idx_IS, :);
Y_Labels_IS = Y_Labels_3D(idx_IS, :);
Expert_IS   = Expert_Active(idx_IS, :);
Y_Crash_IS  = Y_Crash_1D(idx_IS);

% 切分 Out-of-Sample 盲測推論資料
E_time_OOS  = E_time_all(idx_OOS, :, :);
E_space_OOS = E_space_all(idx_OOS, :, :);
Macro_OOS   = Macro_2D(idx_OOS, :);
Expert_OOS  = Expert_Active(idx_OOS, :);

fprintf('✅ 時間軸劃分完畢！IS 訓練區間: %d 天 | OOS 盲測推論區間: %d 天\n', ...
    length(idx_IS), length(idx_OOS));

%% 5. 啟動 GBDT 代理人訓練與 OOF 機率生成
disp('--- 步驟 5：實例化 GBDT 專家並執行 Expanding-Window 交叉驗證 ---');
gbdt_agent = GBDTExpertAgent(configObj);

% ★ Phase 14.25 核心修復：正確傳入 6 個參數 (補上 X_18D_IS，對齊類別簽名與 Raw Baseline)
[P_time_oof_IS, P_space_oof_IS] = gbdt_agent.train_and_predict_oof_cross_sectional(...
    E_time_IS, E_space_IS, X_18D_IS, Macro_IS, Y_Labels_IS, Expert_IS);

% 訓練崩盤護欄與 Platt Scaling 事後校準
P_crash_oof_IS = gbdt_agent.train_and_predict_oof_crash(Macro_IS, Y_Crash_IS);

%% 6. 執行 OOS 盲測期真實前向推論
disp('--- 步驟 6：執行 OOS 盲測期無洩漏推論 (含 Platt 校準應用) ---');
[P_time_oos_sub, P_space_oos_sub, P_crash_oos_sub] = gbdt_agent.predict_oos(...
    E_time_OOS, E_space_OOS, Macro_OOS, Expert_OOS);

%% 7. 矩陣全域縫合 (將 IS-OOF 與 OOS 推論組裝為全歷史面板)
disp('--- 步驟 7：組裝全歷史選股與崩盤機率矩陣 ---');
P_time_all  = 0.5 * ones(numDaysRaw, numT, 'single');
P_space_all = 0.5 * ones(numDaysRaw, numT, 'single');
P_crash_all = zeros(numDaysRaw, 1, 'single');

% 填入 IS 區間 (嚴格 OOF 機率)
P_time_all(idx_IS, :)   = P_time_oof_IS;
P_space_all(idx_IS, :)  = P_space_oof_IS;
P_crash_all(idx_IS)     = P_crash_oof_IS;

% 填入 OOS 區間 (真實前向推論機率)
P_time_all(idx_OOS, :)  = P_time_oos_sub;
P_space_all(idx_OOS, :) = P_space_oos_sub;
P_crash_all(idx_OOS)    = P_crash_oos_sub;

% 非活躍標的強制作為中立 0.5 機率
P_time_all(~Expert_Active)  = 0.5;
P_space_all(~Expert_Active) = 0.5;

%% 8. 快速抽樣 SHAP 解釋性視覺化 (標準學術白底黑字格式)
disp('--- 步驟 8：產出 SHAP 特徵邊際貢獻度視覺化 (白底黑字) ---');
try
    sample_active_t = idx_IS(end);
    sample_active_idx = find(Expert_IS(end, :), 1, 'first');
    
    if ~isempty(sample_active_idx)
        e_sample_t = permute(E_time_IS(end, :, sample_active_idx), [3, 2, 1]);
        mac_sample = Macro_IS(end, :);
        x_query = [e_sample_t, mac_sample];
        
        bg_samples = min(200, length(idx_IS));
        bg_t = randsample(length(idx_IS), bg_samples);
        x_bg = zeros(bg_samples, size(x_query, 2), 'single');
        for b = 1:bg_samples
            act_cand = find(Expert_IS(bg_t(b), :), 1, 'first');
            if isempty(act_cand), act_cand = 1; end
            e_bg = permute(E_time_IS(bg_t(b), :, act_cand), [3, 2, 1]);
            x_bg(b, :) = [e_bg, Macro_IS(bg_t(b), :)];
        end
        
        % 呼叫代理人繪製 SHAP 特徵貢獻圖
        gbdt_agent.explain_shapley(x_query, x_bg, 'time');
        
        % ★ 捕捉當前視窗並全面套用白底黑字與淺灰網格學術樣式
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
            
            % 強化標題與軸標籤對比度
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
        
        % ★ 300 DPI 高解析度白底無失真匯出
        shapFigPath = fullfile(configObj.ModelDir, 'Phase3_SHAP_TimeExpert.png');
        exportgraphics(fig_shap, shapFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
        close(fig_shap);
        fprintf(' 📊 SHAP 特徵歸因圖 (白底黑字) 已儲存至: %s\n', shapFigPath);
    end
catch ME
    warning('⚠️ SHAP 繪圖跳過 (非致命): %s', ME.message);
end

%% 9. 儲存 GBDT 模型與全域機率矩陣
disp('--- 步驟 9：儲存 GBDT 模型實體與機率矩陣 ---');
savePath = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');
save(savePath, 'gbdt_agent', 'P_time_all', 'P_space_all', 'P_crash_all', '-v7.3');
fprintf('💾 顯性風險預測流已安全落地至: %s\n', savePath);

disp('=================================================================');
disp('🎯 [Phase 3] 執行完成！機率矩陣與 Baseline 比對已全部對齊，請執行 Phase 4。');
disp('=================================================================');