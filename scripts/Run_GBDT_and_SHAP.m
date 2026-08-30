% =========================================================================
% 腳本：3_Run_GBDT_and_SHAP.m (階段 3：GBDT 專家網路與可解釋性分析管線)
% 升級：Phase 14.9 (★ 修復維度崩潰、強制 SPY 校驗、擴增 Crash SHAP 歸因)
% 職責：訓練 GBDT 個股勝率與崩盤護欄，產出防溫室幻覺的 OOF 機率，並執行 SHAP
% =========================================================================

% 清除工作區變數、清空命令視窗、關閉所有繪圖視窗，確保記憶體環境純淨與避免變數污染
clear; clc; close all;

%% 0. 環境路徑掛載與全域設定
disp('=================================================================');
disp('🚀 [Phase 14.9] 啟動 GBDT 專家網路訓練與 SHAP 歸因分析管線');
disp('=================================================================');

% 動態抓取當前腳本所在的絕對路徑，確保跨設備執行時的相容性
currentPath = fileparts(mfilename('fullpath'));
% 若在命令列直接單行執行導致路徑為空，則預設為當下工作目錄 (pwd)
if isempty(currentPath), currentPath = pwd; end
projectRoot = currentPath; 

% 檢查 configs 目錄是否存在，若不存在則將根目錄往上一層推導
if ~exist(fullfile(projectRoot, 'configs'), 'dir')
    projectRoot = fullfile(currentPath, '..');
end

% 將專案核心的各個子資料夾遞迴加入 MATLAB 搜尋路徑中
addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'models'))); 
addpath(genpath(fullfile(projectRoot, 'agents'))); 
% 刷新 MATLAB 內部工具箱快取，確保剛加入路徑的類別與函數能被即時識別
rehash toolboxcache;

% 實例化全域設定檔，統一控管超參數與路徑
configObj = Config();

% ★ 計畫書修正 (問題 P3-1 ⚪ P3)：固定全域隨機種子，確保實驗可重現
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

% ★ 核心修復 1：確保大宇宙名單被正確載入
% 呼叫 Config 類別的動態大宇宙解析器，確保接下來的特徵矩陣與股票代號完全對齊
configObj.loadUniverse(); 

%% 1. 載入特徵快取與 DL 降噪表徵
disp('--- 步驟 1：載入 3D 特徵面板與 DL 節點級別 Embedding ---');

% 定義降噪特徵的快取檔案路徑
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
% 防呆機制：確保 Phase 1 的特徵工程已經成功落地
if ~exist(cachePath, 'file')
    error('❌ 找不到 features_denoised.mat，請先執行 Phase 1');
end

% 載入 Phase 1 產出的 3D 標準化特徵、活躍遮罩、價格與對齊後的時間軸
load(cachePath, 'X_norm_3D', 'Expert_Active', 'Prices_Active', 'Dates_Active');
% 剝除時間軸的時區標籤，避免後續進行 datetime 比較時發生時區衝突報錯
Dates_Active.TimeZone = ''; 

% 定義深度學習表徵 (Embedding) 的快取檔案路徑
modelPath = fullfile(configObj.ModelDir, 'DL_Extractors.mat');
% 防呆機制：確保 Phase 2 的 DL 特徵萃取器已經完訓並產出表徵
if ~exist(modelPath, 'file')
    error('❌ 找不到 DL_Extractors.mat，請先執行 Phase 2');
end
% 載入 Phase 2 產出的時序與空間解耦 Embedding 張量 (64維)
load(modelPath, 'E_time_all', 'E_space_all');

% 取得總交易天數與總股票數量，建立矩陣迭代邊界
numDays = length(Dates_Active);
numTickers = configObj.NumTickers;

% 提取 4 維宏觀特徵 [Days, 4]
% 由於宏觀特徵對每一檔個股皆相同，因此只需沿著第 3 維度 (Ticker) 提取第 1 檔的資料即可
Macro_2D = X_norm_3D(:, 19:22, 1);

%% 2. 嚴格構建無洩漏預測標籤 (Labels Construction)
disp('--- 步驟 2：構建橫截面與時序預測標籤 (Zero-Leakage Labels) ---');

% 預先配置一個充滿 NaN 的單精度矩陣來存放次日報酬率 (T+1)
R_fwd = NaN(numDays, numTickers, 'single');
% 計算嚴格無未來洩漏的次日報酬率：(明日收盤 - 今日收盤) / 今日收盤
R_fwd(1:end-1, :) = (Prices_Active(2:end, :) - Prices_Active(1:end-1, :)) ./ Prices_Active(1:end-1, :);
% 將可能因為除以 0 或極端錯誤導致的無限大 (Inf) 替換為 NaN
R_fwd(isinf(R_fwd)) = NaN;

% 預配置二元分類標籤：選股標籤 (3D) 與崩盤護欄標籤 (1D)
Y_Labels_3D = zeros(numDays, numTickers, 'single');
Y_Crash_1D = zeros(numDays, 1, 'single');

% ★ 核心修復 2：嚴格校驗 SPY 基準，找不到直接報錯，絕不沉默代替
% 定位基準指數 SPY 在宇宙清單中的正確索引，用於計算崩盤標準
spy_idx = find(strcmp(configObj.IdxTickers, 'SPY'));
if isempty(spy_idx)
    error('❌ 致命錯誤：在宇宙名單中找不到 SPY！無法計算崩盤護欄標籤。'); 
end

% 遍歷每一天以構建橫截面預測標籤 (最後一天因無 T+1 報酬故捨去)
for t = 1:numDays-1
    % 取得當日允許交易的活躍股票遮罩
    active_mask = Expert_Active(t, :);
    
    % 確保當日有足夠的活躍標的 (大於 10 檔) 才具有統計學上的橫截面意義
    if sum(active_mask) > 10
        % A. 橫截面選股標籤 (Beat the Median)
        % 計算當日有效橫截面次日報酬的「中位數」，能有效抵抗極端值干擾
        med_ret = median(R_fwd(t, active_mask), 'omitnan');
        % 個別報酬大於中位數者標記為 1 (正樣本)，其餘為 0 (負樣本)
        Y_Labels_3D(t, active_mask) = single(R_fwd(t, active_mask) > med_ret);
    end
    
    % B. 大盤崩盤護欄標籤 (預測 SPY 明日跌幅是否超過 0.5%)
    % 如果基準指數次日報酬率小於 -0.5%，則將當日標記為潛在崩盤日 (1.0)
    if R_fwd(t, spy_idx) < -0.005
        Y_Crash_1D(t) = 1.0;
    end
end

%% 3. 切分 IS / OOS 時間軸並解耦 3D 張量
disp('--- 步驟 3：嚴格切分 IS 訓練集與 OOS 盲測集張量 ---');

% 宣告訓練集 (IS) 與盲測集 (OOS) 的絕對時間邊界
Train_Start_Date = datetime('2006-01-01'); % 系統記憶起點同步
OOS_Start_Date = datetime('2022-01-01');

% ★ 限制 IS 訓練樣本的下界與上界
% 嚴格利用邏輯條件提取 In-Sample 的時間索引
idx_IS = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);
% 嚴格提取 Out-Of-Sample 的時間索引
idx_OOS = find(Dates_Active >= OOS_Start_Date);

% 防禦機制：若 IS 的最後一天剛好是歷史資料的最後一天，將其剔除以避免標籤 NaN
if idx_IS(end) == numDays
    idx_IS(end) = [];
end

% 將 3D 表徵與標籤矩陣沿著時間軸精準切分為訓練集 (IS)
E_time_IS = E_time_all(idx_IS, :, :);
E_space_IS = E_space_all(idx_IS, :, :);
Macro_IS = Macro_2D(idx_IS, :);
Y_Labels_IS = Y_Labels_3D(idx_IS, :);
Y_Crash_IS = Y_Crash_1D(idx_IS);
Expert_IS = Expert_Active(idx_IS, :);

% 將 3D 表徵與標籤矩陣沿著時間軸精準切分為盲測集 (OOS)
E_time_OOS = E_time_all(idx_OOS, :, :);
E_space_OOS = E_space_all(idx_OOS, :, :);
Macro_OOS = Macro_2D(idx_OOS, :);
Expert_OOS = Expert_Active(idx_OOS, :);

% 在終端機印出資料切割的驗證結果
fprintf('  -> 訓練集 (IS): %d 天 | 盲測集 (OOS): %d 天\n', length(idx_IS), length(idx_OOS));

%% 4. 初始化 GBDT 專家網路與模型訓練
disp('--- 步驟 4：實例化 GBDT 專家並啟動 Block K-Fold 訓練管線 ---');

% 實例化 GBDT 代理人物件
gbdt_agent = GBDTExpertAgent(configObj);

% 執行時序防洩漏的橫截面 OOF 訓練，取得 IS 期間無過擬合幻覺的乾淨機率
[P_time_oof, P_space_oof] = gbdt_agent.train_and_predict_oof_cross_sectional(...
    E_time_IS, E_space_IS, Macro_IS, Y_Labels_IS, Expert_IS);

% 執行崩盤護欄的 RUSBoost 訓練與 OOF 機率預測
P_crash_oof = gbdt_agent.train_and_predict_oof_crash(Macro_IS, Y_Crash_IS);

%% 5. 執行 OOS 純淨盲測推論
disp('--- 步驟 5：執行 OOS 區間純淨推論 (Zero-Leakage Inference) ---');

% 針對 2022 年以後的 OOS 數據進行前向推論，產出完全未見過資料的機率
[P_time_oos, P_space_oos, P_crash_oos] = gbdt_agent.predict_oos(...
    E_time_OOS, E_space_OOS, Macro_OOS, Expert_OOS);

%% 6. 無縫拼接全歷史機率矩陣供 RL 總管使用
disp('--- 步驟 6：無縫拼接 IS(OOF) 與 OOS 機率矩陣 ---');

% 預配置涵蓋全歷史的機率矩陣，預設 0.5 (不具方向性的中立機率)
P_time_all = zeros(numDays, numTickers, 'single') + 0.5;
P_space_all = zeros(numDays, numTickers, 'single') + 0.5;
P_crash_all = zeros(numDays, 1, 'single');

% 將訓練期的 OOF 驗證機率填回主矩陣，確保後續 RL 環境模擬不帶有未來函數
P_time_all(idx_IS, :) = P_time_oof;
P_space_all(idx_IS, :) = P_space_oof;
P_crash_all(idx_IS) = P_crash_oof;

% 將盲測期的真實推論機率填回主矩陣
P_time_all(idx_OOS, :) = P_time_oos;
P_space_all(idx_OOS, :) = P_space_oos;
P_crash_all(idx_OOS) = P_crash_oos;

%% 7. 準備 SHAP 背景與查詢資料集，啟動歸因分析
disp('--- 步驟 7：資料降維預處理與 SHAP 歸因分析啟動 ---');

% 定義特徵維度常數：DL 表徵 64 維，總體宏觀 4 維
embedDim = 64;
numMacro = 4;

% A. 展平並提取 IS 背景樣本 (X_bg_raw) 作為 SHAP 的對照組
% 將活躍遮罩轉換為線性索引，以方便進行隨機抽樣
active_is_linear = find(Expert_IS);
% 抽樣最多 5000 筆資料作為運算背景，平衡運算速度與代表性
sample_bg_idx = randsample(active_is_linear, min(5000, length(active_is_linear)));
% 將線性索引轉回 [時間, 標的] 的二維座標
[t_bg, tic_bg] = ind2sub(size(Expert_IS), sample_bg_idx);

% 預配置背景資料集的 2D 矩陣
X_bg_time = zeros(length(sample_bg_idx), embedDim + numMacro, 'single');
for i = 1:length(sample_bg_idx)
    % ★ 終極修復：強制將 [1, 64, 1] 的 3D 張量重塑 (Reshape) 為 [1, 64] 的橫列向量
    e_t = reshape(E_time_IS(t_bg(i), :, tic_bg(i)), 1, embedDim);
    % 提取對應的宏觀特徵向量 [1, 4]
    mac = Macro_IS(t_bg(i), :); 
    % 完美水平拼接為 [1, 68] 並寫入矩陣
    X_bg_time(i, :) = [e_t, mac]; 
end

% B. 展平並提取 OOS 查詢樣本 (X_query_raw) 以檢視模型真實關注點
active_oos_linear = find(Expert_OOS);
% 隨機抽取 1000 筆 OOS 活躍數據作為查詢目標
sample_q_idx = randsample(active_oos_linear, min(1000, length(active_oos_linear)));
[t_q, tic_q] = ind2sub(size(Expert_OOS), sample_q_idx);

% 預配置查詢資料集的 2D 矩陣
X_query_time = zeros(length(sample_q_idx), embedDim + numMacro, 'single');
for i = 1:length(sample_q_idx)
    % ★ 終極修復：再次執行嚴格的降維重塑，防禦維度不匹配報錯
    e_t = reshape(E_time_OOS(t_q(i), :, tic_q(i)), 1, embedDim);
    mac = Macro_OOS(t_q(i), :);
    X_query_time(i, :) = [e_t, mac];
end

% A. 執行時序專家選股 SHAP 歸因
disp('  -> 產生時序選股專家 SHAP 分析圖表...');
% 呼叫代理人內部的 explain_shapley 函數進行 K-Means 壓縮與繪圖
gbdt_agent.explain_shapley(X_query_time, X_bg_time, 'time');

% B. ★ 核心修復 4：執行 2022 崩盤護欄 SHAP 壓力測試
disp('  -> 產生 2022 崩盤護欄 (Crash Guard) SHAP 分析圖表...');
% 崩盤護欄只吃 Macro_2D 特徵，直接取用 OOS (2022+) 作為查詢，IS 作為背景
% 從 IS 宏觀特徵中隨機抽取 5000 筆作為背景
bg_crash = Macro_IS(randsample(size(Macro_IS, 1), min(5000, size(Macro_IS, 1))), :);
% 查詢整個 2022 年以後的宏觀狀態，以檢視模型如何抓出當年空頭訊號
query_crash = Macro_OOS; 
gbdt_agent.explain_shapley(query_crash, bg_crash, 'crash');

%% 8. 儲存模型與機率矩陣供 HRL 總管使用
disp('--- 步驟 8：儲存 GBDT_Guards.mat 快取 ---');

% 定義 GBDT 模型的最終落地路徑
gbdtPath = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');
% 使用 -v7.3 格式儲存大型陣列與 GBDT 物件，供強化學習 (Phase 5, Phase 6) 調用
save(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all', 'gbdt_agent', '-v7.3');

fprintf('💾 GBDT 模型與機率矩陣已存至: %s\n', gbdtPath);
disp('=================================================================');
disp('🎯 [Phase 3] 完美完成。OOF 防護與選股矩陣已就緒，請接著進入 HRL 強化學習！');
disp('=================================================================');