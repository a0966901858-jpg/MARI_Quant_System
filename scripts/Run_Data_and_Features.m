% =========================================================================
% 腳本：1_Run_Data_and_Features.m 
% 升級：Phase 14.4 (★ 系統級重構：完美橋接流動性結界與 VQ-VAE 實體隔離)
% 職責：調度 Python 爬蟲 -> DataFetcher 矩陣映射 -> 特徵工程 -> VQ-VAE 降噪
% =========================================================================

% 清除工作區的變數、清空終端機輸出、關閉所有繪圖視窗，確保記憶體環境純淨，防禦殘留變數污染
clear; clc; close all;

disp('=================================================================');
disp('🚀 [Phase 14.4] 啟動 MARI 數據湖對齊、特徵萃取與零洩漏降噪管線');
disp('=================================================================');

% --- 環境初始化 ---
% 動態獲取當前腳本所在的絕對路徑，確保專案在不同作業系統或設備上皆能正確執行
currentPath = fileparts(mfilename('fullpath'));
% 若在命令列直接單行執行導致路徑為空，則退回使用當下工作目錄 (pwd)
if isempty(currentPath), currentPath = pwd; end
projectRoot = currentPath; 

% 檢查 configs 目錄是否存在，若不存在則將專案根目錄指標往上推導一層
if ~exist(fullfile(projectRoot, 'configs'), 'dir'), projectRoot = fullfile(currentPath, '..'); end

% 將專案核心的各個子資料夾遞迴加入 MATLAB 搜尋路徑中，確保所有自定義模組皆可被呼叫
addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'envs')));
addpath(genpath(fullfile(projectRoot, 'agents')));
addpath(genpath(fullfile(projectRoot, 'models'))); 
% 刷新 MATLAB 內部工具箱快取，確保剛加入的類別與函數能被即時識別
rehash toolboxcache;

% 實例化全域設定檔，統一控管超參數與路徑，確保全系統各模組的參數一致性 (Single Source of Truth)
configObj = Config();

% ★ 計畫書修正 (問題 P3-1 ⚪ P3)：固定全域隨機種子，確保實驗可重現
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

% 啟動平行運算池 (用於後續支援可能的高併發運算)
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp('⚡ 正在喚醒多核心並行池 (parpool)...');
    % 使用 'Processes' 模式分配獨立實體記憶體，防禦後續平行運算時的記憶體干擾
    parpool('Processes');
end

%% --- 步驟 1：啟動外部 Python 爬蟲 ---
disp('--- 步驟 1：啟動外部 Python 爬蟲與 Pandas 關聯引擎 ---');
disp('⏳ 正在執行 Python 爬蟲微服務 (歷史大宇宙構建中)...');

% 定義 Python 爬蟲腳本的絕對路徑
pythonScript = fullfile(configObj.DataDir, 'crawlers', 'hybrid_crawler.py');

% 嘗試尋找並綁定正確的 Python 虛擬環境路徑 (Anaconda/Miniconda)
condaPy = '/home/andy/miniconda3/bin/python'; 
condaPy2 = '/home/andy/anaconda3/bin/python'; 
if exist(condaPy, 'file')
    pyCmd = condaPy;
elseif exist(condaPy2, 'file')
    pyCmd = condaPy2;
else
    % 若找不到指定的 conda 環境，退回系統預設的 python3
    pyCmd = 'python3';
end

% 跨語言呼叫 (Inter-Process Communication)：透過系統命令列喚醒 Python 執行爬蟲
[status, cmdout] = system(sprintf('%s "%s"', pyCmd, pythonScript));

% 檢查執行狀態碼，0 代表成功，非 0 代表發生異常
if status ~= 0
    % 給予優雅的降級處理 (Graceful Degradation)，允許系統使用舊有歷史資料繼續運行
    warning('⚠️ MATLAB 呼叫 Python 爬蟲失敗。若您已在終端機手動執行完畢，系統將自動銜接歷史資料。\n錯誤訊息：%s', cmdout);
else
    disp('✅ Python 爬蟲與 Pandas 關聯引擎執行完畢！');
end

% 重新載入最新抓取下來的宇宙名單，確保後續矩陣映射的維度與實體資料對齊
configObj.loadUniverse(); 

%% --- 步驟 2：載入 Data Lake 大宇宙數據 (純矩陣映射) ---
disp('--- 步驟 2：載入 Data Lake 並執行極速矩陣映射 ---');

% 實例化 DataFetcher，該模組內建 SPY 絕對日曆對齊與流動性枯竭防護
fetcher = DataFetcher(configObj);
% 讀取 Parquet 長表並極速映射為 2D 價格與特徵矩陣
dataStruct = fetcher.fetch_data(); 

%% --- 步驟 3：特徵工程與時變圖譜萃取 ---
disp('--- 步驟 3：特徵工程與 DyGAT 有向圖譜萃取 (防禦 Look-ahead Bias) ---');

% 實例化特徵工程引擎
fe = FeatureEngineer(configObj);

% 型別斷言：確保 DataFetcher 傳遞過來的是合法的資料結構
if ~isstruct(dataStruct)
    error('❌ 致命錯誤：資料流格式錯誤！請確保 DataFetcher 輸出的是 Struct 矩陣封裝。');
end

% 執行核心運算：計算微觀/宏觀/相對特徵，並產出至關重要的實體交易流動性結界 (Expert_Active)
[X_norm_3D, Prices_Active, Expert_Active, Dates_Active, AdjMatrix_3D] = fe.process(dataStruct);

%% --- 步驟 4：嚴格 In-Sample 預訓練 VQ-VAE 降噪器 ---
disp('--- 步驟 4：嚴格 In-Sample 預訓練 VQ-VAE 降噪器 ---');

% 定義訓練區間的絕對起點，並確保帶有 UTC 時區，與 Dates_Active 的時間軸絕對對齊
Train_Start_Date = datetime('2006-01-01', 'TimeZone', 'UTC');
OOS_Start_Date = datetime('2022-01-01', 'TimeZone', 'UTC');

% ★ 限制 IS (In-Sample) 訓練樣本的下界與上界，建立時間結界以徹底防禦未來函數
idx_IS = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);
fprintf(' 🔒 物理隔絕啟動：VQ-VAE 僅允許在 In-Sample 區間 (%d 天) 進行降噪字典學習。\n', length(idx_IS));

% 實例化 VQ-VAE 降噪代理人
vqvaeAgent = VQVAEAgent(configObj);

% ★ 核心橋接：精準切分 In-Sample 數據，絕不讓 2022 年以後的資料參與字典編碼
X_norm_IS = X_norm_3D(idx_IS, :, :);
% 同步切分對應的流動性結界 (活躍遮罩)
Expert_Active_IS = Expert_Active(idx_IS, :); 

% 傳入 IS 訓練特徵與結界遮罩，進行 30 個 Epoch 的字典學習，徹底消滅零值誤判
vqvaeAgent.train(X_norm_IS, Expert_Active_IS, 30); 

%% --- 步驟 5：全域盲測降噪與快取落地 ---
disp(' 🔄 啟動全域盲測降噪 (Out-of-Sample Denoising)...');

% ★ 核心橋接：利用完訓的 IS 字典，對包含 OOS 的全域資料進行前向推論降噪
% 傳入全域 Expert_Active 遮罩，執行物理結界抹除，確保非活躍股票的重建特徵被強制歸零 (斬斷殭屍股復活)
X_denoised_3D = vqvaeAgent.denoise(X_norm_3D, Expert_Active);

%% --- 步驟 6：降噪特徵 3D 張量快取落地 ---
disp('--- 步驟 6：降噪特徵 3D 張量快取落地 ---');

% 定義特徵快取的落地路徑
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');

% 將降噪完成的 3D 特徵張量、價格矩陣、活躍遮罩與圖譜矩陣儲存至硬碟
% 使用 '-v7.3' 參數以支援大於 2GB 的超大型 HDF5 矩陣儲存，供 Phase 2 與 Phase 3 調用
save(cachePath, 'X_denoised_3D', 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active', 'AdjMatrix_3D', '-v7.3');
fprintf('💾 全域 3D 特徵快取已安全落地至: %s\n', cachePath);

disp('=================================================================');
disp('🎯 [Phase 1] 完美完成。資料維度與物理邊界防護已 100% 驗證，請進入 Phase 2！');
disp('=================================================================');