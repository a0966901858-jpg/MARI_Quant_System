% =========================================================================
% 腳本：1_Run_Data_and_Features.m 
% 升級：Phase 15.5 (★ 雙爬蟲微服務調度、FRED 總經整合、VQ-VAE 字典物理凍結與大腦實體落地)
% 職責：調度 Python 爬蟲 -> DataFetcher 矩陣映射 -> 特徵工程 (含真實VIX/VRP) -> VQ-VAE 降噪存檔
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 MARI 數據湖對齊、特徵萃取與零洩漏降噪管線');
disp('=================================================================');

%% 0. 環境路徑掛載
currentPath = fileparts(mfilename('fullpath'));
if isempty(currentPath), currentPath = pwd; end
projectRoot = currentPath; 
if ~exist(fullfile(projectRoot, 'configs'), 'dir')
    projectRoot = fullfile(currentPath, '..'); 
end

addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'agents'))); 
addpath(genpath(fullfile(projectRoot, 'models'))); 
addpath(genpath(fullfile(projectRoot, 'utils'))); % ★ 統一掛載共用統計工具箱
rehash toolboxcache;

configObj = Config();

% 固定全域隨機種子
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

% 啟動平行運算池
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp('⚡ 正在喚醒多核心並行池 (parpool)...');
    parpool('Processes');
end

%% --- 步驟 1：啟動外部 Python 雙爬蟲微服務 (Yahoo Finance + FRED API) ---
disp('--- 步驟 1：啟動外部 Python 爬蟲微服務 (美股大宇宙 + FRED 總經) ---');

% 尋找 Python 執行環境
condaPy  = '/home/andy/miniconda3/bin/python'; 
condaPy2 = '/home/andy/anaconda3/bin/python'; 
if exist(condaPy, 'file')
    pyCmd = condaPy;
elseif exist(condaPy2, 'file')
    pyCmd = condaPy2;
else
    pyCmd = 'python3';
end

% 1A. 調度歷史大宇宙與 K 線爬蟲
crawlerHybrid = fullfile(configObj.DataDir, 'crawlers', 'hybrid_crawler.py');
disp('⏳ 正在執行美股大宇宙與 K 線爬蟲 (hybrid_crawler.py)...');
[status1, cmdout1] = system(sprintf('%s "%s"', pyCmd, crawlerHybrid));
if status1 ~= 0
    warning('⚠️ 美股爬蟲回傳警告或失敗。訊息：%s', cmdout1);
else
    disp('✅ 美股大宇宙 K 線長表建立完成！');
end

% 1B. 調度 FRED 宏觀總經爬蟲 (殖利率曲線、信用利差、失業率、VIX)
crawlerFred = fullfile(configObj.DataDir, 'crawlers', 'fred_crawler.py');
disp('⏳ 正在執行 FRED 總經領先指標爬蟲 (fred_crawler.py)...');
[status2, cmdout2] = system(sprintf('%s "%s"', pyCmd, crawlerFred));
if status2 ~= 0
    warning('⚠️ FRED 總經爬蟲回傳警告。若已存在歷史快取將自動銜接。訊息：%s', cmdout2);
else
    disp('✅ FRED 宏觀總經數據抓取與 PiT 延遲校正完成！');
end

% 動態重載宇宙名單
configObj.loadUniverse(); 

%% --- 步驟 2：載入 Data Lake 大宇宙數據 (純矩陣映射) ---
disp('--- 步驟 2：載入 Data Lake 並執行極速矩陣映射 ---');
fetcher = DataFetcher(configObj);
dataStruct = fetcher.fetch_data(); 

%% --- 步驟 3：特徵工程與時變圖譜萃取 ---
disp('--- 步驟 3：特徵工程與 DyGAT 時變圖譜萃取 (防禦 Look-ahead Bias) ---');
fe = FeatureEngineer(configObj);
if ~isstruct(dataStruct)
    error('❌ 致命錯誤：資料流格式錯誤！請確保 DataFetcher 輸出的是 Struct 矩陣封裝。');
end

% 執行核心特徵計算與空間拓撲建構
[X_norm_3D, Prices_Active, Expert_Active, Dates_Active, AdjMatrix_3D] = fe.process(dataStruct);

% 特徵維度斷言校驗
[numDaysCheck, totalFeatsCheck, numTickersCheck] = size(X_norm_3D);
fprintf('  📊 [特徵面板維度校驗] 天數: %d | 節點特徵數: %d 維 (預期 %d 維) | 標的數: %d 檔\n', ...
    numDaysCheck, totalFeatsCheck, fe.TotalNodeFeats, numTickersCheck);
if totalFeatsCheck ~= fe.TotalNodeFeats
    error('❌ 特徵維度不匹配：實際產出 %d 維，預期為 %d 維！', totalFeatsCheck, fe.TotalNodeFeats);
end

%% --- 步驟 4：嚴格 In-Sample 預訓練 VQ-VAE 降噪器與字典凍結 ---
disp('--- 步驟 4：嚴格 In-Sample 預訓練 VQ-VAE 降噪器與字典凍結 ---');
Train_Start_Date = datetime('2006-01-01', 'TimeZone', 'UTC');
OOS_Start_Date   = datetime('2022-01-01', 'TimeZone', 'UTC');
idx_IS = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);
fprintf(' 🔒 物理隔絕啟動：VQ-VAE 僅允許在 In-Sample 區間 (%d 天) 進行降噪字典學習。\n', length(idx_IS));

vqvaeAgent = VQVAEAgent(configObj);
X_norm_IS = X_norm_3D(idx_IS, :, :);
Expert_Active_IS = Expert_Active(idx_IS, :); 
vqvaeAgent.train(X_norm_IS, Expert_Active_IS, 30); 

% 訓練完畢立即凍結字典，物理禁止 OOS 推論時更新編碼簿
vqvaeAgent.Quantizer.freeze();
fprintf(' 🧊 VQ-VAE 編碼簿字典已成功凍結 (Freeze)，徹底杜絕 OOS 洩漏！\n');

%% --- 步驟 5：全域盲測降噪 ---
disp(' 🔄 啟動全域盲測降噪 (Out-of-Sample Denoising)...');
X_denoised_3D = vqvaeAgent.denoise(X_norm_3D, Expert_Active);

%% --- 步驟 6：降噪特徵張量與 VQ-VAE 模型實體快取落地 ---
disp('--- 步驟 6：降噪特徵 3D 張量快取與 VQ-VAE 大腦實體落地 ---');

% 6A. 儲存全域特徵面板與圖譜快取
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
save(cachePath, 'X_denoised_3D', 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active', 'AdjMatrix_3D', '-v7.3');
fprintf('💾 全域 3D 特徵快取已安全落地至: %s\n', cachePath);

% 6B. ★ Phase 15.5 關鍵修復：儲存 VQ-VAE 降噪實體模型 (供 Run_Ablation_VQVAE.m 調用)
vqModelPath = fullfile(configObj.ModelDir, 'VQVAE_Agent.mat');
if ~exist(configObj.ModelDir, 'dir'), mkdir(configObj.ModelDir); end
save(vqModelPath, 'vqvaeAgent', '-v7.3');
fprintf('💾 VQ-VAE 降噪大腦實體已存檔至: %s\n', vqModelPath);

disp('=================================================================');
disp('🎯 [Phase 1] 完美完成。特徵資料庫與降噪模型已 100% 落地，請進入 Phase 2！');
disp('=================================================================');
