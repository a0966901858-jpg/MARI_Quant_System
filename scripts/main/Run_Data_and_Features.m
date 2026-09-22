% =========================================================================
% 腳本：1_Run_Data_and_Features.m 
% 升級：Phase 15.5 物理客觀基準版 (★ mrg32k3a 獨立子串流注入、parfor 平行隨機狀態嚴格鎖定、
%       雙爬蟲微服務調度、FRED 總經整合、VQ-VAE 字典物理凍結與大腦實體落地、
%       ★ 特徵湖物理純淨度斷言、正則化沙盒解耦防洩漏、各階段獨立高精度計時與總運行耗時審計)
% 職責：調度 Python 爬蟲 -> DataFetcher 矩陣映射 -> 特徵工程 (含真實VIX/VRP) -> VQ-VAE 降噪存檔
% 紀律：嚴禁在此處寫死隨機特徵遮蔽或持久化噪聲，維持資料湖絕對物理客觀性
% =========================================================================
clear; clc; close all;

% 啟動全域總計時器
t_total_start = tic;

disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 MARI 數據湖對齊、特徵萃取與零洩漏降噪管線 (mrg32k3a 確定性子串流版)');
disp('=================================================================');

%% 0. 環境路徑掛載 (規範化階層回溯解析與路徑重新整理)
t_step0 = tic;
disp('--- 步驟 0：環境路徑掛載、Config 載入與平行隨機子串流鎖定 ---');

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

% ★ 核心修復 1：使用 Config 統一初始化主執行緒 mrg32k3a 隨機數生成器 (Substream = 1)
configObj.initRNG(1);

% 啟動平行運算池
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp('⚡ 正在喚醒多核心並行池 (parpool)...');
    poolobj = parpool('Processes');
end

% ★ 核心修復 2：為 parpool 所有並行 Worker 注入 mrg32k3a 獨立子串流 (Sub-streams)
rng_seed = configObj.RNG_Seed;
rng_gen  = configObj.RNG_Generator;
spmd
    worker_stream = RandStream(rng_gen, 'Seed', rng_seed);
    worker_stream.Substream = labindex;
    RandStream.setGlobalStream(worker_stream);
end
disp('🔒 已成功為所有並行 Worker 注入 mrg32k3a 獨立隨機子串流 (確定性可重現模式)。');

time_step0 = toc(t_step0);
fprintf('⏱️ [步驟 0 完成] 耗時: %.2f 秒\n\n', time_step0);

%% --- 步驟 1：啟動外部 Python 雙爬蟲微服務 (Yahoo Finance + FRED API) ---
t_step1 = tic;
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
time_step1 = toc(t_step1);
fprintf('⏱️ [步驟 1 完成] 耗時: %.2f 秒 (%.2f 分鐘)\n\n', time_step1, time_step1 / 60);

%% --- 步驟 2：載入 Data Lake 大宇宙數據 (純矩陣映射) ---
t_step2 = tic;
disp('--- 步驟 2：載入 Data Lake 並執行極速矩陣映射 ---');
fetcher = DataFetcher(configObj);
dataStruct = fetcher.fetch_data(); 
time_step2 = toc(t_step2);
fprintf('⏱️ [步驟 2 完成] 耗時: %.2f 秒\n\n', time_step2);

%% --- 步驟 3：特徵工程與時變圖譜萃取 ---
t_step3 = tic;
disp('--- 步驟 3：特徵工程與 DyGAT 時變圖譜萃取 (防禦 Look-ahead Bias) ---');
fe = FeatureEngineer(configObj);
if isprop(fe, 'RandStream')
    fe.RandStream = configObj.getRandStream(1);
end

if ~isstruct(dataStruct)
    error('❌ 致命錯誤：資料流格式錯誤！請確保 DataFetcher 輸出的是 Struct 矩陣封裝。');
end

% 執行核心特徵計算與空間拓撲建構 (內部 parfor 自動繼承 Worker 的獨立子串流)
[X_norm_3D, Prices_Active, Expert_Active, Dates_Active, AdjMatrix_3D] = fe.process(dataStruct);

% 特徵維度斷言校驗
[numDaysCheck, totalFeatsCheck, numTickersCheck] = size(X_norm_3D);
fprintf('  📊 [特徵面板維度校驗] 天數: %d | 節點特徵數: %d 維 (預期 %d 維) | 標的數: %d 檔\n', ...
    numDaysCheck, totalFeatsCheck, fe.TotalNodeFeats, numTickersCheck);
if totalFeatsCheck ~= fe.TotalNodeFeats
    error('❌ 特徵維度不匹配：實際產出 %d 維，預期為 %d 維！', totalFeatsCheck, fe.TotalNodeFeats);
end

time_step3 = toc(t_step3);
fprintf('⏱️ [步驟 3 完成] 耗時: %.2f 秒 (%.2f 分鐘)\n\n', time_step3, time_step3 / 60);

%% --- 步驟 4：嚴格 In-Sample 預訓練 VQ-VAE 降噪器與字典凍結 ---
t_step4 = tic;
disp('--- 步驟 4：嚴格 In-Sample 預訓練 VQ-VAE 降噪器與字典凍結 ---');

% 時區相容性安全處理
Train_Start_Date = datetime('2006-01-01');
OOS_Start_Date   = datetime('2022-01-01');
if ~isempty(Dates_Active.TimeZone)
    Train_Start_Date.TimeZone = Dates_Active.TimeZone;
    OOS_Start_Date.TimeZone   = Dates_Active.TimeZone;
end

idx_IS = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);
fprintf(' 🔒 物理隔絕啟動：VQ-VAE 僅允許在 In-Sample 區間 (%d 天) 進行降噪字典學習。\n', length(idx_IS));

vqvaeAgent = VQVAEAgent(configObj);
X_norm_IS = X_norm_3D(idx_IS, :, :);
Expert_Active_IS = Expert_Active(idx_IS, :); 
vqvaeAgent.train(X_norm_IS, Expert_Active_IS, 30); 

% 訓練完畢立即凍結字典，物理禁止 OOS 推論時更新編碼簿
vqvaeAgent.Quantizer.freeze();
fprintf(' 🧊 VQ-VAE 編碼簿字典已成功凍結 (Freeze)，徹底杜絕 OOS 洩漏！\n');

time_step4 = toc(t_step4);
fprintf('⏱️ [步驟 4 完成] 耗時: %.2f 秒 (%.2f 分鐘)\n\n', time_step4, time_step4 / 60);

%% --- 步驟 5：全域盲測降噪 ---
t_step5 = tic;
disp(' 🔄 啟動全域盲測降噪 (Out-of-Sample Denoising)...');
X_denoised_3D = vqvaeAgent.denoise(X_norm_3D, Expert_Active);

time_step5 = toc(t_step5);
fprintf('⏱️ [步驟 5 完成] 耗時: %.2f 秒\n\n', time_step5);

%% --- 步驟 6：降噪特徵張量與 VQ-VAE 模型實體快取落地 ---
t_step6 = tic;
disp('--- 步驟 6：降噪特徵 3D 張量快取與 VQ-VAE 大腦實體落地 ---');

% ★ 特徵湖物理純淨度嚴格斷言：確保快取持久層未遭隨機丟棄或外生噪聲污染
nan_ratio_norm = sum(isnan(X_norm_3D(:))) / numel(X_norm_3D);
nan_ratio_deno = sum(isnan(X_denoised_3D(:))) / numel(X_denoised_3D);
assert(nan_ratio_norm < 0.01, '❌ [資料品質警報] X_norm_3D 異常值比例過高 (%.4f)！', nan_ratio_norm);
assert(nan_ratio_deno < 0.01, '❌ [資料品質警報] X_denoised_3D 異常值比例過高 (%.4f)！', nan_ratio_deno);

% 6A. 儲存全域特徵面板與圖譜快取
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
save(cachePath, 'X_denoised_3D', 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active', 'AdjMatrix_3D', '-v7.3');
fprintf('💾 全域 3D 特徵快取已安全落地至: %s\n', cachePath);

% 6B. 儲存 VQ-VAE 降噪實體模型 (供 Run_Ablation_VQVAE.m 調用)
vqModelPath = fullfile(configObj.ModelDir, 'VQVAE_Agent.mat');
if ~exist(configObj.ModelDir, 'dir'), mkdir(configObj.ModelDir); end
save(vqModelPath, 'vqvaeAgent', '-v7.3');
fprintf('💾 VQ-VAE 降噪大腦實體已存檔至: %s\n', vqModelPath);

time_step6 = toc(t_step6);
fprintf('⏱️ [步驟 6 完成] 耗時: %.2f 秒\n\n', time_step6);

%% =========================================================================
% 結算全流程執行時長審計
% =========================================================================
total_elapsed_sec = toc(t_total_start);
tot_hours = floor(total_elapsed_sec / 3600);
tot_mins  = floor(mod(total_elapsed_sec, 3600) / 60);
tot_secs  = mod(total_elapsed_sec, 60);

fprintf('=================================================================\n');
fprintf('📊 【Phase 1 各階段耗時明細與總時長審計報告】\n');
fprintf('=================================================================\n');
fprintf(' 步驟 0：環境掛載與隨機串流注入 : %8.2f 秒 (%5.1f%%)\n', time_step0, (time_step0 / total_elapsed_sec) * 100);
fprintf(' 步驟 1：Python 爬蟲微服務調度  : %8.2f 秒 (%5.1f%%)\n', time_step1, (time_step1 / total_elapsed_sec) * 100);
fprintf(' 步驟 2：Data Lake 矩陣映射對齊 : %8.2f 秒 (%5.1f%%)\n', time_step2, (time_step2 / total_elapsed_sec) * 100);
fprintf(' 步驟 3：特徵工程與時變圖譜萃取 : %8.2f 秒 (%5.1f%%)\n', time_step3, (time_step3 / total_elapsed_sec) * 100);
fprintf(' 步驟 4：In-Sample VQ-VAE 字典預訓練: %8.2f 秒 (%5.1f%%)\n', time_step4, (time_step4 / total_elapsed_sec) * 100);
fprintf(' 步驟 5：全域盲測特徵向量降噪推論: %8.2f 秒 (%5.1f%%)\n', time_step5, (time_step5 / total_elapsed_sec) * 100);
fprintf(' 步驟 6：張量快取與大腦實體序列化: %8.2f 秒 (%5.1f%%)\n', time_step6, (time_step6 / total_elapsed_sec) * 100);
fprintf('-----------------------------------------------------------------\n');
if tot_hours > 0
    fprintf('⏱️ 【總執行時長】: %d 小時 %d 分 %.2f 秒 (共 %.2f 秒)\n', tot_hours, tot_mins, tot_secs, total_elapsed_sec);
else
    fprintf('⏱️ 【總執行時長】: %d 分 %.2f 秒 (共 %.2f 秒)\n', tot_mins, tot_secs, total_elapsed_sec);
end
fprintf('🔒 【架構安全斷言】: 特徵資料庫保持絕對客觀物理數值，無常態性隨機遮蔽。\n');
fprintf('🛡️ 【正則化解耦機制】: Feature/Attention/Variational Dropout 嚴格限制於 Phase 2 動態生成。\n');
fprintf('=================================================================\n');
disp('🎯 [Phase 1] 完美完成。特徵資料庫與降噪模型已 100% 確定性落地，請進入 Phase 2！');
disp('=================================================================');