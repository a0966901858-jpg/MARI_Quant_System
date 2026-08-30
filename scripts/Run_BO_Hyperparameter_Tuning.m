% =========================================================================
% 腳本：4_Run_BO_Hyperparameter_Tuning.m
% 升級：Phase 14.22 (★ 加入負 IR 循環斷路器、放寬超參數搜索邊界、對齊統一成本模型)
% 職責：利用貝氏最佳化，在完全無簡化假設的微型模擬器中聯合尋優
% =========================================================================
% 清空工作區變數、清除命令列輸出、關閉所有繪圖視窗，確保記憶體狀態純淨
clear; clc; close all;
disp('=================================================================');
disp('🧠 啟動 MARI 系統：CIO 總管動態路由貝氏超參數尋優 (物理鐵律版)');
disp('=================================================================');

% 取得當下腳本執行的絕對路徑，確保專案在不同設備上皆能正確定位
currentPath = fileparts(mfilename('fullpath'));
if isempty(currentPath), currentPath = pwd; end
projectRoot = currentPath; 

% 檢查 configs 目錄是否存在，藉此判斷是否需要將根目錄指標往上推一層
if ~exist(fullfile(projectRoot, 'configs'), 'dir')
    projectRoot = fullfile(currentPath, '..'); 
end

% 將專案核心的子模組路徑遞迴加入 MATLAB 搜尋環境中
addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'models'))); 
rehash toolboxcache;

% 實例化全域設定物件，作為超參數的單一真理來源 (Single Source of Truth)
configObj = Config();

% 固定全域隨機種子，確保實驗可重現
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入全域快取與 GBDT 預測機率，並動態拉取 Opens 矩陣
disp('--- 步驟 1：載入快取與時間軸嚴格對齊 (引入 Opens 矩陣) ---');
% 定義特徵快取與 GBDT 模型快取的絕對路徑
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat'); 

% 嚴格斷言 (Assertion)：確保前置任務 (Phase 1~3) 已產出必要檔案
if ~exist(cachePath, 'file') || ~exist(gbdtPath, 'file')
    error('❌ 找不到前置快取檔案，請確認 Phase 1 至 Phase 3 已成功執行！');
end

% 1A. 載入 3D 特徵面板的活躍遮罩、收盤價矩陣與絕對時間軸
load(cachePath, 'Prices_Active', 'Expert_Active', 'Dates_Active');
% 載入 Phase 3 產出的 GBDT 大盤崩盤機率與個股勝率預測矩陣
load(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all');

% 剝除時間軸的時區標籤，防禦後續 datetime 比較時的時區衝突
Dates_Active.TimeZone = ''; 

% 1B. 向 DataFetcher 即時索取 T+1 開盤交易所需的 Opens 矩陣 (極速映射)
fetcher = DataFetcher(configObj);
dataStruct = fetcher.fetch_data();
Opens_Raw = dataStruct.Opens;

% 提取時間軸總長度、標的總數與時序特徵回溯長度 (預設 60 天)
numDaysRaw = length(Dates_Active);
numT = configObj.NumTickers;
seqLen = configObj.SeqLen;

% 確保系統有足夠的歷史特徵，捨棄最前面的序列暖機期 (Warm-up Period)，避免 NaN 污染
valid_idx = seqLen : numDaysRaw;
Prices_Active = Prices_Active(valid_idx, :);
Opens_Active  = Opens_Raw(valid_idx, :);
Expert_Active = Expert_Active(valid_idx, :);
Dates_Active  = Dates_Active(valid_idx);

%% 2. 切分 In-Sample (IS) 訓練區間
disp('--- 步驟 2：切分 In-Sample 訓練區間 ---');
Train_Start_Date = datetime('2006-01-01');
OOS_Start_Date   = datetime('2022-01-01');

% 限制 IS 訓練樣本的下界與上界，建立時間結界防禦 Look-ahead Bias
idx_IS = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);

% 對崩盤機率進行 20 日移動平均 (MA20) 平滑處理
P_crash_smooth = movmean(P_crash_all, [19, 0]);

% 萃取 IS 區間的實體數據矩陣
Prices_IS = Prices_Active(idx_IS, :);
Opens_IS  = Opens_Active(idx_IS, :);
Expert_IS = Expert_Active(idx_IS, :);

% 萃取平滑後的 IS 崩盤機率，並轉置為橫列向量以利後續計算
P_crash_IS = P_crash_smooth(valid_idx(idx_IS))'; 

% 將個股預測機率轉置為 [Tickers, Days]，優化後續迴圈在記憶體中的提取連續性 (Cache Hit Rate)
P_time_IS  = P_time_all(valid_idx(idx_IS), :)';
P_space_IS = P_space_all(valid_idx(idx_IS), :)';
Dates_IS   = Dates_Active(idx_IS);

%% 3. 定義貝氏最佳化超參數空間 (擴大邊界版)
disp('--- 步驟 3：定義貝氏最佳化超參數空間 (擴大邊界與邊界安全加固版) ---');
% ★ 二次優化修正 1：放寬下界至 p_50 (中位數)，避免搜索空間被卡死在 75 百分位數
p_50 = prctile(P_crash_IS, 50);
p_99 = prctile(P_crash_IS, 99);

if isnan(p_50) || isnan(p_99) || p_50 >= p_99
    p_50 = 0.30;
    p_99 = 0.95;
end

if (p_99 - p_50) < 0.05
    p_99 = min(1.0, p_50 + 0.05);
end
fprintf('  -> 護欄動態邊界鎖定：[%.4f, %.4f]\n', p_50, p_99);

% 宣告 BayesOpt 的三大決策變數：
% 1. 崩盤護欄閾值 (放寬下界至 p_50)
var_guard  = optimizableVariable('Guardrail_CrashProb', [p_50, p_99], 'Type', 'real');
% 2. 時序專家資金權重 (★ 二次優化修正 2：完全開放至 [0.0, 1.0])
var_weight = optimizableVariable('Expert_Time_Weight', [0.0, 1.0], 'Type', 'real');
% 3. 持股集中度 Top_K (離散型整數)
var_topk   = optimizableVariable('Top_K_Assets', [10, 40], 'Type', 'integer');
bo_vars    = [var_guard, var_weight, var_topk];

%% 4. 啟動 BayesOpt 尋優引擎
disp('--- 步驟 4：啟動 BayesOpt 尋優引擎 (無假設微型實境模擬器 - i5多核全開) ---');
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp('⚡ 偵測到尚未開啟平行池，正在喚醒 CPU 多核心 (parpool)...');
    parpool('Processes');
end

% 封裝代理目標函數
obj_fun = @(x) evaluate_hrl_proxy(x, Prices_IS, Opens_IS, Expert_IS, P_time_IS, P_space_IS, P_crash_IS, configObj);

% 啟動貝氏優化器
results = bayesopt(obj_fun, bo_vars, ...
    'MaxObjectiveEvaluations', 25, ...                       
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'UseParallel', true, ...                                
    'Verbose', 1);

%% 5. 提取最佳參數、循環斷路器診斷與存檔
disp('--- 步驟 5：提取最佳架構參數與循環斷路器檢驗 ---');
if isempty(results.XAtMinObjective) || height(results.XAtMinObjective) == 0
    best_params = table(0.85, 0.5, 20, 'VariableNames', {'Guardrail_CrashProb', 'Expert_Time_Weight', 'Top_K_Assets'});
    warning('⚠️ 優化未收斂，將自動採用預設值');
else
    best_params = results.XAtMinObjective;
end

best_IR = -results.MinObjective;

% 輸出最終尋優結果報表
fprintf('\n🏆 【貝氏最佳化尋優結果】 🏆\n');
fprintf('  > 最佳 崩盤護欄閥值 (Guardrail) : %.4f\n', best_params.Guardrail_CrashProb);
fprintf('  > 最佳 時序專家權重 (Time_W)  : %.4f\n', best_params.Expert_Time_Weight);
fprintf('  > 最佳 集中度標的數 (Top_K)   : %d\n', best_params.Top_K_Assets);
fprintf('  > 最佳 資訊比率 (Best IR)    : %.4f\n', best_IR);

% ★ 二次優化修正 3：加入客觀函數循環斷路器 (Circuit Breaker)
if results.MinObjective > 0
    warning(['\n⚠️ =========================================================================\n' ...
             '⚠️ 【循環斷路器觸發警告】\n' ...
             '   貝氏最佳化在整個搜索空間內未找到任何正 IR 的參數組合 (最佳 IR = %.4f)！\n' ...
             '   這代表當前選股訊號 (P_time / P_space) 在微型實境模擬器中本身不具備超額 Alpha，\n' ...
             '   根因在於 Phase 2 (DL 萃取器) 或 Phase 3 (GBDT 專家) 的訊號品質，而非超參數配置。\n' ...
             '   建議暫停執行 Phase 5 與 Phase 6，優先排查並修復 Phase 2/3！\n' ...
             '⚠️ =========================================================================\n'], best_IR);
else
    disp('✅ 尋優成功！在當前搜索空間中捕獲正超額收益組合 (IR > 0)。');
end

% 將最佳參數實時寫回全域 Config 物件中
configObj.Guardrail_CrashProb = best_params.Guardrail_CrashProb;

% 將優化結果快取落地，供後續強化學習階段 (Phase 5) 調用
boPath = fullfile(configObj.ModelDir, 'BO_Hyperparameters.mat');
save(boPath, 'best_params', 'results');
disp('✅ Phase 4 完成！3D 空間超參數已鎖定，請接著執行 Phase 5。');

%% =====================================================================
% 決定性代理評估函數 (★ 實戰物理鐵律版：包含 Drift、Open 換倉、停牌鎖死)
% =====================================================================
function neg_IR = evaluate_hrl_proxy(params, Prices, Opens, Expert, P_time, P_space, P_crash, config)
    try
        numDays = size(Prices, 1); 
        numTickers = size(Prices, 2);
        
        spy_idx = find(strcmp(config.IdxTickers, 'SPY'));
        if isempty(spy_idx)
            error('❌ 代理函數中找不到基準標的 SPY，無法計算摩擦力與護欄！');
        end
        
        start_t = 21; 
        steps = numDays - start_t - 2; 
        
        prev_assets = zeros(numTickers, 1, 'single'); 
        prev_cash = 1.0; 
        excess_returns = zeros(steps, 1, 'single');
        
        spy_prices = Prices(:, spy_idx);
        spy_rets = [0; diff(spy_prices) ./ spy_prices(1:end-1)];
        spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
        vol20 = movstd(spy_rets, [19, 0], 1) * sqrt(252);
        
        opt_guard = params.Guardrail_CrashProb;
        w_time = params.Expert_Time_Weight;
        w_space = 1.0 - w_time; 
        top_k = round(params.Top_K_Assets);
        
        for step_idx = 1:steps
            t = start_t + step_idx - 1;
            
            % 鐵律 1：投資組合權重漂移 (Weight Drift)
            drift_ret = (Opens(t+1, :) - Opens(t, :)) ./ Opens(t, :);
            drift_ret(isnan(drift_ret) | isinf(drift_ret)) = 0; 
            
            asset_mult = prev_assets .* (1 + drift_ret');
            port_val = sum(asset_mult) + prev_cash; 
            
            if port_val > 0
                w_drift = asset_mult / port_val;
            else
                w_drift = prev_assets;
            end
            
            % 鐵律 2：確認流動性，標記停牌鎖死 (Liquidity Halted Mask)
            halted_mask = isnan(Opens(t+1, :))' | (Opens(t+1, :) <= 0)';
            locked_weights = zeros(numTickers, 1, 'single');
            locked_weights(halted_mask) = w_drift(halted_mask);
            locked_sum = sum(locked_weights);
            
            available_cap = max(0, 1.0 - locked_sum);
            
            % 鐵律 3：T+1 開盤資金重分配與 GBDT 硬熔斷
            target_cash = 0.0; 
            if P_crash(t) > opt_guard
                target_cash = 1.0; 
            end
            
            actual_cash_target = min(target_cash, available_cap);
            rem_cap_for_assets = available_cap - actual_cash_target;
            
            comb_p = P_time(:, t) * w_time + P_space(:, t) * w_space;
            comb_p = comb_p .* Expert(t, :)';
            comb_p(halted_mask) = 0; 
            
            % Top-K 集中度篩選
            if sum(comb_p > 0) > top_k
                [~, sort_idx] = sort(comb_p, 'descend');
                comb_p(comb_p < comb_p(sort_idx(top_k))) = 0;
            end
            
            if sum(comb_p) > 0
                comb_p = comb_p ./ sum(comb_p); 
            else
                comb_p(:) = 0; 
            end
            
            asset_weights = comb_p .* rem_cap_for_assets;
            asset_weights(halted_mask) = locked_weights(halted_mask);
            
            % 鐵律 4：嚴格統一摩擦力計算 (讀取 Config.m 統一參數)
            current_vol = vol20(t);
            if isnan(current_vol) || isinf(current_vol), current_vol = 0; end
            
            if isprop(config, 'BaseFrictionFee') && isprop(config, 'SlippageVolCoeff')
                tc_rate = config.BaseFrictionFee + (config.SlippageVolCoeff * current_vol);
            else
                tc_rate = 0.0005 + (0.10 * current_vol);
            end
            
            frict_cost = sum(abs(asset_weights(~halted_mask) - w_drift(~halted_mask))) * tc_rate;
            
            % 鐵律 5：實體驗收報酬 (Open-to-Open Forward Return)
            fwd_ret = (Opens(t+2, :) - Opens(t+1, :)) ./ Opens(t+1, :);
            fwd_ret(isnan(fwd_ret) | isinf(fwd_ret)) = 0;
            
            port_ret = sum(asset_weights .* fwd_ret') - frict_cost;
            
            spy_ret = (Opens(t+2, spy_idx) - Opens(t+1, spy_idx)) / Opens(t+1, spy_idx);
            if isnan(spy_ret) || isinf(spy_ret), spy_ret = 0; end
            
            excess_returns(step_idx) = port_ret - spy_ret;
            
            prev_assets = asset_weights; 
            prev_cash = actual_cash_target;
        end
        
        avg_excess = mean(excess_returns);
        std_excess = std(excess_returns);
        
        if std_excess < 1e-6
            neg_IR = 1e6; 
        else
            IR = (avg_excess / std_excess) * sqrt(252);
            neg_IR = -IR; 
        end
    catch ME
        disp(ME.message);
        neg_IR = 1e6; 
    end
end