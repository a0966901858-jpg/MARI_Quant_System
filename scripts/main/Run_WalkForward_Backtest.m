% =========================================================================
% 腳本：6_Run_WalkForward_Backtest.m
% 升級：Phase 15.5 生產基準版 (★ 徹底剔除 PPO 強化學習消除牛市現金拖累、
%       Pure-Time 剪枝防訊號稀釋、Phase 4 BO 最佳化參數直接接管資產配置、
%       Config.Horizon/RebalanceStride 調倉步進動態同構、mrg32k3a 確定性串流、
%       Open-to-Open 權重自然漂移與停牌鎖死防護、死區連續縮放護欄、
%       OOS 盲測期獨立歸一化起算與水下回撤歸零結算、全流程高精度計時與時長審計)
% 職責：執行嚴格的因果律前向回測，產出無前視偏差的 IS/OOS 真實績效、交易軌跡與診斷報表
% =========================================================================
clear; clc; close all;

% 啟動全域總計時器
t_total_start = tic;
disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 MARI 嚴格前向滾動回測管線 (規則優化與無RL極速版)');
disp('=================================================================');

%% 0. 環境路徑掛載與隨機串流管理
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

% 使用 Config 統一初始化主執行緒 mrg32k3a 隨機數引擎 (Substream = 1)
stream = configObj.getRandStream(1);
RandStream.setGlobalStream(stream);
disp('🔒 已掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定前向回測環境。');
time_step0 = toc(t_step0);
fprintf('⏱️ [步驟 0 完成] 耗時: %.2f 秒\n\n', time_step0);

%% 1. 載入全域快取、DataFetcher 與模型權重
t_step1 = tic;
disp('--- 步驟 1：載入全域快取、DataFetcher 開盤價與模型參數 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');

if ~exist(cachePath, 'file') || ~exist(gbdtPath, 'file')
    error('❌ 找不到前置快取，請確認 Phase 1 至 Phase 3 已成功執行！');
end

load(cachePath, 'Prices_Active', 'Expert_Active', 'Dates_Active');
if isempty(Dates_Active.TimeZone)
    Dates_Active.TimeZone = 'UTC';
end
tz = Dates_Active.TimeZone;

load(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all');

% ★ 空間專家狀態探針：檢查是否剪枝空間專家 (DyGAT)
enable_space = isprop(configObj, 'EnableSpaceExpertTraining') && configObj.EnableSpaceExpertTraining && any(P_space_all(:) ~= 0);
if enable_space
    disp('  -> 空間專家狀態: 【已啟用】(橫截面選股將綜合時序與空間表徵)');
else
    disp('  -> 空間專家狀態: 【⏩ 已剪枝】(Pure-Time 模式：強制時序權重=1.0，杜絕排序稀釋)');
end

% ★ 優先載入 Phase 4 貝氏最佳化通過 DSR 顯著性之正式生產參數
prod_bo_path = fullfile(configObj.ModelDir, 'BO_Hyperparameters.mat');
if exist(prod_bo_path, 'file')
    try
        bo_data = load(prod_bo_path);
        configObj.Guardrail_CrashProb = bo_data.best_params.Guardrail_CrashProb;
        configObj.Expert_Time_Weight  = bo_data.best_params.Expert_Time_Weight;
        configObj.Top_K_Assets        = bo_data.best_params.Top_K_Assets;
        fprintf('  🎯 [Phase 4 參數載入] 成功繼承通過 DSR 檢定 (DSR=%.4f) 之 BO 最佳參數：\n', bo_data.dsr_val);
        fprintf('     > 崩盤護欄閾值: %.4f | 時序專家權重: %.4f | 集中度 Top-K: %d\n', ...
            configObj.Guardrail_CrashProb, configObj.Expert_Time_Weight, configObj.Top_K_Assets);
    catch ME
        warning('⚠️ 載入 BO 參數失敗 (%s)，將使用 Config 基準參數。', ME.message);
    end
else
    disp('  ℹ️ 未檢測到正式 BO_Hyperparameters.mat，回測將使用 Config 預設基準配置。');
end

% ★ 決策架構徹底剔除 PPO 強化學習 (Phase 5)
% 依據實證與 P2-3 消融結論，PPO 在低信噪比環境中難以產生有效超額，常態性持有 50%+ 現金引發嚴重牛市拖累；
% 本管線全面剔除 PPO Agent 載入，資產配置 100% 由 Phase 4 BO 最優規則與連續死區護欄接管。
has_cio_agents = false;
disp('  🏛️ [決策架構] 已徹底剔除 PPO 強化學習，全面採用 Phase 4 貝氏最佳化規則與連續死區護欄。');

fetcher = DataFetcher(configObj);
dataStruct = fetcher.fetch_data();
Opens_Raw = dataStruct.Opens;
seqLen = configObj.SeqLen;
numDaysRaw = length(Dates_Active);
valid_idx = seqLen : numDaysRaw;

% 對崩盤機率進行 20 日移動平均平滑
P_crash_smooth_all = movmean(P_crash_all, [19, 0]);
P_crash_smooth = P_crash_smooth_all(valid_idx);

Prices_Active = Prices_Active(valid_idx, :);
Opens_Active  = Opens_Raw(valid_idx, :);
Expert_Active = Expert_Active(valid_idx, :);
Dates_Active  = Dates_Active(valid_idx);

P_time_M  = P_time_all(valid_idx, :)'; 
P_space_M = P_space_all(valid_idx, :)';
numDays   = length(Dates_Active);
numTickers = configObj.NumTickers;

spy_idx = find(strcmp(configObj.IdxTickers, 'SPY'));
if isempty(spy_idx), error('❌ 找不到 SPY 基準標的！'); end

time_step1 = toc(t_step1);
fprintf('⏱️ [步驟 1 完成] 耗時: %.2f 秒\n\n', time_step1);

%% 2. 預計算市場宏觀狀態與動態波動度
t_step2 = tic;
disp('--- 步驟 2：預計算市場波動度與宏觀指標 ---');
spy_prices = Prices_Active(:, spy_idx);
spy_rets = [0; diff(spy_prices) ./ (spy_prices(1:end-1) + 1e-8)];
spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
vol20 = movstd(spy_rets, [19, 0], 1) * sqrt(252);

time_step2 = toc(t_step2);
fprintf('⏱️ [步驟 2 完成] 耗時: %.2f 秒\n\n', time_step2);

%% 3. 定義嚴格時間邊界與崩盤護欄死區校準
t_step3 = tic;
disp('--- 步驟 3：時間邊界劃分與校準崩盤護欄死區 ---');
spy_inception_idx = find(spy_prices > 10, 1);
if isempty(spy_inception_idx), spy_inception_idx = 252; end

Train_Start_Date = datetime('2006-01-01', 'TimeZone', tz);
idx_train_start = find(Dates_Active >= Train_Start_Date, 1);
valid_start_t = max(spy_inception_idx + 252, idx_train_start); 

OOS_Date = datetime('2022-01-01', 'TimeZone', tz);
idx_OOS_start = find(Dates_Active >= OOS_Date, 1);

port_values = ones(numDays, 1, 'single');
spy_values  = ones(numDays, 1, 'single');
cash_ratios = zeros(numDays, 1, 'single');
tc_records  = zeros(numDays, 1, 'single');
prev_assets = zeros(numTickers, 1, 'single');
prev_cash   = 1.0;

opt_guard       = configObj.Guardrail_CrashProb;
top_k           = configObj.Top_K_Assets;
fallback_w_time = configObj.Expert_Time_Weight;
base_frict      = configObj.MoE_FrictionMask; 
Verbose_Log     = true; 
is_bankrupt     = false; 

% 動態繼承全域調倉步進 (未指定時動態對齊 Horizon，杜絕硬編碼)
if isprop(configObj, 'RebalanceStride') && ~isempty(configObj.RebalanceStride)
    rebalance_stride = configObj.RebalanceStride;
else
    rebalance_stride = configObj.Horizon;
end

cached_stock_props = zeros(numTickers, 1, 'single');

% 提取 IS 期間 75% 雜訊分位數作為死區下限
vars_gbdt = who('-file', gbdtPath);
if ismember('Y_Crash_all', vars_gbdt)
    data_crash = load(gbdtPath, 'Y_Crash_all');
    Y_Crash_valid = data_crash.Y_Crash_all(valid_idx);
elseif ismember('Y_Crash_1D', vars_gbdt)
    data_crash = load(gbdtPath, 'Y_Crash_1D');
    Y_Crash_valid = data_crash.Y_Crash_1D(valid_idx);
else
    spy_fwd10 = zeros(numDays, 1, 'single');
    spy_fwd10(1:end-10) = (spy_prices(11:end) - spy_prices(1:end-10)) ./ (spy_prices(1:end-10) + 1e-8);
    Y_Crash_valid = (spy_fwd10 < -0.05);
end

is_days = valid_start_t : (idx_OOS_start - 1);
is_non_crash_idx = is_days(Y_Crash_valid(is_days) == 0);
non_crash_p = P_crash_smooth(is_non_crash_idx);
tau_noise = prctile(non_crash_p, 75);

guard_high = opt_guard;
guard_low  = min(guard_high * 0.85, max(tau_noise, guard_high - 0.03));
if guard_low >= guard_high
    guard_low = guard_high * 0.85;
end

fprintf(' 🔍 [護欄與調倉參數校準]\n');
fprintf('    > IS 非危機期 P(Crash) 75%% 雜訊分位數 (tau_noise) : %.4f\n', tau_noise);
fprintf('    > 硬熔斷上限 (Guard_High)                      : %.4f\n', guard_high);
fprintf('    > 死區縮放下限 (Guard_Low)                       : %.4f\n', guard_low);
fprintf('    > 有效連續緩衝區間寬度                           : %.4f\n', guard_high - guard_low);
fprintf('    > 統一調倉步進 (RebalanceStride)                : 每 %d 個交易日定期換手 (對齊 Horizon=%d)\n', ...
    rebalance_stride, configObj.Horizon);

sample_vol_daily = mean(vol20) / sqrt(252);
sample_tc = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff * sample_vol_daily);
fprintf(' 📊 [成本模型檢查] 平均日波動度: %.4f%% | 預期平均換手成本率: %.4f%%\n', ...
    sample_vol_daily*100, sample_tc*100);

time_step3 = toc(t_step3);
fprintf('⏱️ [步驟 3 完成] 耗時: %.2f 秒\n\n', time_step3);

%% 4. 執行逐日回測 (定期調倉 + Open-to-Open 權重漂移 + 停牌鎖死)
t_step4 = tic;
disp('--- 步驟 4：啟動逐日推論與交易結算 (同構前向回測) ---');
fprintf(' 📡 回測正式起點：%s\n', datestr(Dates_Active(valid_start_t)));
port_values(valid_start_t)   = 1.0;
port_values(valid_start_t+1) = 1.0;
spy_values(valid_start_t)    = 1.0;
spy_values(valid_start_t+1)  = 1.0;

for t = valid_start_t : numDays - 2
    step_idx = t - valid_start_t + 1;
    
    % 破產防護機制
    if port_values(t+1) < 0.05
        is_bankrupt = true;
    end
    
    if is_bankrupt
        port_values(t+2) = port_values(t+1);
        spy_ret_bench = (Opens_Active(t+2, spy_idx) - Opens_Active(t+1, spy_idx)) / (Opens_Active(t+1, spy_idx) + 1e-8);
        if isnan(spy_ret_bench) || isinf(spy_ret_bench), spy_ret_bench = 0; end
        spy_values(t+2) = spy_values(t+1) * (1 + spy_ret_bench);
        cash_ratios(t+1) = 1.0;
        continue;
    end
    
    % -------------------------------------------------------------
    % 1. 計算資產權重自然漂移 (Weight Drift from t Open to t+1 Open)
    % -------------------------------------------------------------
    drift_ret = (Opens_Active(t+1, :) - Opens_Active(t, :)) ./ (Opens_Active(t, :) + 1e-8);
    drift_ret(isnan(drift_ret) | isinf(drift_ret)) = 0;
    
    asset_mult = prev_assets .* (1 + drift_ret');
    port_val_drift = sum(asset_mult) + prev_cash;
    
    if port_val_drift > 0
        w_drift = asset_mult / port_val_drift;
    else
        w_drift = prev_assets;
    end
    
    % -------------------------------------------------------------
    % 2. 停牌鎖死防護 (Halted Stocks Cannot Be Traded)
    % -------------------------------------------------------------
    halted_mask = isnan(Opens_Active(t+1, :))' | (Opens_Active(t+1, :) <= 0)';
    locked_weights = zeros(numTickers, 1, 'single');
    locked_weights(halted_mask) = w_drift(halted_mask);
    locked_sum = sum(locked_weights);
    
    available_cap = max(0, 1.0 - locked_sum);
    
    % -------------------------------------------------------------
    % 3. 策略決策：專家權重 (時序專家 100% 保真傳遞)
    % -------------------------------------------------------------
    if ~enable_space
        w_time  = 1.0;
        w_space = 0.0;
    else
        w_time  = fallback_w_time;
        w_space = 1.0 - fallback_w_time;
    end
    
    % -------------------------------------------------------------
    % 4. 崩盤護欄死區連續縮放 (由 Phase 4 BO 參數精確裁決)
    %    ★ 關鍵優化：常態非危機時期基準現金鎖定 0.0，杜絕牛市現金拖累；
    %    僅在宏觀 P(Crash) 突破死區下限時，動態連續拉升現金防禦水位。
    % -------------------------------------------------------------
    p_c = P_crash_smooth(t);
    if p_c <= guard_low
        risk_scale = 0.0;
    elseif p_c >= guard_high
        risk_scale = 1.0;
    else
        risk_scale = (p_c - guard_low) / (guard_high - guard_low);
    end
    
    actual_cash_target = min(risk_scale, available_cap);
    rem_cap_for_assets = available_cap - actual_cash_target;
    
    % -------------------------------------------------------------
    % 5. 定期調倉 vs. 被動價格漂移
    % -------------------------------------------------------------
    is_rebal_day = (step_idx == 1) || (mod(step_idx - 1, rebalance_stride) == 0);
    
    if is_rebal_day
        % 定期調倉日：重新計算專家排序並挑選 Top-K
        if enable_space
            comb_p = P_time_M(:, t) * w_time + P_space_M(:, t) * w_space;
        else
            comb_p = P_time_M(:, t); % 純時序保真傳遞，無雜訊稀釋
        end
        comb_p = comb_p .* Expert_Active(t, :)';
        comb_p(halted_mask) = 0;
        
        if sum(comb_p > 0) > top_k
            [~, sort_idx] = sort(comb_p, 'descend');
            comb_p(comb_p < comb_p(sort_idx(top_k))) = 0;
        end
        
        if sum(comb_p) > 0
            cached_stock_props = comb_p / sum(comb_p);
        else
            cached_stock_props(:) = 0;
        end
        target_active_weights = cached_stock_props .* rem_cap_for_assets;
    else
        % 非調倉日：資產部位隨價格自然漂移；若防禦水位變更，等比例縮放活躍持股
        curr_active_sum = sum(w_drift(~halted_mask));
        if curr_active_sum > 1e-6 && rem_cap_for_assets > 0
            scale_ratio = rem_cap_for_assets / curr_active_sum;
            target_active_weights = w_drift .* scale_ratio;
            target_active_weights(halted_mask) = 0;
        elseif rem_cap_for_assets <= 0
            target_active_weights = zeros(numTickers, 1, 'single');
        else
            target_active_weights = cached_stock_props .* rem_cap_for_assets;
        end
    end
    
    asset_w = target_active_weights;
    asset_w(halted_mask) = locked_weights(halted_mask);
    
    % -------------------------------------------------------------
    % 6. 慣性摩擦過濾 (Inertia Friction Mask)
    % -------------------------------------------------------------
    turnover = abs(asset_w(~halted_mask) - w_drift(~halted_mask));
    ignore_sub = turnover < base_frict;
    unhalted_idx = find(~halted_mask);
    asset_w(unhalted_idx(ignore_sub)) = w_drift(unhalted_idx(ignore_sub));
    
    tot_w = sum(asset_w) + actual_cash_target;
    if tot_w > 0
        asset_w = asset_w / tot_w;
        w_cash  = actual_cash_target / tot_w;
    else
        w_cash = 1.0;
        asset_w(:) = 0;
    end
    
    cash_ratios(t+1) = w_cash;
    
    % -------------------------------------------------------------
    % 7. 交易摩擦成本計算 (日頻動態波動滑價模型)
    % -------------------------------------------------------------
    current_vol_daily = vol20(t) / sqrt(252);
    if isprop(configObj, 'BaseFrictionFee') && isprop(configObj, 'SlippageVolCoeff')
        tc_rate = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff * current_vol_daily);
    else
        tc_rate = 0.0005 + (0.10 * current_vol_daily);
    end
    
    % 僅對可交易標的主動換手收取成本 (非調倉日無主動操作時換手自然為 0)
    cost = sum(abs(asset_w(~halted_mask) - w_drift(~halted_mask))) * tc_rate;
    tc_records(t+1) = cost;
    
    % -------------------------------------------------------------
    % 8. 結算 Open(t+1) 至 Open(t+2) 跨日資產報酬
    % -------------------------------------------------------------
    ret_t1 = (Opens_Active(t+2, :) - Opens_Active(t+1, :)) ./ (Opens_Active(t+1, :) + 1e-8);
    ret_t1(isnan(ret_t1) | isinf(ret_t1)) = 0;
    
    spy_ret_t1 = (Opens_Active(t+2, spy_idx) - Opens_Active(t+1, spy_idx)) / (Opens_Active(t+1, spy_idx) + 1e-8);
    if isnan(spy_ret_t1) || isinf(spy_ret_t1), spy_ret_t1 = 0; end
    
    port_ret = sum(asset_w .* ret_t1') - cost;
    
    port_values(t+2) = port_values(t+1) * (1 + port_ret);
    spy_values(t+2)  = spy_values(t+1) * (1 + spy_ret_t1);
    
    prev_assets = asset_w;
    prev_cash   = w_cash;
    
    if Verbose_Log && mod(t, 252) == 0
        num_holdings = sum(asset_w > 0.001);
        fprintf('[%s] SPY: %+5.2f%% | MARI: %+5.2f%% | 現金: %5.1f%% | 持股: %2d 檔 | 換手成本: %.3f%% | 累計淨值: %.2f\n', ...
            datestr(Dates_Active(t+2)), spy_ret_t1*100, port_ret*100, w_cash*100, num_holdings, cost*100, port_values(t+2));
    end
end
cash_ratios(numDays) = w_cash;
time_step4 = toc(t_step4);
fprintf('⏱️ [步驟 4 完成] 耗時: %.2f 秒 (%.2f 分鐘)\n\n', time_step4, time_step4 / 60);

%% 5. 機構級績效結算 (IS 與 OOS 嚴格隔離)
t_step5 = tic;
disp('--- 步驟 5：嚴格 IS / OOS 績效分離結算 ---');
is_idx  = (valid_start_t + 1) : idx_OOS_start;
oos_idx = idx_OOS_start : numDays;

is_port  = port_values(is_idx);
is_spy   = spy_values(is_idx);
oos_port = port_values(oos_idx);
oos_spy  = spy_values(oos_idx);

is_m   = evaluate_financial_metrics(is_port, is_spy);
oos_m  = evaluate_financial_metrics(oos_port, oos_spy);
full_m = evaluate_financial_metrics(port_values(valid_start_t+1:numDays), spy_values(valid_start_t+1:numDays));

fprintf('\n====================================================================================================\n');
fprintf('📊 【MARI Quant System 機構級前向回測綜合報告 (Phase 15.5 生產基準版)】\n');
fprintf('====================================================================================================\n');
fprintf(' 區間劃分        | 累積總報酬 | 年化報酬(CAGR) | 年化波動度 | 最大回撤(MDD) | 夏普比率 | 卡瑪比率 | 資訊比率(IR) | 日勝率\n');
fprintf('----------------------------------------------------------------------------------------------------\n');
fprintf(' IS 訓練期(MARI) |  %+8.2f%% |     %+6.2f%%   |   %5.2f%%  |    %6.2f%%   |  %6.2f  |  %6.2f  |    %+6.2f    | %5.1f%%\n', ...
    is_m.TotalRet, is_m.CAGR, is_m.AnnVol, is_m.MDD, is_m.Sharpe, is_m.Calmar, is_m.IR, is_m.WinRate);
fprintf(' IS 基準線(SPY)  |  %+8.2f%% |     %+6.2f%%   |   %5.2f%%  |    %6.2f%%   |  %6.2f  |  %6.2f  |      N/A     | %5.1f%%\n', ...
    is_m.BenchTotalRet, is_m.BenchCAGR, is_m.BenchAnnVol, is_m.BenchMDD, is_m.BenchSharpe, is_m.BenchCalmar, is_m.BenchWinRate);
fprintf('----------------------------------------------------------------------------------------------------\n');
fprintf(' OOS 盲測(MARI)  |  %+8.2f%% |     %+6.2f%%   |   %5.2f%%  |    %6.2f%%   |  %6.2f  |  %6.2f  |    %+6.2f    | %5.1f%%\n', ...
    oos_m.TotalRet, oos_m.CAGR, oos_m.AnnVol, oos_m.MDD, oos_m.Sharpe, oos_m.Calmar, oos_m.IR, oos_m.WinRate);
fprintf(' OOS 基準線(SPY) |  %+8.2f%% |     %+6.2f%%   |   %5.2f%%  |    %6.2f%%   |  %6.2f  |  %6.2f  |      N/A     | %5.1f%%\n', ...
    oos_m.BenchTotalRet, oos_m.BenchCAGR, oos_m.BenchAnnVol, oos_m.BenchMDD, oos_m.BenchSharpe, oos_m.BenchCalmar, oos_m.BenchWinRate);
fprintf('----------------------------------------------------------------------------------------------------\n');
fprintf(' 全歷史累計(MARI)|  %+8.2f%% |     %+6.2f%%   |   %5.2f%%  |    %6.2f%%   |  %6.2f  |  %6.2f  |    %+6.2f    | %5.1f%%\n', ...
    full_m.TotalRet, full_m.CAGR, full_m.AnnVol, full_m.MDD, full_m.Sharpe, full_m.Calmar, full_m.IR, full_m.WinRate);
fprintf('====================================================================================================\n\n');
time_step5 = toc(t_step5);
fprintf('⏱️ [步驟 5 完成] 耗時: %.2f 秒\n\n', time_step5);

%% 6. 繪製視覺化診斷報表 (OOS 淨值獨立歸一化起算 & 回撤獨立歸零)
t_step6 = tic;
disp('--- 步驟 6：生成機構級視覺化報表 (白底黑字 - OOS 獨立歸一化與回撤歸零版) ---');
fig_wf = figure('Name', 'MARI Quant Walk-Forward Backtest', ...
    'Color', 'w', 'Position', [100, 100, 1250, 1000], 'Visible', 'off');
set(fig_wf, 'InvertHardcopy', 'off');

% 計算 IS 與 OOS 分別從 1.0 開始起算的淨值曲線
is_port_norm  = is_port  ./ (is_port(1) + 1e-8);
is_spy_norm   = is_spy   ./ (is_spy(1) + 1e-8);
oos_port_norm = oos_port ./ (oos_port(1) + 1e-8);
oos_spy_norm  = oos_spy  ./ (oos_spy(1) + 1e-8);

% 水下回撤嚴格分離計算：OOS 在 2022 年起點強制歸零，杜絕繼承歷史高點
is_mari_dd  = (is_port_norm  - cummax(is_port_norm))  ./ (cummax(is_port_norm)  + 1e-8) * 100;
is_spy_dd   = (is_spy_norm   - cummax(is_spy_norm))   ./ (cummax(is_spy_norm)   + 1e-8) * 100;
oos_mari_dd = (oos_port_norm - cummax(oos_port_norm)) ./ (cummax(oos_port_norm) + 1e-8) * 100;
oos_spy_dd  = (oos_spy_norm  - cummax(oos_spy_norm))  ./ (cummax(oos_spy_norm)  + 1e-8) * 100;

% 子圖 1：對數淨值曲線 (IS 與 OOS 各自獨立從 1.0 起算)
subplot(4, 1, 1);
plot(Dates_Active(is_idx), log10(is_port_norm), 'LineWidth', 1.6, 'Color', '#D95319', 'DisplayName', 'MARI (IS, Base=1.0)'); hold on;
plot(Dates_Active(is_idx), log10(is_spy_norm), 'LineWidth', 1.3, 'Color', '#0072BD', 'DisplayName', 'SPY (IS, Base=1.0)');
plot(Dates_Active(oos_idx), log10(oos_port_norm), 'LineWidth', 2.0, 'Color', '#A2142F', 'DisplayName', 'MARI (OOS, Reset Base=1.0)');
plot(Dates_Active(oos_idx), log10(oos_spy_norm), 'LineWidth', 1.4, 'Color', '#4DBEEE', 'DisplayName', 'SPY (OOS, Reset Base=1.0)');
xline(Dates_Active(idx_OOS_start), '--k', 'OOS Start (Rebased to 1.0)', 'LineWidth', 1.3, ...
    'LabelVerticalAlignment', 'bottom', 'Color', 'k', 'FontName', 'Helvetica', 'FontWeight', 'bold');
yline(0, ':k', 'Wealth = 1.0', 'LineWidth', 1.0, 'HandleVisibility', 'off');
title(sprintf('Log-Scale Cumulative Equity Curve (IS & OOS Rebased to 1.0, Stride=%dD)', rebalance_stride), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Log_{10}(Wealth)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% 子圖 2：水下回撤圖 (%) (IS 與 OOS 嚴格分離，OOS 獨立歸零)
subplot(4, 1, 2);
area(Dates_Active(is_idx), is_mari_dd, 'FaceColor', '#D95319', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'DisplayName', 'MARI DD (IS)'); hold on;
plot(Dates_Active(is_idx), is_spy_dd, 'Color', '#0072BD', 'LineWidth', 1.1, 'DisplayName', 'SPY DD (IS)');
area(Dates_Active(oos_idx), oos_mari_dd, 'FaceColor', '#A2142F', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'DisplayName', 'MARI DD (OOS Reset)');
plot(Dates_Active(oos_idx), oos_spy_dd, 'Color', '#4DBEEE', 'LineWidth', 1.2, 'DisplayName', 'SPY DD (OOS Reset)');
xline(Dates_Active(idx_OOS_start), '--k', 'OOS Start (DD Reset 0%)', 'LineWidth', 1.2, 'Color', 'k', ...
    'LabelVerticalAlignment', 'bottom', 'FontName', 'Helvetica', 'FontWeight', 'bold');
yline(0, '-k', 'LineWidth', 1.0, 'HandleVisibility', 'off');
title('Portfolio Underwater Drawdown Profile (%) [IS & OOS Separated, OOS Re-anchored to 0%]', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Drawdown (%)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'southwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% 子圖 3：動態避險現金比例 (%)
subplot(4, 1, 3);
full_dates = Dates_Active(valid_start_t+1:numDays);
area(full_dates, cash_ratios(valid_start_t+1:numDays) * 100, ...
    'FaceColor', '#77AC30', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Cash Ratio (%)'); hold on;
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 1.2, 'Color', 'k');
title('Dynamic Cash Allocation & Continuous Risk Scaling', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Cash (%)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylim([0, 105]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% 子圖 4：平滑崩盤機率與護欄診斷
subplot(4, 1, 4);
plot(full_dates, P_crash_smooth(valid_start_t+1:numDays), ...
    'Color', '#7E2F8E', 'LineWidth', 1.3, 'DisplayName', 'Smoothed P(Crash)'); hold on;
yline(guard_high, '--r', sprintf('Hard Hedge (%.3f)', guard_high), 'LineWidth', 1.3);
yline(guard_low, ':r', sprintf('Deadband Floor (%.3f)', guard_low), 'LineWidth', 1.1);
yline(tau_noise, '-.b', sprintf('\\tau_{noise} (%.3f)', tau_noise), 'LineWidth', 1.0);
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 1.2, 'Color', 'k');
title('Macro Timing Diagnosis: Crash Probability vs Dynamic Deadband Guardrail', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('P(Crash)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylim([0, 1.05]);
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

if ~exist(configObj.ModelDir, 'dir'), mkdir(configObj.ModelDir); end
wfFigPath = fullfile(configObj.ModelDir, 'Phase6_WalkForward_Backtest.png');
exportgraphics(fig_wf, wfFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf(' 📊 機構級前向回測報表 (白底黑字) 已儲存至: %s\n', wfFigPath);
close(fig_wf);
time_step6 = toc(t_step6);
fprintf('⏱️ [步驟 6 完成] 耗時: %.2f 秒\n\n', time_step6);

%% =========================================================================
% 結算全流程執行時長審計報告
% =========================================================================
total_elapsed_sec = toc(t_total_start);
tot_hours = floor(total_elapsed_sec / 3600);
tot_mins  = floor(mod(total_elapsed_sec, 3600) / 60);
tot_secs  = mod(total_elapsed_sec, 60);

fprintf('=================================================================\n');
fprintf('📊 【Phase 6 各階段耗時明細與總時長審計報告】\n');
fprintf('=================================================================\n');
fprintf(' 步驟 0：環境掛載與隨機串流鎖定   : %8.2f 秒 (%5.1f%%)\n', time_step0, (time_step0 / total_elapsed_sec) * 100);
fprintf(' 步驟 1：快取與開盤價矩陣載入     : %8.2f 秒 (%5.1f%%)\n', time_step1, (time_step1 / total_elapsed_sec) * 100);
fprintf(' 步驟 2：宏觀波動度與狀態空間預計算: %8.2f 秒 (%5.1f%%)\n', time_step2, (time_step2 / total_elapsed_sec) * 100);
fprintf(' 步驟 3：時間邊界劃分與死區校準   : %8.2f 秒 (%5.1f%%)\n', time_step3, (time_step3 / total_elapsed_sec) * 100);
fprintf(' 步驟 4：前向滾動推論與定期調倉撮合: %8.2f 秒 (%5.1f%%)\n', time_step4, (time_step4 / total_elapsed_sec) * 100);
fprintf(' 步驟 5：IS/OOS 財務計量指標分離  : %8.2f 秒 (%5.1f%%)\n', time_step5, (time_step5 / total_elapsed_sec) * 100);
fprintf(' 步驟 6：圖表繪製 (OOS 重設) 與存檔: %8.2f 秒 (%5.1f%%)\n', time_step6, (time_step6 / total_elapsed_sec) * 100);
fprintf('-----------------------------------------------------------------\n');
if tot_hours > 0
    fprintf('⏱️ 【總執行時長】: %d 小時 %d 分 %.2f 秒 (共 %.2f 秒)\n', tot_hours, tot_mins, tot_secs, total_elapsed_sec);
else
    fprintf('⏱️ 【總執行時長】: %d 分 %.2f 秒 (共 %.2f 秒)\n', tot_mins, tot_secs, total_elapsed_sec);
end
fprintf('=================================================================\n');
disp('🎯 [Phase 15.5] 前向回測執行完畢！全管線同構閉環達成。');
disp('=================================================================');

%% =====================================================================
% 輔助函數：機構級財務計量指標結算器
% =====================================================================
function m = evaluate_financial_metrics(v, v_bench)
    n_days = length(v);
    
    r = diff(v) ./ (v(1:end-1) + 1e-8);
    r(isnan(r) | isinf(r)) = 0;
    
    rb = diff(v_bench) ./ (v_bench(1:end-1) + 1e-8);
    rb(isnan(rb) | isinf(rb)) = 0;
    
    m = struct();
    m.TotalRet = (v(end) / v(1) - 1) * 100;
    
    ratio = v(end) / v(1);
    if ratio <= 0
        m.CAGR = -100.0;
    else
        m.CAGR = (real(ratio ^ (252 / max(1, n_days))) - 1) * 100;
    end
    
    m.AnnVol   = std(r) * sqrt(252) * 100;
    m.MDD      = min((v - cummax(v)) ./ (cummax(v) + 1e-8)) * 100;
    m.Sharpe   = (mean(r) / (std(r) + 1e-8)) * sqrt(252);
    m.Calmar   = m.CAGR / max(1e-4, abs(m.MDD));
    m.WinRate  = mean(r > 0) * 100;
    
    m.BenchTotalRet = (v_bench(end) / v_bench(1) - 1) * 100;
    
    b_ratio = v_bench(end) / v_bench(1);
    if b_ratio <= 0
        m.BenchCAGR = -100.0;
    else
        m.BenchCAGR = (real(b_ratio ^ (252 / max(1, n_days))) - 1) * 100;
    end
    
    m.BenchAnnVol   = std(rb) * sqrt(252) * 100;
    m.BenchMDD      = min((v_bench - cummax(v_bench)) ./ (cummax(v_bench) + 1e-8)) * 100;
    m.BenchSharpe   = (mean(rb) / (std(rb) + 1e-8)) * sqrt(252);
    m.BenchCalmar   = m.BenchCAGR / max(1e-4, abs(m.BenchMDD));
    m.BenchWinRate  = mean(rb > 0) * 100;
    
    ex_r = r - rb;
    std_ex = std(ex_r);
    if std_ex > 1e-6
        m.IR = (mean(ex_r) / std_ex) * sqrt(252);
    else
        m.IR = 0.0;
    end
end
