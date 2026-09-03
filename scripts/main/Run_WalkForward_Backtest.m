% =========================================================================
% 腳本：6_Run_WalkForward_Backtest.m
% 升級：Phase 15.5 Stage 4 規範版 (★ mrg32k3a 隨機串流鎖定、Open-to-Open 權重漂移、
%       停牌資產鎖死、死區護欄連續縮放、CIO 彈性退回中立規則、因果時間軸校正)
% 職責：執行嚴格的因果律滾動回測，產出無縫的 IS/OOS 真實績效、交易軌跡與視覺化診斷報表
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 MARI 嚴格前向滾動回測 (權重漂移與停牌鎖死對齊版)');
disp('=================================================================');

%% 0. 環境路徑掛載與隨機串流管理
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

% ★ 核心修復 1：使用 Config 統一初始化主執行緒 mrg32k3a 隨機數引擎 (Substream = 1)
stream = configObj.getRandStream(1);
RandStream.setGlobalStream(stream);
disp('🔒 已掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定前向回測環境。');

%% 1. 載入全域快取、DataFetcher 與大腦權重
disp('--- 步驟 1：載入全域快取、DataFetcher 開盤價與多軌大腦 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');

if ~exist(cachePath, 'file') || ~exist(gbdtPath, 'file')
    error('❌ 找不到前置快取，請確認 Phase 1 至 Phase 3 已成功執行！');
end

load(cachePath, 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC'; 
load(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all');

% ★ 核心修復 2：CIO 權重彈性載入防護 (若未訓練則優雅退回 Config/BO 基準)
has_cio_agents = false;
path_agg = fullfile(configObj.ModelDir, 'CIO_Aggressive.mat');
path_bal = fullfile(configObj.ModelDir, 'CIO_Balanced.mat');
path_con = fullfile(configObj.ModelDir, 'CIO_Conservative.mat');

if exist(path_agg, 'file') && exist(path_bal, 'file') && exist(path_con, 'file')
    try
        agent_aggressive   = load(path_agg).agent_aggressive;
        agent_balanced     = load(path_bal).agent_balanced;
        agent_conservative = load(path_con).agent_conservative;
        has_cio_agents = true;
        disp('  🤖 成功加載 Phase 5 三軌 CIO 強化學習大腦 (Aggressive, Balanced, Conservative)。');
    catch ME
        warning('⚠️ 加載 CIO 代理人失敗 (%s)，將退回使用中立/BO 超參數路由。', ME.message);
    end
else
    disp('  ℹ️ 未檢測到完整 Phase 5 CIO 權重檔，回測將使用 Config / Phase 4 最佳化超參數路由。');
end

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
if isempty(spy_idx), error('❌ 找不到 SPY 基準！'); end

%% 2. 預計算 CIO 5 維狀態空間 
disp('--- 步驟 2：預計算 CIO 5 維狀態空間 ---');
CIO_State = zeros(5, numDays, 'single');
spy_prices = Prices_Active(:, spy_idx);
spy_rets = [0; diff(spy_prices) ./ spy_prices(1:end-1)];
spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
vol20 = movstd(spy_rets, [19, 0], 1) * sqrt(252);

mdd252 = zeros(numDays, 1);
for t = 1:numDays
    start_t = max(1, t - 251);
    cum_ret = cumprod(1 + spy_rets(start_t:t));
    mdd252(t) = min((cum_ret - cummax(cum_ret)) ./ cummax(cum_ret));
end

spy_ret20 = zeros(numDays, 1);
for t = 21:numDays
    spy_ret20(t) = (spy_prices(t) - spy_prices(t-20)) / spy_prices(t-20);
end

CIO_State(1, :) = P_crash_smooth'; 
CIO_State(2, :) = spy_ret20';      
CIO_State(3, :) = vol20';          
CIO_State(4, :) = abs(mdd252)';    
CIO_State(5, :) = 1.0;             

%% 3. 定義嚴格時間邊界
spy_inception_idx = find(spy_prices > 10, 1);
if isempty(spy_inception_idx), spy_inception_idx = 252; end
Train_Start_Date = datetime('2006-01-01', 'TimeZone', 'UTC');
idx_train_start = find(Dates_Active >= Train_Start_Date, 1);
valid_start_t = max(spy_inception_idx + 252, idx_train_start); 

OOS_Date = datetime('2022-01-01', 'TimeZone', 'UTC');
idx_OOS_start = find(Dates_Active >= OOS_Date, 1);

%% 4. 初始化回測沙盒與崩盤護欄死區 (Deadband) 校準
disp('--- 步驟 3：初始化回測沙盒與校準崩盤護欄死區 ---');
port_values = ones(numDays, 1, 'single');
spy_values  = ones(numDays, 1, 'single');
cash_ratios = zeros(numDays, 1, 'single');
tc_records  = zeros(numDays, 1, 'single');

prev_assets = zeros(numTickers, 1, 'single');
prev_cash = 1.0;

opt_guard       = configObj.Guardrail_CrashProb;
top_k           = configObj.Top_K_Assets;
fallback_w_time = configObj.Expert_Time_Weight;
base_frict      = configObj.MoE_FrictionMask; 
Verbose_Log     = true; 
is_bankrupt     = false; 

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

fprintf(' 🔍 [護欄參數校準]\n');
fprintf('    > IS 非危機期 P(Crash) 75%% 雜訊分位數 (tau_noise) : %.4f\n', tau_noise);
fprintf('    > 硬熔斷上限 (Guard_High)                      : %.4f\n', guard_high);
fprintf('    > 死區縮放下限 (Guard_Low)                       : %.4f\n', guard_low);
fprintf('    > 有效連續緩衝區間寬度                           : %.4f\n', guard_high - guard_low);

sample_vol_daily = mean(vol20) / sqrt(252);
sample_tc = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff * sample_vol_daily);
fprintf(' 📊 [成本模型檢查] 平均日波動度: %.4f%% | 預期平均換手成本率: %.4f%%\n', ...
    sample_vol_daily*100, sample_tc*100);

%% 5. 執行逐日回測 (★ 核心修復 3：嚴格權重漂移、停牌鎖死與時間軸因果對齊)
disp('--- 步驟 4：啟動逐日推論與 X 光級交易監控 (Open-to-Open 權重漂移版) ---');
fprintf(' 📡 回測正式起點：%s\n', datestr(Dates_Active(valid_start_t)));

% 初始化起始兩天為現金基準
port_values(valid_start_t)   = 1.0;
port_values(valid_start_t+1) = 1.0;
spy_values(valid_start_t)    = 1.0;
spy_values(valid_start_t+1)  = 1.0;

for t = valid_start_t : numDays - 2
    
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
    % 3. 策略決策：時空權重與現金目標
    % -------------------------------------------------------------
    if has_cio_agents
        state_2d = CIO_State(:, t);
        state_2d(5) = prev_cash;
        
        [act_agg, ~] = agent_aggressive.get_actions(state_2d, 0);
        [act_bal, ~] = agent_balanced.get_actions(state_2d, 0);
        [act_con, ~] = agent_conservative.get_actions(state_2d, 0);
        
        if P_crash_smooth(t) > guard_high
            prob_con = 0.80; prob_bal = 0.15; prob_agg = 0.05;
        elseif vol20(t) > 0.20
            prob_con = 0.20; prob_bal = 0.60; prob_agg = 0.20;
        else
            prob_con = 0.10; prob_bal = 0.30; prob_agg = 0.60;
        end
        
        act_final = act_agg * prob_agg + act_bal * prob_bal + act_con * prob_con;
        act_final(isnan(act_final)) = 0;
        
        % 防禦負值輸出破壞權重比例
        w_time = max(0, act_final(1));
        w_space = max(0, act_final(2));
        target_cash = max(0, min(1, act_final(3)));
        
        if (w_time + w_space) <= 1e-6
            w_time = fallback_w_time; w_space = 1.0 - fallback_w_time;
        else
            sum_w = w_time + w_space;
            w_time = w_time / sum_w; w_space = w_space / sum_w;
        end
    else
        w_time = fallback_w_time;
        w_space = 1.0 - fallback_w_time;
        target_cash = 0.0;
    end
    
    % -------------------------------------------------------------
    % 4. 崩盤護欄死區連續縮放 (Deadband Continuous Risk Scaling)
    % -------------------------------------------------------------
    p_c = P_crash_smooth(t);
    if p_c <= guard_low
        risk_scale = 0.0;
    elseif p_c >= guard_high
        risk_scale = 1.0;
    else
        risk_scale = (p_c - guard_low) / (guard_high - guard_low);
    end
    target_cash = max(target_cash, risk_scale);
    
    actual_cash_target = min(target_cash, available_cap);
    rem_cap_for_assets = available_cap - actual_cash_target;
    
    % -------------------------------------------------------------
    % 5. 專家選股分數融合與 Top-K 篩選
    % -------------------------------------------------------------
    comb_p = P_time_M(:, t) * w_time + P_space_M(:, t) * w_space;
    comb_p = comb_p .* Expert_Active(t, :)';
    comb_p(halted_mask) = 0; % 停牌標的不可新增部位
    
    if sum(comb_p > 0) > top_k
        [~, sort_idx] = sort(comb_p, 'descend');
        comb_p(comb_p < comb_p(sort_idx(top_k))) = 0;
    end
    
    if sum(comb_p) > 0
        comb_p = comb_p / sum(comb_p);
    else
        comb_p(:) = 0;
    end
    
    asset_w = comb_p * rem_cap_for_assets;
    asset_w(halted_mask) = locked_weights(halted_mask);
    
    % -------------------------------------------------------------
    % 6. 機構級摩擦緩衝過濾 (Inertia Friction Mask)
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
    
    % 僅對可交易標的收取開盤換手摩擦
    cost = sum(abs(asset_w(~halted_mask) - w_drift(~halted_mask))) * tc_rate;
    tc_records(t+1) = cost;
    
    % -------------------------------------------------------------
    % 8. 結算 Open(t+1) 至 Open(t+2) 投資組合跨日報酬
    % -------------------------------------------------------------
    ret_t1 = (Opens_Active(t+2, :) - Opens_Active(t+1, :)) ./ (Opens_Active(t+1, :) + 1e-8);
    ret_t1(isnan(ret_t1) | isinf(ret_t1)) = 0;
    
    spy_ret_t1 = (Opens_Active(t+2, spy_idx) - Opens_Active(t+1, spy_idx)) / (Opens_Active(t+1, spy_idx) + 1e-8);
    if isnan(spy_ret_t1) || isinf(spy_ret_t1), spy_ret_t1 = 0; end
    
    port_ret = sum(asset_w .* ret_t1') - cost;
    
    % 報酬結算至 t+2 日開盤淨值
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

%% 6. 機構級績效結算 (IS 與 OOS 嚴格隔離)
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
fprintf('📊 【MARI Quant System 機構級前向回測綜合報告 (Phase 15.5 Open-to-Open)】\n');
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

%% 7. 繪製標準學術白底黑字視覺化報表
disp('--- 步驟 6：生成機構級視覺化報表 (白底黑字) ---');
fig_wf = figure('Name', 'MARI Quant Walk-Forward Backtest', ...
    'Color', 'w', 'Position', [100, 100, 1250, 1000], 'Visible', 'off');
set(fig_wf, 'InvertHardcopy', 'off');

% 子圖 1：全歷史對數淨值曲線 (IS 與 OOS 連續無縫銜接)
subplot(4, 1, 1);
plot(Dates_Active(is_idx), log10(is_port), 'LineWidth', 1.6, 'Color', '#D95319', 'DisplayName', 'MARI (IS)'); hold on;
plot(Dates_Active(is_idx), log10(is_spy), 'LineWidth', 1.3, 'Color', '#0072BD', 'DisplayName', 'SPY (IS)');
plot(Dates_Active(oos_idx), log10(oos_port), 'LineWidth', 2.0, 'Color', '#A2142F', 'DisplayName', 'MARI (OOS)');
plot(Dates_Active(oos_idx), log10(oos_spy), 'LineWidth', 1.4, 'Color', '#4DBEEE', 'DisplayName', 'SPY (OOS)');
xline(Dates_Active(idx_OOS_start), '--k', 'OOS Start', 'LineWidth', 1.3, ...
    'LabelVerticalAlignment', 'bottom', 'Color', 'k', 'FontName', 'Helvetica', 'FontWeight', 'bold');
title('Log-Scale Cumulative Equity Curve (Continuous Realized Open-to-Open)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Log_{10}(Wealth)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% 子圖 2：水下回撤圖 (%)
subplot(4, 1, 2);
full_port_eval = port_values(valid_start_t+1:numDays);
full_spy_eval  = spy_values(valid_start_t+1:numDays);
full_dates     = Dates_Active(valid_start_t+1:numDays);

mari_dd = (full_port_eval - cummax(full_port_eval)) ./ cummax(full_port_eval) * 100;
spy_dd  = (full_spy_eval  - cummax(full_spy_eval))  ./ cummax(full_spy_eval)  * 100;

area(full_dates, mari_dd, 'FaceColor', '#D95319', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'DisplayName', 'MARI Drawdown'); hold on;
plot(full_dates, spy_dd, 'Color', '#0072BD', 'LineWidth', 1.1, 'DisplayName', 'SPY Drawdown');
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 1.2, 'Color', 'k');
title('Portfolio Underwater Drawdown Profile (%)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Drawdown (%)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'southwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% 子圖 3：動態避險現金比例 (%)
subplot(4, 1, 3);
area(full_dates, cash_ratios(valid_start_t+1:numDays) * 100, ...
    'FaceColor', '#77AC30', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Cash Ratio (%)'); hold on;
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 1.2, 'Color', 'k');
title('Dynamic Cash Allocation & Continuous Risk Scaling', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Cash (%)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylim([0, 105]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% 子圖 4：平滑崩盤機率與護欄閾值診斷 (含雜訊底線與死區標註)
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

wfFigPath = fullfile(configObj.ResultDir, 'Phase6_WalkForward_Backtest.png');
exportgraphics(fig_wf, wfFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf(' 📊 機構級前向回測報表 (白底黑字) 已儲存至: %s\n', wfFigPath);
close(fig_wf);

disp('=================================================================');
disp('🎯 [Phase 15.5] 前向回測執行完畢！');
disp('=================================================================');

%% =====================================================================
% 輔助函數：機構級財務計量指標結算器
% =====================================================================
function m = evaluate_financial_metrics(v, v_bench)
    n_days = length(v);
    
    r = diff(v) ./ v(1:end-1);
    r(isnan(r) | isinf(r)) = 0;
    
    rb = diff(v_bench) ./ v_bench(1:end-1);
    rb(isnan(rb) | isinf(rb)) = 0;
    
    m = struct();
    m.TotalRet = (v(end) / v(1) - 1) * 100;
    m.CAGR     = ((v(end) / v(1)) ^ (252 / max(1, n_days)) - 1) * 100;
    m.AnnVol   = std(r) * sqrt(252) * 100;
    m.MDD      = min((v - cummax(v)) ./ cummax(v)) * 100;
    m.Sharpe   = (mean(r) / (std(r) + 1e-8)) * sqrt(252);
    m.Calmar   = m.CAGR / max(1e-4, abs(m.MDD));
    m.WinRate  = mean(r > 0) * 100;
    
    m.BenchTotalRet = (v_bench(end) / v_bench(1) - 1) * 100;
    m.BenchCAGR     = ((v_bench(end) / v_bench(1)) ^ (252 / max(1, n_days)) - 1) * 100;
    m.BenchAnnVol   = std(rb) * sqrt(252) * 100;
    m.BenchMDD      = min((v_bench - cummax(v_bench)) ./ cummax(v_bench)) * 100;
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