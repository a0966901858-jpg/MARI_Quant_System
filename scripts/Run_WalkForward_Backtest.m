% =========================================================================
% 腳本：6_Run_WalkForward_Backtest.m
% 升級：Phase 15 (★ 日頻滑價成本校正、Open-to-Open 嚴格撮合、因果律邊界斷言)
% 職責：執行嚴格的因果律回測，產出無縫的 IS/OOS 真實績效、交易軌跡與視覺化診斷報表
% =========================================================================
clear; clc; close all;
disp('=================================================================');
disp('🚀 [Phase 15] 啟動 MARI 嚴格前向滾動回測與動態體制集成引擎');
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

if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入全域資料、DataFetcher 與多軌預訓練大腦
disp('--- 步驟 1：載入全域快取、DataFetcher 開盤價與多軌預訓練大腦 ---');
load(fullfile(configObj.CacheDir, 'features_denoised.mat'), 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC'; 
load(fullfile(configObj.ModelDir, 'GBDT_Guards.mat'), 'P_crash_all', 'P_time_all', 'P_space_all');

agent_aggressive   = load(fullfile(configObj.ModelDir, 'CIO_Aggressive.mat')).agent_aggressive;
agent_balanced     = load(fullfile(configObj.ModelDir, 'CIO_Balanced.mat')).agent_balanced;
agent_conservative = load(fullfile(configObj.ModelDir, 'CIO_Conservative.mat')).agent_conservative;

% 載入 Opens 以對齊 Phase 4 Open-to-Open 撮合環境
fetcher = DataFetcher(configObj);
dataStruct = fetcher.fetch_data();
Opens_Raw = dataStruct.Opens;

seqLen = configObj.SeqLen;
numDaysRaw = length(Dates_Active);
valid_idx = seqLen : numDaysRaw;

% 先對全歷史進行 20 日移動平均平滑，再截取有效區間
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
disp('--- 步驟 2：預計算 CIO 5 維狀態空間 (MLP 對齊版) ---');
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

%% 4. 初始化回測沙盒與成本模型校驗
disp('--- 步驟 3：初始化連續淨值回測沙盒與成本率合理性檢驗 ---');
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

% 成本率基線抽樣檢查
sample_vol_daily = mean(vol20) / sqrt(252);
sample_tc = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff * sample_vol_daily);
fprintf(' 📊 [成本模型檢查] 平均日波動度: %.4f%% | 預期平均換手成本率: %.4f%% (合理區間: 0.05%%~0.50%%)\n', ...
    sample_vol_daily*100, sample_tc*100);

%% 5. 執行逐日回測 (Open-to-Open 撮合機制)
disp('--- 步驟 4：啟動逐日推論與 X光級交易監控 (Open-to-Open) ---');
fprintf(' 📡 回測正式起點：%s\n', datestr(Dates_Active(valid_start_t)));

% 監測 OOS 起點與護欄機率早期分佈
fprintf('\n[DEBUG] OOS起點=%d (%s)\n', idx_OOS_start, datestr(Dates_Active(idx_OOS_start)));
fprintf('[DEBUG] guard_high=%.4f guard_low=%.4f\n', opt_guard, max(0, opt_guard-0.10));
dbg_days = idx_OOS_start : min(idx_OOS_start+29, numDays-2);
fprintf('[DEBUG] OOS前30天 P_crash_smooth:\n'); disp(P_crash_smooth(dbg_days)');
t_probe = idx_OOS_start + 50;
probe_p = P_time_M(:, t_probe)*fallback_w_time + P_space_M(:, t_probe)*(1-fallback_w_time);
probe_p = probe_p .* Expert_Active(t_probe, :)';
fprintf('[DEBUG] t=%d 非零選股數=%d comb_p總和=%.4f P_crash=%.4f\n\n', ...
    t_probe, sum(probe_p>0), sum(probe_p), P_crash_smooth(t_probe));

% 因 Open-to-Open 需存取 t+2，迴圈上限調整至 numDays - 2
for t = valid_start_t : numDays - 2
    
    % ★ Phase 15 (P1-5)：時間索引因果律邊界斷言防呆
    assert(t + 2 <= numDays, '❌ Open-to-Open 跨日撮合索引溢出邊界！');
    
    % OOS 盲測期破產旗標與資金池強制重置
    if t == idx_OOS_start
        is_bankrupt = false;
        port_values(t) = 1.0;
        spy_values(t) = 1.0; 
        prev_assets = zeros(numTickers, 1, 'single');
        prev_cash = 1.0;
        if Verbose_Log
            fprintf('\n🚨 [%s] 跨越時間邊界，正式進入 OOS 盲測區間！(破產旗標解除，資金重置為1.0，獨立重新起算)\n', datestr(Dates_Active(t)));
        end
    end
    
    % 破產防護機制
    if port_values(t) < 0.05 || is_bankrupt
        is_bankrupt = true;
        port_values(t+1) = port_values(t);
        spy_ret_bench = (Opens_Active(t+2, spy_idx) - Opens_Active(t+1, spy_idx)) / (Opens_Active(t+1, spy_idx) + 1e-8);
        if isnan(spy_ret_bench) || isinf(spy_ret_bench), spy_ret_bench = 0; end
        spy_values(t+1) = spy_values(t) * (1 + spy_ret_bench);
        cash_ratios(t)   = 1.0;
        continue;
    end
    
    state_2d = CIO_State(:, t);
    state_2d(5) = prev_cash;
    
    [act_agg, ~] = agent_aggressive.get_actions(state_2d, 0);
    [act_bal, ~] = agent_balanced.get_actions(state_2d, 0);
    [act_con, ~] = agent_conservative.get_actions(state_2d, 0);
    
    if P_crash_smooth(t) > opt_guard
        prob_con = 0.80; prob_bal = 0.15; prob_agg = 0.05;
    elseif vol20(t) > 0.20
        prob_con = 0.20; prob_bal = 0.60; prob_agg = 0.20;
    else
        prob_con = 0.10; prob_bal = 0.30; prob_agg = 0.60;
    end
    
    act_final = act_agg * prob_agg + act_bal * prob_bal + act_con * prob_con;
    act_final(isnan(act_final)) = 0;
    
    w_time = act_final(1); w_space = act_final(2); target_cash = act_final(3);
    
    if (w_time + w_space) == 0
        w_time = fallback_w_time; w_space = 1.0 - fallback_w_time;
    else
        w_time = w_time / (w_time + w_space); w_space = 1.0 - w_time;
    end
    
    % 崩盤護欄連續縮放 (Continuous Risk Scaling)
    guard_high = opt_guard;
    guard_low  = max(0, opt_guard - 0.10);
    if guard_high > guard_low
        risk_scale = max(0, min(1, (P_crash_smooth(t) - guard_low) / (guard_high - guard_low)));
        target_cash = max(target_cash, risk_scale);
    elseif P_crash_smooth(t) >= guard_high
        target_cash = 1.0;
    end
    
    w_cash = target_cash; rem_w = 1.0 - w_cash;
    
    comb_p = P_time_M(:, t) * w_time + P_space_M(:, t) * w_space;
    comb_p = comb_p .* Expert_Active(t, :)';
    
    if sum(comb_p > 0) > top_k
        [~, sort_idx] = sort(comb_p, 'descend');
        comb_p(comb_p < comb_p(sort_idx(top_k))) = 0;
    end
    
    if sum(comb_p) > 0
        comb_p = comb_p / sum(comb_p);
    else
        comb_p(:) = 0;
    end
    
    asset_w = comb_p * rem_w;
    
    turnover = abs(asset_w - prev_assets);
    ignore_mask = turnover < base_frict;
    asset_w(ignore_mask) = prev_assets(ignore_mask);
    
    active_sum = sum(asset_w);
    if active_sum > 0
        total_cap = active_sum + w_cash;
        asset_w = asset_w / total_cap;
        w_cash  = w_cash / total_cap;
    else
        w_cash = 1.0; asset_w(:) = 0;
    end
    
    cash_ratios(t) = w_cash;
    
    % Open-to-Open 跨日報酬撮合
    ret_t1 = (Opens_Active(t+2, :) - Opens_Active(t+1, :)) ./ (Opens_Active(t+1, :) + 1e-8);
    ret_t1(isnan(ret_t1) | isinf(ret_t1)) = 0;
    
    spy_ret_t1 = (Opens_Active(t+2, spy_idx) - Opens_Active(t+1, spy_idx)) / (Opens_Active(t+1, spy_idx) + 1e-8);
    if isnan(spy_ret_t1) || isinf(spy_ret_t1), spy_ret_t1 = 0; end
    
    % ★ Phase 15 (P0-1)：年化波動度還原為日頻波動度，修正交易成本放大問題
    current_vol_daily = vol20(t) / sqrt(252);
    if isprop(configObj, 'BaseFrictionFee') && isprop(configObj, 'SlippageVolCoeff')
        tc_rate = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff * current_vol_daily);
    else
        tc_rate = 0.0005 + (0.10 * current_vol_daily);
    end
    cost = sum(abs(asset_w - prev_assets)) * tc_rate;
    tc_records(t) = cost;
    
    port_ret = sum(asset_w .* ret_t1') - cost;
    
    port_values(t+1) = port_values(t) * (1 + port_ret);
    spy_values(t+1)  = spy_values(t) * (1 + spy_ret_t1);
    
    prev_assets = asset_w;
    prev_cash   = w_cash;
    
    if Verbose_Log && mod(t, 252) == 0
        num_holdings = sum(asset_w > 0.001);
        fprintf('[%s] SPY: %+5.2f%% | MARI: %+5.2f%% | 現金: %5.1f%% | 持股: %2d 檔 | 成本: %.3f%% | 累計淨值: %.2f\n', ...
            datestr(Dates_Active(t+1)), spy_ret_t1*100, port_ret*100, w_cash*100, num_holdings, cost*100, port_values(t+1));
    end
end
cash_ratios(numDays-1:numDays) = w_cash;
port_values(numDays) = port_values(numDays-1);
spy_values(numDays) = spy_values(numDays-1);

%% 6. 績效結算與視覺化
disp('--- 步驟 5：嚴格 IS / OOS 績效分離結算 ---');
is_port = port_values(valid_start_t:idx_OOS_start-1);
is_spy  = spy_values(valid_start_t:idx_OOS_start-1);
oos_port = port_values(idx_OOS_start:numDays-1);
oos_spy  = spy_values(idx_OOS_start:numDays-1);

calc_metrics = @(v) struct(...
    'Ret', (v(end)/v(1) - 1)*100, ...                                
    'MDD', min((v - cummax(v))./cummax(v))*100, ...                  
    'Sharpe', mean(diff(v)./v(1:end-1),'omitnan') / (std(diff(v)./v(1:end-1),'omitnan')+1e-8) * sqrt(252) ...
);

is_m   = calc_metrics(is_port);  is_sm  = calc_metrics(is_spy);
oos_m  = calc_metrics(oos_port); oos_sm = calc_metrics(oos_spy);

fprintf('\n=======================================================\n');
fprintf('📊 MARI Quant System 嚴格分離盲測報告 (Phase 15 Open-to-Open)\n');
fprintf('=======================================================\n');
fprintf('[In-Sample: %s to %s]\n', datestr(Dates_Active(valid_start_t)), datestr(Dates_Active(idx_OOS_start-1)));
fprintf('  > MARI : 總報酬 %8.2f%% | MDD %7.2f%% | 夏普 %5.2f\n', is_m.Ret, is_m.MDD, is_m.Sharpe);
fprintf('  > SPY  : 總報酬 %8.2f%% | MDD %7.2f%% | 夏普 %5.2f\n', is_sm.Ret, is_sm.MDD, is_sm.Sharpe);
fprintf('-------------------------------------------------------\n');
fprintf('[Out-of-Sample: %s to %s] ☢️ 真實盲測\n', datestr(Dates_Active(idx_OOS_start)), datestr(Dates_Active(numDays-1)));
fprintf('  > MARI : 總報酬 %8.2f%% | MDD %7.2f%% | 夏普 %5.2f\n', oos_m.Ret, oos_m.MDD, oos_m.Sharpe);
fprintf('  > SPY  : 總報酬 %8.2f%% | MDD %7.2f%% | 夏普 %5.2f\n', oos_sm.Ret, oos_sm.MDD, oos_sm.Sharpe);
fprintf('=======================================================\n');

%% 7. 繪製機構級回測視覺化報表
disp('--- 步驟 6：生成機構級視覺化報表 ---');
figure('Name', 'MARI Quant Walk-Forward Backtest (Phase 15)', 'Color', 'w', 'Position', [100, 100, 1200, 950]);

subplot(4,1,1);
plot(Dates_Active(valid_start_t:idx_OOS_start-1), log10(is_port), 'LineWidth', 1.5, 'Color', '#D95319'); hold on;
plot(Dates_Active(valid_start_t:idx_OOS_start-1), log10(is_spy), 'LineWidth', 1.5, 'Color', '#0072BD');
plot(Dates_Active(idx_OOS_start:numDays-1), log10(oos_port), 'LineWidth', 2.0, 'Color', '#A2142F');
plot(Dates_Active(idx_OOS_start:numDays-1), log10(oos_spy), 'LineWidth', 1.5, 'Color', '#4DBEEE');
xline(Dates_Active(idx_OOS_start), '--k', 'OOS Blind Test Start', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
title('Log-Scale Cumulative Equity Curve (MARI vs SPY)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Log10(Value)');
legend('MARI (IS)', 'SPY (IS)', 'MARI (OOS)', 'SPY (OOS)', 'Location', 'northwest');
grid on;

subplot(4,1,2);
mari_dd_is = (is_port - cummax(is_port)) ./ cummax(is_port);
spy_dd_is  = (is_spy  - cummax(is_spy))  ./ cummax(is_spy);
mari_dd_oos = (oos_port - cummax(oos_port)) ./ cummax(oos_port);
area(Dates_Active(valid_start_t:idx_OOS_start-1), mari_dd_is, 'FaceColor', '#D95319', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); hold on;
area(Dates_Active(idx_OOS_start:numDays-1), mari_dd_oos, 'FaceColor', '#A2142F', 'FaceAlpha', 0.7, 'EdgeColor', 'none');
plot(Dates_Active(valid_start_t:idx_OOS_start-1), spy_dd_is, 'Color', '#0072BD', 'LineWidth', 1.2);
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 2);
title('Portfolio Drawdown', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Drawdown (%)');
grid on;

subplot(4,1,3);
area(Dates_Active(valid_start_t:numDays-1), cash_ratios(valid_start_t:numDays-1) * 100, 'FaceColor', '#77AC30', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 2);
title('Dynamic Cash Allocation & Hedge Ratio', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Cash Ratio (%)');
ylim([0, 100]);
grid on;

subplot(4,1,4);
plot(Dates_Active(valid_start_t:numDays-1), P_crash_smooth(valid_start_t:numDays-1), 'Color', '#7E2F8E', 'LineWidth', 1.2); hold on;
yline(opt_guard, '--r', sprintf('Guardrail Target (%.3f)', opt_guard), 'LineWidth', 1.5);
yline(max(0, opt_guard - 0.10), ':r', 'Guardrail Activation', 'LineWidth', 1.0);
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 2);
title('Crash Probability vs Continuous Guardrail (Diagnosis Tool)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('P(Crash)');
ylim([0, 1.05]);
legend('Smoothed P(Crash)', 'Full Hedge Line', 'Start Hedge Line', 'Location', 'northwest');
grid on;
