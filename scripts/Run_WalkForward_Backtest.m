% =========================================================================
% 腳本：6_Run_WalkForward_Backtest.m
% 升級：Phase 14.20 (★ 成本模型真理來源統一、體制切換參數化規範、全域隨機種子對齊)
% 職責：執行嚴格的因果律回測，產出無縫的 IS/OOS 真實績效與交易軌跡
% =========================================================================
% 初始化環境：清空工作區、清除命令列、關閉所有視窗，確保記憶體狀態無殘留污染
clear; clc; close all;
disp('=================================================================');
disp('🚀 [Phase 14.20] 啟動 MARI 嚴格前向滾動回測與動態體制集成引擎');
disp('=================================================================');

% 動態獲取當下腳本的絕對路徑，確保專案在不同作業系統下的路徑相容性
currentPath = fileparts(mfilename('fullpath'));
if isempty(currentPath), currentPath = pwd; end
projectRoot = currentPath; 

% 檢查 configs 目錄是否存在，藉此判斷是否需要將根目錄指標往上推導一層
if ~exist(fullfile(projectRoot, 'configs'), 'dir'), projectRoot = fullfile(currentPath, '..'); end

% 將專案核心子模組路徑遞迴加入 MATLAB 搜尋環境中
addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'envs')));
addpath(genpath(fullfile(projectRoot, 'models')));
addpath(genpath(fullfile(projectRoot, 'agents')));
rehash toolboxcache;

% 實例化全域設定檔，統一控管超參數
configObj = Config();

% ★ 計畫書修正 (問題 P3-1 ⚪ P3)：固定全域隨機種子，確保實驗可重現
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入全域資料與預訓練大腦
disp('--- 步驟 1：載入全域快取與多軌預訓練大腦 ---');
% 載入 Phase 1 產出的標準化價格矩陣、活躍股票遮罩與絕對時間軸
load(fullfile(configObj.CacheDir, 'features_denoised.mat'), 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC'; 

% 載入 Phase 3 GBDT 專家網路產出的：崩盤機率、時序專家勝率、空間專家勝率
load(fullfile(configObj.ModelDir, 'GBDT_Guards.mat'), 'P_crash_all', 'P_time_all', 'P_space_all');

% 載入 Phase 5 完訓的三位具備不同風險偏好的 CIO 強化學習代理人 (大腦)
agent_aggressive   = load(fullfile(configObj.ModelDir, 'CIO_Aggressive.mat')).agent_aggressive;
agent_balanced     = load(fullfile(configObj.ModelDir, 'CIO_Balanced.mat')).agent_balanced;
agent_conservative = load(fullfile(configObj.ModelDir, 'CIO_Conservative.mat')).agent_conservative;

% 提取序列長度，截斷歷史資料前期的無效暖機期
seqLen = configObj.SeqLen;
numDaysRaw = length(Dates_Active);
valid_idx = seqLen : numDaysRaw;

% 套用有效索引，對齊所有的實體矩陣
Prices_Active = Prices_Active(valid_idx, :);
Expert_Active = Expert_Active(valid_idx, :);
Dates_Active  = Dates_Active(valid_idx);

% 對崩盤機率進行 20 日向後移動平均平滑 (Backward Moving Average) 防洩漏
P_crash_smooth = movmean(P_crash_all(valid_idx), [19, 0]);

% 將選股機率矩陣轉置，優化後續迴圈的欄位提取速度
P_time_M  = P_time_all(valid_idx, :)'; 
P_space_M = P_space_all(valid_idx, :)';
numDays   = length(Dates_Active);
numTickers = configObj.NumTickers;

% 尋找大盤基準 SPY
spy_idx = find(strcmp(configObj.IdxTickers, 'SPY'));
if isempty(spy_idx), error('❌ 找不到 SPY 基準！'); end

%% 2. 預計算 CIO 5 維狀態空間 (2D 矩陣對齊 MLP)
disp('--- 步驟 2：預計算 CIO 5 維狀態空間 (MLP 對齊版) ---');
CIO_State = zeros(5, numDays, 'single');
spy_prices = Prices_Active(:, spy_idx);

% 計算大盤每日報酬
spy_rets = [0; diff(spy_prices) ./ spy_prices(1:end-1)];
spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;

% 計算 20 日滾動年化波動率
vol20 = movstd(spy_rets, [19, 0], 1) * sqrt(252);

% 計算滾動 252 日最大回撤 (MDD)
mdd252 = zeros(numDays, 1);
for t = 1:numDays
    start_t = max(1, t - 251);
    cum_ret = cumprod(1 + spy_rets(start_t:t));
    mdd252(t) = min((cum_ret - cummax(cum_ret)) ./ cummax(cum_ret));
end

% 計算 20 日大盤累積報酬率
spy_ret20 = zeros(numDays, 1);
for t = 21:numDays
    spy_ret20(t) = (spy_prices(t) - spy_prices(t-20)) / spy_prices(t-20);
end

% 組合 5 維狀態供 RL 大腦觀測
CIO_State(1, :) = P_crash_smooth'; % 維度 1：崩盤指標
CIO_State(2, :) = spy_ret20';      % 維度 2：大盤動能
CIO_State(3, :) = vol20';          % 維度 3：大盤波動率
CIO_State(4, :) = abs(mdd252)';    % 維度 4：一年期最大回撤
CIO_State(5, :) = 1.0;             % 維度 5：當前現金水位 (初始預設為滿手現金)

%% 3. 定義嚴格時間邊界
spy_inception_idx = find(spy_prices > 10, 1);
if isempty(spy_inception_idx), spy_inception_idx = 252; end

Train_Start_Date = datetime('2006-01-01', 'TimeZone', 'UTC');
idx_train_start = find(Dates_Active >= Train_Start_Date, 1);
if isempty(idx_train_start)
    error('資料庫中找不到 2006 年以後的數據！');
end

% 物理防線：取「SPY 暖機完成日」與「2006 年第一天」兩者中最晚的一天作為真實 IS 績效起點
valid_start_t = max(spy_inception_idx + 252, idx_train_start); 

% 嚴格切分 Out-Of-Sample (OOS) 盲測期
OOS_Date = datetime('2022-01-01', 'TimeZone', 'UTC');
idx_OOS_start = find(Dates_Active >= OOS_Date, 1);

%% 4. 初始化回測沙盒
disp('--- 步驟 3：初始化連續淨值回測沙盒 (無縫模式) ---');
port_values = ones(numDays, 1, 'single');
spy_values  = ones(numDays, 1, 'single');
cash_ratios = zeros(numDays, 1, 'single');

prev_assets = zeros(numTickers, 1, 'single');
prev_cash = 1.0;

opt_guard       = configObj.Guardrail_CrashProb;
top_k           = configObj.Top_K_Assets;
fallback_w_time = configObj.Expert_Time_Weight;
base_frict      = configObj.MoE_FrictionMask; 
Verbose_Log     = true; 
is_bankrupt     = false; 

%% 5. 執行逐日回測
disp('--- 步驟 4：啟動逐日推論與 X光級交易監控 ---');
fprintf(' 📡 回測正式起點：%s\n', datestr(Dates_Active(valid_start_t)));

for t = valid_start_t : numDays - 1
    
    if t == idx_OOS_start
        if Verbose_Log
            fprintf('\n🚨 [%s] 跨越時間邊界，正式進入 Out-of-Sample 盲測區間！\n', datestr(Dates_Active(t)));
        end
    end
    
    % 破產防護機制
    if port_values(t) < 0.05 || is_bankrupt
        is_bankrupt = true;
        port_values(t+1) = port_values(t);
        spy_values(t+1)  = spy_values(t) * (1 + spy_rets(t+1));
        cash_ratios(t)   = 1.0;
        continue;
    end
    
    % 動態建構當日的 2D 狀態向量，注入前一日的真實現金水位
    state_2d = CIO_State(:, t);
    state_2d(5) = prev_cash;
    
    % 呼叫三軌強化學習代理人進行確定性推論 (noise_std = 0)
    [act_agg, ~] = agent_aggressive.get_actions(state_2d, 0);
    [act_bal, ~] = agent_balanced.get_actions(state_2d, 0);
    [act_con, ~] = agent_conservative.get_actions(state_2d, 0);
    
    % -------------------------------------------------------------------------
    % ★ 動態體制集成 (Regime-Switching Mixture of Experts, MoE)
    % -------------------------------------------------------------------------
    if P_crash_smooth(t) > opt_guard
        % 崩盤體制：保守型話語權 80%
        prob_con = 0.80; prob_bal = 0.15; prob_agg = 0.05;
    elseif vol20(t) > 0.20
        % 高波體制：平衡型話語權 60%
        prob_con = 0.20; prob_bal = 0.60; prob_agg = 0.20;
    else
        % 低波牛市：積極型話語權 60%
        prob_con = 0.10; prob_bal = 0.30; prob_agg = 0.60;
    end
    
    % 加權融合出最終的系統決策向量
    act_final = act_agg * prob_agg + act_bal * prob_bal + act_con * prob_con;
    act_final(isnan(act_final)) = 0;
    
    w_time = act_final(1); w_space = act_final(2); target_cash = act_final(3);
    
    if (w_time + w_space) == 0
        w_time = fallback_w_time; w_space = 1.0 - fallback_w_time;
    else
        w_time = w_time / (w_time + w_space); w_space = 1.0 - w_time;
    end
    
    % GBDT 終極硬熔斷機制
    if P_crash_smooth(t) > opt_guard
        target_cash = 1.0; w_time = 0; w_space = 0;
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
    
    % 換倉慣性遮罩 (Turnover Mask)
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
    
    % 次日市場真實報酬 (T 到 T+1 的 Close-to-Close)
    ret_t1 = (Prices_Active(t+1, :) - Prices_Active(t, :)) ./ Prices_Active(t, :);
    ret_t1(isnan(ret_t1) | isinf(ret_t1)) = 0;
    
    % ★ 計畫書修正 (問題 8-2 🟠 P1)：嚴格讀取 Config.m 統一成本模型
    if isprop(configObj, 'BaseFrictionFee') && isprop(configObj, 'SlippageVolCoeff')
        tc_rate = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff * vol20(t));
    else
        tc_rate = 0.0005 + (0.10 * vol20(t));
    end
    cost = sum(abs(asset_w - prev_assets)) * tc_rate;
    
    % 計算投組真實淨報酬 (扣除交易成本)
    port_ret = sum(asset_w .* ret_t1') - cost;
    
    % 更新連續資金淨值曲線
    port_values(t+1) = port_values(t) * (1 + port_ret);
    spy_values(t+1)  = spy_values(t) * (1 + spy_rets(t+1));
    
    prev_assets = asset_w;
    prev_cash   = w_cash;
    
    % 每年輸出一次交易監控日誌
    if Verbose_Log && mod(t, 252) == 0
        num_holdings = sum(asset_w > 0.001);
        fprintf('[%s] SPY: %+5.2f%% | MARI: %+5.2f%% | 現金: %5.1f%% | 持股: %2d 檔 | 累計淨值: %.2f\n', ...
            datestr(Dates_Active(t+1)), spy_rets(t+1)*100, port_ret*100, w_cash*100, num_holdings, port_values(t+1));
    end
end

cash_ratios(numDays) = w_cash;

%% 6. 績效結算與視覺化
disp('--- 步驟 5：嚴格 IS / OOS 績效分離結算 ---');
is_port = port_values(valid_start_t:idx_OOS_start-1);
is_spy  = spy_values(valid_start_t:idx_OOS_start-1);
oos_port = port_values(idx_OOS_start:end);
oos_spy  = spy_values(idx_OOS_start:end);

calc_metrics = @(v) struct(...
    'Ret', (v(end)/v(1) - 1)*100, ...                                
    'MDD', min((v - cummax(v))./cummax(v))*100, ...                  
    'Sharpe', mean(diff(v)./v(1:end-1),'omitnan') / (std(diff(v)./v(1:end-1),'omitnan')+1e-8) * sqrt(252) ...
);

is_m   = calc_metrics(is_port);  is_sm  = calc_metrics(is_spy);
oos_m  = calc_metrics(oos_port); oos_sm = calc_metrics(oos_spy);

fprintf('\n=======================================================\n');
fprintf('📊 MARI Quant System 嚴格分離盲測報告\n');
fprintf('=======================================================\n');
fprintf('[In-Sample: %s to %s]\n', datestr(Dates_Active(valid_start_t)), datestr(Dates_Active(idx_OOS_start-1)));
fprintf('  > MARI : 總報酬 %8.2f%% | MDD %7.2f%% | 夏普 %5.2f\n', is_m.Ret, is_m.MDD, is_m.Sharpe);
fprintf('  > SPY  : 總報酬 %8.2f%% | MDD %7.2f%% | 夏普 %5.2f\n', is_sm.Ret, is_sm.MDD, is_sm.Sharpe);
fprintf('-------------------------------------------------------\n');
fprintf('[Out-of-Sample: %s to %s] ☢️ 真實盲測\n', datestr(Dates_Active(idx_OOS_start)), datestr(Dates_Active(end)));
fprintf('  > MARI : 總報酬 %8.2f%% | MDD %7.2f%% | 夏普 %5.2f\n', oos_m.Ret, oos_m.MDD, oos_m.Sharpe);
fprintf('  > SPY  : 總報酬 %8.2f%% | MDD %7.2f%% | 夏普 %5.2f\n', oos_sm.Ret, oos_sm.MDD, oos_sm.Sharpe);
fprintf('=======================================================\n');

%% 7. 繪製機構級回測視覺化報表
disp('--- 步驟 6：生成機構級視覺化報表 ---');
figure('Name', 'MARI Quant Walk-Forward Backtest', 'Color', 'w', 'Position', [100, 100, 1200, 800]);

% 子圖 1：對數資金曲線
subplot(3,1,1);
plot(Dates_Active(valid_start_t:end), log10(port_values(valid_start_t:end)), 'LineWidth', 1.5, 'Color', '#D95319'); hold on;
plot(Dates_Active(valid_start_t:end), log10(spy_values(valid_start_t:end)), 'LineWidth', 1.5, 'Color', '#0072BD');
xline(Dates_Active(idx_OOS_start), '--k', 'OOS Blind Test Start', 'LineWidth', 2, 'LabelVerticalAlignment', 'bottom');
title('Log-Scale Cumulative Equity Curve (MARI vs SPY)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Log10(Portfolio Value)');
legend('MARI Quant', 'SPY Benchmark', 'Location', 'northwest');
grid on;

% 子圖 2：最大回撤
subplot(3,1,2);
mari_dd = (port_values(valid_start_t:end) - cummax(port_values(valid_start_t:end))) ./ cummax(port_values(valid_start_t:end));
spy_dd  = (spy_values(valid_start_t:end)  - cummax(spy_values(valid_start_t:end)))  ./ cummax(spy_values(valid_start_t:end));
area(Dates_Active(valid_start_t:end), mari_dd, 'FaceColor', '#D95319', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); hold on;
plot(Dates_Active(valid_start_t:end), spy_dd, 'Color', '#0072BD', 'LineWidth', 1.2);
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 2);
title('Portfolio Drawdown', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Drawdown (%)');
legend('MARI Drawdown', 'SPY Drawdown', 'Location', 'southwest');
grid on;

% 子圖 3：動態現金部位與避險水位
subplot(3,1,3);
area(Dates_Active(valid_start_t:end), cash_ratios(valid_start_t:end) * 100, 'FaceColor', '#77AC30', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 2);
title('Dynamic Cash Allocation & Hedge Ratio', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Cash Ratio (%)');
ylim([0, 100]);
grid on;