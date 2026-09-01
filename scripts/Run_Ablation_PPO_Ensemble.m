% =========================================================================
% 腳本：Run_Ablation_PPO_Ensemble.m (P2-3 PPO 三軌動態集成邊際消融實驗)
% 升級：Phase 15.5 (★ 3軌集成 vs 單一 Balanced/Aggressive/Conservative、HAC-DM 檢定)
% 職責：量化強化學習多風險偏好動態路由相較於單一策略大腦的邊際增量與抗震能力
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧪 [P2-3] 啟動 PPO 三軌動態體制集成消融實驗 (HAC 檢定 + 跨體制對比)');
disp('=================================================================');

%% 0. 環境路徑掛載與隨機種子鎖定
currentPath = fileparts(mfilename('fullpath'));
if isempty(currentPath), currentPath = pwd; end
projectRoot = currentPath; 
if ~exist(fullfile(projectRoot, 'configs'), 'dir'), projectRoot = fullfile(currentPath, '..'); end

addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'envs')));
addpath(genpath(fullfile(projectRoot, 'models')));
addpath(genpath(fullfile(projectRoot, 'agents')));
addpath(genpath(fullfile(projectRoot, 'utils'))); % ★ 掛載共用 HAC 統計工具箱
rehash toolboxcache;

configObj = Config();

if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入面板資料、專家矩陣與三軌預訓練大腦
disp('--- 步驟 1：載入全域快取與預訓練 CIO 代理人大腦 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');

if ~exist(cachePath, 'file') || ~exist(gbdtPath, 'file')
    error('❌ 找不到前置快取檔案，請確認 Phase 1 至 Phase 3 已成功執行！');
end

load(cachePath, 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC';
load(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all');

agent_aggressive   = load(fullfile(configObj.ModelDir, 'CIO_Aggressive.mat')).agent_aggressive;
agent_balanced     = load(fullfile(configObj.ModelDir, 'CIO_Balanced.mat')).agent_balanced;
agent_conservative = load(fullfile(configObj.ModelDir, 'CIO_Conservative.mat')).agent_conservative;

fetcher = DataFetcher(configObj);
dataStruct = fetcher.fetch_data();
Opens_Raw = dataStruct.Opens;

seqLen = configObj.SeqLen;
numDaysRaw = length(Dates_Active);
valid_idx = seqLen : numDaysRaw;

% 20 日移動平均平滑崩盤機率
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
if isempty(spy_idx), spy_idx = 1; end

%% 2. 預計算 CIO 5 維宏觀感知狀態
disp('--- 步驟 2：預計算 CIO 5 維宏觀狀態空間 ---');
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

Train_Start_Date = datetime('2006-01-01', 'TimeZone', 'UTC');
idx_train_start = find(Dates_Active >= Train_Start_Date, 1);
valid_start_t = max(252, idx_train_start);
OOS_Date = datetime('2022-01-01', 'TimeZone', 'UTC');
idx_OOS_start = find(Dates_Active >= OOS_Date, 1);

%% 3. 定義四組消融實驗組別
group_names = { ...
    'Group A (3-Track Ensemble - 提議模型)', ...
    'Group B (Single Balanced Agent)', ...
    'Group C (Single Aggressive Agent)', ...
    'Group D (Single Conservative Agent)' ...
};
num_groups = length(group_names);

daily_port_rets = zeros(numDays, num_groups, 'single');
port_curves     = ones(numDays, num_groups, 'single');
cash_curves     = zeros(numDays, num_groups, 'single');

opt_guard       = configObj.Guardrail_CrashProb;
top_k           = configObj.Top_K_Assets;
fallback_w_time = configObj.Expert_Time_Weight;
base_frict      = configObj.MoE_FrictionMask;

%% 4. 執行四組並行回測撮合模擬 (Open-to-Open)
disp('--- 步驟 3：執行嚴格因果律回測模擬 (四組對照) ---');

for g = 1:num_groups
    prev_assets = zeros(numTickers, 1, 'single');
    prev_cash = 1.0;
    
    for t = valid_start_t : numDays - 2
        % OOS 邊界重置
        if t == idx_OOS_start
            port_curves(t, g) = 1.0;
            prev_assets = zeros(numTickers, 1, 'single');
            prev_cash = 1.0;
        end
        
        state_2d = CIO_State(:, t);
        state_2d(5) = prev_cash;
        
        % 提取三位代理人前向決策
        [act_agg, ~] = agent_aggressive.get_actions(state_2d, 0);
        [act_bal, ~] = agent_balanced.get_actions(state_2d, 0);
        [act_con, ~] = agent_conservative.get_actions(state_2d, 0);
        
        % ★ 依組別決定動作合成邏輯
        switch g
            case 1 % Group A: 3-Track Ensemble (動態路由)
                if P_crash_smooth(t) > opt_guard
                    prob_con = 0.80; prob_bal = 0.15; prob_agg = 0.05;
                elseif vol20(t) > 0.20
                    prob_con = 0.20; prob_bal = 0.60; prob_agg = 0.20;
                else
                    prob_con = 0.10; prob_bal = 0.30; prob_agg = 0.60;
                end
                act_final = act_agg * prob_agg + act_bal * prob_bal + act_con * prob_con;
                
            case 2 % Group B: Pure Balanced
                act_final = act_bal;
                
            case 3 % Group C: Pure Aggressive
                act_final = act_agg;
                
            case 4 % Group D: Pure Conservative
                act_final = act_con;
        end
        
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
        
        w_cash = target_cash; 
        rem_w  = 1.0 - w_cash;
        cash_curves(t, g) = w_cash;
        
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
        
        % 慣性摩擦過濾
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
        
        % 日頻交易成本
        current_vol_daily = vol20(t) / sqrt(252);
        tc_rate = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff * current_vol_daily);
        cost = sum(abs(asset_w - prev_assets)) * tc_rate;
        
        % Open-to-Open 跨日報酬結算
        ret_t1 = (Opens_Active(t+2, :) - Opens_Active(t+1, :)) ./ (Opens_Active(t+1, :) + 1e-8);
        ret_t1(isnan(ret_t1) | isinf(ret_t1)) = 0;
        
        step_port_ret = sum(asset_w .* ret_t1') - cost;
        daily_port_rets(t+1, g) = step_port_ret;
        port_curves(t+1, g) = port_curves(t, g) * (1 + step_port_ret);
        
        prev_assets = asset_w;
        prev_cash   = w_cash;
    end
end

%% 5. 統計檢定與消融評估報告 (IS 與 OOS 雙區間並列)
is_ret_idx  = (valid_start_t + 1) : idx_OOS_start;
oos_ret_idx = (idx_OOS_start + 1) : (numDays - 1);

is_curve_idx  = valid_start_t : (idx_OOS_start - 1);
oos_curve_idx = idx_OOS_start : (numDays - 1);

spy_ret_is  = (Opens_Active(is_ret_idx+1, spy_idx) - Opens_Active(is_ret_idx, spy_idx)) ./ (Opens_Active(is_ret_idx, spy_idx) + 1e-8);
spy_ret_oos = (Opens_Active(oos_ret_idx+1, spy_idx) - Opens_Active(oos_ret_idx, spy_idx)) ./ (Opens_Active(oos_ret_idx, spy_idx) + 1e-8);

disp('========================================================================================');
disp('📊 【P2-3 PPO 三軌集成與單代理人消融實驗綜合報告】');
disp('========================================================================================');

% --- 區間 A: In-Sample (2006 ~ 2021) ---
fprintf('\n▶ 【In-Sample 樣本內表現 (2006-01-01 至 2021-12-31，共 %d 天)】\n', length(is_ret_idx));
for g = 1:num_groups
    v = port_curves(is_curve_idx, g);
    r = daily_port_rets(is_ret_idx, g);
    tot_ret = (v(end)/v(1) - 1) * 100;
    mdd = min((v - cummax(v)) ./ cummax(v)) * 100;
    sharpe = mean(r) / (std(r) + 1e-8) * sqrt(252);
    ann_ret = ((1 + tot_ret/100)^(252 / length(is_ret_idx)) - 1) * 100;
    calmar = ann_ret / (abs(mdd) + 1e-8);
    
    fprintf('  %-36s | 總報酬: %+7.2f%% | MDD: %6.2f%% | Sharpe: %+5.2f | Calmar: %5.2f\n', ...
        group_names{g}, tot_ret, mdd, sharpe, calmar);
end

% IS 顯著性檢定
r_ens_is = daily_port_rets(is_ret_idx, 1);
r_bal_is = daily_port_rets(is_ret_idx, 2);
r_agg_is = daily_port_rets(is_ret_idx, 3);
r_con_is = daily_port_rets(is_ret_idx, 4);

[~, p_bal_is_naive, ~, s_bal_is] = ttest(r_ens_is, r_bal_is);
[t_bal_is_hac, p_bal_is_hac]     = hac_significance_test(r_ens_is - r_bal_is);
[t_agg_is_hac, p_agg_is_hac]     = hac_significance_test(r_ens_is - r_agg_is);
[t_con_is_hac, p_con_is_hac]     = hac_significance_test(r_ens_is - r_con_is);

d_ens_bal_is = (r_bal_is - spy_ret_is).^2 - (r_ens_is - spy_ret_is).^2;
[dm_stat_is, p_dm_is] = hac_significance_test(d_ens_bal_is);

fprintf('\n  [IS 顯著性檢定：Group A (三軌集成) vs 單一代理人]\n');
fprintf('  > vs Group B (純 Balanced)     : Naive t=%.3f (p=%.4f) | HAC t=%+.3f (p=%.4f) | HAC-DM=%.3f (p=%.4f)\n', ...
    s_bal_is.tstat, p_bal_is_naive, t_bal_is_hac, p_bal_is_hac, dm_stat_is, p_dm_is);
fprintf('  > vs Group C (純 Aggressive)   : HAC t=%+.3f (p=%.4f)\n', t_agg_is_hac, p_agg_is_hac);
fprintf('  > vs Group D (純 Conservative) : HAC t=%+.3f (p=%.4f)\n', t_con_is_hac, p_con_is_hac);

% --- 區間 B: Out-of-Sample (2022 ~ 至今) ---
fprintf('\n▶ 【Out-of-Sample 樣本外盲測 (2022-01-01 至 至今，共 %d 天)】\n', length(oos_ret_idx));
for g = 1:num_groups
    v = port_curves(oos_curve_idx, g);
    r = daily_port_rets(oos_ret_idx, g);
    tot_ret = (v(end)/v(1) - 1) * 100;
    mdd = min((v - cummax(v)) ./ cummax(v)) * 100;
    sharpe = mean(r) / (std(r) + 1e-8) * sqrt(252);
    ann_ret = ((1 + tot_ret/100)^(252 / length(oos_ret_idx)) - 1) * 100;
    calmar = ann_ret / (abs(mdd) + 1e-8);
    
    fprintf('  %-36s | 總報酬: %+7.2f%% | MDD: %6.2f%% | Sharpe: %+5.2f | Calmar: %5.2f\n', ...
        group_names{g}, tot_ret, mdd, sharpe, calmar);
end

% OOS 顯著性檢定
r_ens_oos = daily_port_rets(oos_ret_idx, 1);
r_bal_oos = daily_port_rets(oos_ret_idx, 2);
r_agg_oos = daily_port_rets(oos_ret_idx, 3);
r_con_oos = daily_port_rets(oos_ret_idx, 4);

[~, p_bal_oos_naive, ~, s_bal_oos] = ttest(r_ens_oos, r_bal_oos);
[t_bal_oos_hac, p_bal_oos_hac]     = hac_significance_test(r_ens_oos - r_bal_oos);
[t_agg_oos_hac, p_agg_oos_hac]     = hac_significance_test(r_ens_oos - r_agg_oos);
[t_con_oos_hac, p_con_oos_hac]     = hac_significance_test(r_ens_oos - r_con_oos);

d_ens_bal_oos = (r_bal_oos - spy_ret_oos).^2 - (r_ens_oos - spy_ret_oos).^2;
[dm_stat_oos, p_dm_oos] = hac_significance_test(d_ens_bal_oos);

fprintf('\n  [OOS 顯著性檢定：Group A (三軌集成) vs 單一代理人]\n');
fprintf('  > vs Group B (純 Balanced)     : Naive t=%.3f (p=%.4f) | HAC t=%+.3f (p=%.4f) | HAC-DM=%.3f (p=%.4f)\n', ...
    s_bal_oos.tstat, p_bal_oos_naive, t_bal_oos_hac, p_bal_oos_hac, dm_stat_oos, p_dm_oos);
fprintf('  > vs Group C (純 Aggressive)   : HAC t=%+.3f (p=%.4f)\n', t_agg_oos_hac, p_agg_oos_hac);
fprintf('  > vs Group D (純 Conservative) : HAC t=%+.3f (p=%.4f)\n', t_con_oos_hac, p_con_oos_hac);

%% 6. 歷史極端危機壓力測試分析
disp('----------------------------------------------------------------------------------------');
disp('🌪️ 【歷史重大下行危機壓力測試剖析 (集成 vs 單一代理人)】');
disp('----------------------------------------------------------------------------------------');

crisis_events = { ...
    struct('name', '2008 金融海嘯', 'start', datetime('2007-10-01','TimeZone','UTC'), 'end', datetime('2009-03-31','TimeZone','UTC')), ...
    struct('name', '2020 COVID 崩盤', 'start', datetime('2020-01-01','TimeZone','UTC'), 'end', datetime('2020-04-30','TimeZone','UTC')), ...
    struct('name', '2022 升息與熊市', 'start', datetime('2022-01-01','TimeZone','UTC'), 'end', datetime('2022-10-31','TimeZone','UTC')) ...
};

for c = 1:length(crisis_events)
    ce = crisis_events{c};
    c_idx = find(Dates_Active >= ce.start & Dates_Active <= ce.end);
    
    if ~isempty(c_idx)
        fprintf('\n📌 危機事件：%s (%s 至 %s)\n', ce.name, datestr(ce.start, 'yyyy-mm-dd'), datestr(ce.end, 'yyyy-mm-dd'));
        
        spy_c_prices = Prices_Active(c_idx, spy_idx);
        spy_c_ret = (spy_c_prices(end)/spy_c_prices(1) - 1) * 100;
        spy_c_mdd = min((spy_c_prices - cummax(spy_c_prices)) ./ cummax(spy_c_prices)) * 100;
        fprintf('  > %-36s | 區間報酬: %+6.2f%% | 區間最大回撤: %6.2f%%\n', 'SPY Benchmark', spy_c_ret, spy_c_mdd);
        
        for g = 1:num_groups
            c_rets = daily_port_rets(c_idx, g);
            c_val  = cumprod(1 + c_rets);
            c_tot  = (c_val(end) - 1) * 100;
            c_mdd  = min((c_val - cummax(c_val)) ./ cummax(c_val)) * 100;
            avg_cash = mean(cash_curves(c_idx, g)) * 100;
            
            fprintf('  > %-36s | 區間報酬: %+6.2f%% | 區間最大回撤: %6.2f%% | 平均避險現金: %5.1f%%\n', ...
                group_names{g}, c_tot, c_mdd, avg_cash);
        end
    end
end

fprintf('\n========================================================================================\n');
fprintf('📌 【方法論與論文定位總結】\n');
fprintf('  1. 若三軌動態集成 (Group A) 與單一 Balanced 代理人 (Group B) 顯著性未達 p < 0.05，\n');
fprintf('     論文中應誠實表述為「三軌集成展示了分層強化學習 (HRL) 處理多風險偏好的工程架構設計」，\n');
fprintf('     並從極端體制（牛市進攻性 vs 熊市防禦性）之自適應切換視角進行質化論述。\n');
fprintf('========================================================================================\n\n');

%% 7. 產出視覺化圖表 (標準學術白底黑字格式)
disp('--- 步驟 7：產出 P2-3 PPO 三軌集成消融實驗報表 (白底黑字) ---');

fig = figure('Name', 'P2-3 PPO Ensemble Study', ...
    'Color', 'w', 'Position', [100, 100, 1150, 850], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

% -------------------------------------------------------------
% 子圖 1：對數淨值曲線對比 (3軌集成 vs 各單一代理人)
% -------------------------------------------------------------
subplot(3, 1, 1);
plot(Dates_Active(is_curve_idx), log10(port_curves(is_curve_idx, 1)), 'Color', '#D95319', 'LineWidth', 1.8, 'DisplayName', 'Group A (Ensemble)'); hold on;
plot(Dates_Active(is_curve_idx), log10(port_curves(is_curve_idx, 2)), 'Color', '#0072BD', 'LineWidth', 1.2, 'DisplayName', 'Group B (Balanced)');
plot(Dates_Active(is_curve_idx), log10(port_curves(is_curve_idx, 3)), 'Color', '#EDB120', 'LineWidth', 1.0, 'DisplayName', 'Group C (Aggressive)');
plot(Dates_Active(is_curve_idx), log10(port_curves(is_curve_idx, 4)), 'Color', '#77AC30', 'LineWidth', 1.0, 'DisplayName', 'Group D (Conservative)');

plot(Dates_Active(oos_curve_idx), log10(port_curves(oos_curve_idx, 1)), 'Color', '#D95319', 'LineWidth', 2.0, 'HandleVisibility', 'off');
plot(Dates_Active(oos_curve_idx), log10(port_curves(oos_curve_idx, 2)), 'Color', '#0072BD', 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(Dates_Active(oos_curve_idx), log10(port_curves(oos_curve_idx, 3)), 'Color', '#EDB120', 'LineWidth', 1.2, 'HandleVisibility', 'off');
plot(Dates_Active(oos_curve_idx), log10(port_curves(oos_curve_idx, 4)), 'Color', '#77AC30', 'LineWidth', 1.2, 'HandleVisibility', 'off');

xline(Dates_Active(idx_OOS_start), '--k', 'OOS Blind Test Start', 'LineWidth', 1.5, ...
    'LabelVerticalAlignment', 'bottom', 'Color', 'k', 'FontName', 'Helvetica', 'FontWeight', 'bold');

title('對數淨值曲線對比 (Log Equity Curve: Ensemble vs Single Agents)', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Log_{10}(Value)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);

set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% -------------------------------------------------------------
% 子圖 2：回撤幅度對比 (%)
% -------------------------------------------------------------
subplot(3, 1, 2);
for g = [3, 4, 2, 1]
    v = [port_curves(is_curve_idx, g); port_curves(oos_curve_idx, g)];
    dd = (v - cummax(v)) ./ cummax(v) * 100;
    plot(Dates_Active(valid_start_t:numDays-1), dd, 'LineWidth', 1.2, 'DisplayName', group_names{g}); hold on;
end
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 1.5, 'Color', 'k');

title('回撤幅度對比 (Drawdown % Profile)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Drawdown (%)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'southwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);

set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% -------------------------------------------------------------
% 子圖 3：持倉現金避險比例對比 (%)
% -------------------------------------------------------------
subplot(3, 1, 3);
plot(Dates_Active(valid_start_t:numDays-1), cash_curves(valid_start_t:numDays-1, 1) * 100, ...
    'Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'Ensemble Cash %'); hold on;
plot(Dates_Active(valid_start_t:numDays-1), cash_curves(valid_start_t:numDays-1, 2) * 100, ...
    'Color', '#0072BD', 'LineWidth', 1.1, 'LineStyle', ':', 'DisplayName', 'Balanced Cash %');
plot(Dates_Active(valid_start_t:numDays-1), cash_curves(valid_start_t:numDays-1, 4) * 100, ...
    'Color', '#77AC30', 'LineWidth', 1.1, 'LineStyle', '--', 'DisplayName', 'Conservative Cash %');
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 1.5, 'Color', 'k');

title('持倉現金避險比例對比 (Dynamic Cash Ratio %)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Cash (%)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylim([0, 105]);
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);

set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% -------------------------------------------------------------
% 300 DPI 高解析度白底匯出
% -------------------------------------------------------------
ensembleFigPath = fullfile(configObj.ResultDir, 'P2_3_PPO_Ensemble_Ablation.png');
exportgraphics(fig, ensembleFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化圖表 (白底黑字) 已儲存至: %s\n', ensembleFigPath);
close(fig);