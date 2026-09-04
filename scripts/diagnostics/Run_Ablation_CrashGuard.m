% =========================================================================
% 腳本：Run_Ablation_CrashGuard.m (P2-2 崩盤護欄與宏觀擇時消融實驗)
% 升級：Phase 15.5 Stage 2.5 修復版 (★ 修正護欄雜訊死區、動態錨定非危機底線、
%       消除機械性全現金失真、HAC-DM 檢定、極端危機壓力測試)
% 職責：驗證宏觀崩盤護欄在 IS / OOS 與歷史危機窗口下的尾端風險防禦邊際
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧪 [P2-2] 啟動崩盤護欄與宏觀擇時邊際消融實驗 (雜訊死區修復版)');
disp('=================================================================');

%% 0. 環境路徑掛載與隨機種子鎖定
currentFile = mfilename('fullpath');
if isempty(currentFile), currentPath = pwd; else, currentPath = fileparts(currentFile); end
projectRoot = currentPath;
while ~exist(fullfile(projectRoot, 'configs'), 'dir')
    parentDir = fileparts(projectRoot);
    if strcmp(parentDir, projectRoot)
        error('❌ 找不到 MARI_Quant_System 專案根目錄 (包含 configs/)！');
    end
    projectRoot = parentDir;
end

addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'models')));
addpath(genpath(fullfile(projectRoot, 'utils')));
rehash toolboxcache;

configObj = Config();
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入面板資料與時間軸嚴格對齊
disp('--- 步驟 1：載入全域快取與時間軸對齊 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');
if ~exist(cachePath, 'file') || ~exist(gbdtPath, 'file')
    error('❌ 找不到前置快取檔案，請確認 Phase 1 至 Phase 3 已成功執行！');
end

load(cachePath, 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC';
load(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all');

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
spy_prices = Prices_Active(:, spy_idx);
spy_rets = [0; diff(spy_prices) ./ spy_prices(1:end-1)];
spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
vol20 = movstd(spy_rets, [19, 0], 1) * sqrt(252);

Train_Start_Date = datetime('2006-01-01', 'TimeZone', 'UTC');
idx_train_start = find(Dates_Active >= Train_Start_Date, 1);
valid_start_t = max(252, idx_train_start);
OOS_Date = datetime('2022-01-01', 'TimeZone', 'UTC');
idx_OOS_start = find(Dates_Active >= OOS_Date, 1);

is_curve_idx  = valid_start_t : (idx_OOS_start - 1);
oos_curve_idx = idx_OOS_start : (numDays - 1);

%% 2. 核心修復：崩盤護欄基準率校準與雜訊死區 (Deadband) 計算
disp('--- 步驟 2：校準崩盤護欄雜訊底線與動態緩衝寬度 ---');

% 取得樣本內真實崩盤標籤（若未存檔則以 SPY 前向 10 日跌幅 < -5% 補齊）
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

% 擷取 IS 期間非崩盤日的日常背景雜訊分佈
is_non_crash_idx = is_curve_idx(Y_Crash_valid(is_curve_idx) == 0);
non_crash_p = P_crash_smooth(is_non_crash_idx);
tau_noise = prctile(non_crash_p, 75); % 75% 背景雜訊分位數

guard_high = configObj.Guardrail_CrashProb;

% 動態錨定下界：必須高於背景雜訊底線，且與 guard_high 保持有效緩衝寬度
guard_low = min(guard_high * 0.85, max(tau_noise, guard_high - 0.03));
if guard_low >= guard_high
    guard_low = guard_high * 0.85;
end

fprintf('  🔍 [護欄參數校準]\n');
fprintf('     > IS 非危機期 P(Crash) 75%% 雜訊分位數 (tau_noise) : %.4f\n', tau_noise);
fprintf('     > 硬熔斷上限 (Guard_High)                      : %.4f\n', guard_high);
fprintf('     > 死區縮放下限 (Guard_Low)                       : %.4f\n', guard_low);
fprintf('     > 有效連續緩衝區間寬度                           : %.4f\n', guard_high - guard_low);

%% 3. 定義三組消融實驗組別與回測參數
group_names = { ...
    'Group A (Continuous Scaling - 提議模型)', ...
    'Group B (Binary Hard Guard - 二元硬熔斷)', ...
    'Group C (Unprotected - 無護欄基線)' ...
};
num_groups = length(group_names);
daily_port_rets = zeros(numDays, num_groups, 'single');
port_curves     = ones(numDays, num_groups, 'single');
cash_curves     = zeros(numDays, num_groups, 'single');

top_k      = configObj.Top_K_Assets;
w_time     = configObj.Expert_Time_Weight;
w_space    = 1.0 - w_time;
base_frict = configObj.MoE_FrictionMask;

%% 4. 執行三組對照回測撮合模擬 (Open-to-Open)
disp('--- 步驟 3：執行嚴格因果律回測模擬 (三組並行) ---');
for g = 1:num_groups
    prev_assets = zeros(numTickers, 1, 'single');
    
    for t = valid_start_t : numDays - 2
        % OOS 邊界重置
        if t == idx_OOS_start
            port_curves(t, g) = 1.0;
            prev_assets = zeros(numTickers, 1, 'single');
        end
        
        p_c = P_crash_smooth(t);
        
        switch g
            case 1 % Group A: Continuous Scaling (具備死區之連續縮放)
                if p_c <= guard_low
                    target_cash = 0.0; % 處於日常雜訊帶，嚴格不減倉
                elseif p_c >= guard_high
                    target_cash = 1.0; % 超過硬熔斷閾值，100% 避險
                else
                    target_cash = (p_c - guard_low) / (guard_high - guard_low);
                end
                
            case 2 % Group B: Binary Hard Guard (二元硬熔斷)
                if p_c >= guard_high
                    target_cash = 1.0;
                else
                    target_cash = 0.0;
                end
                
            case 3 % Group C: Unprotected (無護欄基線)
                target_cash = 0.0;
        end
        
        rem_w = 1.0 - target_cash;
        cash_curves(t, g) = target_cash;
        
        % 專家選股權重混合
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
        
        % 慣性換手摩擦過濾
        turnover = abs(asset_w - prev_assets);
        ignore_mask = turnover < base_frict;
        asset_w(ignore_mask) = prev_assets(ignore_mask);
        
        active_sum = sum(asset_w);
        if active_sum > 0
            total_cap = active_sum + target_cash;
            asset_w = asset_w / total_cap;
        else
            asset_w(:) = 0;
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
    end
end

%% 5. 統計檢定與消融評估報告 (IS 與 OOS 雙區間並列)
is_ret_idx  = (valid_start_t + 1) : idx_OOS_start;
oos_ret_idx = (idx_OOS_start + 1) : (numDays - 1);

spy_ret_is  = (Opens_Active(is_ret_idx+1, spy_idx) - Opens_Active(is_ret_idx, spy_idx)) ./ (Opens_Active(is_ret_idx, spy_idx) + 1e-8);
spy_ret_oos = (Opens_Active(oos_ret_idx+1, spy_idx) - Opens_Active(oos_ret_idx, spy_idx)) ./ (Opens_Active(oos_ret_idx, spy_idx) + 1e-8);

disp('========================================================================================');
disp('📊 【P2-2 崩盤護欄與宏觀擇時消融實驗綜合報告 (雜訊死區修復版)】');
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

% 輸出 Group A 護欄觸發行為統計 (診斷用)
is_cash_A = cash_curves(is_curve_idx, 1);
fprintf('  -> Group A IS 避險統計: 平均現金: %.1f%% | 減倉觸發日 (Cash>0): %.1f%% | 完全避險日 (Cash=1): %.1f%%\n', ...
    mean(is_cash_A) * 100, mean(is_cash_A > 0) * 100, mean(is_cash_A >= 0.999) * 100);

% IS 顯著性檢定
r_base_is = daily_port_rets(is_ret_idx, 1);
r_bin_is  = daily_port_rets(is_ret_idx, 2);
r_none_is = daily_port_rets(is_ret_idx, 3);
[~, p_none_is_naive, ~, s_none_is] = ttest(r_base_is, r_none_is);
[t_none_is_hac, p_none_is_hac]     = hac_significance_test(r_base_is - r_none_is);
[t_bin_is_hac, p_bin_is_hac]       = hac_significance_test(r_base_is - r_bin_is);
d_guard_is = (r_none_is - spy_ret_is).^2 - (r_base_is - spy_ret_is).^2;
[dm_stat_is, p_dm_is] = hac_significance_test(d_guard_is);

fprintf('\n  [IS 顯著性檢定：Group A (連續護欄) vs 對照組]\n');
fprintf('  > vs Group C (無護欄基線) : Naive t=%.3f (p=%.4f) | HAC t=%+.3f (p=%.4f) | HAC-DM=%.3f (p=%.4f)\n', ...
    s_none_is.tstat, p_none_is_naive, t_none_is_hac, p_none_is_hac, dm_stat_is, p_dm_is);
fprintf('  > vs Group B (二元硬熔斷) : HAC t=%+.3f (p=%.4f)\n', t_bin_is_hac, p_bin_is_hac);

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

oos_cash_A = cash_curves(oos_curve_idx, 1);
fprintf('  -> Group A OOS 避險統計: 平均現金: %.1f%% | 減倉觸發日 (Cash>0): %.1f%% | 完全避險日 (Cash=1): %.1f%%\n', ...
    mean(oos_cash_A) * 100, mean(oos_cash_A > 0) * 100, mean(oos_cash_A >= 0.999) * 100);

% OOS 顯著性檢定
r_base_oos = daily_port_rets(oos_ret_idx, 1);
r_bin_oos  = daily_port_rets(oos_ret_idx, 2);
r_none_oos = daily_port_rets(oos_ret_idx, 3);
[~, p_none_oos_naive, ~, s_none_oos] = ttest(r_base_oos, r_none_oos);
[t_none_oos_hac, p_none_oos_hac]     = hac_significance_test(r_base_oos - r_none_oos);
[t_bin_oos_hac, p_bin_oos_hac]       = hac_significance_test(r_base_oos - r_bin_oos);
d_guard_oos = (r_none_oos - spy_ret_oos).^2 - (r_base_oos - spy_ret_oos).^2;
[dm_stat_oos, p_dm_oos] = hac_significance_test(d_guard_oos);

fprintf('\n  [OOS 顯著性檢定：Group A (連續護欄) vs 對照組]\n');
fprintf('  > vs Group C (無護欄基線) : Naive t=%.3f (p=%.4f) | HAC t=%+.3f (p=%.4f) | HAC-DM=%.3f (p=%.4f)\n', ...
    s_none_oos.tstat, p_none_oos_naive, t_none_oos_hac, p_none_oos_hac, dm_stat_oos, p_dm_oos);
fprintf('  > vs Group B (二元硬熔斷) : HAC t=%+.3f (p=%.4f)\n', t_bin_oos_hac, p_bin_oos_hac);

%% 6. 歷史極端危機壓力測試分析 (Crisis Stress Tests)
disp('----------------------------------------------------------------------------------------');
disp('🌪️ 【歷史重大下行危機壓力測試剖析 (MDD 與累計回撤)】');
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
        fprintf('  > %-34s | 區間報酬: %+6.2f%% | 區間最大回撤: %6.2f%%\n', 'SPY Benchmark', spy_c_ret, spy_c_mdd);
        
        for g = 1:num_groups
            c_rets = daily_port_rets(c_idx, g);
            c_val  = cumprod(1 + c_rets);
            c_tot  = (c_val(end) - 1) * 100;
            c_mdd  = min((c_val - cummax(c_val)) ./ cummax(c_val)) * 100;
            avg_cash = mean(cash_curves(c_idx, g)) * 100;
            
            fprintf('  > %-34s | 區間報酬: %+6.2f%% | 區間最大回撤: %6.2f%% | 平均避險現金: %5.1f%%\n', ...
                group_names{g}, c_tot, c_mdd, avg_cash);
        end
    end
end
fprintf('========================================================================================\n\n');

%% 7. 產出視覺化圖表 (標準學術白底黑字格式)
disp('--- 步驟 4：產出 P2-2 崩盤護欄消融實驗報表 (白底黑字) ---');
fig = figure('Name', 'P2-2 Crash Guard Ablation Study (Calibrated)', ...
    'Color', 'w', 'Position', [100, 100, 1150, 850], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

% 子圖 1：對數淨值曲線對比
subplot(3, 1, 1);
plot(Dates_Active(is_curve_idx), log10(port_curves(is_curve_idx, 1)), 'Color', '#D95319', 'LineWidth', 1.6, 'DisplayName', 'Group A (Continuous)'); hold on;
plot(Dates_Active(is_curve_idx), log10(port_curves(is_curve_idx, 2)), 'Color', '#EDB120', 'LineWidth', 1.2, 'DisplayName', 'Group B (Binary)');
plot(Dates_Active(is_curve_idx), log10(port_curves(is_curve_idx, 3)), 'Color', '#7E2F8E', 'LineWidth', 1.2, 'DisplayName', 'Group C (Unprotected)');
plot(Dates_Active(oos_curve_idx), log10(port_curves(oos_curve_idx, 1)), 'Color', '#D95319', 'LineWidth', 2.0, 'HandleVisibility', 'off');
plot(Dates_Active(oos_curve_idx), log10(port_curves(oos_curve_idx, 2)), 'Color', '#EDB120', 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(Dates_Active(oos_curve_idx), log10(port_curves(oos_curve_idx, 3)), 'Color', '#7E2F8E', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(Dates_Active(idx_OOS_start), '--k', 'OOS Blind Test Start', 'LineWidth', 1.5, ...
    'LabelVerticalAlignment', 'bottom', 'Color', 'k', 'FontName', 'Helvetica', 'FontWeight', 'bold');
title('對數淨值曲線對比 (Log Equity Curve: Continuous vs Binary vs Unprotected)', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Log_{10}(Value)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% 子圖 2：回撤幅度對比 (%)
subplot(3, 1, 2);
for g = [3, 2, 1]
    v = [port_curves(is_curve_idx, g); port_curves(oos_curve_idx, g)];
    dd = (v - cummax(v)) ./ cummax(v) * 100;
    plot(Dates_Active(valid_start_t:numDays-1), dd, 'LineWidth', 1.3, 'DisplayName', group_names{g}); hold on;
end
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 1.5, 'Color', 'k');
title('回撤幅度對比 (Drawdown % Profile)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Drawdown (%)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'southwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% 子圖 3：避險現金比例、死區與平滑崩盤機率 (%)
subplot(3, 1, 3);
area(Dates_Active(valid_start_t:numDays-1), cash_curves(valid_start_t:numDays-1, 1) * 100, ...
    'FaceColor', '#77AC30', 'FaceAlpha', 0.45, 'EdgeColor', 'none', 'DisplayName', 'Group A Continuous Cash %'); hold on;
plot(Dates_Active(valid_start_t:numDays-1), P_crash_smooth(valid_start_t:numDays-1) * 100, ...
    'Color', '#A2142F', 'LineWidth', 1.3, 'DisplayName', 'Smoothed P(Crash) %');
yline(guard_high * 100, '--r', sprintf('Hard Hedge Guard High (%.2f%%)', guard_high * 100), ...
    'LineWidth', 1.3, 'DisplayName', 'Hard Hedge Target');
yline(guard_low * 100, ':r', sprintf('Scaling Start Guard Low (%.2f%%)', guard_low * 100), ...
    'LineWidth', 1.1, 'DisplayName', 'Scaling Start (Deadband)');
yline(tau_noise * 100, '-.b', sprintf('Noise Floor \\tau_{noise} (%.2f%%)', tau_noise * 100), ...
    'LineWidth', 1.0, 'DisplayName', 'IS Noise Floor (p75)');
xline(Dates_Active(idx_OOS_start), '--k', 'LineWidth', 1.5, 'Color', 'k');
title('避險現金比例與平滑崩盤機率 (Crash Probability & Dynamic Deadband Cash Allocation)', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Ratio (%)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylim([0, 105]);
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 9);
grid on; box on;

% 匯出圖表
guardFigPath = fullfile(configObj.ResultDir, 'P2_2_CrashGuard_Ablation.png');
exportgraphics(fig, guardFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化圖表已成功儲存至: %s\n', guardFigPath);
close(fig);

disp('=================================================================');
disp('🎯 [P2-2] 消融實驗執行完畢！');
disp('=================================================================');