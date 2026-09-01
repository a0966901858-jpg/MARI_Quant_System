% =========================================================================
% 腳本：Run_Ablation_SpaceExpert.m (P2-1 空間專家邊際貢獻消融實驗)
% 升級：Phase 15.5 (★ HAC-DM 檢定、Naive/HAC 雙版本檢定、IS/OOS 雙區間驗證)
% 職責：對比 Time-Only (1.0)、Balanced (0.5)、Space-Only (0.0) 之 IS/OOS 表現
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧪 [P2-1] 啟動空間專家 (DyGAT) 邊際貢獻消融實驗 (HAC-DM + 雙區間版)');
disp('=================================================================');

currentPath = fileparts(mfilename('fullpath'));
if isempty(currentPath), currentPath = pwd; end
projectRoot = currentPath; 
if ~exist(fullfile(projectRoot, 'configs'), 'dir'), projectRoot = fullfile(currentPath, '..'); end

addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'models')));
addpath(genpath(fullfile(projectRoot, 'utils'))); % ★ 掛載 HAC 統計工具箱
rehash toolboxcache;

configObj = Config();

% ★ 固定全域隨機種子
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入數據與時間軸
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');
load(cachePath, 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC';
load(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all');

fetcher = DataFetcher(configObj);
dataStruct = fetcher.fetch_data();
Opens_Raw = dataStruct.Opens;

seqLen = configObj.SeqLen;
numDaysRaw = length(Dates_Active);
valid_idx = seqLen : numDaysRaw;

P_crash_smooth = movmean(P_crash_all, [19, 0]);
P_crash_smooth = P_crash_smooth(valid_idx);

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

%% 2. 定義三組實驗參數
ablation_weights = [1.0, 0.5, 0.0];
group_names = {'Group A (Pure Time / BO)', 'Group B (Balanced 50/50)', 'Group C (Pure Space)'};
num_groups = length(ablation_weights);

daily_port_rets = zeros(numDays, num_groups, 'single');
port_curves = ones(numDays, num_groups, 'single');

opt_guard  = configObj.Guardrail_CrashProb;
top_k      = configObj.Top_K_Assets;
base_frict = configObj.MoE_FrictionMask;

%% 3. 執行平行回測模擬
for g = 1:num_groups
    w_time_target = ablation_weights(g);
    w_space_target = 1.0 - w_time_target;
    
    prev_assets = zeros(numTickers, 1, 'single');
    
    for t = valid_start_t : numDays - 2
        if t == idx_OOS_start
            port_curves(t, g) = 1.0;
            prev_assets = zeros(numTickers, 1, 'single');
        end
        
        % 護欄連續縮放
        target_cash = 0.0;
        guard_high = opt_guard;
        guard_low  = max(0, opt_guard - 0.10);
        if guard_high > guard_low
            risk_scale = max(0, min(1, (P_crash_smooth(t) - guard_low) / (guard_high - guard_low)));
            target_cash = max(target_cash, risk_scale);
        elseif P_crash_smooth(t) >= guard_high
            target_cash = 1.0;
        end
        
        rem_w = 1.0 - target_cash;
        
        % 空間與時序專家權重混合
        comb_p = P_time_M(:, t) * w_time_target + P_space_M(:, t) * w_space_target;
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
            total_cap = active_sum + target_cash;
            asset_w = asset_w / total_cap;
        else
            asset_w(:) = 0;
        end
        
        % 日頻交易成本
        current_vol_daily = vol20(t) / sqrt(252);
        tc_rate = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff * current_vol_daily);
        cost = sum(abs(asset_w - prev_assets)) * tc_rate;
        
        ret_t1 = (Opens_Active(t+2, :) - Opens_Active(t+1, :)) ./ (Opens_Active(t+1, :) + 1e-8);
        ret_t1(isnan(ret_t1) | isinf(ret_t1)) = 0;
        
        step_port_ret = sum(asset_w .* ret_t1') - cost;
        daily_port_rets(t+1, g) = step_port_ret;
        port_curves(t+1, g) = port_curves(t, g) * (1 + step_port_ret);
        
        prev_assets = asset_w;
    end
end

%% 4. 統計檢定與消融評估報告 (IS 與 OOS 雙區間並列)
is_ret_idx  = (valid_start_t + 1) : idx_OOS_start;
oos_ret_idx = (idx_OOS_start + 1) : (numDays - 1);

is_curve_idx  = valid_start_t : (idx_OOS_start - 1);
oos_curve_idx = idx_OOS_start : (numDays - 1);

spy_ret_is  = (Opens_Active(is_ret_idx+1, spy_idx) - Opens_Active(is_ret_idx, spy_idx)) ./ (Opens_Active(is_ret_idx, spy_idx) + 1e-8);
spy_ret_oos = (Opens_Active(oos_ret_idx+1, spy_idx) - Opens_Active(oos_ret_idx, spy_idx)) ./ (Opens_Active(oos_ret_idx, spy_idx) + 1e-8);

disp('========================================================================================');
disp('📊 【P2-1 空間專家 (DyGAT) 邊際貢獻消融實驗綜合報告】');
disp('========================================================================================');

% --- 區間 A: In-Sample (2006 ~ 2021) ---
fprintf('\n▶ 【In-Sample 樣本內表現 (2006-01-01 至 2021-12-31，共 %d 天)】\n', length(is_ret_idx));
for g = 1:num_groups
    v = port_curves(is_curve_idx, g);
    r = daily_port_rets(is_ret_idx, g);
    tot_ret = (v(end)/v(1) - 1) * 100;
    mdd = min((v - cummax(v)) ./ cummax(v)) * 100;
    sharpe = mean(r) / (std(r) + 1e-8) * sqrt(252);
    fprintf('  %-26s | 總報酬: %+7.2f%% | MDD: %6.2f%% | 年化 Sharpe: %+5.2f\n', ...
        group_names{g}, tot_ret, mdd, sharpe);
end

% IS 顯著性檢定
r_base_is = daily_port_rets(is_ret_idx, 1);
r_symm_is = daily_port_rets(is_ret_idx, 2);
r_spac_is = daily_port_rets(is_ret_idx, 3);

[~, p_symm_is_naive, ~, s_symm_is] = ttest(r_base_is, r_symm_is);
[~, p_spac_is_naive, ~, s_spac_is] = ttest(r_base_is, r_spac_is);
[t_symm_is_hac, p_symm_is_hac]     = hac_significance_test(r_base_is - r_symm_is);
[t_spac_is_hac, p_spac_is_hac]     = hac_significance_test(r_base_is - r_spac_is);

d_symm_is = (r_base_is - spy_ret_is).^2 - (r_symm_is - spy_ret_is).^2;
[dm_stat_is, p_dm_is] = hac_significance_test(d_symm_is);

fprintf('\n  [IS 顯著性檢定：Group A (純時序) vs 其他組別]\n');
fprintf('  > vs Group B (50/50)  : Naive t=%.3f (p=%.4f) | HAC t=%+.3f (p=%.4f) | HAC-DM=%.3f (p=%.4f)\n', ...
    s_symm_is.tstat, p_symm_is_naive, t_symm_is_hac, p_symm_is_hac, dm_stat_is, p_dm_is);
fprintf('  > vs Group C (純空間) : Naive t=%.3f (p=%.4f) | HAC t=%+.3f (p=%.4f)\n', ...
    s_spac_is.tstat, p_spac_is_naive, t_spac_is_hac, p_spac_is_hac);

% --- 區間 B: Out-of-Sample (2022 ~ 至今) ---
fprintf('\n▶ 【Out-of-Sample 樣本外盲測 (2022-01-01 至 至今，共 %d 天)】\n', length(oos_ret_idx));
for g = 1:num_groups
    v = port_curves(oos_curve_idx, g);
    r = daily_port_rets(oos_ret_idx, g);
    tot_ret = (v(end)/v(1) - 1) * 100;
    mdd = min((v - cummax(v)) ./ cummax(v)) * 100;
    sharpe = mean(r) / (std(r) + 1e-8) * sqrt(252);
    fprintf('  %-26s | 總報酬: %+7.2f%% | MDD: %6.2f%% | 年化 Sharpe: %+5.2f\n', ...
        group_names{g}, tot_ret, mdd, sharpe);
end

% OOS 顯著性檢定
r_base_oos = daily_port_rets(oos_ret_idx, 1);
r_symm_oos = daily_port_rets(oos_ret_idx, 2);
r_spac_oos = daily_port_rets(oos_ret_idx, 3);

[~, p_symm_oos_naive, ~, s_symm_oos] = ttest(r_base_oos, r_symm_oos);
[~, p_spac_oos_naive, ~, s_spac_oos] = ttest(r_base_oos, r_spac_oos);
[t_symm_oos_hac, p_symm_oos_hac]     = hac_significance_test(r_base_oos - r_symm_oos);
[t_spac_oos_hac, p_spac_oos_hac]     = hac_significance_test(r_base_oos - r_spac_oos);

d_symm_oos = (r_base_oos - spy_ret_oos).^2 - (r_symm_oos - spy_ret_oos).^2;
[dm_stat_oos, p_dm_oos] = hac_significance_test(d_symm_oos);

fprintf('\n  [OOS 顯著性檢定：Group A (純時序) vs 其他組別]\n');
fprintf('  > vs Group B (50/50)  : Naive t=%.3f (p=%.4f) | HAC t=%+.3f (p=%.4f) | HAC-DM=%.3f (p=%.4f)\n', ...
    s_symm_oos.tstat, p_symm_oos_naive, t_symm_oos_hac, p_symm_oos_hac, dm_stat_oos, p_dm_oos);
fprintf('  > vs Group C (純空間) : Naive t=%.3f (p=%.4f) | HAC t=%+.3f (p=%.4f)\n', ...
    s_spac_oos.tstat, p_spac_oos_naive, t_spac_oos_hac, p_spac_oos_hac);

fprintf('\n----------------------------------------------------------------------------------------\n');
fprintf('📌 【方法論聲明與客觀結論】\n');
fprintf('  1. 本實驗之 Top_K 與護欄參數取自主線針對 Time-Only 尋優之結果，Group C (純空間) 之表現\n');
fprintf('     應客觀解讀為「非自身最佳化條件下的下界估計 (Lower-Bound Estimate)」。\n');
if p_symm_oos_hac > 0.05 && p_symm_is_hac > 0.05
    fprintf('  2. 統計結論：在 HAC 消除序列自相關後，IS 與 OOS 雙區間之 Group B (50/50) 與 Group A (純時序)\n');
    fprintf('     均無統計顯著差異 (p > 0.05)。量化證實在當前特徵訊號強度下，加入 DyGAT 空間專家\n');
    fprintf('     未能提供可偵測的額外邊際貢獻，此結果與 Phase 3 GBDT AUC≈0.50 及 BO 邊界解三方印證！\n');
else
    fprintf('  2. 統計結論：空間專家在部分週期展現出統計顯著影響 (p <= 0.05)。\n');
end
fprintf('========================================================================================\n\n');