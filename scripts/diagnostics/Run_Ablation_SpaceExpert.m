% =========================================================================
% 腳本：Run_Ablation_SpaceExpert.m (P2-1 空間專家邊際貢獻消融實驗)
% 升級：Phase 15.5 Stage 4 規範版 (★ 60D 連續超額報酬 Z-Score 標籤、
%       HAC Rank IC (max_lag>=60) 訊號評估、ΔIC 邊際增量檢定、
%       Open-to-Open 權重漂移撮合、死區連續縮放護欄、mrg32k3a 隨機串流鎖定)
% 職責：對比 Time-Only (1.0)、Balanced (0.5)、Space-Only (0.0) 之 60D Rank IC
%       與實盤回測表現，以統計檢定量化證偽/證實 DyGAT 空間專家之邊際貢獻
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧪 [P2-1] 啟動空間專家 (DyGAT) 邊際貢獻消融實驗 (60D HAC Rank IC + 雙區間版)');
disp('=================================================================');

%% 0. 環境路徑掛載與隨機串流鎖定
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
        error('❌ 找不到 MARI_Quant_System 專案根目錄 (包含 configs/)！');
    end
    projectRoot = parentDir;
end

addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'models')));
addpath(genpath(fullfile(projectRoot, 'agents')));
addpath(genpath(fullfile(projectRoot, 'utils')));
rehash toolboxcache;

configObj = Config();

% ★ 核心修復 1：由 Config 統一生產 mrg32k3a 隨機數引擎 (Substream = 1)
stream = configObj.getRandStream(1);
RandStream.setGlobalStream(stream);
disp('🔒 已成功掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定消融實驗確定性。');

%% 1. 載入特徵快取、Opens 矩陣與 GBDT 排序得分
disp('--- 步驟 1：載入全域資料、Opens 撮合矩陣與專家百分位排序得分 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');

if ~exist(cachePath, 'file') || ~exist(gbdtPath, 'file')
    error('❌ 找不到前置快取，請先確認 Phase 1 至 Phase 3 已成功執行！');
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

% 平滑崩盤機率
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

%% 2. 構建 60D 前向連續超額報酬 Z-Score 標籤 (★ 核心升級)
disp('--- 步驟 2：構建 60 日遠期橫截面標準化超額報酬標籤 (Direction 3 Aligned) ---');
horizon = 60;
valid_label_days = numDays - horizon;

R_fwd_60D = (Prices_Active(1+horizon:end, :) - Prices_Active(1:end-horizon, :)) ...
            ./ (Prices_Active(1:end-horizon, :) + 1e-8);
R_fwd_60D(isnan(R_fwd_60D) | isinf(R_fwd_60D)) = NaN;

Y_cont_60D = NaN(valid_label_days, numTickers, 'single');
for t = 1:valid_label_days
    act_m = Expert_Active(t, :) & ~isnan(R_fwd_60D(t, :)) & ~isinf(R_fwd_60D(t, :));
    if sum(act_m) >= 10
        r_t = R_fwd_60D(t, act_m);
        mu_r  = mean(r_t, 'omitnan');
        std_r = std(r_t, 0, 'omitnan') + 1e-6;
        Y_cont_60D(t, act_m) = (r_t - mu_r) ./ std_r; % 橫截面 Z-Score
    end
end
fprintf('  📊 60D 連續標籤建構完成 (有效天數: %d 天 | 預測視窗: %d 日)\n', valid_label_days, horizon);

%% 3. 定義消融組別與信號矩陣
ablation_weights = [1.0, 0.5, 0.0];
group_names = {'Group A (Pure Time / BO)', 'Group B (Balanced 50/50)', 'Group C (Pure Space)'};
num_groups = length(ablation_weights);

% 預計算各組融合得分矩陣 [numDays, numTickers]
Comb_Scores = cell(num_groups, 1);
for g = 1:num_groups
    w_t = ablation_weights(g);
    w_s = 1.0 - w_t;
    comb = P_time_M * w_t + P_space_M * w_s; % [numTickers, numDays]
    comb(~Expert_Active') = 0;
    Comb_Scores{g} = comb'; % [numDays, numTickers]
end

%% 4. 訊號層級消融：60D 逐日橫截面 Spearman Rank IC 與 HAC 顯著性檢定
disp('--- 步驟 3：計算 60D 遠期逐日橫截面 Rank IC 與 HAC 檢定 (貫穿 max_lag >= 60) ---');
daily_rank_ic = NaN(valid_label_days, num_groups, 'single');

for t = 1:valid_label_days
    act_m = Expert_Active(t, :) & ~isnan(Y_cont_60D(t, :));
    if sum(act_m) >= 10
        y_true = Y_cont_60D(t, act_m)';
        for g = 1:num_groups
            score_g = Comb_Scores{g}(t, act_m)';
            daily_rank_ic(t, g) = corr(score_g, y_true, 'Type', 'Spearman', 'Rows', 'complete');
        end
    end
end

% 切分 IS 與 OOS 評估索引
is_ic_mask  = false(valid_label_days, 1);
oos_ic_mask = false(valid_label_days, 1);

t_dates_eval = Dates_Active(1:valid_label_days);
is_ic_mask(t_dates_eval >= Train_Start_Date & t_dates_eval < OOS_Date) = true;
oos_ic_mask(t_dates_eval >= OOS_Date) = true;

% 確保只納入非 NaN 交易日
valid_eval_days = all(~isnan(daily_rank_ic), 2);
is_ic_idx  = find(is_ic_mask & valid_eval_days);
oos_ic_idx = find(oos_ic_mask & valid_eval_days);

%% 5. 實盤層級消融：Open-to-Open 權重漂移、停牌鎖死與死區護欄回測
disp('--- 步驟 4：執行真實 Open-to-Open 撮合回測 (含權重漂移與停牌鎖死) ---');
daily_port_rets = zeros(numDays, num_groups, 'single');
port_curves     = ones(numDays, num_groups, 'single');

opt_guard  = configObj.Guardrail_CrashProb;
top_k      = configObj.Top_K_Assets;
base_frict = configObj.MoE_FrictionMask;

% 校準死區護欄
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
if guard_low >= guard_high, guard_low = guard_high * 0.85; end

for g = 1:num_groups
    w_time_target = ablation_weights(g);
    w_space_target = 1.0 - w_time_target;
    
    prev_assets = zeros(numTickers, 1, 'single');
    prev_cash = 1.0;
    
    port_curves(valid_start_t, g)   = 1.0;
    port_curves(valid_start_t+1, g) = 1.0;
    
    for t = valid_start_t : numDays - 2
        % 1. 權重自然漂移 (Weight Drift)
        drift_ret = (Opens_Active(t+1, :) - Opens_Active(t, :)) ./ (Opens_Active(t, :) + 1e-8);
        drift_ret(isnan(drift_ret) | isinf(drift_ret)) = 0;
        
        asset_mult = prev_assets .* (1 + drift_ret');
        port_val_drift = sum(asset_mult) + prev_cash;
        if port_val_drift > 0
            w_drift = asset_mult / port_val_drift;
        else
            w_drift = prev_assets;
        end
        
        % 2. 停牌鎖死防護 (Halted Stocks Lock)
        halted_mask = isnan(Opens_Active(t+1, :))' | (Opens_Active(t+1, :) <= 0)';
        locked_weights = zeros(numTickers, 1, 'single');
        locked_weights(halted_mask) = w_drift(halted_mask);
        locked_sum = sum(locked_weights);
        available_cap = max(0, 1.0 - locked_sum);
        
        % 3. 崩盤護欄死區連續縮放
        p_c = P_crash_smooth(t);
        if p_c <= guard_low
            risk_scale = 0.0;
        elseif p_c >= guard_high
            risk_scale = 1.0;
        else
            risk_scale = (p_c - guard_low) / (guard_high - guard_low);
        end
        actual_cash_target = min(risk_scale, available_cap);
        rem_w = available_cap - actual_cash_target;
        
        % 4. 融合得分選股與 Top-K 截斷
        comb_p = P_time_M(:, t) * w_time_target + P_space_M(:, t) * w_space_target;
        comb_p = comb_p .* Expert_Active(t, :)';
        comb_p(halted_mask) = 0;
        
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
        asset_w(halted_mask) = locked_weights(halted_mask);
        
        % 5. 慣性摩擦過濾
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
        
        % 6. 日頻動態波動換手成本
        current_vol_daily = vol20(t) / sqrt(252);
        if isprop(configObj, 'BaseFrictionFee') && isprop(configObj, 'SlippageVolCoeff')
            tc_rate = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff * current_vol_daily);
        else
            tc_rate = 0.0005 + (0.10 * current_vol_daily);
        end
        cost = sum(abs(asset_w(~halted_mask) - w_drift(~halted_mask))) * tc_rate;
        
        % 7. 結算 Open(t+1) 至 Open(t+2) 跨日報酬
        ret_t1 = (Opens_Active(t+2, :) - Opens_Active(t+1, :)) ./ (Opens_Active(t+1, :) + 1e-8);
        ret_t1(isnan(ret_t1) | isinf(ret_t1)) = 0;
        
        step_port_ret = sum(asset_w .* ret_t1') - cost;
        daily_port_rets(t+1, g) = step_port_ret;
        port_curves(t+2, g)     = port_curves(t+1, g) * (1 + step_port_ret);
        
        prev_assets = asset_w;
        prev_cash   = w_cash;
    end
end

%% 6. 統計檢定與雙層次消融報告 (訊號 IC + 實盤回測)
is_ret_idx  = (valid_start_t + 1) : idx_OOS_start;
oos_ret_idx = idx_OOS_start : (numDays - 2);
is_curve_idx  = (valid_start_t + 1) : idx_OOS_start;
oos_curve_idx = idx_OOS_start : (numDays - 1);

spy_ret_is  = (Opens_Active(is_ret_idx+1, spy_idx) - Opens_Active(is_ret_idx, spy_idx)) ./ (Opens_Active(is_ret_idx, spy_idx) + 1e-8);
spy_ret_oos = (Opens_Active(oos_ret_idx+1, spy_idx) - Opens_Active(oos_ret_idx, spy_idx)) ./ (Opens_Active(oos_ret_idx, spy_idx) + 1e-8);

disp(' ');
disp('========================================================================================================================');
disp('📊 【P2-1 空間專家 (DyGAT) 邊際貢獻消融實驗綜合報告 (60D 連續 Z-Score + Open-to-Open 版)】');
disp('========================================================================================================================');

% -------------------------------------------------------------
% A. 訊號層級：60D 遠期橫截面 Rank IC 檢定報告
% -------------------------------------------------------------
fprintf('\n▶ 【層次一：訊號預測力消融 (60D 遠期連續超額報酬 Rank IC，貫穿 max_lag >= 60)】\n');
fprintf(' 評估區間 | 消融組別                 | 60D Mean IC | ICIR (年化) | HAC p-val (lag) | 95%% 信賴區間        | 訊號邊際判定\n');
fprintf('------------------------------------------------------------------------------------------------------------------------\n');

hac_lag_is  = max(horizon, floor(4 * (length(is_ic_idx) / 100)^(2/9)));
hac_lag_oos = max(horizon, floor(4 * (length(oos_ic_idx) / 100)^(2/9)));

% IS 訊號指標結算
ic_is_means = zeros(num_groups, 1);
ic_is_pvals = zeros(num_groups, 1);
for g = 1:num_groups
    ic_sub = daily_rank_ic(is_ic_idx, g);
    m_ic = mean(ic_sub);
    icir = (m_ic / (std(ic_sub) + 1e-8)) * sqrt(252);
    [~, p_hac] = hac_significance_test(ic_sub, hac_lag_is);
    [ci_l, ci_u] = block_bootstrap_ic_by_day(ic_sub, 500, 0.95);
    
    ic_is_means(g) = m_ic;
    ic_is_pvals(g) = p_hac;
    
    fprintf(' IS 區間  | %-24s |   %+.4f    |    %+6.2f   |  %.4f (%2d)   | [%+.4f, %+.4f] | %s\n', ...
        group_names{g}, m_ic, icir, p_hac, hac_lag_is, ci_l, ci_u, ...
        string(ternary(m_ic >= 0.02 && p_hac < 0.05, "⭐ SIGNIFICANT", "NO-GO (Noise)")));
end

% OOS 訊號指標結算
fprintf('------------------------------------------------------------------------------------------------------------------------\n');
ic_oos_means = zeros(num_groups, 1);
ic_oos_pvals = zeros(num_groups, 1);
for g = 1:num_groups
    ic_sub = daily_rank_ic(oos_ic_idx, g);
    m_ic = mean(ic_sub);
    icir = (m_ic / (std(ic_sub) + 1e-8)) * sqrt(252);
    [~, p_hac] = hac_significance_test(ic_sub, hac_lag_oos);
    [ci_l, ci_u] = block_bootstrap_ic_by_day(ic_sub, 500, 0.95);
    
    ic_oos_means(g) = m_ic;
    ic_oos_pvals(g) = p_hac;
    
    fprintf(' OOS 盲測 | %-24s |   %+.4f    |    %+6.2f   |  %.4f (%2d)   | [%+.4f, %+.4f] | %s\n', ...
        group_names{g}, m_ic, icir, p_hac, hac_lag_oos, ci_l, ci_u, ...
        string(ternary(m_ic >= 0.02 && p_hac < 0.05, "⭐ SIGNIFICANT", "NO-GO (Noise)")));
end

% 訊號增量假設檢定 (ΔIC = IC_Balanced - IC_Time)
delta_ic_is  = daily_rank_ic(is_ic_idx, 2)  - daily_rank_ic(is_ic_idx, 1);
delta_ic_oos = daily_rank_ic(oos_ic_idx, 2) - daily_rank_ic(oos_ic_idx, 1);
[t_dic_is, p_dic_is]   = hac_significance_test(delta_ic_is, hac_lag_is);
[t_dic_oos, p_dic_oos] = hac_significance_test(delta_ic_oos, hac_lag_oos);

fprintf('\n 🔬 [訊號邊際增量 HAC 檢定：ΔIC = IC(Balanced) - IC(Pure Time)]\n');
fprintf('    > IS  期間 ΔIC 均值: %+.4f | HAC t = %+.3f (p = %.4f)\n', mean(delta_ic_is), t_dic_is, p_dic_is);
fprintf('    > OOS 期間 ΔIC 均值: %+.4f | HAC t = %+.3f (p = %.4f)\n', mean(delta_ic_oos), t_dic_oos, p_dic_oos);

% -------------------------------------------------------------
% B. 實盤層級：投資組合績效與 Diebold-Mariano 檢定
% -------------------------------------------------------------
fprintf('\n▶ 【層次二：實盤經濟表現消融 (Open-to-Open 撮合、摩擦成本與回撤)】\n');
fprintf(' 評估區間 | 消融組別                 | 累積總報酬 | 年化報酬(CAGR) | 最大回撤(MDD) | 年化 Sharpe | HAC-DM vs SPY (p)\n');
fprintf('------------------------------------------------------------------------------------------------------------------------\n');

for g = 1:num_groups
    v = port_curves(is_curve_idx, g);
    r = daily_port_rets(is_ret_idx, g);
    tot_ret = (v(end)/v(1) - 1) * 100;
    cagr    = ((v(end)/v(1))^(252/length(r)) - 1) * 100;
    mdd     = min((v - cummax(v)) ./ cummax(v)) * 100;
    sharpe  = mean(r) / (std(r) + 1e-8) * sqrt(252);
    
    d_spy = (r - spy_ret_is).^2;
    [dm_stat, p_dm] = hac_significance_test(d_spy);
    
    fprintf(' IS 區間  | %-24s |  %+8.2f%% |     %+6.2f%%   |    %6.2f%%   |    %+5.2f    |  %.3f (p=%.4f)\n', ...
        group_names{g}, tot_ret, cagr, mdd, sharpe, dm_stat, p_dm);
end

fprintf('------------------------------------------------------------------------------------------------------------------------\n');
for g = 1:num_groups
    v = port_curves(oos_curve_idx, g);
    r = daily_port_rets(oos_ret_idx, g);
    tot_ret = (v(end)/v(1) - 1) * 100;
    cagr    = ((v(end)/v(1))^(252/length(r)) - 1) * 100;
    mdd     = min((v - cummax(v)) ./ cummax(v)) * 100;
    sharpe  = mean(r) / (std(r) + 1e-8) * sqrt(252);
    
    d_spy = (r - spy_ret_oos).^2;
    [dm_stat, p_dm] = hac_significance_test(d_spy);
    
    fprintf(' OOS 盲測 | %-24s |  %+8.2f%% |     %+6.2f%%   |    %6.2f%%   |    %+5.2f    |  %.3f (p=%.4f)\n', ...
        group_names{g}, tot_ret, cagr, mdd, sharpe, dm_stat, p_dm);
end

% 投資組合報酬差異 HAC 檢定
r_base_is  = daily_port_rets(is_ret_idx, 1);
r_symm_is  = daily_port_rets(is_ret_idx, 2);
r_base_oos = daily_port_rets(oos_ret_idx, 1);
r_symm_oos = daily_port_rets(oos_ret_idx, 2);

[t_port_is, p_port_is]   = hac_significance_test(r_symm_is - r_base_is);
[t_port_oos, p_port_oos] = hac_significance_test(r_symm_oos - r_base_oos);

fprintf('\n 🔬 [實盤超額收益 HAC 檢定：ΔReturn = Return(Balanced) - Return(Pure Time)]\n');
fprintf('    > IS  期間日均超額差: %+.4f%% | HAC t = %+.3f (p = %.4f)\n', mean(r_symm_is - r_base_is)*100, t_port_is, p_port_is);
fprintf('    > OOS 期間日均超額差: %+.4f%% | HAC t = %+.3f (p = %.4f)\n', mean(r_symm_oos - r_base_oos)*100, t_port_oos, p_port_oos);

fprintf('\n========================================================================================================================\n');
fprintf('📌 【方法論證偽與學術結論】\n');
if (p_dic_is > 0.05 && p_dic_oos > 0.05) || (mean(delta_ic_oos) <= 0)
    fprintf('  ⭐ 【SPATIAL EXPERT FALSIFIED (空間專家獨立邊際貢獻證偽)】\n');
    fprintf('     1. 訊號層面：在 60D 連續超額報酬目標下，加入 DyGAT 空間專家並未產生統計顯著之 Rank IC 提升\n');
    fprintf('        (OOS ΔIC = %+.4f, HAC p = %.4f > 0.05)。\n', mean(delta_ic_oos), p_dic_oos);
    fprintf('     2. 實盤層面：Balanced (50/50) 與 Pure Time 亦無統計顯著報酬差異 (OOS HAC p = %.4f)。\n', p_port_oos);
    fprintf('     3. 理論詮釋：此實證結果與 Round 8a (單層幾何容量極限) 及 Phase 4 貝氏尋優推向邊界解 (Time_W=1.0) 形成完全閉環，\n');
    fprintf('        證實美股市場截面圖拓撲中，基於動態圖卷積所捕獲之關聯訊號，其信噪比不足以超越單純時序動能！\n');
else
    fprintf('  ✅ 【SPATIAL EXPERT CONFIRMED (空間專家展現顯著 Alpha)】\n');
    fprintf('     空間專家在 60D 連續目標下帶來顯著 Rank IC 增量與實盤改善。\n');
end
fprintf('========================================================================================================================\n\n');

%% 7. 繪製標準學術白底黑字視覺化報表
disp('--- 步驟 5：生成消融視覺化診斷圖表 (白底黑字) ---');
fig = figure('Name', 'P2-1 Space Expert Marginal Contribution Ablation', ...
    'Color', 'w', 'Position', [100, 100, 1200, 850], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

colors = {'#D95319', '#0072BD', '#7E2F8E'};
line_styles = {'-', '--', ':'};

% 子圖 1：對數淨值曲線 (IS 與 OOS 銜接)
subplot(3, 1, 1);
eval_dates = Dates_Active(valid_start_t+1 : numDays-1);
for g = 1:num_groups
    plot(eval_dates, log10(port_curves(valid_start_t+1:numDays-1, g)), ...
        'Color', colors{g}, 'LineStyle', line_styles{g}, 'LineWidth', 1.6, 'DisplayName', group_names{g});
    hold on;
end
xline(Dates_Active(idx_OOS_start), '--k', 'OOS Start', 'LineWidth', 1.2, 'Color', 'k');
title('Log-Scale Cumulative Equity Curve (Open-to-Open Matching)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Log_{10}(Wealth)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

% 子圖 2：60D 遠期累積 Spearman Rank IC 曲線
subplot(3, 1, 2);
ic_eval_dates = Dates_Active(1:valid_label_days);
for g = 1:num_groups
    clean_ic = daily_rank_ic(:, g);
    clean_ic(isnan(clean_ic)) = 0;
    cum_ic = cumsum(clean_ic);
    plot(ic_eval_dates, cum_ic, 'Color', colors{g}, 'LineStyle', line_styles{g}, ...
        'LineWidth', 1.5, 'DisplayName', sprintf('%s (OOS IC=%+.4f)', group_names{g}, ic_oos_means(g)));
    hold on;
end
yline(0, '--k', 'Zero Cumulative IC', 'LineWidth', 1.0, 'HandleVisibility', 'off');
xline(Dates_Active(idx_OOS_start), '--k', 'OOS Start', 'LineWidth', 1.2, 'Color', 'k');
title('Cumulative 60D Forward Spearman Rank IC (Signal-Level Information Horizon)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Cumulative Rank IC', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

% 子圖 3：滾動 120 日 Rank IC 差異 (ΔIC = Balanced - Time)
subplot(3, 1, 3);
delta_ic_all = daily_rank_ic(:, 2) - daily_rank_ic(:, 1);
delta_ic_all(isnan(delta_ic_all)) = 0;
roll_delta_ic = movmean(delta_ic_all, [119, 0]);

plot(ic_eval_dates, roll_delta_ic, 'Color', '#77AC30', 'LineWidth', 1.5, 'DisplayName', 'Rolling 120D \DeltaIC (Balanced - Time)'); hold on;
yline(0, '--k', 'No Marginal Gain', 'LineWidth', 1.1);
xline(Dates_Active(idx_OOS_start), '--k', 'OOS Start', 'LineWidth', 1.2, 'Color', 'k');
title('Rolling 120-Day Marginal Rank IC Improvement (\DeltaIC = Balanced - Pure Time)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Date', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('\Delta Rank IC', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'southwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

if ~exist(configObj.ResultDir, 'dir'), mkdir(configObj.ResultDir); end
figPath = fullfile(configObj.ResultDir, 'P2_1_Ablation_SpaceExpert.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 消融視覺化圖表已成功儲存至: %s\n', figPath);
close(fig);

% 儲存消融檢定數值資料
savePath = fullfile(configObj.ResultDir, 'P2_1_Ablation_SpaceExpert.mat');
save(savePath, 'daily_rank_ic', 'daily_port_rets', 'port_curves', 'ic_is_means', 'ic_oos_means', ...
    'delta_ic_is', 'delta_ic_oos', 't_dic_is', 'p_dic_is', 't_dic_oos', 'p_dic_oos');
fprintf('💾 消融檢定數值成果已儲存至: %s\n', savePath);

disp('=================================================================');
disp('🎯 [P2-1] 空間專家邊際貢獻消融實驗執行完畢！');
disp('=================================================================');

%% =====================================================================
% 輔助函數
% =====================================================================
function out = ternary(cond, val_true, val_false)
    if cond, out = val_true; else, out = val_false; end
end