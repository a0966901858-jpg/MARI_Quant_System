% =========================================================================
% 腳本：4_Run_BO_Hyperparameter_Tuning.m
% 升級：Phase 15 (★ 日頻成本校正、DSR 統計檢驗、白底黑字學術視覺化報表)
% 職責：利用貝氏最佳化在多子窗口中聯合尋優，並以 DSR 量化選擇偏誤
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧠 [Phase 15] 啟動 CIO 總管動態路由貝氏超參數尋優 (白底黑字學術版)');
disp('=================================================================');

%% 0. 環境路徑掛載 (規範化階層回溯解析與路徑重新整理)
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

% 固定全域隨機種子，確保實驗可重現
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入全域快取與 GBDT 預測機率，並動態拉取 Opens 矩陣
disp('--- 步驟 1：載入快取與時間軸嚴格對齊 (引入 Opens 矩陣) ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat'); 

if ~exist(cachePath, 'file') || ~exist(gbdtPath, 'file')
    error('❌ 找不到前置快取檔案，請確認 Phase 1 至 Phase 3 已成功執行！');
end

% 1A. 載入特徵面板、收盤價與時間軸
load(cachePath, 'Prices_Active', 'Expert_Active', 'Dates_Active');
load(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all');
Dates_Active.TimeZone = 'UTC'; 

% 1B. 向 DataFetcher 即時索取 T+1 開盤交易所需的 Opens 矩陣
fetcher = DataFetcher(configObj);
dataStruct = fetcher.fetch_data();
Opens_Raw = dataStruct.Opens;

numDaysRaw = length(Dates_Active);
numT = configObj.NumTickers;
seqLen = configObj.SeqLen;
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

%% 2. 切分 In-Sample (IS) 訓練區間
disp('--- 步驟 2：切分 In-Sample 訓練區間 ---');
Train_Start_Date = datetime('2006-01-01', 'TimeZone', 'UTC');
OOS_Start_Date   = datetime('2022-01-01', 'TimeZone', 'UTC');

idx_IS = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);

Prices_IS  = Prices_Active(idx_IS, :);
Opens_IS   = Opens_Active(idx_IS, :);
Expert_IS  = Expert_Active(idx_IS, :);
P_crash_IS = P_crash_smooth(idx_IS)'; 
P_time_IS  = P_time_M(:, idx_IS);
P_space_IS = P_space_M(:, idx_IS);
Dates_IS   = Dates_Active(idx_IS);

%% 3. 定義貝氏最佳化超參數空間 (自適應邊界加固)
disp('--- 步驟 3：定義貝氏最佳化超參數空間 (自適應邊界與間距保證) ---');

p_80 = prctile(P_crash_IS, 80);
p_99 = prctile(P_crash_IS, 99);

if isnan(p_80) || isnan(p_99) || p_80 >= p_99
    p_80 = 0.05;
    p_99 = 0.30;
end

p_80 = max(0.01, min(0.90, p_80));

if (p_99 - p_80) < 0.05
    p_99 = p_80 + 0.05;
end

if p_99 > 0.99
    p_99 = 0.99;
    p_80 = min(p_80, p_99 - 0.05);
end

fprintf('  -> 護欄動態邊界鎖定：[%.4f (下界), %.4f (上界)]\n', p_80, p_99);

% 宣告 BayesOpt 決策變數
var_guard  = optimizableVariable('Guardrail_CrashProb', [p_80, p_99], 'Type', 'real');
var_weight = optimizableVariable('Expert_Time_Weight', [0.0, 1.0], 'Type', 'real');
var_topk   = optimizableVariable('Top_K_Assets', [10, 40], 'Type', 'integer');
bo_vars    = [var_guard, var_weight, var_topk];

%% 4. 啟動 BayesOpt 尋優引擎 (穩健多子窗口評估)
disp('--- 步驟 4：啟動 BayesOpt 尋優引擎 (4 子窗口穩健目標函數 - 多核平行) ---');
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp('⚡ 正在喚醒 CPU 多核心 (parpool)...');
    parpool('Processes');
end

obj_fun = @(x) evaluate_hrl_proxy_robust(x, Prices_IS, Opens_IS, Expert_IS, P_time_IS, P_space_IS, P_crash_IS, configObj);
results = bayesopt(obj_fun, bo_vars, ...
    'MaxObjectiveEvaluations', 25, ...                       
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'UseParallel', true, ...                                
    'Verbose', 1);

%% 5. 提取最佳參數、DSR 統計校正與存檔
disp('--- 步驟 5：提取最佳參數、DSR 統計檢驗與落地存檔 ---');
if isempty(results.XAtMinObjective) || height(results.XAtMinObjective) == 0
    best_params = table(p_99, 0.5, 20, 'VariableNames', {'Guardrail_CrashProb', 'Expert_Time_Weight', 'Top_K_Assets'});
    warning('⚠️ 優化未收斂，將自動採用預設值');
else
    best_params = results.XAtMinObjective;
end

best_robust_score = -results.MinObjective;

% 執行單一完整 IS 模擬以提取最佳逐日超額報酬序列
spy_idx = find(strcmp(configObj.IdxTickers, 'SPY'));
if isempty(spy_idx), spy_idx = 1; end
spy_prices = Prices_IS(:, spy_idx);
spy_rets = [0; diff(spy_prices) ./ spy_prices(1:end-1)];
spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
vol20_is = movstd(spy_rets, [19, 0], 1) * sqrt(252);

[~, best_excess_returns] = run_subwindow_simulation(best_params, Prices_IS, Opens_IS, Expert_IS, ...
    P_time_IS, P_space_IS, P_crash_IS, vol20_is, spy_idx, 21, size(Prices_IS, 1) - 2, configObj);

% 計算 IS 期間護欄觸發天數比例
pct_days_guarded = mean(P_crash_IS > best_params.Guardrail_CrashProb) * 100;

fprintf('\n🏆 【Phase 15 貝氏最佳化尋優結果】 🏆\n');
fprintf('  > 最佳 崩盤護欄閥值 (Guardrail) : %.4f\n', best_params.Guardrail_CrashProb);
fprintf('  > 最佳 時序專家權重 (Time_W)  : %.4f\n', best_params.Expert_Time_Weight);
fprintf('  > 最佳 集中度標的數 (Top_K)   : %d\n', best_params.Top_K_Assets);
fprintf('  > 穩健綜合指標 (Robust Score) : %.4f\n', best_robust_score);
fprintf('  > 護欄觸發比例 (IS 期間)      : %.1f%% 的交易日啟用完全避險\n', pct_days_guarded);

% Deflated Sharpe Ratio 多重試驗選擇偏誤校正
trial_scores = -results.ObjectiveTrace; % 各輪試驗的穩健 IR 分數
[dsr_val, psr_val, sr0_val] = compute_deflated_sharpe(trial_scores, best_excess_returns);

% 診斷警示
if pct_days_guarded > 15.0
    warning('⚠️ 警告：護欄觸發比例 (%.1f%%) 偏高，可能導致系統過度避險！', pct_days_guarded);
end

boundary_tol = 0.02;
if abs(best_params.Expert_Time_Weight - 0.0) < boundary_tol || abs(best_params.Expert_Time_Weight - 1.0) < boundary_tol
    warning('⚠️ 警告：Expert_Time_Weight (%.4f) 落在搜索邊界，可能代表 P_time/P_space 高度相關！', ...
        best_params.Expert_Time_Weight);
end

if best_params.Top_K_Assets <= 11 || best_params.Top_K_Assets >= 39
    warning('⚠️ 警告：Top_K_Assets (%d) 落在搜索邊界，可能代表選股訊號排序集中度邊界！', best_params.Top_K_Assets);
end

if results.MinObjective > 0
    warning(['\n⚠️ =========================================================================\n' ...
             '⚠️ 【循環斷路器觸發警告】\n' ...
             '   貝氏最佳化在整個搜索空間內未找到任何正穩健 IR 組合 (Score = %.4f)！\n' ...
             '   代表當前選股訊號在多子窗口環境中均無超額 Alpha，問題在於 Phase 2/3 的特徵品質。\n' ...
             '⚠️ =========================================================================\n'], best_robust_score);
else
    disp('✅ 尋優成功！在多子窗口下捕獲具備泛化能力的正超額收益參數組合。');
end

% 寫回 Config 物件並落地
configObj.Guardrail_CrashProb = best_params.Guardrail_CrashProb;
configObj.Expert_Time_Weight  = best_params.Expert_Time_Weight;
configObj.Top_K_Assets        = best_params.Top_K_Assets;

boPath = fullfile(configObj.ModelDir, 'BO_Hyperparameters.mat');
save(boPath, 'best_params', 'results', 'dsr_val', 'psr_val', 'sr0_val');
disp('💾 Phase 4 超參數與 DSR 檢驗結果已存檔至 BO_Hyperparameters.mat');

%% 6. 產出 BayesOpt 收斂軌跡與超額收益視覺化報表 (標準學術白底黑字)
disp('--- 步驟 6：產出 BayesOpt 尋優收斂與超額收益報表 (白底黑字) ---');

fig_bo = figure('Name', 'Phase 4: BayesOpt Tuning Report', ...
    'Position', [100, 100, 1200, 500], 'Color', 'w', 'Visible', 'off');
set(fig_bo, 'InvertHardcopy', 'off');

% 子圖 1：尋優目標函數收斂軌跡
subplot(1, 2, 1);
iter_evals = 1:length(results.ObjectiveTrace);
best_so_far = cummin(results.ObjectiveTrace);

plot(iter_evals, -results.ObjectiveTrace, 'o', 'Color', '#7E2F8E', 'MarkerFaceColor', '#DDA0DD', ...
    'LineWidth', 1.0, 'DisplayName', 'Trial Robust IR'); hold on;
plot(iter_evals, -best_so_far, '-d', 'Color', '#D95319', 'MarkerFaceColor', '#D95319', ...
    'LineWidth', 2.0, 'DisplayName', 'Best Robust IR');
yline(0.0, '--k', 'Zero Bound', 'LineWidth', 1.0, 'HandleVisibility', 'off');

title('Bayesian Optimization Convergence', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Trial Evaluations', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Robust Score (Mean IR - 0.5 \times Std IR)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'southeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);

set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 10);
grid on; box on;

% 子圖 2：最佳超參數在 IS 區間累積超額報酬曲線
subplot(1, 2, 2);
cum_excess = cumsum(best_excess_returns) * 100;
sim_dates = Dates_IS(21 : size(Prices_IS, 1) - 2);

plot(sim_dates, cum_excess, 'Color', '#0072BD', 'LineWidth', 2.0, ...
    'DisplayName', sprintf('Optimal Setup (TopK=%d, Guard=%.2f)', best_params.Top_K_Assets, best_params.Guardrail_CrashProb));
hold on;
yline(0.0, '--k', 'Zero Benchmark', 'LineWidth', 1.0, 'HandleVisibility', 'off');

title('In-Sample Cumulative Excess Return (vs SPY)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Date', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Cumulative Excess Return (%)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);

set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 10);
grid on; box on;

boFigPath = fullfile(configObj.ModelDir, 'Phase4_BayesOpt_Convergence.png');
exportgraphics(fig_bo, boFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf(' 📊 尋優收斂與超額收益圖 (白底黑字) 已儲存至: %s\n', boFigPath);
close(fig_bo);

disp('=================================================================');
disp('🎯 [Phase 4] 完美完成。穩健超參數與視覺化報表已鎖定，請執行 Phase 5！');
disp('=================================================================');

%% =====================================================================
% 多子窗口穩健代理評估函數
% =====================================================================
function neg_robust_IR = evaluate_hrl_proxy_robust(params, Prices, Opens, Expert, P_time, P_space, P_crash, config)
    try
        numDays = size(Prices, 1);
        n_folds = 4;
        fold_edges = round(linspace(1, numDays, n_folds + 1));
        fold_IRs = zeros(n_folds, 1, 'single');
        
        spy_idx = find(strcmp(config.IdxTickers, 'SPY'));
        if isempty(spy_idx), spy_idx = 1; end
        
        spy_prices = Prices(:, spy_idx);
        spy_rets = [0; diff(spy_prices) ./ spy_prices(1:end-1)];
        spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
        vol20 = movstd(spy_rets, [19, 0], 1) * sqrt(252);
        
        for f = 1:n_folds
            f_start = fold_edges(f);
            f_end   = fold_edges(f+1) - 2;
            
            if f_start < 21, f_start = 21; end
            
            if f_end <= f_start
                fold_IRs(f) = -10.0;
                continue;
            end
            
            [fold_IRs(f), ~] = run_subwindow_simulation(params, Prices, Opens, Expert, P_time, P_space, P_crash, ...
                vol20, spy_idx, f_start, f_end, config);
        end
        
        mean_ir = mean(fold_IRs);
        std_ir  = std(fold_IRs);
        robust_score = mean_ir - (0.5 * std_ir);
        
        neg_robust_IR = -robust_score;
    catch ME
        disp(ME.message);
        neg_robust_IR = 1e6;
    end
end

% =====================================================================
% 微型子窗口撮合模擬器 (★ Phase 15 日頻成本校正 + 序列回傳)
% =====================================================================
function [ir_val, excess_returns] = run_subwindow_simulation(params, ~, Opens, Expert, P_time, P_space, P_crash, vol20, spy_idx, start_t, end_t, config)
    numTickers = size(Opens, 2);
    steps = end_t - start_t + 1;
    
    prev_assets = zeros(numTickers, 1, 'single');
    prev_cash = 1.0;
    excess_returns = zeros(steps, 1, 'single');
    
    opt_guard = params.Guardrail_CrashProb;
    w_time    = params.Expert_Time_Weight;
    w_space   = 1.0 - w_time;
    top_k     = round(params.Top_K_Assets);
    
    for s_i = 1:steps
        t = start_t + s_i - 1;
        
        % 1. 權重漂移
        drift_ret = (Opens(t+1, :) - Opens(t, :)) ./ (Opens(t, :) + 1e-8);
        drift_ret(isnan(drift_ret) | isinf(drift_ret)) = 0;
        
        asset_mult = prev_assets .* (1 + drift_ret');
        port_val = sum(asset_mult) + prev_cash;
        
        if port_val > 0
            w_drift = asset_mult / port_val;
        else
            w_drift = prev_assets;
        end
        
        % 2. 停牌鎖死
        halted_mask = isnan(Opens(t+1, :))' | (Opens(t+1, :) <= 0)';
        locked_weights = zeros(numTickers, 1, 'single');
        locked_weights(halted_mask) = w_drift(halted_mask);
        locked_sum = sum(locked_weights);
        
        available_cap = max(0, 1.0 - locked_sum);
        
        % 3. 連續縮放護欄 (Continuous Risk Scaling)
        target_cash = 0.0;
        guard_high = opt_guard;
        guard_low  = max(0, opt_guard - 0.10);
        if guard_high > guard_low
            risk_scale = max(0, min(1, (P_crash(t) - guard_low) / (guard_high - guard_low)));
            target_cash = max(target_cash, risk_scale);
        elseif P_crash(t) >= guard_high
            target_cash = 1.0;
        end
        
        actual_cash_target = min(target_cash, available_cap);
        rem_cap_for_assets = available_cap - actual_cash_target;
        
        comb_p = P_time(:, t) * w_time + P_space(:, t) * w_space;
        comb_p = comb_p .* Expert(t, :)';
        comb_p(halted_mask) = 0;
        
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
        
        % 4. 交易摩擦成本計算 (日頻波動度基礎)
        current_vol = vol20(t);
        if isnan(current_vol) || isinf(current_vol), current_vol = 0; end
        current_vol_daily = current_vol / sqrt(252);
        
        if isprop(config, 'BaseFrictionFee') && isprop(config, 'SlippageVolCoeff')
            tc_rate = config.BaseFrictionFee + (config.SlippageVolCoeff * current_vol_daily);
        else
            tc_rate = 0.0005 + (0.10 * current_vol_daily);
        end
        
        frict_cost = sum(abs(asset_weights(~halted_mask) - w_drift(~halted_mask))) * tc_rate;
        
        % 5. 驗收 Open-to-Open 報酬
        fwd_ret = (Opens(t+2, :) - Opens(t+1, :)) ./ (Opens(t+1, :) + 1e-8);
        fwd_ret(isnan(fwd_ret) | isinf(fwd_ret)) = 0;
        
        port_ret = sum(asset_weights .* fwd_ret') - frict_cost;
        spy_ret  = (Opens(t+2, spy_idx) - Opens(t+1, spy_idx)) / (Opens(t+1, spy_idx) + 1e-8);
        if isnan(spy_ret) || isinf(spy_ret), spy_ret = 0; end
        
        excess_returns(s_i) = port_ret - spy_ret;
        
        prev_assets = asset_weights;
        prev_cash   = actual_cash_target;
    end
    
    avg_ex = mean(excess_returns);
    std_ex = std(excess_returns);
    
    if std_ex < 1e-6
        ir_val = -10.0;
    else
        ir_val = (avg_ex / std_ex) * sqrt(252);
    end
end

% =====================================================================
% Deflated Sharpe Ratio 統計校正函數
% =====================================================================
function [dsr, psr, sr0_noise_floor] = compute_deflated_sharpe(all_trial_scores, best_excess_returns)
    N = length(all_trial_scores);
    var_trials = var(all_trial_scores);
    euler_gamma = 0.5772156649;

    if N > 1 && var_trials > 0
        sr0_noise_floor = sqrt(var_trials) * ((1 - euler_gamma) * norminv(1 - 1/N) + ...
                                              euler_gamma * norminv(1 - 1/(N * exp(1))));
    else
        sr0_noise_floor = 0;
    end

    observed_sr = mean(best_excess_returns) / (std(best_excess_returns) + 1e-8) * sqrt(252);
    n_obs = length(best_excess_returns);
    skew_val = skewness(best_excess_returns);
    kurt_val = kurtosis(best_excess_returns);

    z_num = (observed_sr - sr0_noise_floor) * sqrt(n_obs - 1);
    z_den = sqrt(max(1e-8, 1 - skew_val * observed_sr + ((kurt_val - 1) / 4) * observed_sr^2));

    psr = normcdf(z_num / z_den);
    dsr = psr;

    fprintf('\n=================================================================\n');
    fprintf('📊 【Phase 15 Deflated Sharpe Ratio (DSR) 選擇偏誤校正報告】\n');
    fprintf('=================================================================\n');
    fprintf('  > BayesOpt 試驗次數 N         : %d 次\n', N);
    fprintf('  > 試驗噪聲基準 (Sharpe0 Floor): %.4f (高於此值才具統計顯著性)\n', sr0_noise_floor);
    fprintf('  > 最佳參數觀測年化 Sharpe     : %.4f\n', observed_sr);
    fprintf('  > Deflated Sharpe Ratio (DSR) : %.4f\n', dsr);
    fprintf('-----------------------------------------------------------------\n');
    if dsr > 0.95
        fprintf('  ✅ 在 95%% 信心水準下，尋優結果統計上顯著超越多重試驗運氣！\n');
    else
        fprintf('  ⚠️ 未能拒絕「此結果僅為 %d 次試驗中運氣最好者」之虛無假設！\n', N);
    end
    fprintf('=================================================================\n\n');
end