% =========================================================================
% 腳本：4_Run_BO_Hyperparameter_Tuning.m
% 升級：Phase 15.5 60D 基準線版 (★ 20日定期調倉步進對齊、消除慢速訊號過度換手、
%       平行池全域注入 mrg32k3a 獨立子串流、DSR 統計顯著性熔斷機制、
%       防無效參數污染生產環境、死區護欄模擬器對齊、白底黑字學術視覺化報表)
% 職責：利用貝氏最佳化在多子窗口中聯合尋優，以 DSR 量化選擇偏誤；
%       若未達顯著則啟動熔斷，防止邊界病態解覆寫生產管線。
% =========================================================================
clear; clc; close all;
disp('=================================================================');
disp('🧠 [Phase 15.5] 啟動 CIO 總管動態路由貝氏超參數尋優 (60D 基準線與定期調倉版)');
disp('=================================================================');

%% 0. 環境路徑掛載 (規範化階層回溯解析與路徑重新整理)
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
configObj.initRNG(1);

%% 1. 載入全域快取與 GBDT 預測機率，並動態拉取 Opens 矩陣
disp('--- 步驟 1：載入快取與時間軸嚴格對齊 (引入 Opens 矩陣) ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat'); 

if ~exist(cachePath, 'file') || ~exist(gbdtPath, 'file')
    error('❌ 找不到前置快取檔案，請確認 Phase 1 至 Phase 3 已成功執行！');
end

load(cachePath, 'Prices_Active', 'Expert_Active', 'Dates_Active');
load(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all');

if isempty(Dates_Active.TimeZone)
    Dates_Active.TimeZone = 'UTC';
end

fetcher = DataFetcher(configObj);
dataStruct = fetcher.fetch_data();
Opens_Raw = dataStruct.Opens;

numDaysRaw = length(Dates_Active);
numT = configObj.NumTickers;
seqLen = configObj.SeqLen;
valid_idx = seqLen : numDaysRaw;

P_crash_smooth_all = movmean(P_crash_all, [19, 0]);
P_crash_smooth = P_crash_smooth_all(valid_idx);
Prices_Active = Prices_Active(valid_idx, :);
Opens_Active  = Opens_Raw(valid_idx, :);
Expert_Active = Expert_Active(valid_idx, :);
Dates_Active  = Dates_Active(valid_idx);
P_time_M  = P_time_all(valid_idx, :)';
P_space_M = P_space_all(valid_idx, :)';

%% 2. 切分 In-Sample (IS) 訓練區間與雜訊底線校準
disp('--- 步驟 2：切分 In-Sample 訓練區間與校準背景雜訊底線 ---');
Train_Start_Date = datetime('2006-01-01', 'TimeZone', Dates_Active.TimeZone);
OOS_Start_Date   = datetime('2022-01-01', 'TimeZone', Dates_Active.TimeZone);

idx_IS = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);
Prices_IS  = Prices_Active(idx_IS, :);
Opens_IS   = Opens_Active(idx_IS, :);
Expert_IS  = Expert_Active(idx_IS, :);
P_crash_IS = P_crash_smooth(idx_IS)'; 
P_time_IS  = P_time_M(:, idx_IS);
P_space_IS = P_space_M(:, idx_IS);
Dates_IS   = Dates_Active(idx_IS);

% 提取 IS 期間 75% 雜訊分位數，作為子窗口回測模擬的死區基準
tau_noise = prctile(P_crash_IS, 75);
fprintf('  -> IS 期間 P(Crash) 75%% 背景雜訊分位數 (tau_noise): %.4f\n', tau_noise);

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
var_guard  = optimizableVariable('Guardrail_CrashProb', [p_80, p_99], 'Type', 'real');
var_weight = optimizableVariable('Expert_Time_Weight', [0.0, 1.0], 'Type', 'real');
var_topk   = optimizableVariable('Top_K_Assets', [10, 40], 'Type', 'integer');
bo_vars    = [var_guard, var_weight, var_topk];

%% 4. 啟動 BayesOpt 尋優引擎 (穩健多子窗口評估 + mrg32k3a 平行子串流鎖定)
disp('--- 步驟 4：啟動 BayesOpt 尋優引擎 (4 子窗口穩健目標函數 - 多核平行) ---');
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp('⚡ 正在喚醒 CPU 多核心 (parpool)...');
    poolobj = parpool('Processes');
end

rng_seed = configObj.RNG_Seed;
rng_gen  = configObj.RNG_Generator;
spmd
    worker_stream = RandStream(rng_gen, 'Seed', rng_seed);
    worker_stream.Substream = labindex;
    RandStream.setGlobalStream(worker_stream);
end
disp('🔒 已成功為平行池所有 Worker 注入 mrg32k3a 獨立隨機子串流 (確定性可重現模式)。');

obj_fun = @(x) evaluate_hrl_proxy_robust(x, Prices_IS, Opens_IS, Expert_IS, P_time_IS, P_space_IS, P_crash_IS, tau_noise, configObj);
results = bayesopt(obj_fun, bo_vars, ...
    'MaxObjectiveEvaluations', 25, ...                       
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'UseParallel', true, ...                                
    'Verbose', 1);

%% 5. 提取最佳參數、DSR 統計校正與斷路器檢驗
disp('--- 步驟 5：提取最佳參數、DSR 統計檢驗與熔斷裁決 ---');
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
spy_rets = [0; diff(spy_prices) ./ (spy_prices(1:end-1) + 1e-8)];
spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
vol20_is = movstd(spy_rets, [19, 0], 1) * sqrt(252);

warmup_days = max(60, configObj.Horizon);
sim_start_t = warmup_days + 1;
sim_end_t   = size(Prices_IS, 1) - 2;

[~, best_excess_returns] = run_subwindow_simulation(best_params, Prices_IS, Opens_IS, Expert_IS, ...
    P_time_IS, P_space_IS, P_crash_IS, vol20_is, spy_idx, sim_start_t, sim_end_t, tau_noise, configObj);

pct_days_guarded = mean(P_crash_IS > best_params.Guardrail_CrashProb) * 100;
fprintf('\n🏆 【Phase 15.5 貝氏最佳化候選結果 (60D 定期調倉)】 🏆\n');
fprintf('  > 候選 崩盤護欄閥值 (Guardrail) : %.4f\n', best_params.Guardrail_CrashProb);
fprintf('  > 候選 時序專家權重 (Time_W)  : %.4f\n', best_params.Expert_Time_Weight);
fprintf('  > 候選 集中度標的數 (Top_K)   : %d\n', best_params.Top_K_Assets);
fprintf('  > 穩健綜合指標 (Robust Score) : %.4f\n', best_robust_score);
fprintf('  > 護欄觸發比例 (IS 期間)      : %.1f%% 的交易日啟用完全避險\n', pct_days_guarded);

% Deflated Sharpe Ratio 多重試驗選擇偏誤校正
trial_scores = -results.ObjectiveTrace;
[dsr_val, psr_val, sr0_val] = compute_deflated_sharpe(trial_scores, best_excess_returns);

% 診斷邊界警示
boundary_tol = 0.02;
if abs(best_params.Expert_Time_Weight - 0.0) < boundary_tol || abs(best_params.Expert_Time_Weight - 1.0) < boundary_tol
    warning('⚠️ 警告：Expert_Time_Weight (%.4f) 落在搜索邊界，可能代表 P_time/P_space 高度相關！', ...
        best_params.Expert_Time_Weight);
end
if best_params.Top_K_Assets <= 11 || best_params.Top_K_Assets >= 39
    warning('⚠️ 警告：Top_K_Assets (%d) 落在搜索邊界，可能代表選股訊號排序集中度邊界！', best_params.Top_K_Assets);
end

% =========================================================================
% DSR 門檻斷路器 (防止無效參數污染生產環境)
% =========================================================================
is_statistically_significant = (dsr_val >= 0.95) && (best_robust_score > 0.0);
prod_bo_path = fullfile(configObj.ModelDir, 'BO_Hyperparameters.mat');
diag_bo_path = fullfile(configObj.ModelDir, 'BO_Hyperparameters_FAILED_DIAGNOSTIC.mat');

if is_statistically_significant
    disp('✅ 尋優成功！通過 DSR 顯著性檢驗 (DSR >= 0.95 且 Robust Score > 0)，落地為正式生產參數。');
    configObj.Guardrail_CrashProb = best_params.Guardrail_CrashProb;
    configObj.Expert_Time_Weight  = best_params.Expert_Time_Weight;
    configObj.Top_K_Assets        = best_params.Top_K_Assets;
    
    save(prod_bo_path, 'best_params', 'results', 'dsr_val', 'psr_val', 'sr0_val', 'best_robust_score');
    fprintf('💾 生產環境 BO 參數檔已成功寫入: %s\n', prod_bo_path);
else
    fprintf('\n⚠️ =========================================================================\n');
    fprintf('⚠️ 【循環斷路器熔斷生效】\n');
    fprintf('   尋優結果未達統計顯著性 (DSR = %.4f < 0.95 或 Robust Score = %.4f <= 0)！\n', dsr_val, best_robust_score);
    fprintf('   未能拒絕「尋優結果純屬多重試驗過擬合噪聲」之虛無假設。\n');
    fprintf('   ❌ 拒絕將病態邊界參數寫入生產環境配置！\n');
    fprintf('   🔄 系統退回學術中立基準 (Time_W=0.50, Top_K=20, Guard=0.0850)。\n');
    fprintf('⚠️ =========================================================================\n\n');
    
    configObj.Guardrail_CrashProb = 0.0850;
    configObj.Expert_Time_Weight  = 0.5000;
    configObj.Top_K_Assets        = 20;
    
    if exist(prod_bo_path, 'file')
        delete(prod_bo_path);
        fprintf('🗑️ 已清除舊有生產參數檔: %s\n', prod_bo_path);
    end
    
    save(diag_bo_path, 'best_params', 'results', 'dsr_val', 'psr_val', 'sr0_val', 'best_robust_score');
    fprintf('💾 尋優診斷數據已獨立存檔至: %s (供論文假說 H1d 證偽引用)\n', diag_bo_path);
end

%% 6. 產出 BayesOpt 收斂軌跡與超額收益視覺化報表 (標準學術白底黑字)
disp('--- 步驟 6：產出 BayesOpt 尋優收斂與超額收益報表 (白底黑字) ---');
fig_bo = figure('Name', 'Phase 4: BayesOpt Tuning Report', ...
    'Position', [100, 100, 1200, 500], 'Color', 'w', 'Visible', 'off');
set(fig_bo, 'InvertHardcopy', 'off');

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

subplot(1, 2, 2);
cum_excess = cumsum(best_excess_returns) * 100;
sim_dates = Dates_IS(sim_start_t : sim_end_t);
plot(sim_dates, cum_excess, 'Color', '#0072BD', 'LineWidth', 2.0, ...
    'DisplayName', sprintf('Candidate (TopK=%d, Guard=%.2f)', best_params.Top_K_Assets, best_params.Guardrail_CrashProb));
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
disp('🎯 [Phase 4] 執行完畢！熔斷檢定與參數狀態已鎖定。');
disp('=================================================================');

%% =====================================================================
% 多子窗口穩健代理評估函數
% =====================================================================
function neg_robust_IR = evaluate_hrl_proxy_robust(params, Prices, Opens, Expert, P_time, P_space, P_crash, tau_noise, config)
    try
        numDays = size(Prices, 1);
        n_folds = 4;
        fold_edges = round(linspace(1, numDays, n_folds + 1));
        fold_IRs = zeros(n_folds, 1, 'single');
        
        spy_idx = find(strcmp(config.IdxTickers, 'SPY'));
        if isempty(spy_idx), spy_idx = 1; end
        
        spy_prices = Prices(:, spy_idx);
        spy_rets = [0; diff(spy_prices) ./ (spy_prices(1:end-1) + 1e-8)];
        spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
        vol20 = movstd(spy_rets, [19, 0], 1) * sqrt(252);
        
        warmup = max(60, config.Horizon);
        
        for f = 1:n_folds
            f_start = fold_edges(f);
            f_end   = fold_edges(f+1) - 2;
            
            if f_start < warmup, f_start = warmup; end
            
            if f_end <= f_start
                fold_IRs(f) = -10.0;
                continue;
            end
            
            [fold_IRs(f), ~] = run_subwindow_simulation(params, Prices, Opens, Expert, P_time, P_space, P_crash, ...
                vol20, spy_idx, f_start, f_end, tau_noise, config);
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

%% =====================================================================
% 微型子窗口撮合模擬器 (定期調倉 + 死區連續縮放護欄 + 動態摩擦成本)
% =====================================================================
function [ir_val, excess_returns] = run_subwindow_simulation(params, ~, Opens, Expert, P_time, P_space, P_crash, vol20, spy_idx, start_t, end_t, tau_noise, config)
    numTickers = size(Opens, 2);
    steps = end_t - start_t + 1;
    
    prev_assets = zeros(numTickers, 1, 'single');
    prev_cash = 1.0;
    excess_returns = zeros(steps, 1, 'single');
    
    opt_guard = params.Guardrail_CrashProb;
    w_time    = params.Expert_Time_Weight;
    w_space   = 1.0 - w_time;
    top_k     = round(params.Top_K_Assets);
    
    % 校準死區緩衝帶
    guard_high = opt_guard;
    guard_low  = min(guard_high * 0.85, max(tau_noise, guard_high - 0.03));
    if guard_low >= guard_high
        guard_low = guard_high * 0.85;
    end
    
    % 讀取定期調倉步進 (預設 20 日)
    if isprop(config, 'RebalanceStride') && ~isempty(config.RebalanceStride)
        stride = config.RebalanceStride;
    else
        stride = 20;
    end
    
    cached_stock_props = zeros(numTickers, 1, 'single');
    
    for s_i = 1:steps
        t = start_t + s_i - 1;
        
        % 1. 權重隨開盤價漂移 (Open(t) -> Open(t+1))
        drift_ret = (Opens(t+1, :) - Opens(t, :)) ./ (Opens(t, :) + 1e-8);
        drift_ret(isnan(drift_ret) | isinf(drift_ret)) = 0;
        
        asset_mult = prev_assets .* (1 + drift_ret');
        port_val = sum(asset_mult) + prev_cash;
        
        if port_val > 0
            w_drift = asset_mult / port_val;
        else
            w_drift = prev_assets;
        end
        
        % 2. 停牌標的鎖死檢查
        halted_mask = isnan(Opens(t+1, :))' | (Opens(t+1, :) <= 0)';
        locked_weights = zeros(numTickers, 1, 'single');
        locked_weights(halted_mask) = w_drift(halted_mask);
        locked_sum = sum(locked_weights);
        
        available_cap = max(0, 1.0 - locked_sum);
        
        % 3. 帶死區之連續縮放護欄 (每日監控宏觀風險)
        p_c = P_crash(t);
        if p_c <= guard_low
            risk_scale = 0.0;
        elseif p_c >= guard_high
            risk_scale = 1.0;
        else
            risk_scale = (p_c - guard_low) / (guard_high - guard_low);
        end
        
        actual_cash_target = min(risk_scale, available_cap);
        rem_cap_for_assets = available_cap - actual_cash_target;
        
        % 4. 判斷是否為定期調倉日 (Rebalance Day)
        is_rebal_day = (s_i == 1) || (mod(s_i - 1, stride) == 0);
        
        if is_rebal_day
            % 定期調倉：重新計算專家加權得分與 Top-K 選股
            comb_p = P_time(:, t) * w_time + P_space(:, t) * w_space;
            comb_p = comb_p .* Expert(t, :)';
            comb_p(halted_mask) = 0;
            
            if sum(comb_p > 0) > top_k
                [~, sort_idx] = sort(comb_p, 'descend');
                comb_p(comb_p < comb_p(sort_idx(top_k))) = 0;
            end
            
            if sum(comb_p) > 0
                cached_stock_props = comb_p ./ sum(comb_p);
            else
                cached_stock_props(:) = 0;
            end
            
            target_active_weights = cached_stock_props .* rem_cap_for_assets;
        else
            % 非調倉日：資產被動漂移；若護欄現金需求變動，等比例縮放活躍部位
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
        
        asset_weights = target_active_weights;
        asset_weights(halted_mask) = locked_weights(halted_mask);
        
        % 5. 交易摩擦成本計算 (日頻波動度基礎)
        current_vol = vol20(t);
        if isnan(current_vol) || isinf(current_vol), current_vol = 0; end
        current_vol_daily = current_vol / sqrt(252);
        
        if isprop(config, 'BaseFrictionFee') && isprop(config, 'SlippageVolCoeff')
            tc_rate = config.BaseFrictionFee + (config.SlippageVolCoeff * current_vol_daily);
        else
            tc_rate = 0.0005 + (0.10 * current_vol_daily);
        end
        
        % 僅對非停牌主動換手收取成本 (非調倉日且無風控動作時換手率自然為 0)
        turnover = sum(abs(asset_weights(~halted_mask) - w_drift(~halted_mask)));
        frict_cost = turnover * tc_rate;
        
        % 6. 驗收 Open-to-Open 報酬
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

%% =====================================================================
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
    fprintf('📊 【Phase 15.5 Deflated Sharpe Ratio (DSR) 選擇偏誤校正報告】\n');
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