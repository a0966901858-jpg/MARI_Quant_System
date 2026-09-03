% =========================================================================
% 腳本：5_Run_CIO_HRL_Train.m (原 5_Run_CIO_Awakening.m)
% 升級：Phase 15.5 Stage 4 規範版 (★ PPO 動作探索綁定 mrg32k3a 獨立子串流、
%       狀態空間對齊橫截面百分位排序得分、Open-to-Open 撮合與權重漂移同構化、
%       死區連續縮放護欄對齊、日頻波動動態交易摩擦模型、多體制驗證早停)
% 職責：在向量化平行模擬環境中，訓練三位具備不同風險偏好的 CIO 總管，追求高夏普與低回撤
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 CIO 三軌訓練 (mrg32k3a 確定性子串流與動態成本同構版)');
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

rehash path;
rehash;

if exist('Config', 'class') ~= 8
    error('❌ 已掛載路徑但仍找不到 Config 類別，請檢查 configs/Config.m 權限或語法錯誤！');
end

configObj = Config();

% ★ 核心修復 1：由 Config 統一生產 mrg32k3a 隨機數引擎，並設為全域主串流 (Substream = 1)
rng_seed = configObj.RNG_Seed;
rng_gen  = configObj.RNG_Generator;
stream_main = configObj.getRandStream(1);
RandStream.setGlobalStream(stream_main);
disp('🔒 已成功掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定 CIO 訓練環境。');

%% 0. 啟動 MATLAB 平行運算池
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp(' ⚙️ 正在啟動 CPU 平行運算池 (Parallel Pool)...');
    parpool('Processes'); 
end

%% 1. 載入特徵快取、DataFetcher 開盤價與專家百分位選股矩陣
disp('--- 步驟 1：載入全域資料庫、Opens 矩陣與專家百分位選股分數 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');

if ~exist(cachePath, 'file') || ~exist(gbdtPath, 'file')
    error('❌ 找不到前置快取檔案，請先確認 Phase 1 至 Phase 3 已成功執行！');
end

load(cachePath, 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC'; 
load(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all');

% 載入 Opens 矩陣以實現嚴格 Open-to-Open 撮合
fetcher = DataFetcher(configObj);
dataStruct = fetcher.fetch_data();
Opens_Raw = dataStruct.Opens;

seqLen = configObj.SeqLen;
numDaysRaw = length(Dates_Active);
valid_idx = seqLen : numDaysRaw;

% 對崩盤機率進行 20 日移動平均平滑
P_crash_smooth_all = movmean(P_crash_all, [19, 0]);
P_crash_M     = P_crash_smooth_all(valid_idx)';
Prices_Active = Prices_Active(valid_idx, :);
Opens_Active  = Opens_Raw(valid_idx, :);
Expert_Active = Expert_Active(valid_idx, :);
Dates_Active  = Dates_Active(valid_idx);

% ★ 核心修復 2：P_time_M 與 P_space_M 為 Phase 3 產出之 (0, 1] 橫截面百分位排序得分
P_time_M  = P_time_all(valid_idx, :)'; 
P_space_M = P_space_all(valid_idx, :)';
numDays   = length(Dates_Active);

%% 2. 建構 CIO 五維宏觀感知狀態
disp('--- 步驟 2：預計算 CIO 5 維宏觀狀態空間 ---');
CIO_State = zeros(5, numDays, 'single');
spy_idx = find(strcmp(configObj.IdxTickers, 'SPY'));
if isempty(spy_idx), spy_idx = 1; end

spy_prices = Prices_Active(:, spy_idx);
spy_rets = [0; diff(spy_prices) ./ spy_prices(1:end-1)];
spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
vol20 = movstd(spy_rets, [19, 0], 1) * sqrt(252);

mdd252 = zeros(numDays, 1);
for t = 1:numDays
    start_t = max(1, t - 251);
    window_rets = spy_rets(start_t:t);
    cum_ret = cumprod(1 + window_rets);
    running_max = cummax(cum_ret);
    drawdowns = (cum_ret - running_max) ./ running_max;
    mdd252(t) = min(drawdowns);
end

spy_ret20 = zeros(numDays, 1);
for t = 21:numDays
    spy_ret20(t) = (spy_prices(t) - spy_prices(t-20)) / spy_prices(t-20);
end

CIO_State(1, :) = P_crash_M;          % 維度 1：大盤崩盤護欄機率 (Platt 校準平滑版)
CIO_State(2, :) = spy_ret20';         % 維度 2：大盤中期趨勢動能
CIO_State(3, :) = vol20';             % 維度 3：大盤年化波動率
CIO_State(4, :) = abs(mdd252)';       % 維度 4：一年期最大回撤絕對值
CIO_State(5, :) = 1.0;                % 維度 5：當前持倉現金比例

%% 3. 校準崩盤護欄死區下限並實例化三軌代理人
disp('--- 步驟 3：校準崩盤護欄死區下限並實例化三軌 RL 代理人 ---');
spy_inception_idx = find(spy_prices > 10, 1);
if isempty(spy_inception_idx), spy_inception_idx = 1; end
Train_Start_Date = datetime('2006-01-01', 'TimeZone', 'UTC');
idx_train_start = find(Dates_Active >= Train_Start_Date, 1);
if isempty(idx_train_start)
    error('❌ 資料庫中找不到 2006 年以後的數據！');
end
valid_start_t = max(spy_inception_idx + 252, idx_train_start); 
idx_OOS_start = find(Dates_Active >= datetime('2022-01-01', 'TimeZone', 'UTC'), 1);

% 提取 IS 期間 75% 雜訊分位數 (tau_noise)
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
non_crash_p = P_crash_M(is_non_crash_idx);
tau_noise = prctile(non_crash_p, 75);
fprintf('  -> IS 非危機期 P(Crash) 75%% 雜訊分位數 (tau_noise): %.4f\n', tau_noise);

agent_aggressive   = Agent_PPO();   % 積極型：牛市主攻
agent_balanced     = Agent_PPO();   % 平衡型：震盪市輪動
agent_conservative = Agent_PPO();   % 保守型：熊市防禦

base_lr    = configObj.HRL_LR;
base_frict = configObj.MoE_FrictionMask; 
base_guard = configObj.Guardrail_CrashProb;

cfg_agg = struct('Frict', base_frict*0.5, 'GuardHigh', min(0.99, base_guard*1.15), 'LR', base_lr*1.2, 'TauNoise', tau_noise);
cfg_bal = struct('Frict', base_frict,     'GuardHigh', base_guard,                 'LR', base_lr,     'TauNoise', tau_noise);
cfg_con = struct('Frict', base_frict*1.5, 'GuardHigh', base_guard*0.85,            'LR', base_lr*0.8, 'TauNoise', tau_noise);

%% 4. 動態對齊時間軸並構建跨體制驗證池
disp('--- 步驟 4：切分訓練池與多體制驗證池 (解決短回合過擬合) ---');
RolloutSteps = 60; 
% Open-to-Open 需存取 current_t + 2，因此上限嚴格截斷至 idx_OOS_start - RolloutSteps - 2
valid_starts = valid_start_t : (idx_OOS_start - RolloutSteps - 2);

% 跨體制驗證窗口 (僅限 IS 範圍內)
val_windows = { ...
    struct('start', datetime('2008-01-01','TimeZone','UTC'), 'end', datetime('2009-06-01','TimeZone','UTC')), ...
    struct('start', datetime('2020-01-01','TimeZone','UTC'), 'end', datetime('2020-12-01','TimeZone','UTC')), ...
    struct('start', datetime('2018-09-01','TimeZone','UTC'), 'end', datetime('2019-03-01','TimeZone','UTC')) ...
};

val_starts = [];
for i = 1:numel(val_windows)
    idx_w = find(Dates_Active >= val_windows{i}.start & Dates_Active <= val_windows{i}.end);
    val_starts = [val_starts; intersect(idx_w, valid_starts')];
end
val_starts = unique(val_starts)';
train_starts = setdiff(valid_starts, val_starts);

fprintf(' 📡 時間軸切分完畢！訓練起點池: %d 天 | 跨體制驗證起點池: %d 天\n', ...
    length(train_starts), length(val_starts));

epochs = configObj.HRL_Epochs;
batch_size = 128; 

% 白底黑字學術視覺化畫布設定
fig_train = figure('Name', 'Ensemble Agents Training Progress', ...
    'Position', [100, 100, 950, 520], 'Color', 'w');
set(fig_train, 'InvertHardcopy', 'off');

ax = gca;
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, ...
    'FontName', 'Helvetica', 'FontSize', 10);
grid(ax, 'on'); 
box(ax, 'on');

hLineAgg = animatedline('Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'Aggressive (Train)');
hLineBal = animatedline('Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'Balanced (Train)');
hLineCon = animatedline('Color', '#EDB120', 'LineWidth', 1.5, 'DisplayName', 'Conservative (Train)');
hLineVal = animatedline('Color', '#7E2F8E', 'LineWidth', 2.0, 'LineStyle', '--', 'DisplayName', 'Ensemble (Multi-Regime Val)');

title('HRL 3-Track CIO Training Progress & Multi-Regime Validation', ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k'); 
ylabel('Avg Step Reward', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k'); 
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]); 

best_val_ensemble_reward = -inf;
patience_counter = 0;
patience_limit = 30; 
best_agent_agg = []; best_agent_bal = []; best_agent_con = [];

for ep = 1:epochs
    % ★ 核心修復 3：使用 stream_main 進行起點池確定性隨機抽樣
    start_indices = randsample(stream_main, train_starts, batch_size)'; 
    noise_std = max(0.01, 0.2 - (0.2 / (epochs * 0.8)) * ep);
    
    lr_decay = 0.8 ^ floor((ep - 1) / 50);
    cfg_agg.LR = base_lr * 1.2 * lr_decay;
    cfg_bal.LR = base_lr * lr_decay;
    cfg_con.LR = base_lr * 0.8 * lr_decay;
    
    agents_in = {agent_aggressive, agent_balanced, agent_conservative};
    cfgs_in   = {cfg_agg, cfg_bal, cfg_con};
    
    rews_out   = zeros(1, 3);
    agents_out = cell(1, 3);
    
    % ★ 核心修復 4：為 3 位並行訓練代理人注入獨立正交之 mrg32k3a 子串流 (Substream = ep * 10 + a)
    streams_train = cell(1, 3);
    for a = 1:3
        s_obj = RandStream(rng_gen, 'Seed', rng_seed);
        s_obj.Substream = ep * 10 + a;
        streams_train{a} = s_obj;
    end
    
    parfor a = 1:3
        [rews_out(a), agents_out{a}] = simulate_and_update(agents_in{a}, cfgs_in{a}, ...
            start_indices, RolloutSteps, CIO_State, P_time_M, P_space_M, P_crash_M, ...
            Opens_Active, Expert_Active, noise_std, configObj, false, streams_train{a});
    end
    
    agent_aggressive   = agents_out{1};
    agent_balanced     = agents_out{2};
    agent_conservative = agents_out{3};
    
    rew_agg = rews_out(1);
    rew_bal = rews_out(2);
    rew_con = rews_out(3);
    
    % 獨立驗證池評估多體制泛化表現 (noise_std = 0)
    val_batch_size = min(length(val_starts), batch_size);
    val_sample_indices = randsample(stream_main, val_starts, val_batch_size)';
    val_rews_out = zeros(1, 3);
    
    agents_eval = {agent_aggressive, agent_balanced, agent_conservative};
    streams_val = cell(1, 3);
    for a = 1:3
        s_obj = RandStream(rng_gen, 'Seed', rng_seed);
        s_obj.Substream = ep * 10 + 5 + a;
        streams_val{a} = s_obj;
    end
    
    parfor a = 1:3
        [val_rews_out(a), ~] = simulate_and_update(agents_eval{a}, cfgs_in{a}, ...
            val_sample_indices, RolloutSteps, CIO_State, P_time_M, P_space_M, P_crash_M, ...
            Opens_Active, Expert_Active, 0.0, configObj, true, streams_val{a});
    end
    
    val_ensemble_reward = mean(val_rews_out);
    
    addpoints(hLineAgg, ep, rew_agg); 
    addpoints(hLineBal, ep, rew_bal); 
    addpoints(hLineCon, ep, rew_con);
    addpoints(hLineVal, ep, val_ensemble_reward);
    drawnow limitrate;
    
    if mod(ep, 10) == 0 || ep == 1
        fprintf('Ep %3d | Agg R:%+6.2f | Bal R:%+6.2f | Con R:%+6.2f | Val R:%+6.2f | LR Decay: %.2f\n', ...
            ep, rew_agg, rew_bal, rew_con, val_ensemble_reward, lr_decay);
    end
    
    % Early Stopping 僅依賴多體制驗證集 Reward
    if val_ensemble_reward > best_val_ensemble_reward + 1e-4
        best_val_ensemble_reward = val_ensemble_reward;
        patience_counter = 0;
        
        best_agent_agg = agent_aggressive;
        best_agent_bal = agent_balanced;
        best_agent_con = agent_conservative;
    else
        patience_counter = patience_counter + 1;
    end
    
    if patience_counter >= patience_limit && ep > 80
        fprintf('\n🛑 [Early Stopping] 跨體制驗證集連續 %d 輪未改善，提前於 Epoch %d 終止並回滾最佳快照！\n', patience_limit, ep);
        agent_aggressive   = best_agent_agg;
        agent_balanced     = best_agent_bal;
        agent_conservative = best_agent_con;
        break;
    end
end

if ~isempty(best_agent_agg)
    agent_aggressive   = best_agent_agg;
    agent_balanced     = best_agent_bal;
    agent_conservative = best_agent_con;
end

%% --- 步驟 5：儲存覺醒完成的三軌 CIO 大腦與訓練曲線 ---
disp('--- 步驟 5：儲存覺醒完成的三軌 CIO 大腦與訓練曲線 (白底黑字) ---');
trainFigPath = fullfile(configObj.ModelDir, 'Phase5_CIO_Training_Curve.png');
exportgraphics(fig_train, trainFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf(' 📊 訓練進度曲線 (白底黑字) 已儲存至: %s\n', trainFigPath);
close(fig_train);

save(fullfile(configObj.ModelDir, 'CIO_Aggressive.mat'), 'agent_aggressive');
save(fullfile(configObj.ModelDir, 'CIO_Balanced.mat'), 'agent_balanced');
save(fullfile(configObj.ModelDir, 'CIO_Conservative.mat'), 'agent_conservative');
disp('✅ Phase 5 訓練完成！最佳大腦已具備跨體制穩健性並安全落地。');

%% =====================================================================
% ★ 向量化環境模擬核心函數 (★ Phase 15.5 權重漂移、死區縮放與子串流約束)
% =====================================================================
function [avg_reward, agent] = simulate_and_update(agent, cfg, start_indices, steps, ...
    CIO_State, P_time, P_space, P_crash, Opens, Expert, noise_std, configObj, skip_update, stream)
    
    if nargin < 13, skip_update = false; end
    
    % ★ 核心修復 5：切換 Worker 全域隨機串流，保證 Actor 探索高斯噪聲嚴格受控
    if nargin >= 14 && ~isempty(stream)
        old_stream = RandStream.setGlobalStream(stream);
        cleanupObj = onCleanup(@() RandStream.setGlobalStream(old_stream));
    end
    
    batch_size     = length(start_indices); 
    num_tickers    = configObj.NumTickers;
    num_days_total = size(Opens, 1);
    
    top_k_val       = configObj.Top_K_Assets;
    fallback_w_time = configObj.Expert_Time_Weight;
    
    ep_states    = zeros(5, batch_size, steps, 'single'); 
    ep_actions   = zeros(3, batch_size, steps, 'single'); 
    ep_rewards   = zeros(1, batch_size, steps, 'single'); 
    ep_port_rets = zeros(1, batch_size, steps, 'single'); 
    
    prev_assets = zeros(num_tickers, batch_size, 'single'); 
    prev_cash   = ones(1, batch_size, 'single'); 
    
    spy_idx = find(strcmp(configObj.IdxTickers, 'SPY')); 
    if isempty(spy_idx), spy_idx = 1; end
    
    % 計算死區緩衝下限 (GuardLow)
    guard_high = cfg.GuardHigh;
    guard_low  = min(guard_high * 0.85, max(cfg.TauNoise, guard_high - 0.03));
    if guard_low >= guard_high
        guard_low = guard_high * 0.85;
    end
    
    for step_idx = 1:steps
        current_t = start_indices + step_idx - 1; 
        
        % 時間索引邊界斷言防呆 (Open-to-Open 需存取 current_t + 2)
        assert(max(current_t) + 2 <= num_days_total, '❌ 向量化模擬環境時間索引溢出邊界！');
        
        current_state = CIO_State(:, current_t);  
        current_state(5, :) = prev_cash;
        
        % 取得動作 (Actor 探索噪聲自動採用當前受控串流)
        try
            [acts_raw, ~] = agent.get_actions(current_state, noise_std, stream);
        catch
            [acts_raw, ~] = agent.get_actions(current_state, noise_std);
        end
        acts_raw(isnan(acts_raw)) = 0;
        
        % -------------------------------------------------------------
        % 1. 計算資產權重自然漂移 (Weight Drift from t Open to t+1 Open)
        % -------------------------------------------------------------
        drift_ret = (Opens(current_t + 1, :)' - Opens(current_t, :)') ./ (Opens(current_t, :)' + 1e-8);
        drift_ret(isnan(drift_ret) | isinf(drift_ret)) = 0;
        
        asset_mult = prev_assets .* (1 + drift_ret);
        port_val_drift = sum(asset_mult, 1) + prev_cash;
        w_drift = asset_mult ./ max(1e-6, port_val_drift);
        
        % -------------------------------------------------------------
        % 2. 停牌鎖死防護 (Halted Stocks Cannot Be Traded)
        % -------------------------------------------------------------
        halted_mask = isnan(Opens(current_t + 1, :)') | (Opens(current_t + 1, :)' <= 0);
        locked_weights = zeros(num_tickers, batch_size, 'single');
        locked_weights(halted_mask) = w_drift(halted_mask);
        locked_sum = sum(locked_weights, 1);
        available_cap = max(0, 1.0 - locked_sum);
        
        % -------------------------------------------------------------
        % 3. 動作正規化與死區連續縮放護欄
        % -------------------------------------------------------------
        w_time = max(0, acts_raw(1, :));   
        w_space = max(0, acts_raw(2, :));  
        target_cash = max(0, min(1, acts_raw(3, :)));
        
        sum_w = w_time + w_space;
        zero_w_mask = (sum_w <= 1e-6);
        w_time(zero_w_mask)  = fallback_w_time; 
        w_space(zero_w_mask) = 1.0 - fallback_w_time;
        sum_w = w_time + w_space;
        w_time = w_time ./ sum_w;
        w_space = w_space ./ sum_w;
        
        % 帶死區之連續縮放護欄 (Deadband Continuous Risk Scaling)
        p_c = P_crash(current_t);
        risk_scale = zeros(1, batch_size, 'single');
        low_mask  = (p_c <= guard_low);
        high_mask = (p_c >= guard_high);
        mid_mask  = ~low_mask & ~high_mask;
        
        risk_scale(low_mask)  = 0.0;
        risk_scale(high_mask) = 1.0;
        if guard_high > guard_low
            risk_scale(mid_mask) = (p_c(mid_mask) - guard_low) / (guard_high - guard_low);
        end
        
        target_cash = max(target_cash, risk_scale);
        actual_cash_target = min(target_cash, available_cap);
        rem_weight = available_cap - actual_cash_target;
        
        % -------------------------------------------------------------
        % 4. 專家連續排序得分 (Percentile Rank Scores) 融合與 Top-K 篩選
        % -------------------------------------------------------------
        comb_p = P_time(:, current_t) .* w_time + P_space(:, current_t) .* w_space; 
        active_mask_batch = Expert(current_t, :)';
        comb_p = comb_p .* active_mask_batch; 
        comb_p(halted_mask) = 0; % 停牌標的不可新增部位
        
        for b = 1:batch_size
            col_p = comb_p(:, b);
            pos_cnt = sum(col_p > 0);
            if pos_cnt > top_k_val
                [sorted_vals, ~] = sort(col_p, 'descend');
                cutoff = sorted_vals(top_k_val);
                col_p(col_p < cutoff) = 0;
            end
            s_val = sum(col_p);
            if s_val > 0
                comb_p(:, b) = col_p / s_val;
            else
                comb_p(:, b) = 0;
            end
        end
        
        asset_weights = comb_p .* rem_weight;
        asset_weights(halted_mask) = locked_weights(halted_mask);
        
        % -------------------------------------------------------------
        % 5. 慣性摩擦過濾 (Inertia Friction Mask)
        % -------------------------------------------------------------
        turnover_temp = abs(asset_weights(~halted_mask) - w_drift(~halted_mask)); 
        ignore_sub = turnover_temp < cfg.Frict; 
        unhalted_indices = find(~halted_mask);
        asset_weights(unhalted_indices(ignore_sub)) = w_drift(unhalted_indices(ignore_sub));
        
        active_sum = sum(asset_weights, 1); 
        total_cap  = active_sum + actual_cash_target;
        zero_mask  = (total_cap == 0); 
        asset_weights = asset_weights ./ max(1e-6, total_cap); 
        w_cash        = actual_cash_target ./ max(1e-6, total_cap);
        w_cash(zero_mask) = 1.0; 
        asset_weights(:, zero_mask) = 0;
        
        % -------------------------------------------------------------
        % 6. 撮合報酬與交易成本 (Open-to-Open 動態波動滑價模型)
        % -------------------------------------------------------------
        ret = (Opens(current_t + 2, :)' - Opens(current_t + 1, :)') ./ (Opens(current_t + 1, :)' + 1e-8);
        ret(isnan(ret) | isinf(ret)) = 0; 
        
        spy_ret = (Opens(current_t + 2, spy_idx)' - Opens(current_t + 1, spy_idx)') ./ (Opens(current_t + 1, spy_idx)' + 1e-8);
        spy_ret(isnan(spy_ret) | isinf(spy_ret)) = 0;
        
        current_vol = current_state(3, :); 
        current_vol(isnan(current_vol) | isinf(current_vol)) = 0;
        current_vol_daily = current_vol / sqrt(252);
        
        if isprop(configObj, 'BaseFrictionFee') && isprop(configObj, 'SlippageVolCoeff')
            tc_rate = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff .* current_vol_daily);
        else
            tc_rate = 0.0005 + (0.10 .* current_vol_daily);
        end
        
        % 僅對非停牌可交易標的收取換手成本
        tc = tc_rate .* sum(abs(asset_weights(~halted_mask) - w_drift(~halted_mask)), 1); 
        
        port_ret = sum(asset_weights .* ret, 1) - tc; 
        excess_return = port_ret - spy_ret;
        
        ep_port_rets(1, :, step_idx) = port_ret;
        
        % Huber-like 獎勵計算
        rew_cio = zeros(1, batch_size, 'single');
        pos_mask = excess_return >= 0;
        neg_mask = ~pos_mask;
        
        rew_cio(pos_mask) = excess_return(pos_mask) * 100.0;
        
        loss_threshold = 0.02; 
        small_loss_mask = neg_mask & (abs(excess_return) <= loss_threshold);
        large_loss_mask = neg_mask & (abs(excess_return) > loss_threshold);
        
        rew_cio(small_loss_mask) = excess_return(small_loss_mask) * 100.0;
        raw_large_penalty = loss_threshold * 100.0 + ((abs(excess_return(large_loss_mask)) - loss_threshold) * 100.0).^2;
        rew_cio(large_loss_mask) = -min(raw_large_penalty, 100.0);
        
        % 解除護欄主動避險時的現金懲罰矛盾
        is_hedging = (risk_scale > 0.5);
        cash_penalty_coeff = 0.05;
        rew_cio = rew_cio - cash_penalty_coeff * single(w_cash > 0.95 & ~is_hedging);
        
        ep_states(:, :, step_idx)  = current_state; 
        ep_actions(:, :, step_idx) = acts_raw; 
        ep_rewards(1, :, step_idx) = rew_cio;
        
        prev_assets = asset_weights; 
        prev_cash   = w_cash;
    end
    
    % 注入 Rollout 已實現波動度懲罰項
    port_rets_2d = reshape(permute(ep_port_rets, [2, 3, 1]), [batch_size, steps]);
    rollout_vol = std(port_rets_2d, 0, 2)' * sqrt(252);
    vol_penalty_coeff = 0.02;
    vol_penalty_per_step = (vol_penalty_coeff * rollout_vol) / steps;
    ep_rewards = ep_rewards - reshape(vol_penalty_per_step, [1, batch_size, 1]);
    
    if isprop(configObj, 'HRL_Gamma')
        gamma = configObj.HRL_Gamma;
    else
        gamma = 0.96; 
    end
    
    discounted_returns = zeros(1, batch_size, steps, 'single');
    R = zeros(1, batch_size, 'single');
    for t = steps:-1:1
        R = ep_rewards(1, :, t) + gamma * R;
        discounted_returns(1, :, t) = R;
    end
    
    flat_returns = reshape(discounted_returns, 1, []);
    ret_mean = mean(flat_returns);
    ret_std = std(flat_returns) + 1e-8;
    norm_returns = (flat_returns - ret_mean) ./ ret_std;
    
    flat_states = reshape(ep_states, 5, []);
    flat_actions = reshape(ep_actions, 3, []); 
    
    if ~skip_update
        agent.update_weights(flat_states, flat_actions, norm_returns, cfg.LR);
    end
    
    avg_reward = mean(reshape(ep_rewards, 1, []));
end