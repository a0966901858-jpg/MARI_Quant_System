% =========================================================================
% 腳本：5_Run_CIO_Awakening.m
% 升級：Phase 14.22 (★ 引入 Early Stopping 早停機制、LR 衰減排程、防範過擬合)
% 職責：在向量化平行模擬環境中，極速訓練三位具備不同風險偏好的 CIO 總管
% =========================================================================
% 清空工作區變數、清除命令列輸出、關閉所有繪圖視窗，確保記憶體環境純淨
clear; clc; close all;
disp('=========================================================');
disp('🚀 [Phase 14.22] 啟動 CIO 三軌訓練 (12核心 CPU 極速並列架構)');
disp('=========================================================');

% 動態獲取當下腳本的絕對路徑，確保跨設備執行時的檔案路徑相容性
currentPath = fileparts(mfilename('fullpath'));
if isempty(currentPath), currentPath = pwd; end
projectRoot = currentPath; 

% 檢查 configs 目錄是否存在，若不存在則將專案根目錄向上一層推導
if ~exist(fullfile(projectRoot, 'configs'), 'dir'), projectRoot = fullfile(currentPath, '..'); end

% 將專案核心的各個子資料夾遞迴加入 MATLAB 搜尋路徑中
addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'envs')));
addpath(genpath(fullfile(projectRoot, 'models')));
addpath(genpath(fullfile(projectRoot, 'agents')));
rehash toolboxcache;

% 實例化全域設定檔，統一控管超參數
configObj = Config();

% 固定全域隨機種子，確保實驗可重現
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 0. 啟動 MATLAB 平行運算池 (利用 i5-13420H 的多核心)
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp(' ⚙️ 正在啟動 CPU 平行運算池 (Parallel Pool)...');
    parpool('Processes'); 
end

%% 1. 載入特徵快取與 GBDT/DL 選股矩陣
disp('--- 步驟 1：載入全域資料庫與專家選股矩陣 ---');
load(fullfile(configObj.CacheDir, 'features_denoised.mat'), 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC'; 

load(fullfile(configObj.ModelDir, 'GBDT_Guards.mat'), 'P_crash_all', 'P_time_all', 'P_space_all');

seqLen = configObj.SeqLen;
numDaysRaw = length(Dates_Active);
valid_idx = seqLen : numDaysRaw;

Prices_Active = Prices_Active(valid_idx, :);
Expert_Active = Expert_Active(valid_idx, :);
Dates_Active  = Dates_Active(valid_idx);

% 對崩盤機率進行 20 日移動平均平滑，並轉置為 [1, Days] 的橫向向量
P_crash_smooth = movmean(P_crash_all(valid_idx), [19, 0]);
P_crash_M = P_crash_smooth';

% 將選股機率矩陣轉置為 [Tickers, Days]，以利後續 MKL 記憶體連續提取加速
P_time_M = P_time_all(valid_idx, :)'; 
P_space_M = P_space_all(valid_idx, :)';
numDays = length(Dates_Active);

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

CIO_State(1, :) = P_crash_M;          % 維度 1：大盤崩盤護欄機率
CIO_State(2, :) = spy_ret20';         % 維度 2：大盤中期趨勢動能
CIO_State(3, :) = vol20';             % 維度 3：大盤年化波動率
CIO_State(4, :) = abs(mdd252)';       % 維度 4：一年期最大回撤絕對值
CIO_State(5, :) = 1.0;                % 維度 5：當前持倉現金比例

%% 3. 實例化風險偏好三軌代理人
disp('--- 步驟 3：實例化三位具備獨立風險偏好的 RL 代理人 ---');
agent_aggressive   = Agent_PPO();   % 積極型：牛市主攻
agent_balanced     = Agent_PPO();   % 平衡型：震盪市輪動
agent_conservative = Agent_PPO();   % 保守型：熊市防禦

base_lr    = configObj.HRL_LR;
base_frict = configObj.MoE_FrictionMask; 
base_guard = configObj.Guardrail_CrashProb;

% 針對不同風險偏好，客製化物理環境與學習超參數
cfg_agg = struct('Frict', base_frict*0.5, 'Guard', min(0.99, base_guard*1.15), 'LR', base_lr*1.2);
cfg_bal = struct('Frict', base_frict,     'Guard', base_guard,                 'LR', base_lr);
cfg_con = struct('Frict', base_frict*1.5, 'Guard', base_guard*0.85,            'LR', base_lr*0.8);

%% 4. 動態對齊有效時間軸並啟動訓練
disp('--- 步驟 4：啟動三軌並行強化學習全域訓練 (CPU 多核 Parfor + 早停機制) ---');
spy_inception_idx = find(spy_prices > 10, 1);
if isempty(spy_inception_idx), spy_inception_idx = 1; end

Train_Start_Date = datetime('2006-01-01', 'TimeZone', 'UTC');
idx_train_start = find(Dates_Active >= Train_Start_Date, 1);
if isempty(idx_train_start)
    error('資料庫中找不到 2006 年以後的數據！');
end

valid_start_t = max(spy_inception_idx + 252, idx_train_start); 
idx_OOS_start = find(Dates_Active >= datetime('2022-01-01', 'TimeZone', 'UTC'), 1);

fprintf(' 📡 SPY 傳感器暖機完成，真實訓練區間：%s 至 %s\n', datestr(Dates_Active(valid_start_t)), datestr(Dates_Active(idx_OOS_start-1)));

epochs = configObj.HRL_Epochs;
batch_size = 128; 
RolloutSteps = 60; 
valid_starts = valid_start_t : (idx_OOS_start - RolloutSteps - 1);

figure('Name', 'Ensemble Agents Training Progress', 'Position', [100, 100, 900, 500]);
hLineAgg = animatedline('Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'Aggressive (Bull)');
hLineBal = animatedline('Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'Balanced (Shock)');
hLineCon = animatedline('Color', '#EDB120', 'LineWidth', 1.5, 'DisplayName', 'Conservative (Bear)');
xlabel('Epoch'); ylabel('Avg Step Reward'); grid on; legend('Location', 'northwest'); 

% ★ 二次優化修正 1：Early Stopping 狀態變數初始化
best_ensemble_reward = -inf;
patience_counter = 0;
patience_limit = 30; % 若連續 30 個 Epoch 未改善則觸發早停
best_agent_agg = []; best_agent_bal = []; best_agent_con = [];

for ep = 1:epochs
    start_indices = randsample(valid_starts, batch_size)'; 
    noise_std = max(0.01, 0.2 - (0.2 / (epochs * 0.8)) * ep);
    
    % ★ 二次優化修正 2：加入學習率排程衰減 (每 50 個 Epoch 衰減 20%)
    lr_decay = 0.8 ^ floor((ep - 1) / 50);
    cfg_agg.LR = base_lr * 1.2 * lr_decay;
    cfg_bal.LR = base_lr * lr_decay;
    cfg_con.LR = base_lr * 0.8 * lr_decay;
    
    agents_in = {agent_aggressive, agent_balanced, agent_conservative};
    cfgs_in = {cfg_agg, cfg_bal, cfg_con};
    
    rews_out = zeros(1, 3);
    agents_out = cell(1, 3);
    
    % 平行訓練三位代理人
    parfor a = 1:3
        [rews_out(a), agents_out{a}] = simulate_and_update(agents_in{a}, cfgs_in{a}, ...
            start_indices, RolloutSteps, CIO_State, P_time_M, P_space_M, P_crash_M, ...
            Prices_Active, Expert_Active, noise_std, configObj);
    end
    
    agent_aggressive   = agents_out{1};
    agent_balanced     = agents_out{2};
    agent_conservative = agents_out{3};
    
    rew_agg = rews_out(1);
    rew_bal = rews_out(2);
    rew_con = rews_out(3);
    
    addpoints(hLineAgg, ep, rew_agg); 
    addpoints(hLineBal, ep, rew_bal); 
    addpoints(hLineCon, ep, rew_con);
    drawnow limitrate;
    
    % 計算三軌平均獎勵，用於判斷早停
    ensemble_avg_reward = (rew_agg + rew_bal + rew_con) / 3;
    
    if mod(ep, 10) == 0 || ep == 1
        fprintf('Ep %3d | Agg R:%+6.2f | Bal R:%+6.2f | Con R:%+6.2f | LR Decay: %.2f\n', ...
            ep, rew_agg, rew_bal, rew_con, lr_decay);
    end
    
    % ★ 二次優化修正 3：Early Stopping 檢查與最優模型快照保存
    if ensemble_avg_reward > best_ensemble_reward + 1e-4
        best_ensemble_reward = ensemble_avg_reward;
        patience_counter = 0;
        
        % 保存當下表現最好的大腦實體
        best_agent_agg = agent_aggressive;
        best_agent_bal = agent_balanced;
        best_agent_con = agent_conservative;
    else
        patience_counter = patience_counter + 1;
    end
    
    if patience_counter >= patience_limit && ep > 100
        fprintf('\n🛑 [Early Stopping] 連續 %d 個 Epoch 未能顯著提升獎勵，提早結束訓練以防過擬合！(停於 Epoch %d)\n', patience_limit, ep);
        
        % 載入最佳快照
        agent_aggressive   = best_agent_agg;
        agent_balanced     = best_agent_bal;
        agent_conservative = best_agent_con;
        break;
    end
end

% 若跑滿 Epoch 未觸發早停，仍採用過程中的最佳快照
if ~isempty(best_agent_agg)
    agent_aggressive   = best_agent_agg;
    agent_balanced     = best_agent_bal;
    agent_conservative = best_agent_con;
end

disp('--- 步驟 5：儲存覺醒完成的三軌 CIO 大腦 ---');
save(fullfile(configObj.ModelDir, 'CIO_Aggressive.mat'), 'agent_aggressive');
save(fullfile(configObj.ModelDir, 'CIO_Balanced.mat'), 'agent_balanced');
save(fullfile(configObj.ModelDir, 'CIO_Conservative.mat'), 'agent_conservative');
disp('✅ Phase 5 訓練完成！最佳大腦已具備真實市場認知並安全落地。');

%% =====================================================================
% ★ 向量化環境模擬核心函數 (完全拔除 for 迴圈，MKL 矩陣極速版)
% =====================================================================
function [avg_reward, agent] = simulate_and_update(agent, cfg, start_indices, steps, CIO_State, P_time, P_space, P_crash, Prices, Expert, noise_std, configObj)
    batch_size = length(start_indices); 
    num_tickers = configObj.NumTickers;
    
    top_k_val = configObj.Top_K_Assets;
    fallback_w_time = configObj.Expert_Time_Weight;
    
    ep_states  = zeros(5, batch_size, steps, 'single'); 
    ep_actions = zeros(3, batch_size, steps, 'single'); 
    ep_rewards = zeros(1, batch_size, steps, 'single'); 
    
    prev_assets = zeros(num_tickers, batch_size, 'single'); 
    prev_cash = ones(1, batch_size, 'single'); 
    
    spy_idx = find(strcmp(configObj.IdxTickers, 'SPY')); 
    if isempty(spy_idx), spy_idx = 1; end
    
    for step_idx = 1:steps
        current_t = start_indices + step_idx - 1; 
        
        current_state = CIO_State(:, current_t);  
        current_state(5, :) = prev_cash;
        
        [acts_raw, ~] = agent.get_actions(current_state, noise_std); 
        acts_raw(isnan(acts_raw)) = 0;
        
        w_time = acts_raw(1, :);   
        w_space = acts_raw(2, :);  
        
        sum_w = w_time + w_space;
        zero_w_mask = (sum_w == 0);
        w_time(zero_w_mask) = fallback_w_time; 
        w_space(zero_w_mask) = 1.0 - fallback_w_time;
        
        w_time = w_time ./ (w_time + w_space);
        w_space = 1.0 - w_time;
        
        target_cash = acts_raw(3, :);
        
        c_mask = P_crash(current_t) > cfg.Guard; 
        target_cash(c_mask) = 1.0; 
        w_time(c_mask) = 0.0;       
        w_space(c_mask) = 0.0;
        
        w_cash = target_cash; 
        rem_weight = 1 - w_cash;
        
        comb_p = P_time(:, current_t) .* w_time + P_space(:, current_t) .* w_space; 
        
        active_mask_batch = Expert(current_t, :)';
        comb_p = comb_p .* active_mask_batch; 
        
        [~, sort_idx] = sort(comb_p, 1, 'descend');
        linear_indices = sub2ind(size(comb_p), sort_idx(top_k_val, :), 1:batch_size);
        thresholds = comb_p(linear_indices);
        comb_p(comb_p < thresholds) = 0;
        
        comb_p = comb_p ./ (sum(comb_p, 1) + 1e-8);
        asset_weights = comb_p .* rem_weight; 
        
        turnover_temp = sum(abs(asset_weights - prev_assets), 1); 
        ignore_mask = turnover_temp < cfg.Frict; 
        asset_weights(:, ignore_mask) = prev_assets(:, ignore_mask); 
        w_cash(ignore_mask) = prev_cash(ignore_mask);
        
        active_sum = sum(asset_weights, 1); 
        total_cap = active_sum + w_cash;
        asset_weights = asset_weights ./ total_cap; 
        w_cash = w_cash ./ total_cap;
        
        zero_mask = (total_cap == 0); 
        w_cash(zero_mask) = 1.0; 
        asset_weights(:, zero_mask) = 0;
        
        ret = (Prices(current_t + 1, :) ./ Prices(current_t, :) - 1)';
        ret(isnan(ret) | isinf(ret)) = 0; 
        
        current_vol = current_state(3, :); 
        current_vol(isnan(current_vol) | isinf(current_vol)) = 0;
        
        % 讀取 Config.m 統一摩擦力與衝擊成本模型
        if isprop(configObj, 'BaseFrictionFee') && isprop(configObj, 'SlippageVolCoeff')
            tc_rate = configObj.BaseFrictionFee + (configObj.SlippageVolCoeff .* current_vol);
        else
            tc_rate = 0.0005 + (0.10 .* current_vol);
        end
        
        tc = tc_rate .* sum(abs(asset_weights - prev_assets), 1); 
        
        port_ret = sum(asset_weights .* ret, 1) - tc; 
        spy_ret = ret(spy_idx, :);
        excess_return = port_ret - spy_ret;
        
        % 非對稱懲罰獎勵函數封頂防爆
        rew_cio = zeros(1, batch_size, 'single');
        pos_mask = excess_return >= 0;
        neg_mask = ~pos_mask;
        
        rew_cio(pos_mask) = excess_return(pos_mask) * 100.0;
        raw_penalty = (abs(excess_return(neg_mask)) * 100.0).^2;
        rew_cio(neg_mask) = -min(raw_penalty, 100.0); % 封頂在 -100
        
        ep_states(:, :, step_idx)  = current_state; 
        ep_actions(:, :, step_idx) = acts_raw; 
        ep_rewards(1, :, step_idx) = rew_cio;
        
        prev_assets = asset_weights; 
        prev_cash = w_cash;
    end
    
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
    
    agent.update_weights(flat_states, flat_actions, norm_returns, cfg.LR);
    avg_reward = mean(reshape(ep_rewards, 1, []));
end