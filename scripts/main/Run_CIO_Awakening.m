% =========================================================================
% 腳本：5_Run_CIO_HRL_Train.m (原 5_Run_CIO_Awakening.m)
% 升級：Phase 15.5 生產基準版 (★ 相容 Pure-Time 剪枝防訊號稀釋、
%       Config.m 全域 Horizon/RebalanceStride 動態繼承、
%       支援單軌穩健決策與三軌並行雙模式、Headless 伺服器友善無 GUI 繪圖、
%       PPO 動作探索綁定 mrg32k3a 獨立子串流、Open-to-Open 撮合與權重漂移同構、
%       死區連續縮放護欄、跨體制驗證早停、各階段獨立計時與總耗時審計)
% 職責：在向量化平行模擬環境中，訓練具備跨體制穩健性之 CIO 總管，動態管理權益與防禦部位
% =========================================================================
clear; clc; close all;

% 啟動全域總計時器
t_total_start = tic;
disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 CIO 強化學習訓練管線 (Pure-Time 剪枝與參數動態對齊版)');
disp('=================================================================');

%% 0. 環境路徑掛載與隨機串流鎖定
t_step0 = tic;
disp('--- 步驟 0：環境路徑掛載、Config 載入與平行池啟動 ---');
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

% 由 Config 統一生產 mrg32k3a 隨機數引擎，並設為全域主串流 (Substream = 1)
rng_seed = configObj.RNG_Seed;
rng_gen  = configObj.RNG_Generator;
stream_main = configObj.getRandStream(1);
RandStream.setGlobalStream(stream_main);
disp('🔒 已成功掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定 CIO 訓練環境確定性。');

% 啟動 CPU 平行運算池
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp(' ⚙️ 正在啟動 CPU 平行運算池 (Parallel Pool)...');
    parpool('Processes'); 
end
time_step0 = toc(t_step0);
fprintf('⏱️ [步驟 0 完成] 耗時: %.2f 秒\n\n', time_step0);

%% 1. 載入特徵快取、DataFetcher 開盤價與專家百分位選股矩陣
t_step1 = tic;
disp('--- 步驟 1：載入全域資料庫、Opens 矩陣與專家百分位選股分數 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
gbdtPath  = fullfile(configObj.ModelDir, 'GBDT_Guards.mat');

if ~exist(cachePath, 'file') || ~exist(gbdtPath, 'file')
    error('❌ 找不到前置快取檔案，請先確認 Phase 1 至 Phase 3 已成功執行！');
end

load(cachePath, 'Prices_Active', 'Expert_Active', 'Dates_Active');
load(gbdtPath, 'P_crash_all', 'P_time_all', 'P_space_all');

if isempty(Dates_Active.TimeZone)
    Dates_Active.TimeZone = 'UTC';
end
tz = Dates_Active.TimeZone;

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

% P_time_M 與 P_space_M 為 (0, 1] 橫截面百分位排序得分
P_time_M  = P_time_all(valid_idx, :)'; 
P_space_M = P_space_all(valid_idx, :)';
numDays   = length(Dates_Active);

% ★ 空間專家狀態探針：檢查是否剪枝空間專家
enable_space = isprop(configObj, 'EnableSpaceExpertTraining') && configObj.EnableSpaceExpertTraining && any(P_space_all(:) ~= 0);
if enable_space
    disp('  -> 空間專家狀態: 【已啟用】(RL 將動態調配時序與空間專家權重)');
else
    disp('  -> 空間專家狀態: 【⏩ 已剪枝】(Pure-Time 模式：強制時序權重=1.0，杜絕排序稀釋)');
end

time_step1 = toc(t_step1);
fprintf('⏱️ [步驟 1 完成] 耗時: %.2f 秒\n\n', time_step1);

%% 2. 建構 CIO 五維宏觀感知狀態
t_step2 = tic;
disp('--- 步驟 2：預計算 CIO 5 維宏觀狀態空間 ---');
CIO_State = zeros(5, numDays, 'single');

spy_idx = find(strcmp(configObj.IdxTickers, 'SPY'));
if isempty(spy_idx), spy_idx = 1; end

spy_prices = Prices_Active(:, spy_idx);
spy_rets = [0; diff(spy_prices) ./ (spy_prices(1:end-1) + 1e-8)];
spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
vol20 = movstd(spy_rets, [19, 0], 1) * sqrt(252);

mdd252 = zeros(numDays, 1);
for t = 1:numDays
    start_t = max(1, t - 251);
    window_rets = spy_rets(start_t:t);
    cum_ret = cumprod(1 + window_rets);
    running_max = cummax(cum_ret);
    drawdowns = (cum_ret - running_max) ./ (running_max + 1e-8);
    mdd252(t) = min(drawdowns);
end

spy_ret20 = zeros(numDays, 1);
for t = 21:numDays
    spy_ret20(t) = (spy_prices(t) - spy_prices(t-20)) / (spy_prices(t-20) + 1e-8);
end

CIO_State(1, :) = P_crash_M;          % 維度 1：大盤崩盤護欄機率 (平滑版)
CIO_State(2, :) = spy_ret20';         % 維度 2：大盤中期趨勢動能
CIO_State(3, :) = vol20';             % 維度 3：大盤年化波動率
CIO_State(4, :) = abs(mdd252)';       % 維度 4：一年期最大回撤絕對值
CIO_State(5, :) = 1.0;                % 維度 5：當前持倉現金比例

time_step2 = toc(t_step2);
fprintf('⏱️ [步驟 2 完成] 耗時: %.2f 秒\n\n', time_step2);

%% 3. 校準崩盤護欄死區下限並實例化代理人
t_step3 = tic;
disp('--- 步驟 3：校準崩盤護欄死區下限並實例化 RL 代理人 ---');
spy_inception_idx = find(spy_prices > 10, 1);
if isempty(spy_inception_idx), spy_inception_idx = 1; end

Train_Start_Date = datetime('2006-01-01', 'TimeZone', tz);
idx_train_start = find(Dates_Active >= Train_Start_Date, 1);
if isempty(idx_train_start)
    error('❌ 資料庫中找不到 2006 年以後的數據！');
end

valid_start_t = max(spy_inception_idx + 252, idx_train_start); 
idx_OOS_start = find(Dates_Active >= datetime('2022-01-01', 'TimeZone', tz), 1);

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

% ★ 判斷是否採用單軌穩健 CIO 模式 (依據消融實驗結論，預設單軌大幅提升訓練效率)
single_track_mode = isprop(configObj, 'EnableSingleTrackCIO') && configObj.EnableSingleTrackCIO;
if single_track_mode
    disp('  💡 [決策架構] 啟用【單軌穩健 CIO 決策模式】(消除無效三軌動態震盪，提升泛化力)。');
    num_agents_train = 1;
else
    disp('  💡 [決策架構] 啟用【標準三軌自適應 CIO 模式】(Aggressive, Balanced, Conservative)。');
    num_agents_train = 3;
end

base_lr    = configObj.HRL_LR;
base_frict = configObj.MoE_FrictionMask; 
base_guard = configObj.Guardrail_CrashProb;

cfg_agg = struct('Frict', base_frict*0.5, 'GuardHigh', min(0.99, base_guard*1.15), 'LR', base_lr*1.2, 'TauNoise', tau_noise);
cfg_bal = struct('Frict', base_frict,     'GuardHigh', base_guard,                 'LR', base_lr,     'TauNoise', tau_noise);
cfg_con = struct('Frict', base_frict*1.5, 'GuardHigh', base_guard*0.85,            'LR', base_lr*0.8, 'TauNoise', tau_noise);

agent_aggressive   = Agent_PPO(); 
agent_balanced     = Agent_PPO(); 
agent_conservative = Agent_PPO(); 

time_step3 = toc(t_step3);
fprintf('⏱️ [步驟 3 完成] 耗時: %.2f 秒\n\n', time_step3);

%% 4. 動態對齊時間軸並構建跨體制驗證池 (PPO 主迴圈)
t_step4 = tic;
disp('--- 步驟 4：切分訓練池與多體制驗證池並啟動訓練 ---');

% 動態對齊 RolloutSteps (根據全域 Horizon 自適應，不再寫死 60)
if isprop(configObj, 'RolloutSteps') && ~isempty(configObj.RolloutSteps)
    RolloutSteps = configObj.RolloutSteps;
else
    RolloutSteps = max(40, configObj.Horizon * 2);
end
fprintf('  -> 向量化訓練步長 (RolloutSteps): %d 步 (動態對齊 Horizon=%d)\n', RolloutSteps, configObj.Horizon);

% Open-to-Open 需存取 current_t + 2，嚴格截斷
valid_starts = valid_start_t : (idx_OOS_start - RolloutSteps - 2);

% 跨體制驗證窗口
val_windows = { ...
    struct('start', datetime('2008-01-01','TimeZone',tz), 'end', datetime('2009-06-01','TimeZone',tz)), ...
    struct('start', datetime('2020-01-01','TimeZone',tz), 'end', datetime('2020-12-01','TimeZone',tz)), ...
    struct('start', datetime('2018-09-01','TimeZone',tz), 'end', datetime('2019-03-01','TimeZone',tz)) ...
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

% 初始化記錄容器 (Headless 環境預設不可見)
fig_train = figure('Name', 'CIO Training Progress', ...
    'Position', [100, 100, 950, 520], 'Color', 'w', 'Visible', 'off');
set(fig_train, 'InvertHardcopy', 'off');

history_train_rews = zeros(epochs, 3);
history_val_rews   = zeros(epochs, 1);

best_val_ensemble_reward = -inf;
patience_counter = 0;
patience_limit = 25; 
best_agent_agg = []; best_agent_bal = []; best_agent_con = [];

for ep = 1:epochs
    start_indices = randsample(stream_main, train_starts, batch_size)'; 
    noise_std = max(0.01, 0.2 - (0.2 / (epochs * 0.8)) * ep);
    lr_decay = 0.8 ^ floor((ep - 1) / 50);
    
    cfg_agg.LR = base_lr * 1.2 * lr_decay;
    cfg_bal.LR = base_lr * lr_decay;
    cfg_con.LR = base_lr * 0.8 * lr_decay;
    
    if single_track_mode
        agents_in = {agent_balanced};
        cfgs_in   = {cfg_bal};
        cur_n     = 1;
    else
        agents_in = {agent_aggressive, agent_balanced, agent_conservative};
        cfgs_in   = {cfg_agg, cfg_bal, cfg_con};
        cur_n     = 3;
    end
    
    rews_out   = zeros(1, cur_n);
    agents_out = cell(1, cur_n);
    streams_train = cell(1, cur_n);
    
    for a = 1:cur_n
        s_obj = RandStream(rng_gen, 'Seed', rng_seed);
        s_obj.Substream = ep * 10 + a;
        streams_train{a} = s_obj;
    end
    
    parfor a = 1:cur_n
        [rews_out(a), agents_out{a}] = simulate_and_update(agents_in{a}, cfgs_in{a}, ...
            start_indices, RolloutSteps, CIO_State, P_time_M, P_space_M, P_crash_M, ...
            Opens_Active, Expert_Active, noise_std, configObj, false, streams_train{a}, enable_space);
    end
    
    if single_track_mode
        agent_balanced     = agents_out{1};
        agent_aggressive   = agent_balanced;
        agent_conservative = agent_balanced;
        rew_bal = rews_out(1);
        rew_agg = rew_bal; 
        rew_con = rew_bal;
    else
        agent_aggressive   = agents_out{1};
        agent_balanced     = agents_out{2};
        agent_conservative = agents_out{3};
        rew_agg = rews_out(1);
        rew_bal = rews_out(2);
        rew_con = rews_out(3);
    end
    
    history_train_rews(ep, :) = [rew_agg, rew_bal, rew_con];
    
    % 獨立多體制驗證 (無隨機噪聲)
    val_batch_size = min(length(val_starts), batch_size);
    val_sample_indices = randsample(stream_main, val_starts, val_batch_size)';
    val_rews_out = zeros(1, cur_n);
    
    agents_eval = agents_out;
    streams_val = cell(1, cur_n);
    for a = 1:cur_n
        s_obj = RandStream(rng_gen, 'Seed', rng_seed);
        s_obj.Substream = ep * 10 + 5 + a;
        streams_val{a} = s_obj;
    end
    
    parfor a = 1:cur_n
        [val_rews_out(a), ~] = simulate_and_update(agents_eval{a}, cfgs_in{a}, ...
            val_sample_indices, RolloutSteps, CIO_State, P_time_M, P_space_M, P_crash_M, ...
            Opens_Active, Expert_Active, 0.0, configObj, true, streams_val{a}, enable_space);
    end
    
    val_ensemble_reward = mean(val_rews_out);
    history_val_rews(ep) = val_ensemble_reward;
    
    if mod(ep, 10) == 0 || ep == 1
        if single_track_mode
            fprintf('Ep %3d | Unified R:%+6.2f | Val R:%+6.2f | LR Decay: %.2f\n', ...
                ep, rew_bal, val_ensemble_reward, lr_decay);
        else
            fprintf('Ep %3d | Agg R:%+6.2f | Bal R:%+6.2f | Con R:%+6.2f | Val R:%+6.2f | LR Decay: %.2f\n', ...
                ep, rew_agg, rew_bal, rew_con, val_ensemble_reward, lr_decay);
        end
    end
    
    % 早停依據：多體制驗證集 Reward
    if val_ensemble_reward > best_val_ensemble_reward + 1e-4
        best_val_ensemble_reward = val_ensemble_reward;
        patience_counter = 0;
        best_agent_agg = agent_aggressive;
        best_agent_bal = agent_balanced;
        best_agent_con = agent_conservative;
    else
        patience_counter = patience_counter + 1;
    end
    
    if patience_counter >= patience_limit && ep > 50
        fprintf('\n🛑 [Early Stopping] 跨體制驗證集連續 %d 輪未改善，提前於 Epoch %d 終止並回滾最佳快照！\n', patience_limit, ep);
        agent_aggressive   = best_agent_agg;
        agent_balanced     = best_agent_bal;
        agent_conservative = best_agent_con;
        epochs_completed = ep;
        break;
    end
    epochs_completed = ep;
end

if ~isempty(best_agent_agg)
    agent_aggressive   = best_agent_agg;
    agent_balanced     = best_agent_bal;
    agent_conservative = best_agent_con;
end

time_step4 = toc(t_step4);
fprintf('⏱️ [步驟 4 完成] 耗時: %.2f 秒 (%.2f 分鐘)\n\n', time_step4, time_step4 / 60);

%% 5. 儲存 CIO 模型權重與訓練曲線報表
t_step5 = tic;
disp('--- 步驟 5：儲存 CIO 代理人大腦與訓練曲線 (標準白底黑字) ---');

% 繪製標準白底黑字訓練曲線
ep_range = 1:epochs_completed;
plot(ep_range, history_train_rews(ep_range, 1), 'Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'Aggressive (Train)'); hold on;
plot(ep_range, history_train_rews(ep_range, 2), 'Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'Balanced (Train)');
plot(ep_range, history_train_rews(ep_range, 3), 'Color', '#EDB120', 'LineWidth', 1.5, 'DisplayName', 'Conservative (Train)');
plot(ep_range, history_val_rews(ep_range), 'Color', '#7E2F8E', 'LineWidth', 2.0, 'LineStyle', '--', 'DisplayName', 'Multi-Regime Val');

title(sprintf('HRL CIO Training Progress & Multi-Regime Validation (%dD Horizon)', configObj.Horizon), ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k'); 
ylabel('Avg Step Reward', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k'); 
legend('Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]); 
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 10);
grid on; box on;

if ~exist(configObj.ModelDir, 'dir'), mkdir(configObj.ModelDir); end
trainFigPath = fullfile(configObj.ModelDir, 'Phase5_CIO_Training_Curve.png');
exportgraphics(fig_train, trainFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf(' 📊 訓練進度曲線 (白底黑字) 已儲存至: %s\n', trainFigPath);
close(fig_train);

% 儲存三軌大腦 (下游 Phase 6 嚴格相容)
save(fullfile(configObj.ModelDir, 'CIO_Aggressive.mat'), 'agent_aggressive');
save(fullfile(configObj.ModelDir, 'CIO_Balanced.mat'), 'agent_balanced');
save(fullfile(configObj.ModelDir, 'CIO_Conservative.mat'), 'agent_conservative');
disp('💾 CIO 代理人大腦權重已安全落地！');

time_step5 = toc(t_step5);
fprintf('⏱️ [步驟 5 完成] 耗時: %.2f 秒\n\n', time_step5);

%% =========================================================================
% 結算全流程執行時長審計報告
% =========================================================================
total_elapsed_sec = toc(t_total_start);
tot_hours = floor(total_elapsed_sec / 3600);
tot_mins  = floor(mod(total_elapsed_sec, 3600) / 60);
tot_secs  = mod(total_elapsed_sec, 60);

fprintf('=================================================================\n');
fprintf('📊 【Phase 5 各階段耗時明細與總時長審計報告】\n');
fprintf('=================================================================\n');
fprintf(' 步驟 0：環境掛載與平行池啟動     : %8.2f 秒 (%5.1f%%)\n', time_step0, (time_step0 / total_elapsed_sec) * 100);
fprintf(' 步驟 1：快取載入與開盤價矩陣對齊 : %8.2f 秒 (%5.1f%%)\n', time_step1, (time_step1 / total_elapsed_sec) * 100);
fprintf(' 步驟 2：CIO 5 維宏觀狀態空間計算 : %8.2f 秒 (%5.1f%%)\n', time_step2, (time_step2 / total_elapsed_sec) * 100);
fprintf(' 步驟 3：死區下限校準與代理人實例化: %8.2f 秒 (%5.1f%%)\n', time_step3, (time_step3 / total_elapsed_sec) * 100);
fprintf(' 步驟 4：CIO 訓練與多體制驗證早停 : %8.2f 秒 (%5.1f%%)\n', time_step4, (time_step4 / total_elapsed_sec) * 100);
fprintf(' 步驟 5：訓練曲線輸出與大腦權重存檔: %8.2f 秒 (%5.1f%%)\n', time_step5, (time_step5 / total_elapsed_sec) * 100);
fprintf('-----------------------------------------------------------------\n');
if tot_hours > 0
    fprintf('⏱️ 【總執行時長】: %d 小時 %d 分 %.2f 秒 (共 %.2f 秒)\n', tot_hours, tot_mins, tot_secs, total_elapsed_sec);
else
    fprintf('⏱️ 【總執行時長】: %d 分 %.2f 秒 (共 %.2f 秒)\n', tot_mins, tot_secs, total_elapsed_sec);
end
fprintf('=================================================================\n');
disp('🎯 [Phase 5] CIO 訓練完成！請進入 Phase 6 前向回測。');
disp('=================================================================');

%% =====================================================================
% 向量化環境模擬核心函數 (★ Pure-Time 剪枝防稀釋 + 定期調倉 + 死區縮放)
% =====================================================================
function [avg_reward, agent] = simulate_and_update(agent, cfg, start_indices, steps, ...
    CIO_State, P_time, P_space, P_crash, Opens, Expert, noise_std, configObj, skip_update, stream, enable_space)
    
    if nargin < 13, skip_update = false; end
    if nargin < 15, enable_space = true; end
    
    if nargin >= 14 && ~isempty(stream)
        old_stream = RandStream.setGlobalStream(stream);
        cleanupObj = onCleanup(@() RandStream.setGlobalStream(old_stream));
    end
    
    batch_size     = length(start_indices); 
    num_tickers    = configObj.NumTickers;
    num_days_total = size(Opens, 1);
    
    top_k_val       = configObj.Top_K_Assets;
    fallback_w_time = configObj.Expert_Time_Weight;
    
    % 動態繼承全域調倉步進 (未指定則動態對齊 Horizon)
    if isprop(configObj, 'RebalanceStride') && ~isempty(configObj.RebalanceStride)
        stride = configObj.RebalanceStride;
    else
        stride = configObj.Horizon;
    end
    
    ep_states    = zeros(5, batch_size, steps, 'single'); 
    ep_actions   = zeros(3, batch_size, steps, 'single'); 
    ep_rewards   = zeros(1, batch_size, steps, 'single'); 
    ep_port_rets = zeros(1, batch_size, steps, 'single'); 
    
    prev_assets = zeros(num_tickers, batch_size, 'single'); 
    prev_cash   = ones(1, batch_size, 'single'); 
    cached_stock_props = zeros(num_tickers, batch_size, 'single');
    
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
        assert(max(current_t) + 2 <= num_days_total, '❌ 向量化模擬環境時間索引溢出邊界！');
        
        current_state = CIO_State(:, current_t);  
        current_state(5, :) = prev_cash;
        
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
        % 3. 動作正規化 (★ Pure-Time 剪枝防稀釋：強制時序權重=1.0)
        % -------------------------------------------------------------
        if ~enable_space
            w_time  = ones(1, batch_size, 'single');
            w_space = zeros(1, batch_size, 'single');
        else
            w_time  = max(0, acts_raw(1, :));   
            w_space = max(0, acts_raw(2, :));  
            sum_w   = w_time + w_space;
            zero_w_mask = (sum_w <= 1e-6);
            w_time(zero_w_mask)  = fallback_w_time; 
            w_space(zero_w_mask) = 1.0 - fallback_w_time;
            sum_w = w_time + w_space;
            w_time  = w_time ./ sum_w;
            w_space = w_space ./ sum_w;
        end
        target_cash = max(0, min(1, acts_raw(3, :)));
        
        % 崩盤護欄死區連續縮放
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
        % 4. 定期調倉 vs. 被動漂移資產配置
        % -------------------------------------------------------------
        is_rebal_step = (step_idx == 1) || (mod(step_idx - 1, stride) == 0);
        
        if is_rebal_step
            if enable_space
                comb_p = P_time(:, current_t) .* w_time + P_space(:, current_t) .* w_space; 
            else
                comb_p = P_time(:, current_t); % 純時序保真傳遞
            end
            
            active_mask_batch = Expert(current_t, :)';
            comb_p = comb_p .* active_mask_batch; 
            comb_p(halted_mask) = 0;
            
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
                    cached_stock_props(:, b) = col_p / s_val;
                else
                    cached_stock_props(:, b) = 0;
                end
            end
            target_active_weights = cached_stock_props .* rem_weight;
        else
            curr_active_sum = sum(w_drift .* (~halted_mask), 1);
            target_active_weights = zeros(num_tickers, batch_size, 'single');
            for b = 1:batch_size
                if curr_active_sum(b) > 1e-6 && rem_weight(b) > 0
                    scale_ratio = rem_weight(b) / curr_active_sum(b);
                    target_active_weights(:, b) = w_drift(:, b) .* scale_ratio;
                    target_active_weights(halted_mask(:, b), b) = 0;
                elseif rem_weight(b) <= 0
                    target_active_weights(:, b) = 0;
                else
                    target_active_weights(:, b) = cached_stock_props(:, b) .* rem_weight(b);
                end
            end
        end
        
        asset_weights = target_active_weights;
        asset_weights(halted_mask) = locked_weights(halted_mask);
        
        % -------------------------------------------------------------
        % 5. 慣性摩擦過濾 (Inertia Friction Mask)
        % -------------------------------------------------------------
        turnover_temp = abs(asset_weights - w_drift); 
        ignore_sub = (turnover_temp < cfg.Frict) & (~halted_mask); 
        asset_weights(ignore_sub) = w_drift(ignore_sub);
        
        active_sum = sum(asset_weights, 1); 
        total_cap  = active_sum + actual_cash_target;
        zero_mask  = (total_cap <= 1e-6); 
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
        
        is_hedging = (risk_scale > 0.5);
        cash_penalty_coeff = 0.05;
        rew_cio = rew_cio - cash_penalty_coeff * single(w_cash > 0.95 & ~is_hedging);
        
        ep_states(:, :, step_idx)  = current_state; 
        ep_actions(:, :, step_idx) = acts_raw; 
        ep_rewards(1, :, step_idx) = rew_cio;
        
        prev_assets = asset_weights; 
        prev_cash   = w_cash;
    end
    
    % Rollout 波動度懲罰
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
    ret_std  = std(flat_returns) + 1e-8;
    norm_returns = (flat_returns - ret_mean) ./ ret_std;
    
    flat_states  = reshape(ep_states, 5, []);
    flat_actions = reshape(ep_actions, 3, []); 
    
    if ~skip_update
        agent.update_weights(flat_states, flat_actions, norm_returns, cfg.LR);
    end
    
    avg_reward = mean(reshape(ep_rewards, 1, []));
end
