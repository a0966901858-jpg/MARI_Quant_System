% =========================================================================
% 腳本：Round7_ICGateAblation.m (Round 7: IC 注意力閘門開關消融實驗)
% 升級：Phase 15.5 Stage 0 診斷方案 (★ 顯存防 OOM 分塊驗證、空間雙輸入解耦、
%       HAC-Paired 統計檢定版、無 Linter 警告)
% 職責：檢驗時變 IC 注意力閘門是否引入非平穩輸入分佈，致使神經網路難以優化
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧪 [Round 7] 啟動 IC 注意力閘門開關消融 (Stage 0 方法論修復版)');
disp('=================================================================');

%% 0. 環境路徑掛載 (規範化絕對路徑解析) 與隨機種子鎖定
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

if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入特徵快取、子集抽樣與記憶體型態壓縮 (Memory Downcasting)
disp('--- 步驟 1：載入 3D 特徵快取並進行型態壓縮與記憶體釋放 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
if ~exist(cachePath, 'file')
    error('❌ 找不到特徵快取檔案，請先執行 1_Run_Data_and_Features.m！');
end

load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active', 'AdjMatrix_3D');
Dates_Active.TimeZone = 'UTC';

seqLen = configObj.SeqLen;
numDaysRaw = length(Dates_Active);
valid_idx = seqLen : numDaysRaw;

% 活躍標的抽樣 (Smoke 規模: 60 檔)
sample_T = min(60, configObj.NumTickers);
active_sums = sum(Expert_Active(valid_idx, :), 1);
[~, top_active_tickers] = sort(active_sums, 'descend');
sub_tickers = top_active_tickers(1:sample_T);

numExtractorFeats = 3 + configObj.NumMicroFeatures; % 18 維 (Rel 3 + Micro 15)

% ★ 資料型態下轉 (single / logical)
Prices_sub    = single(Prices_Active(valid_idx, sub_tickers));
Expert_sub    = logical(Expert_Active(valid_idx, sub_tickers));
X_sub_raw_18D = single(X_norm_3D(valid_idx, 1:numExtractorFeats, sub_tickers));
Adj_sub       = single(AdjMatrix_3D(sub_tickers, sub_tickers, valid_idx));
Dates_Active  = Dates_Active(valid_idx);
numDays       = length(Dates_Active);

% ★ 立即銷毀 10GB+ 全域大陣列
clear X_norm_3D Prices_Active Expert_Active AdjMatrix_3D;

fprintf('  📊 抽樣規模：標的 %d 檔 | 有效天數 %d 天 | 輸入維度 %d 維 (記憶體已釋放)\n', ...
    sample_T, numDays, numExtractorFeats);

%% 2. 構建時變 IC 注意力加權特徵矩陣 (Group A: With Gate)
disp('--- 步驟 2：計算 5D Horizon 橫截面 IC 能量守恆注意力閘門 ---');
evaluator = FeatureEvaluator(configObj);
[~, IC_Weights_3D] = evaluator.compute_confidence(X_sub_raw_18D, Prices_sub, Expert_sub, 5);

% 套用注意力閘門加權 (型態壓縮)
X_sub_gated_18D = X_sub_raw_18D .* single(IC_Weights_3D);
clear IC_Weights_3D;

%% 3. 構建 5 日 Beat-the-Median 目標標籤
disp('--- 步驟 3：構建 5 日 Beat-the-Median 目標標籤 ---');
horizon = 5;
R_fwd_5D = (Prices_sub(1+horizon:end, :) - Prices_sub(1:end-horizon, :)) ...
           ./ (Prices_sub(1:end-horizon, :) + 1e-8);
R_fwd_5D(isnan(R_fwd_5D) | isinf(R_fwd_5D)) = NaN;

Y_5D = false(numDays, sample_T);
for t = 1:numDays-horizon
    act_m = Expert_sub(t, :);
    if sum(act_m) > 5
        med_r = median(R_fwd_5D(t, act_m), 'omitnan');
        Y_5D(t, act_m) = (R_fwd_5D(t, act_m) > med_r);
    end
end
clear R_fwd_5D Prices_sub;

%% 4. 切分訓練集與跨體制驗證集 (Purged Embargo)
disp('--- 步驟 4：切分訓練集與驗證集 (Embargo 20 天) ---');
Train_Start_Date = datetime('2006-01-01', 'TimeZone', 'UTC');
idx_train_start = find(Dates_Active >= Train_Start_Date, 1);
if isempty(idx_train_start)
    idx_train_start = 252;
end

valid_start_t = max(252, idx_train_start);
idx_OOS_start = find(Dates_Active >= datetime('2022-01-01', 'TimeZone', 'UTC'), 1);

train_days = valid_start_t : (idx_OOS_start - 20);
val_days   = (idx_OOS_start + 20) : (numDays - 1);

fprintf('  -> 訓練天數: %d 天 | 驗證天數: %d 天\n', length(train_days), length(val_days));

use_gpu = canUseGPU();
if use_gpu
    gpu_dev = gpuDevice();
    fprintf('  🎮 偵測到硬體加速卡: 【%s】，啟用 GPU 向量化張量並行加速。\n', gpu_dev.Name);
else
    disp('  💻 使用 CPU 執行前向與反向運算。');
end

%% 5. 兩組對照實驗平行訓練 (Group A: 有閘門 vs Group B: 無閘門)
disp('--- 步驟 5：啟動 30 Epochs 向量化對照訓練 (Group A vs Group B) ---');

epochs = 30;
lr = 1e-3;
batch_size = 64;
batches_per_ep = floor(length(train_days) / batch_size);

feat_dim_space = numExtractorFeats * sample_T; % 1080
adj_dim_space  = sample_T * sample_T;          % 3600

exp_groups = { ...
    struct('name', 'Group A (With IC Gate)', 'X', X_sub_gated_18D), ...
    struct('name', 'Group B (Without Gate - Raw)', 'X', X_sub_raw_18D) ...
};

num_groups = length(exp_groups);
history_loss = struct();

% 固定全量驗證集評估基準天數
val_eval_days = val_days;

for g = 1:num_groups
    grp = exp_groups{g};
    fprintf('\n▶ 正在訓練 %s ...\n', grp.name);
    
    % 構造適配抽樣標的數之 Config 傳入工廠建立網路
    smokeConfig = configObj;
    smokeConfig.NumTickers = sample_T;
    extractorFactory = BuildDecoupledExtractors(smokeConfig);
    
    if ismethod(extractorFactory, 'buildNetworks')
        [net_t, net_s] = extractorFactory.buildNetworks();
    elseif ismethod(extractorFactory, 'build')
        [net_t, net_s] = extractorFactory.build();
    else
        net_t = extractorFactory.TimeNet;
        net_s = extractorFactory.SpaceNet;
    end
    
    W_t = dlarray(randn(64, 1, 'single') * 0.01);
    b_t = dlarray(zeros(1, 1, 'single'));
    W_s = dlarray(randn(64, 1, 'single') * 0.01);
    b_s = dlarray(zeros(1, 1, 'single'));
    
    avgG_T = []; avgsqG_T = [];
    avgG_S = []; avgsqG_S = [];
    
    tr_loss_t  = zeros(epochs, 1);
    val_loss_t = zeros(epochs, 1);
    tr_loss_s  = zeros(epochs, 1);
    val_loss_s = zeros(epochs, 1);
    
    X_input = grp.X;
    
    tic;
    for ep = 1:epochs
        shuffled_days = train_days(randperm(length(train_days)));
        ep_loss_t_sum = 0;
        ep_loss_s_sum = 0;
        
        for b = 1:batches_per_ep
            iter_idx = (ep - 1) * batches_per_ep + b;
            b_days = shuffled_days((b - 1) * batch_size + 1 : b * batch_size);
            B = length(b_days);
            
            % 時序專家批次向量化
            X_batch_T   = zeros(numExtractorFeats, sample_T * B, seqLen, 'single');
            Y_batch_T   = zeros(sample_T * B, 1, 'single');
            Act_batch_T = false(sample_T * B, 1);
            for i = 1:B
                t_curr = b_days(i);
                t_seq = (t_curr - seqLen + 1) : t_curr;
                col_range = (i - 1) * sample_T + 1 : i * sample_T;
                X_batch_T(:, col_range, :) = permute(X_input(t_seq, :, :), [2, 3, 1]);
                Y_batch_T(col_range)       = single(Y_5D(t_curr, :)');
                Act_batch_T(col_range)     = Expert_sub(t_curr, :)';
            end
            dl_x_t = dlarray(X_batch_T, 'CBT');
            if use_gpu, dl_x_t = gpuArray(dl_x_t); end
            
            [loss_t_b, grad_net_t, grad_w_t, grad_b_t] = dlfeval(@compute_vectorized_bce_time, ...
                net_t, W_t, b_t, dl_x_t, Y_batch_T, Act_batch_T);
            [net_t, avgG_T, avgsqG_T] = adamupdate(net_t, grad_net_t, avgG_T, avgsqG_T, iter_idx, lr);
            W_t = W_t - lr * grad_w_t;
            b_t = b_t - lr * grad_b_t;
            ep_loss_t_sum = ep_loss_t_sum + double(extractdata(loss_t_b));
            
            % 空間專家批次向量化 (雙輸入)
            X_batch_S_feat = zeros(feat_dim_space, B, 'single');
            X_batch_S_adj  = zeros(adj_dim_space, B, 'single');
            Y_batch_S      = single(Y_5D(b_days, :)');
            Act_batch_S    = Expert_sub(b_days, :)';
            for i = 1:B
                t_curr = b_days(i);
                x_c   = squeeze(X_input(t_curr, :, :));
                adj_c = squeeze(Adj_sub(:, :, t_curr));
                X_batch_S_feat(:, i) = x_c(:);
                X_batch_S_adj(:, i)  = adj_c(:);
            end
            dl_s_feat = dlarray(X_batch_S_feat, 'CB');
            dl_s_adj  = dlarray(X_batch_S_adj, 'CB');
            if use_gpu
                dl_s_feat = gpuArray(dl_s_feat);
                dl_s_adj  = gpuArray(dl_s_adj);
            end
            
            [loss_s_b, grad_net_s, grad_w_s, grad_b_s] = dlfeval(@compute_vectorized_bce_space, ...
                net_s, W_s, b_s, dl_s_feat, dl_s_adj, Y_batch_S(:), Act_batch_S(:), sample_T, B);
            [net_s, avgG_S, avgsqG_S] = adamupdate(net_s, grad_net_s, avgG_S, avgsqG_S, iter_idx, lr);
            W_s = W_s - lr * grad_w_s;
            b_s = b_s - lr * grad_b_s;
            ep_loss_s_sum = ep_loss_s_sum + double(extractdata(loss_s_b));
        end
        
        tr_loss_t(ep) = ep_loss_t_sum / batches_per_ep;
        tr_loss_s(ep) = ep_loss_s_sum / batches_per_ep;
        
        % -------------------------------------------------------------
        % 5A. 驗證集評估 (★ 分塊防 OOM 機制評估全量固定驗證集)
        % -------------------------------------------------------------
        val_loss_t(ep) = evaluate_val_loss_chunked_time(net_t, W_t, b_t, ...
            X_input, Y_5D, Expert_sub, val_eval_days, seqLen, sample_T, use_gpu);
        val_loss_s(ep) = evaluate_val_loss_chunked_space(net_s, W_s, b_s, ...
            X_input, Adj_sub, Y_5D, Expert_sub, val_eval_days, sample_T, feat_dim_space, adj_dim_space, use_gpu);
        
        if mod(ep, 10) == 0 || ep == 1
            fprintf('  Ep %2d/%2d | Val Loss (Time: %.4f | Space: %.4f)\n', ...
                ep, epochs, val_loss_t(ep), val_loss_s(ep));
        end
    end
    t_duration = toc;
    fprintf('  ⚡ %s 訓練完成！耗時: %.2f 秒\n', grp.name, t_duration);
    
    % -------------------------------------------------------------
    % 5B. 提取逐日驗證期損失 (用於 Stage 0 統計檢定)
    % -------------------------------------------------------------
    [daily_l_t, daily_l_s] = compute_day_level_val_losses(net_t, net_s, W_t, b_t, W_s, b_s, ...
        X_input, Adj_sub, Expert_sub, Y_5D, val_eval_days, seqLen, sample_T, use_gpu);
    
    % 儲存訓練歷史與計算線性探針 AUC
    history_loss(g).name         = grp.name;
    history_loss(g).train_t      = tr_loss_t;
    history_loss(g).val_t        = val_loss_t;
    history_loss(g).train_s      = tr_loss_s;
    history_loss(g).val_s        = val_loss_s;
    history_loss(g).daily_loss_t = daily_l_t;
    history_loss(g).daily_loss_s = daily_l_s;
    
    [auc_t, auc_s] = evaluate_linear_probe_batched(net_t, net_s, X_input, Adj_sub, Expert_sub, Y_5D, ...
        val_days, seqLen, sample_T, feat_dim_space, adj_dim_space, use_gpu);
    history_loss(g).auc_t = auc_t;
    history_loss(g).auc_s = auc_s;
end

%% 6. 統計顯著性檢定與因果報告 (★ Stage 0: Paired t & HAC 檢定)
disp('========================================================================================');
disp('📊 【Round 7 IC 注意力閘門消融實驗統計檢定報告】');
disp('========================================================================================');

val_loss_t_gate = history_loss(1).val_t(end);
val_loss_t_raw  = history_loss(2).val_t(end);
val_loss_s_gate = history_loss(1).val_s(end);
val_loss_s_raw  = history_loss(2).val_s(end);

delta_t = val_loss_t_raw - val_loss_t_gate; % Raw - Gated (> 0 代表 Gated 較優, < 0 代表 Raw 較優)
delta_s = val_loss_s_raw - val_loss_s_gate;

% ★ 對 Held-out 驗證期逐日損失差進行 Paired t-test 與 HAC 自相關校正檢定
diff_daily_t = history_loss(2).daily_loss_t - history_loss(1).daily_loss_t;
diff_daily_s = history_loss(2).daily_loss_s - history_loss(1).daily_loss_s;

[~, p_pair_t] = ttest(diff_daily_t);
[~, p_pair_s] = ttest(diff_daily_s);

[~, p_hac_t] = hac_significance_test(diff_daily_t);
[~, p_hac_s] = hac_significance_test(diff_daily_s);

fprintf('  > 時序專家 (Time Expert)  :\n');
fprintf('     - With Gate Val Loss : %.4f | Raw Val Loss : %.4f\n', val_loss_t_gate, val_loss_t_raw);
fprintf('     - 損失差值 (Raw-Gate): %+.4f (Paired t p-val = %.4f | HAC p-val = %.4f)\n', delta_t, p_pair_t, p_hac_t);
fprintf('     - 探針 AUC 對比      : With Gate AUC = %.4f | Raw AUC = %.4f (Delta AUC = %+.4f)\n', ...
    history_loss(1).auc_t, history_loss(2).auc_t, history_loss(1).auc_t - history_loss(2).auc_t);
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('  > 空間專家 (Space Expert) :\n');
fprintf('     - With Gate Val Loss : %.4f | Raw Val Loss : %.4f\n', val_loss_s_gate, val_loss_s_raw);
fprintf('     - 損失差值 (Raw-Gate): %+.4f (Paired t p-val = %.4f | HAC p-val = %.4f)\n', delta_s, p_pair_s, p_hac_s);
fprintf('     - 探針 AUC 對比      : With Gate AUC = %.4f | Raw AUC = %.4f (Delta AUC = %+.4f)\n', ...
    history_loss(1).auc_s, history_loss(2).auc_s, history_loss(1).auc_s - history_loss(2).auc_s);
fprintf('----------------------------------------------------------------------------------------\n');

% 決策判讀邏輯 (基於統計顯著性標準 alpha = 0.05)
if (p_hac_t < 0.05 && delta_t < 0) || (p_hac_s < 0.05 && delta_s < 0)
    fprintf('  ⚠️ 【GATE HARMFUL】IC 注意力閘門顯著惡化神經網路優化 (HAC p < 0.05 且 Raw 組損失顯著更低)！\n');
    fprintf('     -> 證明時變 IC 引入非平穩輸入雜訊，建議正式管線移除動態 IC 閘門。\n');
elseif (p_hac_t < 0.05 && delta_t > 0) || (p_hac_s < 0.05 && delta_s > 0)
    fprintf('  ⭐ 【GATE BENEFICIAL】IC 注意力閘門顯著改善表徵泛化能力 (HAC p < 0.05 且 Gated 損失顯著更低)！\n');
    fprintf('     -> 保留能量守恆注意力機制。\n');
else
    fprintf('  ✅ 【GATE NEUTRAL】IC 閘門影響未達統計顯著水準 (HAC p-value: Time=%.4f, Space=%.4f >= 0.05)。\n', p_hac_t, p_hac_s);
    fprintf('     -> 排除「時變 IC 閘門阻礙模型優化」假說，閘門權重變化處於雜訊容忍區間內。\n');
end
fprintf('========================================================================================\n\n');

%% 7. 產出標準學術白底黑字視覺化圖表
fig = figure('Name', 'Round 7 IC Gate Ablation Study', ...
    'Color', 'w', 'Position', [100, 100, 1150, 800], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

% 子圖 1：時序專家驗證損失曲線
subplot(2, 2, 1);
plot(1:epochs, history_loss(1).val_t, '-o', 'Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'Group A (With IC Gate)'); hold on;
plot(1:epochs, history_loss(2).val_t, '-s', 'Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'Group B (Raw 18D)');
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
title('Time Expert Validation BCE Loss (Fixed Set)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Val BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

% 子圖 2：空間專家驗證損失曲線
subplot(2, 2, 2);
plot(1:epochs, history_loss(1).val_s, '-o', 'Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'Group A (With IC Gate)'); hold on;
plot(1:epochs, history_loss(2).val_s, '-s', 'Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'Group B (Raw 18D)');
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
title('Space Expert Validation BCE Loss (Fixed Set)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Val BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

% 子圖 3：線性探針 AUC 對比
subplot(2, 2, 3);
auc_data = [history_loss(1).auc_t, history_loss(2).auc_t; ...
            history_loss(1).auc_s, history_loss(2).auc_s];
b = bar(auc_data, 'grouped');
b(1).FaceColor = [0.8500, 0.3250, 0.0980];
b(2).FaceColor = [0.0000, 0.4470, 0.7410];
for i = 1:2, b(i).EdgeColor = 'none'; end
set(gca, 'XTick', 1:2, 'XTickLabel', {'Time Expert', 'Space Expert'});
yline(0.50, '--k', 'Random Guess (0.50)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
ylim([0.45, max(0.55, max(auc_data, [], 'all') + 0.03)]);
title('Held-out Linear Probe AUC Comparison', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Probe AUC', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend({'With IC Gate', 'Raw 18D'}, 'Location', 'northeast', ...
    'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

% 子圖 4：驗證期逐日損失差分佈 (Raw - Gated)
subplot(2, 2, 4);
h_data = [diff_daily_t, diff_daily_s];
boxplot(h_data, {'Time Delta', 'Space Delta'}, 'Colors', 'k', 'Symbol', 'r+');
yline(0.0, '--k', 'No Difference (0.0)', 'LineWidth', 1.1);
title(sprintf('Daily Val Loss Difference [Raw - Gated]\n(HAC p: T=%.3f, S=%.3f)', p_hac_t, p_hac_s), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('\Delta Loss (Raw - Gated)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

figPath = fullfile(configObj.ResultDir, 'Round7_ICGateAblation.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化圖表 (白底黑字) 已儲存至: %s\n', figPath);
close(fig);

disp('=================================================================');
disp('🎯 [Round 7] 執行完畢！');
disp('=================================================================');

%% =====================================================================
% 輔助函數：空間專家雙輸入自適應轉發器 (防止連接埠順序顛倒)
% =====================================================================
function emb = forward_space_wrapper(net, dl_feat, dl_adj)
    input_names = net.InputNames;
    if length(input_names) >= 2 && contains(input_names{1}, 'adj')
        emb = forward(net, dl_adj, dl_feat);
    else
        emb = forward(net, dl_feat, dl_adj);
    end
end

%% =====================================================================
% 輔助函數：時序專家批次向量化訓練損失與梯度
% =====================================================================
function [loss, grad_net, grad_w, grad_b] = compute_vectorized_bce_time(net, W, b, dl_x, y_true, act_m)
    emb = forward(net, dl_x);
    emb = stripdims(emb);
    logits = W' * emb + b;
    probs = sigmoid(logits);
    
    y_t = y_true(act_m);
    y_t = y_t(:)';
    p_t = probs(act_m);
    p_t = p_t(:)';
    
    bce = -mean(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8));
    
    emb_act = emb(:, act_m);
    std_e = sqrt(var(emb_act, 0, 2) + 1e-4);
    var_loss = mean(max(0, 1.0 - std_e));
    
    loss = bce + 0.1 * var_loss;
    [grad_net, grad_w, grad_b] = dlgradient(loss, net.Learnables, W, b);
end

%% =====================================================================
% 輔助函數：空間專家批次向量化訓練損失與梯度 (雙輸入)
% =====================================================================
function [loss, grad_net, grad_w, grad_b] = compute_vectorized_bce_space(net, W, b, dl_feat, dl_adj, y_true, act_m, numT, B)
    emb_raw = forward_space_wrapper(net, dl_feat, dl_adj);
    emb_raw = stripdims(emb_raw);
    emb = reshape(emb_raw, 64, numT * B);
    
    logits = W' * emb + b;
    probs = sigmoid(logits);
    
    y_t = y_true(act_m);
    y_t = y_t(:)';
    p_t = probs(act_m);
    p_t = p_t(:)';
    
    bce = -mean(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8));
    
    emb_act = emb(:, act_m);
    std_e = sqrt(var(emb_act, 0, 2) + 1e-4);
    var_loss = mean(max(0, 1.0 - std_e));
    
    loss = bce + 0.1 * var_loss;
    [grad_net, grad_w, grad_b] = dlgradient(loss, net.Learnables, W, b);
end

%% =====================================================================
% 輔助函數：時序專家驗證純前向計算
% =====================================================================
function loss = compute_val_forward_time(net, W, b, dl_x, y_true, act_m)
    emb = forward(net, dl_x);
    emb = stripdims(emb);
    logits = W' * emb + b;
    probs = sigmoid(logits);
    y_t = y_true(act_m);
    y_t = y_t(:)';
    p_t = probs(act_m);
    p_t = p_t(:)';
    loss = -mean(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8));
end

%% =====================================================================
% 輔助函數：空間專家驗證純前向計算 (雙輸入)
% =====================================================================
function loss = compute_val_forward_space(net, W, b, dl_feat, dl_adj, y_true, act_m, numT, B)
    emb_raw = forward_space_wrapper(net, dl_feat, dl_adj);
    emb_raw = stripdims(emb_raw);
    emb = reshape(emb_raw, 64, numT * B);
    logits = W' * emb + b;
    probs = sigmoid(logits);
    y_t = y_true(act_m);
    y_t = y_t(:)';
    p_t = probs(act_m);
    p_t = p_t(:)';
    loss = -mean(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8));
end

%% =====================================================================
% 輔助函數：時序專家分塊驗證損失計算 (★ 防 OOM 核心函式)
% =====================================================================
function val_loss = evaluate_val_loss_chunked_time(net, W, b, X_input, Y_5D, Expert_sub, val_days, seqLen, sample_T, use_gpu)
    V = length(val_days);
    chunk_size = 32;
    loss_sum = 0;
    count = 0;
    numFeats = size(X_input, 2);
    
    for c = 1:ceil(V / chunk_size)
        c_idx = (c - 1) * chunk_size + 1 : min(c * chunk_size, V);
        c_days = val_days(c_idx);
        C = length(c_days);
        
        X_chunk   = zeros(numFeats, sample_T * C, seqLen, 'single');
        Y_chunk   = zeros(sample_T * C, 1, 'single');
        Act_chunk = false(sample_T * C, 1);
        
        for i = 1:C
            t_curr = c_days(i);
            t_seq = (t_curr - seqLen + 1) : t_curr;
            col_range = (i - 1) * sample_T + 1 : i * sample_T;
            X_chunk(:, col_range, :) = permute(X_input(t_seq, :, :), [2, 3, 1]);
            Y_chunk(col_range)       = single(Y_5D(t_curr, :)');
            Act_chunk(col_range)     = Expert_sub(t_curr, :)';
        end
        
        dl_x = dlarray(X_chunk, 'CBT');
        if use_gpu
            dl_x = gpuArray(dl_x);
        end
        
        emb = forward(net, dl_x);
        emb = stripdims(emb);
        logits = W' * emb + b;
        probs = sigmoid(logits);
        
        y_t = Y_chunk(Act_chunk);
        y_t = y_t(:);
        p_t = probs(Act_chunk);
        p_t = p_t(:);
        
        bce_c = -sum(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8), 'all');
        loss_sum = loss_sum + double(extractdata(gather(bce_c)));
        count = count + sum(Act_chunk, 'all');
    end
    val_loss = loss_sum / max(1, count);
end

%% =====================================================================
% 輔助函數：空間專家分塊驗證損失計算 (★ 防 OOM 核心函式)
% =====================================================================
function val_loss = evaluate_val_loss_chunked_space(net, W, b, X_input, Adj_sub, Y_5D, Expert_sub, val_days, sample_T, feat_dim, adj_dim, use_gpu)
    V = length(val_days);
    chunk_size = 32;
    loss_sum = 0;
    count = 0;
    
    for c = 1:ceil(V / chunk_size)
        c_idx = (c - 1) * chunk_size + 1 : min(c * chunk_size, V);
        c_days = val_days(c_idx);
        C = length(c_days);
        
        X_feat_c  = zeros(feat_dim, C, 'single');
        X_adj_c   = zeros(adj_dim, C, 'single');
        Y_chunk   = single(Y_5D(c_days, :)');
        Act_chunk = Expert_sub(c_days, :)';
        
        for i = 1:C
            t_curr = c_days(i);
            x_c   = squeeze(X_input(t_curr, :, :));
            adj_c = squeeze(Adj_sub(:, :, t_curr));
            X_feat_c(:, i) = x_c(:);
            X_adj_c(:, i)  = adj_c(:);
        end
        
        dl_f = dlarray(X_feat_c, 'CB');
        dl_a = dlarray(X_adj_c, 'CB');
        if use_gpu
            dl_f = gpuArray(dl_f);
            dl_a = gpuArray(dl_a);
        end
        
        emb_raw = forward_space_wrapper(net, dl_f, dl_a);
        emb_raw = stripdims(emb_raw);
        emb = reshape(emb_raw, 64, sample_T * C);
        logits = W' * emb + b;
        probs = sigmoid(logits);
        
        y_t = Y_chunk(Act_chunk);
        y_t = y_t(:);
        p_t = probs(Act_chunk);
        p_t = p_t(:);
        
        bce_c = -sum(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8), 'all');
        loss_sum = loss_sum + double(extractdata(gather(bce_c)));
        count = count + sum(Act_chunk, 'all');
    end
    val_loss = loss_sum / max(1, count);
end

%% =====================================================================
% 輔助函數：提取逐日驗證期損失 (Day-Level BCE Loss) - 已消除警告版
% =====================================================================
function [daily_l_t, daily_l_s] = compute_day_level_val_losses(net_t, net_s, W_t, b_t, W_s, b_s, ...
    X_input, Adj_sub, Expert_sub, Y_5D, val_days, seqLen, sample_T, use_gpu)
    V = length(val_days);
    daily_l_t = zeros(V, 1);
    daily_l_s = zeros(V, 1);
    
    for i = 1:V
        t_curr = val_days(i);
        t_seq = (t_curr - seqLen + 1) : t_curr;
        y_true = single(Y_5D(t_curr, :)');
        act_m = Expert_sub(t_curr, :)';
        
        % 時序天損失 (單日 60 檔標的，無 OOM 風險)
        x_in = permute(X_input(t_seq, :, :), [2, 3, 1]);
        dl_x_t = dlarray(x_in, 'CBT');
        if use_gpu, dl_x_t = gpuArray(dl_x_t); end
        l_t = compute_val_forward_time(net_t, W_t, b_t, dl_x_t, y_true, act_m);
        daily_l_t(i) = double(extractdata(l_t));
        
        % 空間天損失 (單日 60 檔標的，無 OOM 風險)
        x_c = squeeze(X_input(t_curr, :, :));
        adj_c = squeeze(Adj_sub(:, :, t_curr));
        dl_s_feat = dlarray(x_c(:), 'CB');
        dl_s_adj  = dlarray(adj_c(:), 'CB');
        if use_gpu
            dl_s_feat = gpuArray(dl_s_feat);
            dl_s_adj  = gpuArray(dl_s_adj);
        end
        l_s = compute_val_forward_space(net_s, W_s, b_s, dl_s_feat, dl_s_adj, y_true, act_m, sample_T, 1);
        daily_l_s(i) = double(extractdata(l_s));
    end
end

%% =====================================================================
% 輔助函數：批次分塊線性探針 AUC 評估
% =====================================================================
function [auc_t, auc_s] = evaluate_linear_probe_batched(net_t, net_s, X_input, Adj_sub, Expert_sub, Y_5D, ...
    val_days, seqLen, sample_T, feat_dim_space, adj_dim_space, use_gpu)
    total_val = sum(Expert_sub(val_days, :), 'all');
    E_t = zeros(total_val, 64, 'single');
    E_s = zeros(total_val, 64, 'single');
    Y_v = zeros(total_val, 1, 'single');
    
    num_val_days = length(val_days);
    eval_chunk_size = 128;
    cur_row = 1;
    numExtractorFeats = size(X_input, 2);
    
    for c = 1:ceil(num_val_days / eval_chunk_size)
        c_idx = (c - 1) * eval_chunk_size + 1 : min(c * eval_chunk_size, num_val_days);
        c_days = val_days(c_idx);
        C = length(c_days);
        
        % 時序批次提取
        X_chunk_T = zeros(numExtractorFeats, sample_T * C, seqLen, 'single');
        for i = 1:C
            t_curr = c_days(i);
            t_seq = (t_curr - seqLen + 1) : t_curr;
            col_range = (i - 1) * sample_T + 1 : i * sample_T;
            X_chunk_T(:, col_range, :) = permute(X_input(t_seq, :, :), [2, 3, 1]);
        end
        dl_x_c_t = dlarray(X_chunk_T, 'CBT');
        if use_gpu
            dl_x_c_t = gpuArray(dl_x_c_t);
        end
        emb_t_all = extractdata(gather(forward(net_t, dl_x_c_t)));
        
        % 空間批次提取 (雙輸入)
        X_chunk_S_feat = zeros(feat_dim_space, C, 'single');
        X_chunk_S_adj  = zeros(adj_dim_space, C, 'single');
        for i = 1:C
            t_curr = c_days(i);
            x_c   = squeeze(X_input(t_curr, :, :));
            adj_c = squeeze(Adj_sub(:, :, t_curr));
            X_chunk_S_feat(:, i) = x_c(:);
            X_chunk_S_adj(:, i)  = adj_c(:);
        end
        dl_c_s_feat = dlarray(X_chunk_S_feat, 'CB');
        dl_c_s_adj  = dlarray(X_chunk_S_adj, 'CB');
        if use_gpu
            dl_c_s_feat = gpuArray(dl_c_s_feat);
            dl_c_s_adj  = gpuArray(dl_c_s_adj);
        end
        
        emb_s_raw = forward_space_wrapper(net_s, dl_c_s_feat, dl_c_s_adj);
        emb_s_raw = extractdata(gather(emb_s_raw));
        emb_s_all = reshape(emb_s_raw, 64, sample_T * C);
        
        for i = 1:C
            t_curr = c_days(i);
            act_ids = find(Expert_sub(t_curr, :));
            n_act = length(act_ids);
            if n_act > 0
                cols_selected = (i - 1) * sample_T + act_ids;
                E_t(cur_row : cur_row + n_act - 1, :) = emb_t_all(:, cols_selected)';
                E_s(cur_row : cur_row + n_act - 1, :) = emb_s_all(:, cols_selected)';
                Y_v(cur_row : cur_row + n_act - 1)    = single(Y_5D(t_curr, act_ids)');
                cur_row = cur_row + n_act;
            end
        end
    end
    
    mdl_t = fitclinear(E_t, Y_v, 'Learner', 'logistic');
    mdl_s = fitclinear(E_s, Y_v, 'Learner', 'logistic');
    
    [~, score_t] = predict(mdl_t, E_t);
    [~, score_s] = predict(mdl_s, E_s);
    
    [~, ~, ~, auc_t] = perfcurve(Y_v, score_t(:, 2), 1);
    [~, ~, ~, auc_s] = perfcurve(Y_v, score_s(:, 2), 1);
end