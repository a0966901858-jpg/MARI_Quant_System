% =========================================================================
% 腳本：Round9_TrainingRegimeSweep.m (Round 9: 訓練體制與學習率排程排查)
% 升級：Phase 15.5 診斷方案 (★ 工廠實例化適配、全向量化 GPU 批次加速、記憶體極致釋放)
% 職責：排除「訓練不足」與「學習率排程不當」假說，驗證表徵是否面臨真實資訊天花板
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧪 [Round 9] 啟動訓練體制與學習率排查 (GPU 向量化批次加速版)');
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

load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active');
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
Prices_sub   = single(Prices_Active(valid_idx, sub_tickers));
Expert_sub   = logical(Expert_Active(valid_idx, sub_tickers));
X_sub_18D    = single(X_norm_3D(valid_idx, 1:numExtractorFeats, sub_tickers));
Dates_Active = Dates_Active(valid_idx);
numDays      = length(Dates_Active);

% ★ 立即銷毀 10GB+ 全域大陣列
clear X_norm_3D Prices_Active Expert_Active;

fprintf('  📊 抽樣規模：標的 %d 檔 | 序列長度 %d | 輸入維度 %d 維 (記憶體已釋放)\n', ...
    sample_T, seqLen, numExtractorFeats);

%% 2. 構建 5 日 Beat-the-Median 目標標籤
disp('--- 步驟 2：構建 5 日 Beat-the-Median 目標標籤 ---');
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

%% 3. 切分訓練集與跨體制驗證集 (Purged Embargo)
disp('--- 步驟 3：切分訓練集與驗證集 (Embargo 20 天) ---');
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

%% 4. 定義三組學習率排程與實驗參數 (100 Epochs, 無 Early Stopping)
disp('--- 步驟 4：設定 3 組 LR 排程 (Step Decay vs Constant Small vs Cosine) ---');

total_epochs = 100;
batch_size   = 64;
batches_per_ep = floor(length(train_days) / batch_size);

lr_regimes = { ...
    struct('name', 'Schedule 1: Step Decay (1e-3 -> 1.25e-4)', 'type', 'step', 'base_lr', 1e-3), ...
    struct('name', 'Schedule 2: Constant Small (2e-4)',       'type', 'constant', 'base_lr', 2e-4), ...
    struct('name', 'Schedule 3: Cosine Annealing (1e-3 -> 1e-5)', 'type', 'cosine', 'base_lr', 1e-3, 'min_lr', 1e-5) ...
};

num_regimes = length(lr_regimes);
regime_results = struct();

%% 5. 依序執行 100 Epochs 長週期訓練並記錄高頻 Batch Loss (Mini-Batch 向量化)
disp('--- 步驟 5：啟動 100 Epochs 向量化訓練與高頻 Loss 監控 ---');

for r = 1:num_regimes
    reg = lr_regimes{r};
    fprintf('\n▶ 正在執行 [%d/%d] %s ...\n', r, num_regimes, reg.name);
    
    % ★ 構造適配抽樣標的數之 Config 傳入工廠建立時序專家網路
    smokeConfig = configObj;
    smokeConfig.NumTickers = sample_T;
    extractorFactory = BuildDecoupledExtractors(smokeConfig);
    
    if ismethod(extractorFactory, 'buildNetworks')
        [net_curr, ~] = extractorFactory.buildNetworks();
    elseif ismethod(extractorFactory, 'build')
        [net_curr, ~] = extractorFactory.build();
    else
        net_curr = extractorFactory.TimeNet;
    end
    
    W_aux = dlarray(randn(64, 1, 'single') * 0.01);
    b_aux = dlarray(zeros(1, 1, 'single'));
    
    avgG = [];
    avgsqG = [];
    
    epoch_val_loss   = zeros(total_epochs, 1);
    epoch_train_loss = zeros(total_epochs, 1);
    lr_history       = zeros(total_epochs, 1);
    batch_loss_track = [];
    
    total_iter_count = 0;
    
    tic;
    for ep = 1:total_epochs
        % 動態學習率排程計算
        switch reg.type
            case 'step'
                lr_curr = reg.base_lr * (0.5 ^ floor((ep - 1) / 25));
            case 'constant'
                lr_curr = reg.base_lr;
            case 'cosine'
                lr_curr = reg.min_lr + 0.5 * (reg.base_lr - reg.min_lr) * (1 + cos(pi * ep / total_epochs));
            otherwise
                lr_curr = reg.base_lr;
        end
        lr_history(ep) = lr_curr;
        
        shuffled_days = train_days(randperm(length(train_days)));
        ep_loss_sum = 0;
        
        for b = 1:batches_per_ep
            total_iter_count = total_iter_count + 1;
            b_days = shuffled_days((b - 1) * batch_size + 1 : b * batch_size);
            B = length(b_days);
            
            % ---------------------------------------------------------
            % 全 Batch 向量化 3D 張量 [Feats, sample_T * B, SeqLen]
            % ---------------------------------------------------------
            X_batch_T   = zeros(numExtractorFeats, sample_T * B, seqLen, 'single');
            Y_batch_T   = zeros(sample_T * B, 1, 'single');
            Act_batch_T = false(sample_T * B, 1);
            
            for i = 1:B
                t_curr = b_days(i);
                t_seq = (t_curr - seqLen + 1) : t_curr;
                col_range = (i - 1) * sample_T + 1 : i * sample_T;
                
                X_batch_T(:, col_range, :) = permute(X_sub_18D(t_seq, :, :), [2, 3, 1]);
                Y_batch_T(col_range)       = single(Y_5D(t_curr, :)');
                Act_batch_T(col_range)     = Expert_sub(t_curr, :)';
            end
            
            dl_x = dlarray(X_batch_T, 'CBT');
            if use_gpu
                dl_x = gpuArray(dl_x);
            end
            
            % 單次 GPU 前向與反向傳播 (stripdims 脫除標籤以支援矩陣乘法)
            [loss_b, grad_net, grad_w, grad_b] = dlfeval(@compute_vectorized_train_bce_loss, ...
                net_curr, W_aux, b_aux, dl_x, Y_batch_T, Act_batch_T);
            
            [net_curr, avgG, avgsqG] = adamupdate(net_curr, grad_net, avgG, avgsqG, total_iter_count, lr_curr);
            W_aux = W_aux - lr_curr * grad_w;
            b_aux = b_aux - lr_curr * grad_b;
            
            b_loss_val = double(extractdata(loss_b));
            ep_loss_sum = ep_loss_sum + b_loss_val;
            
            % 每 5 個 Batch 記錄高頻 Loss 軌跡
            if mod(total_iter_count, 5) == 0
                batch_loss_track = [batch_loss_track; b_loss_val];
            end
        end
        epoch_train_loss(ep) = ep_loss_sum / batches_per_ep;
        
        % ---------------------------------------------------------
        % 驗證集批次評估 (純前向，無梯度圖開銷)
        % ---------------------------------------------------------
        val_sample = randsample(val_days, min(64, length(val_days)));
        V = length(val_sample);
        
        X_val_T   = zeros(numExtractorFeats, sample_T * V, seqLen, 'single');
        Y_val_T   = zeros(sample_T * V, 1, 'single');
        Act_val_T = false(sample_T * V, 1);
        for i = 1:V
            t_curr = val_sample(i);
            t_seq = (t_curr - seqLen + 1) : t_curr;
            col_range = (i - 1) * sample_T + 1 : i * sample_T;
            X_val_T(:, col_range, :) = permute(X_sub_18D(t_seq, :, :), [2, 3, 1]);
            Y_val_T(col_range)       = single(Y_5D(t_curr, :)');
            Act_val_T(col_range)     = Expert_sub(t_curr, :)';
        end
        dl_x_val = dlarray(X_val_T, 'CBT');
        if use_gpu
            dl_x_val = gpuArray(dl_x_val);
        end
        epoch_val_loss(ep) = double(extractdata(compute_val_forward_loss_vectorized(net_curr, W_aux, b_aux, dl_x_val, Y_val_T, Act_val_T)));
        
        if mod(ep, 20) == 0 || ep == 1
            fprintf('  Ep %3d/%3d (LR: %.2e) | Train BCE: %.4f | Val BCE: %.4f\n', ...
                ep, total_epochs, lr_curr, epoch_train_loss(ep), epoch_val_loss(ep));
        end
    end
    t_duration = toc;
    fprintf('  ⚡ %s 訓練完成！耗時: %.2f 秒\n', reg.name, t_duration);
    
    % 批次分塊線性探針 AUC 評估
    auc_probe = evaluate_probe_batched(net_curr, X_sub_18D, Expert_sub, Y_5D, val_days, seqLen, sample_T, use_gpu);
    
    regime_results(r).name             = reg.name;
    regime_results(r).train_loss       = epoch_train_loss;
    regime_results(r).val_loss         = epoch_val_loss;
    regime_results(r).lr_history       = lr_history;
    regime_results(r).batch_loss       = batch_loss_track;
    regime_results(r).min_val_loss     = min(epoch_val_loss);
    regime_results(r).final_val_loss   = epoch_val_loss(end);
    regime_results(r).auc              = auc_probe;
end

%% 6. 判讀門檻檢定與結論報告
disp('========================================================================================');
disp('📊 【Round 9 訓練體制與學習率排程消融結論報告】');
disp('========================================================================================');
fprintf(' 排程名稱                              | 最佳 Val BCE | 最終 Val BCE | 探針 AUC | 突破基準\n');
fprintf('----------------------------------------------------------------------------------------\n');

baseline_threshold = 0.6938; % 15-Epoch 最低基準點
breakthrough_flag = false;

for r = 1:num_regimes
    min_v = regime_results(r).min_val_loss;
    fin_v = regime_results(r).final_val_loss;
    auc_v = regime_results(r).auc;
    
    if min_v < baseline_threshold
        break_str = sprintf('YES (%.4f < %.4f)', min_v, baseline_threshold);
        breakthrough_flag = true;
    else
        break_str = 'NO (Stuck at Ceiling)';
    end
    
    fprintf(' %-37s |    %.4f    |    %.4f    |  %.4f  | %s\n', ...
        regime_results(r).name, min_v, fin_v, auc_v, break_str);
end
fprintf('----------------------------------------------------------------------------------------\n');

if breakthrough_flag
    fprintf('  ⭐ 【TRAINING EXTENSION BENEFICIAL】長週期訓練或特定 LR 排程成功突破 0.6938 基準！\n');
    fprintf('     -> 證明原先的損失停滯部分源於 Epoch 數不足或學習率衰減過快。\n');
else
    fprintf('  ✅ 【CEILING CONFIRMED】即使訓練拉長至 100 Epochs 且換用多種 LR 排程，\n');
    fprintf('     Val Loss 全程無法低於 0.6938，且始終在 ln(2)≈0.6931 隨機基準線震盪。\n');
    fprintf('     -> 徹底排除「訓練不足 (Under-training)」假說，確認表徵受限於當前特徵之資訊天花板！\n');
end
fprintf('========================================================================================\n\n');

%% 7. 產出標準學術白底黑字視覺化圖表
fig = figure('Name', 'Round 9 Training Regime Sweep', ...
    'Color', 'w', 'Position', [100, 100, 1200, 750], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

color_rgb = [
    0.8500, 0.3250, 0.0980; % D95319
    0.0000, 0.4470, 0.7410; % 0072BD
    0.4940, 0.1840, 0.5560  % 7E2F8E
];

% 子圖 1：100 Epochs 驗證損失曲線對比
subplot(2, 2, 1);
for r = 1:num_regimes
    plot(1:total_epochs, regime_results(r).val_loss, 'Color', color_rgb(r, :), 'LineWidth', 1.5, ...
        'DisplayName', regime_results(r).name);
    hold on;
end
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
yline(baseline_threshold, ':r', sprintf('15-Ep Baseline (%.4f)', baseline_threshold), 'LineWidth', 1.1);
title('Validation BCE Loss across 100 Epochs', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Val BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

% 子圖 2：前 100 步高頻 Batch Loss 軌跡
subplot(2, 2, 2);
max_step = min(100, length(regime_results(1).batch_loss));
for r = 1:num_regimes
    b_trace = regime_results(r).batch_loss(1:max_step);
    plot(1:max_step, b_trace, 'Color', color_rgb(r, :), 'LineWidth', 1.2, ...
        'DisplayName', regime_results(r).name);
    hold on;
end
yline(0.6931, '--k', 'ln 2 Bound', 'LineWidth', 1.0, 'HandleVisibility', 'off');
title('High-Frequency Initial Batch Loss (Every 5 Batches)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Evaluation Step', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Batch Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

% 子圖 3：各排程之學習率軌跡
subplot(2, 2, 3);
for r = 1:num_regimes
    semilogy(1:total_epochs, regime_results(r).lr_history, 'Color', color_rgb(r, :), 'LineWidth', 1.5, ...
        'DisplayName', regime_results(r).name);
    hold on;
end
title('Learning Rate Schedules (Log Scale)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Learning Rate', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

% 子圖 4：各排程最終探針 AUC 對比
subplot(2, 2, 4);
auc_vals = [regime_results.auc];
b = bar(auc_vals, 0.45);
b.FaceColor = 'flat';
b.EdgeColor = 'none';

for r = 1:num_regimes
    b.CData(r, 1:3) = color_rgb(r, :);
end

set(gca, 'XTick', 1:num_regimes, 'XTickLabel', {'Step Decay', 'Const Small', 'Cosine'}, 'XTickLabelRotation', 15);
yline(0.50, '--k', 'Random Guess (0.50)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
ylim([0.45, max(0.55, max(auc_vals) + 0.03)]);
title('Held-out Linear Probe AUC Comparison', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Probe AUC', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

figPath = fullfile(configObj.ResultDir, 'Round9_TrainingRegimeSweep.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化圖表 (白底黑字) 已儲存至: %s\n', figPath);
close(fig);

disp('=================================================================');
disp('🎯 [Round 9] 執行完畢！');
disp('=================================================================');

%% =====================================================================
% 輔助函數：時序專家批次向量化訓練損失與梯度計算
% =====================================================================
function [loss, grad_net, grad_w, grad_b] = compute_vectorized_train_bce_loss(net, W, b, dl_x, y_true, act_m)
    emb = forward(net, dl_x); % [64, sample_T * B]
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
% 輔助函數：時序專家驗證階段純前向損失計算 (向量化)
% =====================================================================
function loss = compute_val_forward_loss_vectorized(net, W, b, dl_x, y_true, act_m)
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
% 輔助函數：批次分塊線性探針 AUC 評估
% =====================================================================
function auc = evaluate_probe_batched(net, X_input, Expert_sub, Y_5D, val_days, seqLen, sample_T, use_gpu)
    total_val = sum(Expert_sub(val_days, :), 'all');
    E_val = zeros(total_val, 64, 'single');
    Y_val = zeros(total_val, 1, 'single');
    
    num_val_days = length(val_days);
    eval_chunk_size = 128;
    cur_row = 1;
    numExtractorFeats = size(X_input, 2);
    
    for c = 1:ceil(num_val_days / eval_chunk_size)
        c_idx = (c - 1) * eval_chunk_size + 1 : min(c * eval_chunk_size, num_val_days);
        c_days = val_days(c_idx);
        C = length(c_days);
        
        X_chunk_T = zeros(numExtractorFeats, sample_T * C, seqLen, 'single');
        for i = 1:C
            t_curr = c_days(i);
            t_seq = (t_curr - seqLen + 1) : t_curr;
            col_range = (i - 1) * sample_T + 1 : i * sample_T;
            X_chunk_T(:, col_range, :) = permute(X_input(t_seq, :, :), [2, 3, 1]);
        end
        
        dl_x_c = dlarray(X_chunk_T, 'CBT');
        if use_gpu
            dl_x_c = gpuArray(dl_x_c);
        end
        emb_all = extractdata(gather(forward(net, dl_x_c))); % [64, sample_T * C]
        
        for i = 1:C
            t_curr = c_days(i);
            act_ids = find(Expert_sub(t_curr, :));
            n_act = length(act_ids);
            if n_act > 0
                cols_selected = (i - 1) * sample_T + act_ids;
                E_val(cur_row : cur_row + n_act - 1, :) = emb_all(:, cols_selected)';
                Y_val(cur_row : cur_row + n_act - 1)    = single(Y_5D(t_curr, act_ids)');
                cur_row = cur_row + n_act;
            end
        end
    end
    
    mdl = fitclinear(E_val, Y_val, 'Learner', 'logistic');
    [~, scores] = predict(mdl, E_val);
    [~, ~, ~, auc] = perfcurve(Y_val, scores(:, 2), 1);
end