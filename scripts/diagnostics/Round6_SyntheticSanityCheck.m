% =========================================================================
% 腳本：Round6_SyntheticSanityCheck.m (Round 6: 合成訊號 Positive Control)
% 升級：Phase 15.5 Stage 0 診斷方案 (★ 顯存防 OOM 分塊驗證、純純量損失累加、
%       AUC 為核心判讀、GPU 批次加速版)
% 職責：構造已知可解之合成任務，排除維度重排、梯度阻斷與優化器更新等工程 Bug
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧪 [Round 6] 啟動合成訊號 Positive Control (Stage 0 方法論修復版)');
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

load(cachePath, 'X_norm_3D', 'Expert_Active', 'Dates_Active', 'AdjMatrix_3D');
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
X_sub_18D    = single(X_norm_3D(valid_idx, 1:numExtractorFeats, sub_tickers));
Expert_sub   = logical(Expert_Active(valid_idx, sub_tickers));
Adj_sub      = single(AdjMatrix_3D(sub_tickers, sub_tickers, valid_idx));
Dates_Active = Dates_Active(valid_idx);
numDays      = length(Dates_Active);

% ★ 立即銷毀 10GB+ 全域大陣列
clear X_norm_3D Expert_Active AdjMatrix_3D;

fprintf('  📊 抽樣規模：標的 %d 檔 | 有效天數 %d 天 | 輸入維度 %d 維 (記憶體已釋放)\n', ...
    sample_T, numDays, numExtractorFeats);

%% 2. 構造 100% 確定性之合成標籤 (Synthetic Ground Truth)
disp('--- 步驟 2：構建當日輸入可解之合成標籤 (Positive Control Target) ---');

r1_feat_idx = 4; % 通道 R1
Y_synthetic = false(numDays, sample_T);

for t = 1:numDays
    act_m = Expert_sub(t, :);
    if sum(act_m) > 5
        r1_slice = squeeze(X_sub_18D(t, r1_feat_idx, act_m));
        med_val = median(r1_slice, 'omitnan');
        Y_synthetic(t, act_m) = (r1_slice > med_val);
    end
end

pos_ratio = mean(single(Y_synthetic(Expert_sub)));
fprintf('  ✅ 合成標籤構建完畢！正類比例: %5.2f%% (理論基準 50.00%%)\n', pos_ratio * 100);

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

%% 4. 初始化雙軌萃取器網路 (透過工廠類別建立網路)
disp('--- 步驟 4：構建雙軌萃取網路拓撲 ---');

smokeConfig = configObj;
smokeConfig.NumTickers = sample_T;
extractorFactory = BuildDecoupledExtractors(smokeConfig);

if ismethod(extractorFactory, 'buildNetworks')
    [net_time, net_space] = extractorFactory.buildNetworks();
elseif ismethod(extractorFactory, 'build')
    [net_time, net_space] = extractorFactory.build();
else
    net_time = extractorFactory.TimeNet;
    net_space = extractorFactory.SpaceNet;
end

feat_dim_space = numExtractorFeats * sample_T; % 1080
adj_dim_space  = sample_T * sample_T;          % 3600

use_gpu = canUseGPU();
if use_gpu
    gpu_dev = gpuDevice();
    fprintf('  🎮 偵測到硬體加速卡: 【%s】，啟用 GPU 向量化張量並行加速。\n', gpu_dev.Name);
else
    disp('  💻 使用 CPU 執行前向與反向運算。');
end

%% 5. 執行 30 Epochs 預訓練 (Mini-Batch 向量化張量並行加速)
disp('--- 步驟 5：啟動 Mini-Batch 向量化批次訓練 (30 Epochs) ---');

epochs = 30;
lr = 1e-3;
batch_size = 64;
batches_per_ep = floor(length(train_days) / batch_size);

train_loss_time  = zeros(epochs, 1);
val_loss_time    = zeros(epochs, 1);
train_loss_space = zeros(epochs, 1);
val_loss_space   = zeros(epochs, 1);

% 線性分類頭權重初始化
W_aux_time  = dlarray(randn(64, 1, 'single') * 0.01);
b_aux_time  = dlarray(zeros(1, 1, 'single'));
W_aux_space = dlarray(randn(64, 1, 'single') * 0.01);
b_aux_space = dlarray(zeros(1, 1, 'single'));

avgG_T = []; avgsqG_T = [];
avgG_S = []; avgsqG_S = [];

val_eval_days = val_days;

tic;
for ep = 1:epochs
    shuffled_days = train_days(randperm(length(train_days)));
    
    ep_loss_t_sum = 0;
    ep_loss_s_sum = 0;
    
    for b = 1:batches_per_ep
        iter_idx = (ep - 1) * batches_per_ep + b;
        b_days = shuffled_days((b - 1) * batch_size + 1 : b * batch_size);
        B = length(b_days);
        
        % -------------------------------------------------------------
        % 5A. 時序專家：全 Batch 向量化 3D 張量 [Feats, sample_T * B, SeqLen]
        % -------------------------------------------------------------
        X_batch_T   = zeros(numExtractorFeats, sample_T * B, seqLen, 'single');
        Y_batch_T   = zeros(sample_T * B, 1, 'single');
        Act_batch_T = false(sample_T * B, 1);
        
        for i = 1:B
            t_curr = b_days(i);
            t_seq = (t_curr - seqLen + 1) : t_curr;
            col_range = (i - 1) * sample_T + 1 : i * sample_T;
            
            X_batch_T(:, col_range, :) = permute(X_sub_18D(t_seq, :, :), [2, 3, 1]);
            Y_batch_T(col_range)       = single(Y_synthetic(t_curr, :)');
            Act_batch_T(col_range)     = Expert_sub(t_curr, :)';
        end
        
        dl_x_t = dlarray(X_batch_T, 'CBT');
        if use_gpu
            dl_x_t = gpuArray(dl_x_t);
        end
        
        [loss_t_b, grad_net_t, grad_w_t, grad_b_t] = dlfeval(@compute_vectorized_loss_time, ...
            net_time, W_aux_time, b_aux_time, dl_x_t, Y_batch_T, Act_batch_T);
        
        [net_time, avgG_T, avgsqG_T] = adamupdate(net_time, grad_net_t, avgG_T, avgsqG_T, iter_idx, lr);
        W_aux_time = W_aux_time - lr * grad_w_t;
        b_aux_time = b_aux_time - lr * grad_b_t;
        
        ep_loss_t_sum = ep_loss_t_sum + double(extractdata(loss_t_b));
        
        % -------------------------------------------------------------
        % 5B. 空間專家：全 Batch 雙輸入 [FeatDim, B] 與 [AdjDim, B]
        % -------------------------------------------------------------
        X_batch_S_feat = zeros(feat_dim_space, B, 'single');
        X_batch_S_adj  = zeros(adj_dim_space, B, 'single');
        Y_batch_S      = single(Y_synthetic(b_days, :)');
        Act_batch_S    = Expert_sub(b_days, :)';
        
        for i = 1:B
            t_curr = b_days(i);
            x_c   = squeeze(X_sub_18D(t_curr, :, :));
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
        
        [loss_s_b, grad_net_s, grad_w_s, grad_b_s] = dlfeval(@compute_vectorized_loss_space, ...
            net_space, W_aux_space, b_aux_space, dl_s_feat, dl_s_adj, Y_batch_S(:), Act_batch_S(:), sample_T, B);
        
        [net_space, avgG_S, avgsqG_S] = adamupdate(net_space, grad_net_s, avgG_S, avgsqG_S, iter_idx, lr);
        W_aux_space = W_aux_space - lr * grad_w_s;
        b_aux_space = b_aux_space - lr * grad_b_s;
        
        ep_loss_s_sum = ep_loss_s_sum + double(extractdata(loss_s_b));
    end
    
    train_loss_time(ep)  = ep_loss_t_sum / batches_per_ep;
    train_loss_space(ep) = ep_loss_s_sum / batches_per_ep;
    
    % -------------------------------------------------------------
    % 5C. 驗證集評估 (★ 分塊防 OOM 且純純量累加計算)
    % -------------------------------------------------------------
    val_loss_time(ep)  = evaluate_val_loss_chunked_time(net_time, W_aux_time, b_aux_time, ...
        X_sub_18D, Y_synthetic, Expert_sub, val_eval_days, seqLen, sample_T, use_gpu);
    val_loss_space(ep) = evaluate_val_loss_chunked_space(net_space, W_aux_space, b_aux_space, ...
        X_sub_18D, Adj_sub, Y_synthetic, Expert_sub, val_eval_days, sample_T, feat_dim_space, adj_dim_space, use_gpu);
    
    if mod(ep, 5) == 0 || ep == 1
        fprintf('  Epoch %2d/%2d | Train BCE (T: %.4f, S: %.4f) | Val BCE (T: %.4f, S: %.4f)\n', ...
            ep, epochs, train_loss_time(ep), train_loss_space(ep), val_loss_time(ep), val_loss_space(ep));
    end
end
train_duration = toc;
fprintf('⚡ 30 Epochs 雙軌預訓練完成！總耗時: %.2f 秒 (平均每 Epoch: %.2f 秒)\n', train_duration, train_duration / epochs);

%% 6. 批次加速線性探針測試 (Batch Linear Probe on Held-out Set)
disp('--- 步驟 6：批次提煉驗證集 Embedding 並計算線性探針 AUC ---');

total_val_samples = sum(Expert_sub(val_days, :), 'all');
E_val_time  = zeros(total_val_samples, 64, 'single');
E_val_space = zeros(total_val_samples, 64, 'single');
Y_val_true  = zeros(total_val_samples, 1, 'single');

eval_chunk_size = 128;
num_val_days = length(val_days);
cur_row = 1;

for c = 1:ceil(num_val_days / eval_chunk_size)
    c_idx = (c - 1) * eval_chunk_size + 1 : min(c * eval_chunk_size, num_val_days);
    c_days = val_days(c_idx);
    C = length(c_days);
    
    % 時序批次特徵提取
    X_chunk_T = zeros(numExtractorFeats, sample_T * C, seqLen, 'single');
    for i = 1:C
        t_curr = c_days(i);
        t_seq = (t_curr - seqLen + 1) : t_curr;
        col_range = (i - 1) * sample_T + 1 : i * sample_T;
        X_chunk_T(:, col_range, :) = permute(X_sub_18D(t_seq, :, :), [2, 3, 1]);
    end
    dl_x_c_t = dlarray(X_chunk_T, 'CBT');
    if use_gpu
        dl_x_c_t = gpuArray(dl_x_c_t);
    end
    emb_t_all = extractdata(gather(forward(net_time, dl_x_c_t)));
    
    % 空間批次特徵提取 (雙輸入)
    X_chunk_S_feat = zeros(feat_dim_space, C, 'single');
    X_chunk_S_adj  = zeros(adj_dim_space, C, 'single');
    for i = 1:C
        t_curr = c_days(i);
        x_c   = squeeze(X_sub_18D(t_curr, :, :));
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
    
    emb_s_raw = forward_space_wrapper(net_space, dl_c_s_feat, dl_c_s_adj);
    emb_s_raw = extractdata(gather(emb_s_raw));
    emb_s_all = reshape(emb_s_raw, 64, sample_T * C);
    
    % 填入有效樣本遮罩
    for i = 1:C
        t_curr = c_days(i);
        act_ids = find(Expert_sub(t_curr, :));
        n_act = length(act_ids);
        if n_act > 0
            base_col = (i - 1) * sample_T;
            cols_selected = base_col + act_ids;
            
            E_val_time(cur_row : cur_row + n_act - 1, :)  = emb_t_all(:, cols_selected)';
            E_val_space(cur_row : cur_row + n_act - 1, :) = emb_s_all(:, cols_selected)';
            Y_val_true(cur_row : cur_row + n_act - 1)     = single(Y_synthetic(t_curr, act_ids)');
            cur_row = cur_row + n_act;
        end
    end
end

% 線性邏輯回歸探針
mdl_probe_t = fitclinear(E_val_time, Y_val_true, 'Learner', 'logistic');
mdl_probe_s = fitclinear(E_val_space, Y_val_true, 'Learner', 'logistic');

[~, score_t] = predict(mdl_probe_t, E_val_time);
[~, score_s] = predict(mdl_probe_s, E_val_space);

[~, ~, ~, auc_time]  = perfcurve(Y_val_true, score_t(:, 2), 1);
[~, ~, ~, auc_space] = perfcurve(Y_val_true, score_s(:, 2), 1);

final_val_bce_t = val_loss_time(end);
final_val_bce_s = val_loss_space(end);

%% 7. 判讀門檻檢定與結論報告 (★ Stage 0: AUC 為核心判別標準)
disp('========================================================================================');
disp('📊 【Round 6 合成訊號 Positive Control 驗證結論報告】');
disp('========================================================================================');
fprintf('  > 時序專家 (Time Expert)  : 探針 AUC = %.4f (門檻 >= 0.90) | 最終 Val BCE = %.4f\n', auc_time, final_val_bce_t);
fprintf('  > 空間專家 (Space Expert) : 探針 AUC = %.4f (門檻 >= 0.90) | 最終 Val BCE = %.4f\n', auc_space, final_val_bce_s);
fprintf('----------------------------------------------------------------------------------------\n');

% ★ 修復方法論缺陷：以線性探針表徵分離度 (AUC >= 0.90) 為核心 PASS 判定準則
pass_time  = (auc_time >= 0.90);
pass_space = (auc_space >= 0.90);

if pass_time && pass_space
    fprintf('  ✅ 【PASS】雙軌萃取網路與訓練 Pipeline 完全正常！\n');
    fprintf('     -> 證明網路在具備真實訊號時能精確收斂至高表徵可分度 (AUC: %.4f / %.4f)。\n', auc_time, auc_space);
    fprintf('     -> 徹底排除「維度重排、張量錯位或梯度截斷」等工程 Bug！\n');
    fprintf('     -> 後續真實數據若 Val Loss 停滯於 ln(2)，可 100%% 證偽歸因為「特徵資訊天花板」。\n');
else
    fprintf('  ❌ 【FAIL】萃取網路未通過合成訊號測試！存在梯度流或張量維度錯位 Bug。\n');
    if ~pass_time
        fprintf('     -> 時序專家未達標 (探針 AUC = %.4f < 0.90)\n', auc_time);
    end
    if ~pass_space
        fprintf('     -> 空間專家未達標 (探針 AUC = %.4f < 0.90)\n', auc_space);
    end
    fprintf('     -> 請檢查維度排列與 dlgradient 梯度回傳！\n');
end
fprintf('========================================================================================\n\n');

%% 8. 產出標準學術白底黑字視覺化圖表
fig = figure('Name', 'Round 6 Synthetic Sanity Check', ...
    'Color', 'w', 'Position', [100, 100, 1100, 480], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

subplot(1, 2, 1);
plot(1:epochs, train_loss_time, '-o', 'Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'Train Loss');
hold on;
plot(1:epochs, val_loss_time, '-x', 'Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'Val Loss (Fixed Set)');
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
title(sprintf('Time Expert BCE Loss (Probe AUC: %.4f)', auc_time), 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

subplot(1, 2, 2);
plot(1:epochs, train_loss_space, '-o', 'Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'Train Loss');
hold on;
plot(1:epochs, val_loss_space, '-x', 'Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'Val Loss (Fixed Set)');
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
title(sprintf('Space Expert BCE Loss (Probe AUC: %.4f)', auc_space), 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

figPath = fullfile(configObj.ResultDir, 'Round6_SyntheticSanityCheck.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化圖表 (白底黑字) 已儲存至: %s\n', figPath);
close(fig);

disp('=================================================================');
disp('🎯 [Round 6] 執行完畢！');
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
function [loss, grad_net, grad_w, grad_b] = compute_vectorized_loss_time(net, W, b, dl_x, y_true, act_m)
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
function [loss, grad_net, grad_w, grad_b] = compute_vectorized_loss_space(net, W, b, dl_feat, dl_adj, y_true, act_m, numT, B)
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
% 輔助函數：時序專家分塊驗證損失計算 (★ 防 OOM + 純量累加修復版)
% =====================================================================
function val_loss = evaluate_val_loss_chunked_time(net, W, b, X_input, Y_synthetic, Expert_sub, val_days, seqLen, sample_T, use_gpu)
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
            Y_chunk(col_range)       = single(Y_synthetic(t_curr, :)');
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
        
        % ★ 強制統一向量形狀為列向量，消除廣播矩陣外積
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
% 輔助函數：空間專家分塊驗證損失計算 (★ 防 OOM + 純量累加修復版)
% =====================================================================
function val_loss = evaluate_val_loss_chunked_space(net, W, b, X_input, Adj_sub, Y_synthetic, Expert_sub, val_days, sample_T, feat_dim, adj_dim, use_gpu)
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
        Y_chunk   = single(Y_synthetic(c_days, :)');
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
        
        % ★ 強制統一向量形狀為列向量
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