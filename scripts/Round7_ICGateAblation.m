% =========================================================================
% 腳本：Round7_ICGateAblation.m (Round 7: IC 注意力閘門開關消融實驗)
% 升級：Phase 15.5 診斷方案 (★ 空間雙輸入解耦、工廠實例化適配、GPU 批次加速版)
% 職責：檢驗時變 IC 注意力閘門是否引入非平穩輸入分佈，致使神經網路難以優化
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧪 [Round 7] 啟動 IC 注意力閘門開關消融 (GPU 向量化批次加速版)');
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

for g = 1:num_groups
    grp = exp_groups{g};
    fprintf('\n▶ 正在訓練 %s ...\n', grp.name);
    
    % ★ 構造適配抽樣標的數之 Config 傳入工廠建立網路
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
    
    avgG_T = [];
    avgsqG_T = [];
    avgG_S = [];
    avgsqG_S = [];
    
    tr_loss_t = zeros(epochs, 1);
    val_loss_t = zeros(epochs, 1);
    tr_loss_s = zeros(epochs, 1);
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
                
                X_batch_T(:, col_range, :) = permute(X_input(t_seq, :, :), [2, 3, 1]);
                Y_batch_T(col_range)       = single(Y_5D(t_curr, :)');
                Act_batch_T(col_range)     = Expert_sub(t_curr, :)';
            end
            
            dl_x_t = dlarray(X_batch_T, 'CBT');
            if use_gpu
                dl_x_t = gpuArray(dl_x_t);
            end
            
            [loss_t_b, grad_net_t, grad_w_t, grad_b_t] = dlfeval(@compute_vectorized_bce_time, ...
                net_t, W_t, b_t, dl_x_t, Y_batch_T, Act_batch_T);
            
            [net_t, avgG_T, avgsqG_T] = adamupdate(net_t, grad_net_t, avgG_T, avgsqG_T, iter_idx, lr);
            W_t = W_t - lr * grad_w_t;
            b_t = b_t - lr * grad_b_t;
            ep_loss_t_sum = ep_loss_t_sum + double(extractdata(loss_t_b));
            
            % -------------------------------------------------------------
            % 5B. 空間專家：全 Batch 雙輸入 [FeatDim, B] 與 [AdjDim, B]
            % -------------------------------------------------------------
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
        % 5C. 驗證集批次評估 (純前向計算)
        % -------------------------------------------------------------
        val_sample = randsample(val_days, min(64, length(val_days)));
        V = length(val_sample);
        
        % 時序驗證批次
        X_val_T   = zeros(numExtractorFeats, sample_T * V, seqLen, 'single');
        Y_val_T   = zeros(sample_T * V, 1, 'single');
        Act_val_T = false(sample_T * V, 1);
        for i = 1:V
            t_curr = val_sample(i);
            t_seq = (t_curr - seqLen + 1) : t_curr;
            col_range = (i - 1) * sample_T + 1 : i * sample_T;
            X_val_T(:, col_range, :) = permute(X_input(t_seq, :, :), [2, 3, 1]);
            Y_val_T(col_range)       = single(Y_5D(t_curr, :)');
            Act_val_T(col_range)     = Expert_sub(t_curr, :)';
        end
        dl_x_val_t = dlarray(X_val_T, 'CBT');
        if use_gpu
            dl_x_val_t = gpuArray(dl_x_val_t);
        end
        val_loss_t(ep) = double(extractdata(compute_val_forward_time(net_t, W_t, b_t, dl_x_val_t, Y_val_T, Act_val_T)));
        
        % 空間驗證批次 (雙輸入)
        X_val_S_feat = zeros(feat_dim_space, V, 'single');
        X_val_S_adj  = zeros(adj_dim_space, V, 'single');
        Y_val_S      = single(Y_5D(val_sample, :)');
        Act_val_S    = Expert_sub(val_sample, :)';
        for i = 1:V
            t_curr = val_sample(i);
            x_c   = squeeze(X_input(t_curr, :, :));
            adj_c = squeeze(Adj_sub(:, :, t_curr));
            X_val_S_feat(:, i) = x_c(:);
            X_val_S_adj(:, i)  = adj_c(:);
        end
        dl_val_s_feat = dlarray(X_val_S_feat, 'CB');
        dl_val_s_adj  = dlarray(X_val_S_adj, 'CB');
        if use_gpu
            dl_val_s_feat = gpuArray(dl_val_s_feat);
            dl_val_s_adj  = gpuArray(dl_val_s_adj);
        end
        val_loss_space(ep) = double(extractdata(compute_val_forward_space(net_s, W_s, b_s, ...
            dl_val_s_feat, dl_val_s_adj, Y_val_S(:), Act_val_S(:), sample_T, V)));
        
        if mod(ep, 10) == 0 || ep == 1
            fprintf('  Ep %2d/%2d | Val Loss (Time: %.4f | Space: %.4f)\n', ...
                ep, epochs, val_loss_t(ep), val_loss_s(ep));
        end
    end
    t_duration = toc;
    fprintf('  ⚡ %s 訓練完成！耗時: %.2f 秒\n', grp.name, t_duration);
    
    % 儲存訓練歷史與計算線性探針 AUC
    history_loss(g).name    = grp.name;
    history_loss(g).train_t = tr_loss_t;
    history_loss(g).val_t   = val_loss_t;
    history_loss(g).train_s = tr_loss_s;
    history_loss(g).val_s   = val_loss_s;
    
    [auc_t, auc_s] = evaluate_linear_probe_batched(net_t, net_s, X_input, Adj_sub, Expert_sub, Y_5D, ...
        val_days, seqLen, sample_T, feat_dim_space, adj_dim_space, use_gpu);
    history_loss(g).auc_t = auc_t;
    history_loss(g).auc_s = auc_s;
end

%% 6. 判讀門檻檢定與結論報告
disp('========================================================================================');
disp('📊 【Round 7 IC 注意力閘門消融實驗結論報告】');
disp('========================================================================================');

val_loss_t_gate = history_loss(1).val_t(end);
val_loss_t_raw  = history_loss(2).val_t(end);
val_loss_s_gate = history_loss(1).val_s(end);
val_loss_s_raw  = history_loss(2).val_s(end);

delta_t = val_loss_t_raw - val_loss_t_gate; % Raw - Gated
delta_s = val_loss_s_raw - val_loss_s_gate;

fprintf('  > 時序專家 (Time Expert)  : With Gate Val Loss = %.4f | Raw Val Loss = %.4f (Delta = %+.4f)\n', ...
    val_loss_t_gate, val_loss_t_raw, delta_t);
fprintf('  > 空間專家 (Space Expert) : With Gate Val Loss = %.4f | Raw Val Loss = %.4f (Delta = %+.4f)\n', ...
    val_loss_s_gate, val_loss_s_raw, delta_s);
fprintf('  > 探針 AUC 對比 (Time)    : With Gate AUC = %.4f      | Raw AUC = %.4f\n', ...
    history_loss(1).auc_t, history_loss(2).auc_t);
fprintf('  > 探針 AUC 對比 (Space)   : With Gate AUC = %.4f      | Raw AUC = %.4f\n', ...
    history_loss(1).auc_s, history_loss(2).auc_s);
fprintf('----------------------------------------------------------------------------------------\n');

% 判定標準：若 Raw 組 Val Loss 比 Gated 組低超過 0.005，視為閘門有害
if delta_t < -0.005 || delta_s < -0.005
    fprintf('  ⚠️ 【GATE HARMFUL】IC 注意力閘門對神經網路優化產生顯著干擾 (Val Loss 上升 > 0.005)！\n');
    fprintf('     -> 建議後續移除時變 IC 閘門，或改採全樣本靜態注意力加權。\n');
elseif abs(delta_t) <= 0.003 && abs(delta_s) <= 0.003
    fprintf('  ✅ 【GATE NEUTRAL】IC 閘門影響落於雜訊範圍內 (|Delta Loss| <= 0.003)。\n');
    fprintf('     -> 排除「時變 IC 引入非平穩雜訊阻礙優化」假說，保留特徵注意力機制！\n');
else
    fprintf('  ℹ️ 【GATE BENEFICIAL / MINOR】IC 閘門略微改善或維持表徵品質。\n');
end
fprintf('========================================================================================\n\n');

%% 7. 產出標準學術白底黑字視覺化圖表
fig = figure('Name', 'Round 7 IC Gate Ablation Study', ...
    'Color', 'w', 'Position', [100, 100, 1100, 500], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

% 時序專家對比
subplot(1, 2, 1);
plot(1:epochs, history_loss(1).val_t, '-o', 'Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'Group A (With IC Gate)');
hold on;
plot(1:epochs, history_loss(2).val_t, '-s', 'Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'Group B (Raw 18D)');
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
title('Time Expert Validation BCE Loss', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Val BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

% 空間專家對比
subplot(1, 2, 2);
plot(1:epochs, history_loss(1).val_s, '-o', 'Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'Group A (With IC Gate)');
hold on;
plot(1:epochs, history_loss(2).val_s, '-s', 'Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'Group B (Raw 18D)');
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
title('Space Expert Validation BCE Loss', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Val BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

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