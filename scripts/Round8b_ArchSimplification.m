% =========================================================================
% 腳本：Round8b_ArchSimplification.m (Round 8b: 時序架構複雜度消融實驗)
% 升級：Phase 15.5 診斷方案 (★ 工廠實例化適配、全向量化 GPU 批次加速、記憶體極致釋放)
% 職責：檢驗移除 Self-Attention 是否影響表徵品質，驗證模型容量過剩假說並優化 VRAM 開銷
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧪 [Round 8b] 啟動時序架構複雜度消融 (GPU 向量化批次加速版)');
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
Prices_sub    = single(Prices_Active(valid_idx, sub_tickers));
Expert_sub    = logical(Expert_Active(valid_idx, sub_tickers));
X_sub_18D     = single(X_norm_3D(valid_idx, 1:numExtractorFeats, sub_tickers));
Dates_Active  = Dates_Active(valid_idx);
numDays       = length(Dates_Active);

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

%% 4. 構建兩組對照網路拓撲 (Complex vs Simplified)
disp('--- 步驟 4：初始化複雜度對照網路拓撲 ---');

% ★ Group A：原版 Transformer-LSTM (透過工廠實例化構建)
smokeConfig = configObj;
smokeConfig.NumTickers = sample_T;
extractorFactory = BuildDecoupledExtractors(smokeConfig);

if ismethod(extractorFactory, 'buildNetworks')
    [net_complex, ~] = extractorFactory.buildNetworks();
elseif ismethod(extractorFactory, 'build')
    [net_complex, ~] = extractorFactory.build();
else
    net_complex = extractorFactory.TimeNet;
end

% ★ Group B：簡化版純 LSTM (移除 Self-Attention)
layers_simple = [ ...
    sequenceInputLayer(numExtractorFeats, 'Name', 'in_time', 'Normalization', 'none')
    fullyConnectedLayer(64, 'Name', 'fc_in', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu_in')
    lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm_core')
    layerNormalizationLayer('Name', 'ln_post_lstm')
    dropoutLayer(0.2, 'Name', 'drop_lstm')
    fullyConnectedLayer(64, 'Name', 'fc_out', 'WeightsInitializer', 'he')
    layerNormalizationLayer('Name', 'ln_time_out') ...
];
lgraph_simple = layerGraph(layers_simple);
net_simple = dlnetwork(lgraph_simple);

use_gpu = canUseGPU();
if use_gpu
    gpu_dev = gpuDevice();
    fprintf('  🎮 偵測到硬體加速卡: 【%s】，啟用 GPU 向量化張量並行加速。\n', gpu_dev.Name);
else
    disp('  💻 使用 CPU 執行前向與反向運算。');
end

%% 5. 兩組對照實驗平行訓練 (30 Epochs Mini-Batch 向量化)
disp('--- 步驟 5：啟動 30 Epochs 向量化對照訓練 (Trans-LSTM vs Pure-LSTM) ---');

epochs = 30;
lr = 1e-3;
batch_size = 64;
batches_per_ep = floor(length(train_days) / batch_size);

models_under_test = { ...
    struct('name', 'Group A (Transformer-LSTM)', 'net', net_complex), ...
    struct('name', 'Group B (Simplified Pure-LSTM)', 'net', net_simple) ...
};

num_models = length(models_under_test);
history_results = struct();

for m = 1:num_models
    m_info = models_under_test{m};
    fprintf('\n▶ 正在訓練 %s ...\n', m_info.name);
    
    net_curr = m_info.net;
    W_aux = dlarray(randn(64, 1, 'single') * 0.01);
    b_aux = dlarray(zeros(1, 1, 'single'));
    avgG = [];
    avgsqG = [];
    
    tr_loss = zeros(epochs, 1);
    va_loss = zeros(epochs, 1);
    
    tic;
    for ep = 1:epochs
        shuffled_days = train_days(randperm(length(train_days)));
        ep_loss_sum = 0;
        
        for b = 1:batches_per_ep
            iter_idx = (ep - 1) * batches_per_ep + b;
            b_days = shuffled_days((b - 1) * batch_size + 1 : b * batch_size);
            B = length(b_days);
            
            % 全 Batch 向量化 3D 張量 [Feats, sample_T * B, SeqLen]
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
            [loss_b, grad_net, grad_w, grad_b] = dlfeval(@compute_vectorized_loss, ...
                net_curr, W_aux, b_aux, dl_x, Y_batch_T, Act_batch_T);
            
            [net_curr, avgG, avgsqG] = adamupdate(net_curr, grad_net, avgG, avgsqG, iter_idx, lr);
            W_aux = W_aux - lr * grad_w;
            b_aux = b_aux - lr * grad_b;
            
            ep_loss_sum = ep_loss_sum + double(extractdata(loss_b));
        end
        tr_loss(ep) = ep_loss_sum / batches_per_ep;
        
        % 驗證集批次評估 (純前向，無梯度圖開銷)
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
        va_loss(ep) = double(extractdata(compute_val_forward(net_curr, W_aux, b_aux, dl_x_val, Y_val_T, Act_val_T)));
        
        if mod(ep, 10) == 0 || ep == 1
            fprintf('  Ep %2d/%2d | Train BCE: %.4f | Val BCE: %.4f\n', ...
                ep, epochs, tr_loss(ep), va_loss(ep));
        end
    end
    t_duration = toc;
    fprintf('  ⚡ %s 訓練完成！耗時: %.2f 秒\n', m_info.name, t_duration);
    
    % 批次分塊線性探針 AUC 評估
    auc_probe = evaluate_probe_batched(net_curr, X_sub_18D, Expert_sub, Y_5D, val_days, seqLen, sample_T, use_gpu);
    
    history_results(m).name      = m_info.name;
    history_results(m).tr_loss   = tr_loss;
    history_results(m).va_loss   = va_loss;
    history_results(m).auc       = auc_probe;
    history_results(m).final_val = va_loss(end);
end

%% 6. 判讀門檻檢定與結論報告
disp('========================================================================================');
disp('📊 【Round 8b 時序架構複雜度消融實驗結論報告】');
disp('========================================================================================');

val_complex = history_results(1).final_val;
val_simple  = history_results(2).final_val;
delta_val   = val_simple - val_complex; % Simple - Complex

auc_complex = history_results(1).auc;
auc_simple  = history_results(2).auc;

fprintf('  > 原版架構 (Transformer-LSTM) : 最終 Val BCE = %.4f | 探針 AUC = %.4f\n', val_complex, auc_complex);
fprintf('  > 簡化架構 (Pure-LSTM)        : 最終 Val BCE = %.4f | 探針 AUC = %.4f\n', val_simple, auc_simple);
fprintf('  > 驗證損失差值 (Delta Loss)   : %+.4f (門檻: |Delta| <= 0.003)\n', delta_val);
fprintf('----------------------------------------------------------------------------------------\n');

% 判定標準：差距在 ±0.003 內視為持平
if abs(delta_val) <= 0.003
    fprintf('  ✅ 【SIMPLIFY RECOMMENDED】簡化架構與複雜架構表現持平 (|Delta Loss| <= 0.003)！\n');
    fprintf('     -> 證明「架構複雜度非瓶頸」，Self-Attention 在當前樣本規模下未帶來增量。\n');
    fprintf('     -> 建議後續全量訓練直接切換為純 LSTM+LayerNorm 輕量拓撲，節省 VRAM 並加速運算！\n');
elseif delta_val < -0.003
    fprintf('  ⭐ 【SIMPLIFY SUPERIOR】簡化架構顯著優於複雜架構 (Val Loss 降低 > 0.003)！\n');
    fprintf('     -> 複雜架構存在過度參數化 (Over-parameterization) 與樣本不足問題，強烈建議簡化。\n');
else
    fprintf('  ℹ️ 【COMPLEXITY BENEFICIAL】原版 Transformer-LSTM 顯著優於簡化版 (Val Loss 較低 > 0.003)。\n');
    fprintf('     -> 保留 Self-Attention 注意力機制，另行探尋資料增強或正則化手段。\n');
end
fprintf('========================================================================================\n\n');

%% 7. 產出標準學術白底黑字視覺化圖表
fig = figure('Name', 'Round 8b Architecture Simplification Study', ...
    'Color', 'w', 'Position', [100, 100, 1100, 480], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

subplot(1, 2, 1);
plot(1:epochs, history_results(1).tr_loss, '-o', 'Color', '#D95319', 'LineWidth', 1.4, 'DisplayName', 'Trans-LSTM (Train)');
hold on;
plot(1:epochs, history_results(1).va_loss, '--o', 'Color', '#D95319', 'LineWidth', 1.6, 'DisplayName', 'Trans-LSTM (Val)');
plot(1:epochs, history_results(2).tr_loss, '-s', 'Color', '#0072BD', 'LineWidth', 1.4, 'DisplayName', 'Pure-LSTM (Train)');
plot(1:epochs, history_results(2).va_loss, '--s', 'Color', '#0072BD', 'LineWidth', 1.6, 'DisplayName', 'Pure-LSTM (Val)');
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');

title('Training & Validation BCE Loss Convergence', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

subplot(1, 2, 2);
b = bar([auc_complex, auc_simple], 0.5);
b.FaceColor = 'flat';
b.EdgeColor = 'none';
b.CData(1, 1:3) = [0.8500, 0.3250, 0.0980];
b.CData(2, 1:3) = [0.0000, 0.4470, 0.7410];

set(gca, 'XTick', 1:2, 'XTickLabel', {'Trans-LSTM', 'Pure-LSTM'});
yline(0.50, '--k', 'Random Guess (0.50)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
ylim([0.45, max(0.55, max([auc_complex, auc_simple]) + 0.03)]);

title('Held-out Linear Probe AUC Comparison', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Probe AUC', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

figPath = fullfile(configObj.ResultDir, 'Round8b_ArchSimplification.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化圖表 (白底黑字) 已儲存至: %s\n', figPath);
close(fig);

disp('=================================================================');
disp('🎯 [Round 8b] 執行完畢！');
disp('=================================================================');

%% =====================================================================
% 輔助函數：時序專家批次向量化損失與梯度計算
% =====================================================================
function [loss, grad_net, grad_w, grad_b] = compute_vectorized_loss(net, W, b, dl_x, y_true, act_m)
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
% 輔助函數：時序專家驗證純前向計算
% =====================================================================
function loss = compute_val_forward(net, W, b, dl_x, y_true, act_m)
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