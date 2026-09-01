% =========================================================================
% 腳本：Round8a_OverfitCapacityCheck.m (Round 8a: 小樣本記憶力過擬合測試)
% 升級：Phase 15.5 診斷方案 (★ 全靜態張量 GPU 單步狂飆 + 空間雙輸入解耦版)
% 職責：驗證雙軌 DL 萃取網路在真實數據分佈與標籤下是否具備基本的記憶容量與梯度可優化性
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧠 [Round 8a] 啟動小樣本記憶力過擬合測試 (全靜態張量 GPU 向量化版)');
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
        error('❌ 找不到專案根目錄 (包含 configs/)！');
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

%% 1. 載入特徵快取、子集抽樣與即刻記憶體釋放 (Memory Downcasting)
disp('--- 步驟 1：載入 3D 特徵快取並執行型態壓縮 ---');
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

% 資料型態下轉 (single / logical)
Prices_sub   = single(Prices_Active(valid_idx, sub_tickers));
Expert_sub   = logical(Expert_Active(valid_idx, sub_tickers));
X_sub_18D    = single(X_norm_3D(valid_idx, 1:numExtractorFeats, sub_tickers));
Adj_sub      = single(AdjMatrix_3D(sub_tickers, sub_tickers, valid_idx));
Dates_Active = Dates_Active(valid_idx);
numDays      = length(Dates_Active);

% 立即銷毀全域大陣列
clear X_norm_3D Prices_Active Expert_Active AdjMatrix_3D;

fprintf('  📊 抽樣規模：標的 %d 檔 | 序列長度 %d | 輸入維度 %d 維 (記憶體已釋放)\n', ...
    sample_T, seqLen, numExtractorFeats);

%% 2. 構建真實 5 日 Beat-the-Median 目標標籤
disp('--- 步驟 2：構建真實 5 日 Beat-the-Median 目標標籤 ---');
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

%% 3. 強制鎖定極小樣本訓練池並預先載入 GPU 靜態張量 (N = 25 天)
disp('--- 步驟 3：鎖定 N = 25 天小樣本並預先載入 GPU 靜態張量 ---');
Train_Start_Date = datetime('2010-01-01', 'TimeZone', 'UTC');
idx_start = find(Dates_Active >= Train_Start_Date, 1);
if isempty(idx_start)
    idx_start = 252;
end

num_overfit_days = 25;
train_days_overfit = round(linspace(idx_start, idx_start + 300, num_overfit_days));
N = num_overfit_days;
total_flat_samples = sample_T * N;

use_gpu = canUseGPU();
if use_gpu
    gpu_dev = gpuDevice();
    fprintf('  🎮 偵測到硬體加速卡: 【%s】，啟用 GPU 全靜態顯存加速。\n', gpu_dev.Name);
else
    disp('  💻 使用 CPU 執行訓練。');
end

% 預先構建時序專家全量靜態張量 [Feats, sample_T * N, SeqLen]
X_static_T = zeros(numExtractorFeats, total_flat_samples, seqLen, 'single');
Y_static   = zeros(total_flat_samples, 1, 'single');
Act_static = false(total_flat_samples, 1);

for i = 1:N
    t_curr = train_days_overfit(i);
    t_seq = (t_curr - seqLen + 1) : t_curr;
    col_range = (i - 1) * sample_T + 1 : i * sample_T;
    
    X_static_T(:, col_range, :) = permute(X_sub_18D(t_seq, :, :), [2, 3, 1]);
    Y_static(col_range)         = single(Y_5D(t_curr, :)');
    Act_static(col_range)       = Expert_sub(t_curr, :)';
end

dl_x_static_time = dlarray(X_static_T, 'CBT');
if use_gpu
    dl_x_static_time = gpuArray(dl_x_static_time);
end

% 預先構建空間專家雙輸入全量靜態張量
feat_dim_space = numExtractorFeats * sample_T; % 1080
adj_dim_space  = sample_T * sample_T;          % 3600

X_static_S_feat = zeros(feat_dim_space, N, 'single');
X_static_S_adj  = zeros(adj_dim_space, N, 'single');

for i = 1:N
    t_curr = train_days_overfit(i);
    x_c   = squeeze(X_sub_18D(t_curr, :, :));
    adj_c = squeeze(Adj_sub(:, :, t_curr));
    X_static_S_feat(:, i) = x_c(:);
    X_static_S_adj(:, i)  = adj_c(:);
end

dl_static_s_feat = dlarray(X_static_S_feat, 'CB');
dl_static_s_adj  = dlarray(X_static_S_adj, 'CB');
if use_gpu
    dl_static_s_feat = gpuArray(dl_static_s_feat);
    dl_static_s_adj  = gpuArray(dl_static_s_adj);
end

clear X_sub_18D Adj_sub X_static_T X_static_S_feat X_static_S_adj;

fprintf('  🔒 25 天靜態資料已常駐顯存 (總觀測數: %d 筆)。\n', total_flat_samples);

%% 4. 初始化零正則化網路拓撲 (Dropout = 0)
disp('--- 步驟 4：構建零正則化雙軌網路 (Dropout = 0) ---');

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

% 移除時序專家 Dropout
lgraph_time = layerGraph(net_time);
layers_t = lgraph_time.Layers;
for i = 1:length(layers_t)
    if isa(layers_t(i), 'nnet.cnn.layer.DropoutLayer')
        lgraph_time = replaceLayer(lgraph_time, layers_t(i).Name, dropoutLayer(0.0, 'Name', layers_t(i).Name));
    end
end
net_time = dlnetwork(lgraph_time);

% 移除空間專家 Dropout
lgraph_space = layerGraph(net_space);
layers_s = lgraph_space.Layers;
for i = 1:length(layers_s)
    if isa(layers_s(i), 'nnet.cnn.layer.DropoutLayer')
        lgraph_space = replaceLayer(lgraph_space, layers_s(i).Name, dropoutLayer(0.0, 'Name', layers_s(i).Name));
    end
end
net_space = dlnetwork(lgraph_space);

%% 5. 執行 150 Epochs 全向量化過擬合訓練
disp('--- 步驟 5：啟動 150 Epochs 全向量化記憶力訓練 ---');

epochs = 150;
lr = 2e-3;

train_loss_time  = zeros(epochs, 1);
train_loss_space = zeros(epochs, 1);

W_aux_time  = dlarray(randn(64, 1, 'single') * 0.01);
b_aux_time  = dlarray(zeros(1, 1, 'single'));
W_aux_space = dlarray(randn(64, 1, 'single') * 0.01);
b_aux_space = dlarray(zeros(1, 1, 'single'));

avgG_T = [];
avgsqG_T = [];
avgG_S = [];
avgsqG_S = [];

tic;
for ep = 1:epochs
    % 時序專家：單步完成全局前向與反向傳播
    [loss_t, grad_net_t, grad_w_t, grad_b_t] = dlfeval(@compute_pure_bce_time_vectorized, ...
        net_time, W_aux_time, b_aux_time, dl_x_static_time, Y_static, Act_static);
    
    [net_time, avgG_T, avgsqG_T] = adamupdate(net_time, grad_net_t, avgG_T, avgsqG_T, ep, lr);
    W_aux_time = W_aux_time - lr * grad_w_t;
    b_aux_time = b_aux_time - lr * grad_b_t;
    train_loss_time(ep) = double(extractdata(loss_t));
    
    % 空間專家：單步雙輸入全局前向與反向傳播
    [loss_s, grad_net_s, grad_w_s, grad_b_s] = dlfeval(@compute_pure_bce_space_vectorized, ...
        net_space, W_aux_space, b_aux_space, dl_static_s_feat, dl_static_s_adj, Y_static, Act_static, sample_T, N);
    
    [net_space, avgG_S, avgsqG_S] = adamupdate(net_space, grad_net_s, avgG_S, avgsqG_S, ep, lr);
    W_aux_space = W_aux_space - lr * grad_w_s;
    b_aux_space = b_aux_space - lr * grad_b_s;
    train_loss_space(ep) = double(extractdata(loss_s));
    
    if mod(ep, 25) == 0 || ep == 1
        fprintf('  Epoch %3d/%3d | Train BCE (Time: %.5f | Space: %.5f)\n', ...
            ep, epochs, train_loss_time(ep), train_loss_space(ep));
    end
end
elapsed_sec = toc;
fprintf('⚡ 150 Epochs 雙軌過擬合訓練完成！總耗時: %.2f 秒 (平均每 Epoch: %.3f 秒)\n', elapsed_sec, elapsed_sec / epochs);

final_bce_time  = train_loss_time(end);
final_bce_space = train_loss_space(end);

%% 6. 判讀門檻檢定與結論報告
disp('========================================================================================');
disp('📊 【Round 8a 小樣本記憶力過擬合測試結論報告】');
disp('========================================================================================');
fprintf('  > 時序專家 (Time Expert)  : 最終 Train BCE = %.5f\n', final_bce_time);
fprintf('  > 空間專家 (Space Expert) : 最終 Train BCE = %.5f\n', final_bce_space);
fprintf('----------------------------------------------------------------------------------------\n');

pass_time  = (final_bce_time < 0.05);
pass_space = (final_bce_space < 0.05);

if pass_time && pass_space
    fprintf('  ✅ 【PASS】雙軌萃取網路通過記憶力容量測試！\n');
    fprintf('     -> 網路在極小樣本下能迅速過擬合至近乎 0 誤差 (BCE < 0.05)。\n');
    fprintf('     -> 徹底排除「網路容量不足」或「梯度被飽和函數鎖死」之假設！\n');
else
    fprintf('  ❌ 【FAIL】網路連極小樣本都無法記憶！\n');
    if ~pass_time
        fprintf('     -> 時序專家未達標 (最終 BCE = %.5f >= 0.05)\n', final_bce_time);
    end
    if ~pass_space
        fprintf('     -> 空間專家未達標 (最終 BCE = %.5f >= 0.05)\n', final_bce_space);
    end
    fprintf('     -> 存在梯度截斷、學習率過小或神經網路層輸出飽和問題！\n');
end
fprintf('========================================================================================\n\n');

%% 7. 產出標準學術白底黑字視覺化圖表
fig = figure('Name', 'Round 8a Overfit Capacity Check', ...
    'Color', 'w', 'Position', [100, 100, 1100, 480], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

subplot(1, 2, 1);
plot(1:epochs, train_loss_time, '-o', 'Color', '#D95319', 'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'Train BCE');
hold on;
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
yline(0.0500, ':r', 'Overfit Threshold (0.05)', 'LineWidth', 1.2, 'DisplayName', 'Target (< 0.05)');
title('Time Expert Overfit Training Loss', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

subplot(1, 2, 2);
plot(1:epochs, train_loss_space, '-o', 'Color', '#0072BD', 'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'Train BCE');
hold on;
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
yline(0.0500, ':r', 'Overfit Threshold (0.05)', 'LineWidth', 1.2, 'DisplayName', 'Target (< 0.05)');
title('Space Expert Overfit Training Loss', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

figPath = fullfile(configObj.ResultDir, 'Round8a_OverfitCapacityCheck.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化圖表 (白底黑字) 已儲存至: %s\n', figPath);
close(fig);

disp('=================================================================');
disp('🎯 [Round 8a] 執行完畢！');
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
% 輔助函數：時序專家純 BCE 全向量化損失與梯度計算
% =====================================================================
function [loss, grad_net, grad_w, grad_b] = compute_pure_bce_time_vectorized(net, W, b, dl_x, y_true, act_m)
    emb = forward(net, dl_x);
    emb = stripdims(emb);
    logits = W' * emb + b;
    probs = sigmoid(logits);
    
    y_t = y_true(act_m);
    y_t = y_t(:)';
    p_t = probs(act_m);
    p_t = p_t(:)';
    
    loss = -mean(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8));
    [grad_net, grad_w, grad_b] = dlgradient(loss, net.Learnables, W, b);
end

%% =====================================================================
% 輔助函數：空間專家純 BCE 全向量化損失與梯度計算 (雙輸入)
% =====================================================================
function [loss, grad_net, grad_w, grad_b] = compute_pure_bce_space_vectorized(net, W, b, dl_feat, dl_adj, y_true, act_m, numT, N)
    emb_raw = forward_space_wrapper(net, dl_feat, dl_adj);
    emb_raw = stripdims(emb_raw);
    emb = reshape(emb_raw, 64, numT * N);
    
    logits = W' * emb + b;
    probs = sigmoid(logits);
    
    y_t = y_true(act_m);
    y_t = y_t(:)';
    p_t = probs(act_m);
    p_t = p_t(:)';
    
    loss = -mean(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8));
    [grad_net, grad_w, grad_b] = dlgradient(loss, net.Learnables, W, b);
end