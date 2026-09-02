% =========================================================================
% 腳本：Round8a_OverfitCapacityCheck.m (Round 8a-v2: 小樣本記憶力與 DyGAT 隔離診斷)
% 升級：Phase 15.5 Stage 1 診斷方案 (★ 三軌對照: Time vs Pure GCN vs Full DyGAT、
%       32 天分塊防 OOM 驗證、純量損失累加、dlfeval 外梯度範數提取、無 Linter 警告)
% 職責：隔離診斷空間專家學習瓶頸，嚴格判定係動態圖注意力 (DyGAT) 實作瑕疵、
%       底層圖卷積重組問題，或模型容量過剩與梯度飽和
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧠 [Round 8a-v2] 啟動小樣本記憶力與 DyGAT 架構隔離診斷 (Stage 0/1 防 OOM 版)');
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

% 構建 Held-out 驗證區間 (用於 32 天分塊防 OOM 評估)
idx_OOS_start = find(Dates_Active >= datetime('2022-01-01', 'TimeZone', 'UTC'), 1);
val_eval_days = (idx_OOS_start + 20) : (numDays - 1);

use_gpu = canUseGPU();
if use_gpu
    gpu_dev = gpuDevice();
    fprintf('  🎮 偵測到硬體加速卡: 【%s】，啟用 GPU 全靜態顯存加速。\n', gpu_dev.Name);
else
    disp('  💻 使用 CPU 執行訓練。');
end

% -----------------------------------------------------------------
% 3A. 時序專家靜態張量 [Feats, sample_T * N, SeqLen]
% -----------------------------------------------------------------
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

% -----------------------------------------------------------------
% 3B. 空間專家靜態張量 (雙輸入)
% -----------------------------------------------------------------
feat_dim_space = numExtractorFeats * sample_T; % 1080
adj_dim_space  = sample_T * sample_T;          % 3600

X_static_S_feat = zeros(feat_dim_space, N, 'single');
X_static_S_adj  = zeros(adj_dim_space, N, 'single');
X_static_3D_raw = zeros(numExtractorFeats, sample_T, N, 'single');
Adj_static_3D   = zeros(sample_T, sample_T, N, 'single');

for i = 1:N
    t_curr = train_days_overfit(i);
    x_c   = squeeze(X_sub_18D(t_curr, :, :));
    adj_c = squeeze(Adj_sub(:, :, t_curr));
    
    X_static_S_feat(:, i) = x_c(:);
    X_static_S_adj(:, i)  = adj_c(:);
    X_static_3D_raw(:, :, i) = x_c;
    Adj_static_3D(:, :, i)   = adj_c;
end

dl_static_s_feat = dlarray(X_static_S_feat, 'CB');
dl_static_s_adj  = dlarray(X_static_S_adj, 'CB');
dl_static_3D_x   = dlarray(X_static_3D_raw);
dl_static_3D_adj = dlarray(Adj_static_3D);

if use_gpu
    dl_static_s_feat = gpuArray(dl_static_s_feat);
    dl_static_s_adj  = gpuArray(dl_static_s_adj);
    dl_static_3D_x   = gpuArray(dl_static_3D_x);
    dl_static_3D_adj = gpuArray(dl_static_3D_adj);
end

clear X_static_T X_static_S_feat X_static_S_adj X_static_3D_raw Adj_static_3D;
fprintf('  🔒 25 天靜態資料已常駐顯存 (總觀測數: %d 筆)。\n', total_flat_samples);

%% 4. 初始化三組零正則化網路 (Time vs Pure GCN vs Full DyGAT)
disp('--- 步驟 4：構建零正則化三軌網路 (Dropout = 0) 與連接埠斷言 ---');

smokeConfig = configObj;
smokeConfig.NumTickers = sample_T;
extractorFactory = BuildDecoupledExtractors(smokeConfig);

if ismethod(extractorFactory, 'buildNetworks')
    [net_time, net_dygat] = extractorFactory.buildNetworks();
elseif ismethod(extractorFactory, 'build')
    [net_time, net_dygat] = extractorFactory.build();
else
    net_time  = extractorFactory.TimeNet;
    net_dygat = extractorFactory.SpaceNet;
end

% -----------------------------------------------------------------
% 4A. 連接埠斷言 (Port Assertion)
% -----------------------------------------------------------------
input_names = net_dygat.InputNames;
fprintf('  🔍 空間專家 (DyGAT) 連接埠探針: [%s]\n', strjoin(input_names, ', '));
assert(length(input_names) >= 2, '❌ 空間專家輸入埠數量不足 2 個！');

% -----------------------------------------------------------------
% 4B. 零正則化改造 (Dropout = 0)
% -----------------------------------------------------------------
lgraph_time = layerGraph(net_time);
layers_t = lgraph_time.Layers;
for i = 1:length(layers_t)
    if isa(layers_t(i), 'nnet.cnn.layer.DropoutLayer')
        lgraph_time = replaceLayer(lgraph_time, layers_t(i).Name, dropoutLayer(0.0, 'Name', layers_t(i).Name));
    end
end
net_time = dlnetwork(lgraph_time);

lgraph_dygat = layerGraph(net_dygat);
layers_s = lgraph_dygat.Layers;
for i = 1:length(layers_s)
    if isa(layers_s(i), 'nnet.cnn.layer.DropoutLayer')
        lgraph_dygat = replaceLayer(lgraph_dygat, layers_s(i).Name, dropoutLayer(0.0, 'Name', layers_s(i).Name));
    end
end
net_dygat = dlnetwork(lgraph_dygat);

% -----------------------------------------------------------------
% 4C. 初始化純 GCN 對照參數 (無 Attention，僅靜態鄰接矩陣聚合)
% -----------------------------------------------------------------
params_pure_gcn = struct();
params_pure_gcn.W1 = dlarray(randn(64, numExtractorFeats, 'single') * sqrt(2 / numExtractorFeats));
params_pure_gcn.b1 = dlarray(zeros(64, 1, 'single'));
params_pure_gcn.W2 = dlarray(randn(64, 64, 'single') * sqrt(2 / 64));
params_pure_gcn.b2 = dlarray(zeros(64, 1, 'single'));
params_pure_gcn.W_aux = dlarray(randn(64, 1, 'single') * 0.01);
params_pure_gcn.b_aux = dlarray(zeros(1, 1, 'single'));

if use_gpu
    fn = fieldnames(params_pure_gcn);
    for k = 1:length(fn)
        params_pure_gcn.(fn{k}) = gpuArray(params_pure_gcn.(fn{k}));
    end
end

%% 5. 執行 150 Epochs 全向量化過擬合訓練與梯度範數監控
disp('--- 步驟 5：啟動 150 Epochs 全向量化記憶力訓練與逐層梯度監控 ---');

epochs = 150;
lr = 2e-3;

train_loss_time     = zeros(epochs, 1);
train_loss_pure_gcn = zeros(epochs, 1);
train_loss_dygat    = zeros(epochs, 1);

val_loss_time_track  = zeros(epochs, 1);
val_loss_dygat_track = zeros(epochs, 1);

% 梯度範數追蹤器 [Weights, W_attn_1, W_attn_2, MixLogit]
grad_norms_dygat = zeros(epochs, 4);

W_aux_time  = dlarray(randn(64, 1, 'single') * 0.01);
b_aux_time  = dlarray(zeros(1, 1, 'single'));
W_aux_dygat = dlarray(randn(64, 1, 'single') * 0.01);
b_aux_dygat = dlarray(zeros(1, 1, 'single'));

if use_gpu
    W_aux_time  = gpuArray(W_aux_time);
    b_aux_time  = gpuArray(b_aux_time);
    W_aux_dygat = gpuArray(W_aux_dygat);
    b_aux_dygat = gpuArray(b_aux_dygat);
end

avgG_T = []; avgsqG_T = [];
avgG_G = []; avgsqG_G = [];
avgG_S = []; avgsqG_S = [];

tic;
for ep = 1:epochs
    % -------------------------------------------------------------
    % 5A. 時序專家 (Time Expert)
    % -------------------------------------------------------------
    [loss_t, grad_net_t, grad_w_t, grad_b_t] = dlfeval(@compute_pure_bce_time_vectorized, ...
        net_time, W_aux_time, b_aux_time, dl_x_static_time, Y_static, Act_static);
    
    [net_time, avgG_T, avgsqG_T] = adamupdate(net_time, grad_net_t, avgG_T, avgsqG_T, ep, lr);
    W_aux_time = W_aux_time - lr * grad_w_t;
    b_aux_time = b_aux_time - lr * grad_b_t;
    train_loss_time(ep) = double(extractdata(loss_t));
    
    % -------------------------------------------------------------
    % 5B. 純 GCN 對照組 (Pure GCN - No Attention)
    % -------------------------------------------------------------
    [loss_g, grad_gcn] = dlfeval(@compute_pure_bce_pure_gcn_vectorized, ...
        params_pure_gcn, dl_static_3D_x, dl_static_3D_adj, Y_static, Act_static, sample_T, N);
    
    [params_pure_gcn, avgG_G, avgsqG_G] = adamupdate(params_pure_gcn, grad_gcn, avgG_G, avgsqG_G, ep, lr);
    train_loss_pure_gcn(ep) = double(extractdata(loss_g));
    
    % -------------------------------------------------------------
    % 5C. 完整 DyGAT (Full DyGAT with Dynamic Attention)
    % -------------------------------------------------------------
    [loss_s, grad_net_s, grad_w_s, grad_b_s] = dlfeval(@compute_pure_bce_dygat_vectorized, ...
        net_dygat, W_aux_dygat, b_aux_dygat, dl_static_s_feat, dl_static_s_adj, Y_static, Act_static, sample_T, N);
    
    [net_dygat, avgG_S, avgsqG_S] = adamupdate(net_dygat, grad_net_s, avgG_S, avgsqG_S, ep, lr);
    W_aux_dygat = W_aux_dygat - lr * grad_w_s;
    b_aux_dygat = b_aux_dygat - lr * grad_b_s;
    train_loss_dygat(ep) = double(extractdata(loss_s));
    
    % ★ 在 dlfeval 外安全提取梯度範數
    gnorms_ep = extract_dygat_grad_norms(grad_net_s);
    grad_norms_dygat(ep, :) = gnorms_ep;
    
    % -------------------------------------------------------------
    % 5D. 32 天分塊防 OOM 驗證評估 (每 25 Epochs 監控一次)
    % -------------------------------------------------------------
    if mod(ep, 25) == 0 || ep == 1
        val_l_t = evaluate_val_loss_chunked_time(net_time, W_aux_time, b_aux_time, ...
            X_sub_18D, Y_5D, Expert_sub, val_eval_days, seqLen, sample_T, use_gpu);
        val_l_s = evaluate_val_loss_chunked_dygat(net_dygat, W_aux_dygat, b_aux_dygat, ...
            X_sub_18D, Adj_sub, Y_5D, Expert_sub, val_eval_days, sample_T, feat_dim_space, adj_dim_space, use_gpu);
        
        val_loss_time_track(ep)  = val_l_t;
        val_loss_dygat_track(ep) = val_l_s;
        
        fprintf('  Epoch %3d/%3d | Train BCE [T: %.4f | GCN: %.4f | DyGAT: %.4f] | Val BCE [T: %.4f | DyGAT: %.4f] | GradNorm(Attn): %.2e\n', ...
            ep, epochs, train_loss_time(ep), train_loss_pure_gcn(ep), train_loss_dygat(ep), val_l_t, val_l_s, gnorms_ep(2));
    end
end
elapsed_sec = toc;
fprintf('⚡ 150 Epochs 三軌隔離訓練完成！總耗時: %.2f 秒 (平均每 Epoch: %.3f 秒)\n', elapsed_sec, elapsed_sec / epochs);

final_bce_time     = train_loss_time(end);
final_bce_pure_gcn = train_loss_pure_gcn(end);
final_bce_dygat    = train_loss_dygat(end);

final_val_time  = evaluate_val_loss_chunked_time(net_time, W_aux_time, b_aux_time, ...
    X_sub_18D, Y_5D, Expert_sub, val_eval_days, seqLen, sample_T, use_gpu);
final_val_dygat = evaluate_val_loss_chunked_dygat(net_dygat, W_aux_dygat, b_aux_dygat, ...
    X_sub_18D, Adj_sub, Y_5D, Expert_sub, val_eval_days, sample_T, feat_dim_space, adj_dim_space, use_gpu);

%% 6. 判讀門檻檢定與因果鏈診斷報告
disp('========================================================================================');
disp('📊 【Round 8a-v2 小樣本記憶力與 DyGAT 架構隔離診斷報告】');
disp('========================================================================================');
fprintf('  > 時序專家 (Time Expert)   : Train BCE = %.5f (目標 < 0.05) | Held-out Val BCE = %.4f\n', final_bce_time, final_val_time);
fprintf('  > 純圖卷積 (Pure GCN)      : Train BCE = %.5f (目標 < 0.05)\n', final_bce_pure_gcn);
fprintf('  > 完整空間 (Full DyGAT)    : Train BCE = %.5f (目標 < 0.05) | Held-out Val BCE = %.4f\n', final_bce_dygat, final_val_dygat);
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('  > DyGAT 梯度範數末期均值   : [Weights: %.2e | Attn_1: %.2e | Attn_2: %.2e | MixLogit: %.2e]\n', ...
    mean(grad_norms_dygat(end-10:end, 1)), mean(grad_norms_dygat(end-10:end, 2)), ...
    mean(grad_norms_dygat(end-10:end, 3)), mean(grad_norms_dygat(end-10:end, 4)));
fprintf('----------------------------------------------------------------------------------------\n');

pass_time     = (final_bce_time < 0.05);
pass_pure_gcn = (final_bce_pure_gcn < 0.05);
pass_dygat    = (final_bce_dygat < 0.05);

if pass_time && pass_dygat
    fprintf('  ✅ 【CASE 1: ALL PASS】雙軌架構具備 100%% 記憶容量！\n');
    fprintf('     -> 時序與空間專家皆能過擬合至近乎 0 誤差 (BCE < 0.05)。\n');
    fprintf('     -> 徹底排除「模型容量不足」與「工程梯度中斷」假設！真實資料停滯全因訊噪比不足。\n');
elseif pass_pure_gcn && ~pass_dygat
    fprintf('  ⚠️ 【CASE 2: DYGAT ATTENTION DEFECT】純 GCN 通過但 DyGAT 失敗！\n');
    fprintf('     -> 純鄰接矩陣特徵聚合具備記憶能力 (BCE < 0.05)，但動態注意力分支失效。\n');
    fprintf('     -> 確診為注意力權重初始化尺度 (W_attn) 或 MixLogit Sigmoid 飽和鎖死梯度！\n');
    fprintf('     -> 建議：調整 MixLogit 初值或切換為靜態 Normalized GCN。\n');
elseif ~pass_pure_gcn && ~pass_dygat
    fprintf('  ❌ 【CASE 3: SPATIAL TENSOR DEFECT】空間專家圖卷積底層重組存在工程缺陷！\n');
    fprintf('     -> 純 GCN 與 DyGAT 均無法背下 25 天小樣本。\n');
    fprintf('     -> 確診為跨標的維度展平 (pagemtimes/reshape) 或空間前向梯度傳遞中斷。\n');
else
    fprintf('  ⚠️ 【CASE 4: TIME ONLY】時序專家正常，空間架構需重新核對。\n');
end
fprintf('========================================================================================\n\n');

%% 7. 產出標準學術白底黑字視覺化圖表
fig = figure('Name', 'Round 8a-v2 DyGAT Isolation Diagnosis', ...
    'Color', 'w', 'Position', [100, 100, 1200, 500], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

% 子圖 1：三軌訓練損失收斂曲線對比
subplot(1, 2, 1);
plot(1:epochs, train_loss_time, '-o', 'Color', '#D95319', 'LineWidth', 1.5, 'MarkerSize', 2, 'DisplayName', 'Time Expert');
hold on;
plot(1:epochs, train_loss_pure_gcn, '-^', 'Color', '#77AC30', 'LineWidth', 1.5, 'MarkerSize', 2, 'DisplayName', 'Pure GCN (No Attn)');
plot(1:epochs, train_loss_dygat, '-s', 'Color', '#0072BD', 'LineWidth', 1.5, 'MarkerSize', 2, 'DisplayName', 'Full DyGAT');
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
yline(0.0500, ':r', 'Overfit Threshold (0.05)', 'LineWidth', 1.2, 'DisplayName', 'Target (< 0.05)');

title('N=25 Small-Sample Overfit Training Loss', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on;
box on;

% 子圖 2：DyGAT 逐層梯度範數追蹤
subplot(1, 2, 2);
semilogy(1:epochs, grad_norms_dygat(:, 1) + 1e-12, 'Color', '#0072BD', 'LineWidth', 1.3, 'DisplayName', 'Weights (GCN)');
hold on;
semilogy(1:epochs, grad_norms_dygat(:, 2) + 1e-12, 'Color', '#D95319', 'LineWidth', 1.3, 'DisplayName', 'W\_attn\_1');
semilogy(1:epochs, grad_norms_dygat(:, 3) + 1e-12, 'Color', '#EDB120', 'LineWidth', 1.3, 'DisplayName', 'W\_attn\_2');
semilogy(1:epochs, grad_norms_dygat(:, 4) + 1e-12, 'Color', '#7E2F8E', 'LineWidth', 1.3, 'DisplayName', 'MixLogit');
yline(1e-5, ':r', 'Vanishing Gradient (1e-5)', 'LineWidth', 1.1, 'HandleVisibility', 'off');

title('DyGAT Layer-wise Gradient Norm Evolution', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Gradient L2-Norm (Log Scale)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
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
disp('🎯 [Round 8a-v2] 執行完畢！');
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
    y_t = y_t(:);
    p_t = probs(act_m);
    p_t = p_t(:);
    
    loss = -mean(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8));
    [grad_net, grad_w, grad_b] = dlgradient(loss, net.Learnables, W, b);
end

%% =====================================================================
% 輔助函數：純 GCN 向量化前向與梯度計算 (無 Attention 機制)
% =====================================================================
function [loss, grad_p] = compute_pure_bce_pure_gcn_vectorized(params, dl_x, dl_adj, y_true, act_m, numT, N)
    I_N = dlarray(eye(numT, 'single'));
    A_tilde = dl_adj + I_N;
    deg = sum(abs(A_tilde), 2);
    deg_inv_sqrt = 1.0 ./ sqrt(deg + 1e-4);
    A_norm = deg_inv_sqrt .* A_tilde .* permute(deg_inv_sqrt, [2, 1, 3]);
    
    X_perm = permute(dl_x, [2, 1, 3]);
    X_agg = pagemtimes(A_norm, X_perm);
    X_agg_t = permute(X_agg, [2, 1, 3]);
    
    H1 = relu(pagemtimes(params.W1, X_agg_t) + params.b1);
    H2 = pagemtimes(params.W2, H1) + params.b2;
    emb = reshape(H2, 64, numT * N);
    emb = stripdims(emb);
    
    logits = params.W_aux' * emb + params.b_aux;
    probs = sigmoid(logits);
    
    y_t = y_true(act_m);
    y_t = y_t(:);
    p_t = probs(act_m);
    p_t = p_t(:);
    
    loss = -mean(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8));
    grad_p = dlgradient(loss, params);
end

%% =====================================================================
% 輔助函數：完整 DyGAT 向量化前向與梯度計算 (雙輸入)
% =====================================================================
function [loss, grad_net, grad_w, grad_b] = compute_pure_bce_dygat_vectorized(net, W, b, dl_feat, dl_adj, y_true, act_m, numT, N)
    emb_raw = forward_space_wrapper(net, dl_feat, dl_adj);
    emb_raw = stripdims(emb_raw);
    emb = reshape(emb_raw, 64, numT * N);
    
    logits = W' * emb + b;
    probs = sigmoid(logits);
    
    y_t = y_true(act_m);
    y_t = y_t(:);
    p_t = probs(act_m);
    p_t = p_t(:);
    
    loss = -mean(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8));
    [grad_net, grad_w, grad_b] = dlgradient(loss, net.Learnables, W, b);
end

%% =====================================================================
% 輔助函數：安全提取 DyGAT 逐層梯度範數 (在 dlfeval 外部執行)
% =====================================================================
function gnorms = extract_dygat_grad_norms(grad_net)
    gnorm_w = 0; gnorm_a1 = 0; gnorm_a2 = 0; gnorm_mix = 0;
    param_table = grad_net;
    for p = 1:height(param_table)
        p_name = param_table.Parameter{p};
        p_val  = param_table.Value{p};
        if ~isempty(p_val)
            p_mat = double(extractdata(gather(p_val)));
            g_val = norm(p_mat(:));
            if contains(p_name, 'Weights') || contains(p_name, 'Weight')
                gnorm_w = gnorm_w + g_val;
            elseif contains(p_name, 'W_attn_1') || contains(p_name, 'Attn1')
                gnorm_a1 = gnorm_a1 + g_val;
            elseif contains(p_name, 'W_attn_2') || contains(p_name, 'Attn2')
                gnorm_a2 = gnorm_a2 + g_val;
            elseif contains(p_name, 'MixLogit') || contains(p_name, 'Mix')
                gnorm_mix = gnorm_mix + g_val;
            end
        end
    end
    gnorms = [gnorm_w, gnorm_a1, gnorm_a2, gnorm_mix];
end

%% =====================================================================
% 輔助函數：時序專家 32 天分塊驗證損失計算 (★ 防 OOM 核心函式)
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
% 輔助函數：空間專家 32 天分塊驗證損失計算 (★ 防 OOM 核心函式)
% =====================================================================
function val_loss = evaluate_val_loss_chunked_dygat(net, W, b, X_input, Adj_sub, Y_5D, Expert_sub, val_days, sample_T, feat_dim, adj_dim, use_gpu)
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