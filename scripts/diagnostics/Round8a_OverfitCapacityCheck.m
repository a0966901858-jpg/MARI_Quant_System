% =========================================================================
% 腳本：Round8a_OverfitCapacityCheck.m (Round 8a-v3: 空間專家容量與注意力徹底解耦診斷)
% 升級：Phase 15.5 Stage 2.5 方案 (★ mrg32k3a 獨立子串流注入、GPU 權重 He 初始化顯式串流鎖定、
%       四軌乾淨對照: Time vs 2-Layer GCN vs 1-Layer GCN-only vs 1-Layer Dynamic、
%       32 天分塊防 OOM 驗證、純量損失累加、dlfeval 外梯度範數提取、無 Linter 警告)
% 職責：徹底解耦空間專家的學習瓶頸，精準判定係「單層幾何容量不足」、「動態注意力分支缺陷」、
%       還是「底層圖張量工程瑕疵」
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧠 [Round 8a-v3] 啟動空間專家幾何容量與注意力解耦隔離診斷 (mrg32k3a 確定性串流版)');
disp('=================================================================');

%% 0. 環境路徑掛載與隨機種子鎖定
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

% ★ 核心修復 1：由 Config 統一生產 mrg32k3a 隨機數引擎，並設為全域主串流 (Substream = 1)
stream = configObj.getRandStream(1);
RandStream.setGlobalStream(stream);
disp('🔒 已成功掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定空間容量診斷確定性。');

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
    act_m = Expert_sub(t, :) & ~isnan(R_fwd_5D(t, :)) & ~isinf(R_fwd_5D(t, :));
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

% 構建 Held-out 驗證區間
idx_OOS_start = find(Dates_Active >= datetime('2022-01-01', 'TimeZone', 'UTC'), 1);
val_eval_days = (idx_OOS_start + 20) : (numDays - 1);

use_gpu = canUseGPU();
if use_gpu
    gpu_dev = gpuDevice();
    fprintf('  🎮 偵測到硬體加速卡: 【%s】，啟用 GPU 全靜態顯存加速。\n', gpu_dev.Name);
else
    disp('  💻 使用 CPU 執行訓練。');
end

% 3A. 時序專家靜態張量 [Feats, sample_T * N, SeqLen]
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
if use_gpu, dl_x_static_time = gpuArray(dl_x_static_time); end

% 3B. 空間專家靜態張量
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

%% 4. 初始化四組零正則化網路 (Time vs 2L-GCN vs 1L-GCN-only vs 1L-Dynamic)
disp('--- 步驟 4：構建零正則化四軌解耦對照網路 (Dropout = 0) ---');

% -----------------------------------------------------------------
% Group 1: 時序基準 (Pure LSTM)
% -----------------------------------------------------------------
smokeConfig = configObj;
smokeConfig.NumTickers = sample_T;
extractorFactory = BuildDecoupledExtractors(smokeConfig, numExtractorFeats, 'pure_lstm');
[net_time, ~] = extractorFactory.buildNetworks();

lgraph_time = layerGraph(net_time);
layers_t = lgraph_time.Layers;
for i = 1:length(layers_t)
    if isa(layers_t(i), 'nnet.cnn.layer.DropoutLayer')
        lgraph_time = replaceLayer(lgraph_time, layers_t(i).Name, dropoutLayer(0.0, 'Name', layers_t(i).Name));
    end
end
net_time = dlnetwork(lgraph_time);

% -----------------------------------------------------------------
% Group 2: 手動雙層純 GCN (2-Layer Pure GCN，幾何深度容量上限)
% -----------------------------------------------------------------
% ★ 核心修復 2：手動 He 權重初始化顯式使用 stream 隨機子串流
params_2l_gcn = struct();
params_2l_gcn.W1 = dlarray(randn(stream, 64, numExtractorFeats, 'single') * sqrt(2 / numExtractorFeats));
params_2l_gcn.b1 = dlarray(zeros(64, 1, 'single'));
params_2l_gcn.W2 = dlarray(randn(stream, 64, 64, 'single') * sqrt(2 / 64));
params_2l_gcn.b2 = dlarray(zeros(64, 1, 'single'));
params_2l_gcn.W_aux = dlarray(randn(stream, 64, 1, 'single') * 0.01);
params_2l_gcn.b_aux = dlarray(zeros(1, 1, 'single'));

if use_gpu
    fn = fieldnames(params_2l_gcn);
    for k = 1:length(fn), params_2l_gcn.(fn{k}) = gpuArray(params_2l_gcn.(fn{k})); end
end

% -----------------------------------------------------------------
% Group 3: 單層 GraphSpatialFusionLayer (顯式指定 MixMode = 'gcn_only')
% -----------------------------------------------------------------
layer_gcn_only = GraphSpatialFusionLayer('gat_gcn_only', sample_T, 64, numExtractorFeats, 'gcn_only');
lgraph_gcn1 = layerGraph();
lgraph_gcn1 = addLayers(lgraph_gcn1, featureInputLayer(feat_dim_space, 'Name', 'in_space_feat'));
lgraph_gcn1 = addLayers(lgraph_gcn1, featureInputLayer(adj_dim_space,  'Name', 'in_space_adj'));
lgraph_gcn1 = addLayers(lgraph_gcn1, layer_gcn_only);
lgraph_gcn1 = connectLayers(lgraph_gcn1, 'in_space_feat', 'gat_gcn_only/in1');
lgraph_gcn1 = connectLayers(lgraph_gcn1, 'in_space_adj',  'gat_gcn_only/in2');
net_gcn_single = dlnetwork(lgraph_gcn1);

% -----------------------------------------------------------------
% Group 4: 單層 GraphSpatialFusionLayer (顯式指定 MixMode = 'dynamic')
% -----------------------------------------------------------------
layer_dynamic = GraphSpatialFusionLayer('gat_dynamic', sample_T, 64, numExtractorFeats, 'dynamic');
lgraph_dyn1 = layerGraph();
lgraph_dyn1 = addLayers(lgraph_dyn1, featureInputLayer(feat_dim_space, 'Name', 'in_space_feat'));
lgraph_dyn1 = addLayers(lgraph_dyn1, featureInputLayer(adj_dim_space,  'Name', 'in_space_adj'));
lgraph_dyn1 = addLayers(lgraph_dyn1, layer_dynamic);
lgraph_dyn1 = connectLayers(lgraph_dyn1, 'in_space_feat', 'gat_dynamic/in1');
lgraph_dyn1 = connectLayers(lgraph_dyn1, 'in_space_adj',  'gat_dynamic/in2');
net_gat_dynamic = dlnetwork(lgraph_dyn1);

fprintf('✅ 四軌網路建構完畢！\n');
fprintf('   -> G1: Time (Pure LSTM)\n');
fprintf('   -> G2: 2-Layer Pure GCN (Manual Depth Bound)\n');
fprintf('   -> G3: 1-Layer GCN Only (GraphSpatialFusionLayer, MixMode=gcn_only)\n');
fprintf('   -> G4: 1-Layer Full DyGAT (GraphSpatialFusionLayer, MixMode=dynamic)\n');

%% 5. 執行 150 Epochs 全向量化過擬合訓練與梯度範數監控
disp('--- 步驟 5：啟動 150 Epochs 四軌向量化過擬合對照訓練 ---');
epochs = 150;
lr = 2e-3;

train_loss_time     = zeros(epochs, 1);
train_loss_2l_gcn   = zeros(epochs, 1);
train_loss_1l_gcn   = zeros(epochs, 1);
train_loss_1l_dyn   = zeros(epochs, 1);
grad_norms_dynamic  = zeros(epochs, 4); % [Weights, W_attn_1, W_attn_2, MixLogit]
grad_norms_gcn_only = zeros(epochs, 4);

% ★ 核心修復 3：輔助線性頭初始化顯式使用 stream 隨機子串流
W_aux_time   = dlarray(randn(stream, 64, 1, 'single') * 0.01);
b_aux_time   = dlarray(zeros(1, 1, 'single'));
W_aux_1l_gcn = dlarray(randn(stream, 64, 1, 'single') * 0.01);
b_aux_1l_gcn = dlarray(zeros(1, 1, 'single'));
W_aux_1l_dyn = dlarray(randn(stream, 64, 1, 'single') * 0.01);
b_aux_1l_dyn = dlarray(zeros(1, 1, 'single'));

if use_gpu
    W_aux_time   = gpuArray(W_aux_time);   b_aux_time   = gpuArray(b_aux_time);
    W_aux_1l_gcn = gpuArray(W_aux_1l_gcn); b_aux_1l_gcn = gpuArray(b_aux_1l_gcn);
    W_aux_1l_dyn = gpuArray(W_aux_1l_dyn); b_aux_1l_dyn = gpuArray(b_aux_1l_dyn);
end

avgG_T = []; avgsqG_T = [];
avgG_2L = []; avgsqG_2L = [];
avgG_1G = []; avgsqG_1G = [];
avgG_1D = []; avgsqG_1D = [];

tic;
for ep = 1:epochs
    % 5A. Group 1: Time Expert (Pure LSTM)
    [loss_t, grad_net_t, grad_w_t, grad_b_t] = dlfeval(@compute_pure_bce_time_vectorized, ...
        net_time, W_aux_time, b_aux_time, dl_x_static_time, Y_static, Act_static);
    [net_time, avgG_T, avgsqG_T] = adamupdate(net_time, grad_net_t, avgG_T, avgsqG_T, ep, lr);
    W_aux_time = W_aux_time - lr * grad_w_t;
    b_aux_time = b_aux_time - lr * grad_b_t;
    train_loss_time(ep) = double(extractdata(loss_t));
    
    % 5B. Group 2: Pure 2-Layer GCN
    [loss_2g, grad_2gcn] = dlfeval(@compute_pure_bce_pure_gcn_vectorized, ...
        params_2l_gcn, dl_static_3D_x, dl_static_3D_adj, Y_static, Act_static, sample_T, N);
    [params_2l_gcn, avgG_2L, avgsqG_2L] = adamupdate(params_2l_gcn, grad_2gcn, avgG_2L, avgsqG_2L, ep, lr);
    train_loss_2l_gcn(ep) = double(extractdata(loss_2g));
    
    % 5C. Group 3: 1-Layer GCN Only
    [loss_1g, grad_net_1g, grad_w_1g, grad_b_1g] = dlfeval(@compute_pure_bce_dygat_vectorized, ...
        net_gcn_single, W_aux_1l_gcn, b_aux_1l_gcn, dl_static_s_feat, dl_static_s_adj, Y_static, Act_static, sample_T, N);
    [net_gcn_single, avgG_1G, avgsqG_1G] = adamupdate(net_gcn_single, grad_net_1g, avgG_1G, avgsqG_1G, ep, lr);
    W_aux_1l_gcn = W_aux_1l_gcn - lr * grad_w_1g;
    b_aux_1l_gcn = b_aux_1l_gcn - lr * grad_b_1g;
    train_loss_1l_gcn(ep) = double(extractdata(loss_1g));
    grad_norms_gcn_only(ep, :) = extract_dygat_grad_norms(grad_net_1g);
    
    % 5D. Group 4: 1-Layer Full DyGAT (Dynamic Attention)
    [loss_1d, grad_net_1d, grad_w_1d, grad_b_1d] = dlfeval(@compute_pure_bce_dygat_vectorized, ...
        net_gat_dynamic, W_aux_1l_dyn, b_aux_1l_dyn, dl_static_s_feat, dl_static_s_adj, Y_static, Act_static, sample_T, N);
    [net_gat_dynamic, avgG_1D, avgsqG_1D] = adamupdate(net_gat_dynamic, grad_net_1d, avgG_1D, avgsqG_1D, ep, lr);
    W_aux_1l_dyn = W_aux_1l_dyn - lr * grad_w_1d;
    b_aux_1l_dyn = b_aux_1l_dyn - lr * grad_b_1d;
    train_loss_1l_dyn(ep) = double(extractdata(loss_1d));
    grad_norms_dynamic(ep, :) = extract_dygat_grad_norms(grad_net_1d);
    
    if mod(ep, 25) == 0 || ep == 1
        fprintf('  Epoch %3d/%3d | BCE [Time: %.4f | 2L-GCN: %.4f | 1L-GCN: %.4f | 1L-Dyn: %.4f] | Dyn-Attn-Grad: %.2e\n', ...
            ep, epochs, train_loss_time(ep), train_loss_2l_gcn(ep), train_loss_1l_gcn(ep), train_loss_1l_dyn(ep), grad_norms_dynamic(ep, 2));
    end
end
elapsed_sec = toc;
fprintf('⚡ 150 Epochs 四軌隔離訓練完成！總耗時: %.2f 秒\n', elapsed_sec);

final_bce_time   = train_loss_time(end);
final_bce_2l_gcn = train_loss_2l_gcn(end);
final_bce_1l_gcn = train_loss_1l_gcn(end);
final_bce_1l_dyn = train_loss_1l_dyn(end);

% 驗證集防 OOM 損失計算
final_val_time   = evaluate_val_loss_chunked_time(net_time, W_aux_time, b_aux_time, ...
    X_sub_18D, Y_5D, Expert_sub, val_eval_days, seqLen, sample_T, use_gpu);
final_val_1l_dyn = evaluate_val_loss_chunked_dygat(net_gat_dynamic, W_aux_1l_dyn, b_aux_1l_dyn, ...
    X_sub_18D, Adj_sub, Y_5D, Expert_sub, val_eval_days, sample_T, feat_dim_space, adj_dim_space, use_gpu);

%% 6. 判讀門檻檢定與因果鏈診斷報告
disp('========================================================================================');
disp('📊 【Round 8a-v3 空間專家幾何容量與注意力機制解耦診斷報告】');
disp('========================================================================================');
fprintf('  > Group 1: 時序專家 (Pure LSTM)     : Train BCE = %.5f (目標 < 0.05) | Held-out Val = %.4f\n', final_bce_time, final_val_time);
fprintf('  > Group 2: 手動雙層 (2-Layer Pure GCN): Train BCE = %.5f (目標 < 0.05)\n', final_bce_2l_gcn);
fprintf('  > Group 3: 單層靜態 (1-Layer GCN-Only): Train BCE = %.5f (目標 < 0.05)\n', final_bce_1l_gcn);
fprintf('  > Group 4: 單層動態 (1-Layer Dynamic) : Train BCE = %.5f (目標 < 0.05) | Held-out Val = %.4f\n', final_bce_1l_dyn, final_val_1l_dyn);
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('  > G4 (Dynamic)  梯度範數末期均值: [Weights: %.2e | Attn_1: %.2e | Attn_2: %.2e | MixLogit: %.2e]\n', ...
    mean(grad_norms_dynamic(end-10:end, 1)), mean(grad_norms_dynamic(end-10:end, 2)), ...
    mean(grad_norms_dynamic(end-10:end, 3)), mean(grad_norms_dynamic(end-10:end, 4)));
fprintf('  > G3 (GCN-only) 梯度範數末期均值: [Weights: %.2e | Attn_1: %.2e | Attn_2: %.2e | MixLogit: %.2e]\n', ...
    mean(grad_norms_gcn_only(end-10:end, 1)), mean(grad_norms_gcn_only(end-10:end, 2)), ...
    mean(grad_norms_gcn_only(end-10:end, 3)), mean(grad_norms_gcn_only(end-10:end, 4)));
fprintf('----------------------------------------------------------------------------------------\n');

pass_time   = (final_bce_time < 0.05);
pass_2l_gcn = (final_bce_2l_gcn < 0.05);
pass_1l_gcn = (final_bce_1l_gcn < 0.05);
pass_1l_dyn = (final_bce_1l_dyn < 0.05);

if pass_2l_gcn && pass_1l_gcn && pass_1l_dyn
    fprintf('  ✅ 【CASE 1: ALL PASS】空間專家各層級架構皆具備充足記憶容量！\n');
    fprintf('     -> 單層與雙層、靜態與動態皆能過擬合 (BCE < 0.05)。\n');
    fprintf('     -> 排除容量與工程問題；P2-1 空間專家無效純屬橫截面圖譜無 Alpha。\n');
elseif pass_2l_gcn && pass_1l_gcn && ~pass_1l_dyn
    fprintf('  ⚠️ 【CASE 2: ATTENTION DEFECT】靜態圖卷積正常，但動態注意力分支失效！\n');
    fprintf('     -> 1L-GCN 可背下 (BCE < 0.05)，但 1L-Dynamic 失敗。\n');
    fprintf('     -> 確診為 ASTGCN 注意力權重或 MixLogit 飽和引發梯度消失！\n');
    fprintf('     -> 支持 Stage 4 全面切換為 gcn_only。\n');
elseif pass_2l_gcn && ~pass_1l_gcn
    fprintf('  ⚠️ 【CASE 3: SINGLE-LAYER CAPACITY DEFICIT】單層幾何容量不足（深度限制）！\n');
    fprintf('     -> 2-Layer GCN 可過擬合至 %.4f，但 1-Layer GCN (%.4f) 與 Dynamic (%.4f) 均失敗。\n', ...
        final_bce_2l_gcn, final_bce_1l_gcn, final_bce_1l_dyn);
    fprintf('     -> 確診為「單層圖卷積對於 1080 維節點圖譜而言存在結構性容量瓶頸」，而非張量工程 Bug！\n');
    fprintf('     -> 建議：若欲保留空間專家，需擴展為 2 層 GCN；若欲追求極致精簡，則直接裁撤空間專家。\n');
elseif ~pass_2l_gcn
    fprintf('  ❌ 【CASE 4: SPATIAL TENSOR DEFECT】空間專家圖張量存在底層工程缺陷！\n');
    fprintf('     -> 連手動雙層 GCN 亦無法過擬合小樣本。\n');
    fprintf('     -> 確診為度數矩陣正規化或維度重組中斷了空間前向梯度傳導。\n');
end
fprintf('========================================================================================\n\n');

%% 7. 產出標準學術白底黑字視覺化圖表
fig = figure('Name', 'Round 8a-v3 Spatial Capacity Decoupled Diagnosis', ...
    'Color', 'w', 'Position', [100, 100, 1200, 500], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

% 子圖 1：四軌訓練損失收斂曲線對比
subplot(1, 2, 1);
plot(1:epochs, train_loss_time, '-o', 'Color', '#D95319', 'LineWidth', 1.5, 'MarkerSize', 2, 'DisplayName', 'G1: Time (LSTM)'); hold on;
plot(1:epochs, train_loss_2l_gcn, '-^', 'Color', '#77AC30', 'LineWidth', 1.5, 'MarkerSize', 2, 'DisplayName', 'G2: 2L Pure GCN (Manual)');
plot(1:epochs, train_loss_1l_gcn, '-d', 'Color', '#EDB120', 'LineWidth', 1.5, 'MarkerSize', 2, 'DisplayName', 'G3: 1L GCN-Only (Layer)');
plot(1:epochs, train_loss_1l_dyn, '-s', 'Color', '#0072BD', 'LineWidth', 1.5, 'MarkerSize', 2, 'DisplayName', 'G4: 1L Dynamic DyGAT');
yline(0.6931, '--k', 'Random Guess (ln 2)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
yline(0.0500, ':r', 'Target (< 0.05)', 'LineWidth', 1.2, 'DisplayName', 'Overfit Target');
title('N=25 Small-Sample Overfit Training Loss', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('BCE Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

% 子圖 2：動態 vs 靜態注意力梯度對比
subplot(1, 2, 2);
semilogy(1:epochs, grad_norms_dynamic(:, 1) + 1e-12, 'Color', '#0072BD', 'LineWidth', 1.3, 'DisplayName', 'Dynamic Weights'); hold on;
semilogy(1:epochs, grad_norms_dynamic(:, 2) + 1e-12, 'Color', '#D95319', 'LineWidth', 1.3, 'DisplayName', 'Dynamic W\_attn\_1');
semilogy(1:epochs, grad_norms_dynamic(:, 4) + 1e-12, 'Color', '#7E2F8E', 'LineWidth', 1.3, 'DisplayName', 'Dynamic MixLogit');
semilogy(1:epochs, grad_norms_gcn_only(:, 1) + 1e-12, '--', 'Color', '#EDB120', 'LineWidth', 1.3, 'DisplayName', 'GCN-only Weights');
semilogy(1:epochs, grad_norms_gcn_only(:, 2) + 1e-12, ':', 'Color', '#77AC30', 'LineWidth', 1.3, 'DisplayName', 'GCN-only Attn (Frozen=0)');
yline(1e-5, ':r', 'Vanishing Threshold (1e-5)', 'LineWidth', 1.1, 'HandleVisibility', 'off');
title('Gradient Norm Evolution: Dynamic vs GCN-Only', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Gradient L2-Norm (Log Scale)', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8, 0.8, 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

figPath = fullfile(configObj.ResultDir, 'Round8a_OverfitCapacityCheck.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化圖表已儲存至: %s\n', figPath);
close(fig);

disp('=================================================================');
disp('🎯 [Round 8a-v3] 診斷完畢！');
disp('=================================================================');

%% =====================================================================
% 輔助函數區
% =====================================================================
function emb = forward_space_wrapper(net, dl_feat, dl_adj)
    input_names = net.InputNames;
    if length(input_names) >= 2 && contains(input_names{1}, 'adj')
        emb = forward(net, dl_adj, dl_feat);
    else
        emb = forward(net, dl_feat, dl_adj);
    end
end

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
        if use_gpu, dl_x = gpuArray(dl_x); end
        
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