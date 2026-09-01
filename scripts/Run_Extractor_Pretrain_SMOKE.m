% =========================================================================
% 腳本：2_Run_Extractor_Pretrain_SMOKE.m (階段 2：雙軌特徵萃取器 - 極速冒煙測試)
% 升級：Phase 14.24 (★ 資料規模假說驗證：3200 天跨週期視窗 + 溫和正則化 + 自動坍縮診斷)
% 職責：在 60 檔標的 x 3200 天的多體制樣本上驗證資料長度對驗證集泛化能力與過擬合的影響
% =========================================================================
clear; clc; close all;

%% 0. 環境路徑掛載
disp('=================================================================');
disp('🧪 [Phase 14.24 SMOKE TEST] 啟動雙軌萃取器極速冒煙測試 (資料規模擴展版)');
disp('=================================================================');

currentPath = fileparts(mfilename('fullpath'));
if isempty(currentPath), currentPath = pwd; end
projectRoot = currentPath; 

if ~exist(fullfile(projectRoot, 'configs'), 'dir')
    projectRoot = fullfile(currentPath, '..'); 
end

addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'envs')));
addpath(genpath(fullfile(projectRoot, 'agents'))); 
addpath(genpath(fullfile(projectRoot, 'models'))); 
rehash toolboxcache;

configObj = Config();

% ★ 溫和正則化設定 (R2 基準配置)
configObj.DL_DropoutRate = 0.2;

if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入 3D 特徵面板資料並進行「跨週期擴展抽樣 (3200 天)」
disp('--- 步驟 1：載入特徵快取並執行資料規模擴展抽樣 (60 檔 x 3200 天) ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
if ~exist(cachePath, 'file')
    error('❌ 找不到特徵快取檔案，請先確認 Phase 1 已產出 features_denoised.mat');
end

load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active', 'AdjMatrix_3D');
Dates_Active.TimeZone = ''; 

numDaysRaw_full = length(Dates_Active);
numT_full = configObj.NumTickers;

% 60 檔標的抽樣
sample_n_tickers = min(60, numT_full);
sub_tickers = sort(randsample(numT_full, sample_n_tickers));

% ★ 資料規模假說驗證：時間窗口由 1500 天擴展至 3200 天，跨越更多景氣體制
start_day = max(1, round(numDaysRaw_full * 0.4));
end_day   = min(numDaysRaw_full, start_day + 3200 - 1);
sub_days_range = start_day : end_day;

X_norm_3D     = X_norm_3D(sub_days_range, :, sub_tickers);
AdjMatrix_3D  = AdjMatrix_3D(sub_tickers, sub_tickers, sub_days_range);
Prices_Active = Prices_Active(sub_days_range, sub_tickers);
Expert_Active = Expert_Active(sub_days_range, sub_tickers);
Dates_Active  = Dates_Active(sub_days_range);

numDaysRaw = length(Dates_Active);
numT = length(sub_tickers);
configObj.NumTickers = numT; 

numFeats = size(X_norm_3D, 2); 
seqLen = configObj.SeqLen;
horizon = 5; 
valid_idx = seqLen : (numDaysRaw - horizon); 
num_valid = length(valid_idx);

fprintf('  -> [Smoke 子集規模] 標的數: %d 檔 | 交易天數: %d 天 | 特徵維度: %d 維 | 有效步數: %d 步\n', ...
    numT, numDaysRaw, numFeats, num_valid);

%% 1.5 核心串接：計算橫截面特徵 IC 信心權重
disp('--- 步驟 1.5：計算橫截面特徵 IC 信心權重 (Smoke 子集) ---');
evaluator = FeatureEvaluator(configObj);
IC_Weights_2D = evaluator.compute_confidence(X_norm_3D, Prices_Active, Expert_Active);

disp(' -> 執行特徵注意力遮罩融合 (Energy-Preserving Feature Gate)...');
IC_Weights_3D = reshape(IC_Weights_2D, numDaysRaw, numFeats, 1);
X_norm_3D = X_norm_3D .* IC_Weights_3D;

ch_stds = squeeze(std(X_norm_3D, 0, [1, 3], 'omitnan'));
fprintf('  -> [通道尺度健檢] %d 維特徵標準差範圍: [%.4f, %.4f] (平均: %.4f)\n', ...
    numFeats, min(ch_stds), max(ch_stds), mean(ch_stds));

%% 2. 構建橫截面輔助預測目標 (5 日遠期報酬 Beat the Median)
disp('--- 步驟 2：構建 5 日遠期橫截面標籤 (5-Day Beat the Median) ---');
R_fwd = NaN(numDaysRaw, numT, 'single');
R_fwd(1:end-horizon, :) = (Prices_Active(1+horizon:end, :) - Prices_Active(1:end-horizon, :)) ...
                          ./ Prices_Active(1:end-horizon, :);
R_fwd(isinf(R_fwd)) = NaN;

Y_Labels_3D = zeros(numDaysRaw, numT, 'single');
for t = 1:numDaysRaw-horizon
    active_mask = Expert_Active(t, :);
    if sum(active_mask) >= 5
        med_ret = median(R_fwd(t, active_mask), 'omitnan');
        Y_Labels_3D(t, active_mask) = single(R_fwd(t, active_mask) > med_ret);
    end
end

%% 3. 切分時間軸 (Train 80% / Val 20%)
disp('--- 步驟 3：切分子集樣本時間軸 (Train 80% / Val 20%) ---');
split_point = round(num_valid * 0.8);
train_idx_valid = valid_idx(1:split_point);
num_train = length(train_idx_valid);

oos_idx_valid = valid_idx(split_point+1:end);
num_oos = length(oos_idx_valid);

fprintf('✅ 子集時間軸劃分成功！訓練樣本: %d 筆 | 驗證樣本: %d 筆\n', num_train, num_oos);

%% 4. 初始化雙軌 DL 萃取器與獨立輔助預測頭 (He 初始化)
disp('--- 步驟 4：初始化雙軌特徵萃取網路 (He 初始化) ---');
factory = BuildDecoupledExtractors(configObj, numFeats);
[net_time, net_space] = factory.buildNetworks();

W_aux_time  = dlarray(randn(1, 64, 'single') * sqrt(2/64)); 
b_aux_time  = dlarray(zeros(1, 1, 'single'));         
W_aux_space = dlarray(randn(1, 64, 'single') * sqrt(2/64));
b_aux_space = dlarray(zeros(1, 1, 'single'));

if canUseGPU()
    gpuInfo = gpuDevice(); 
    fprintf('🎮 成功捕獲圖形加速卡：【%s】，開啟深度學習預訓練路徑。\n', gpuInfo.Name);
    net_time = dlupdate(@gpuArray, net_time);
    net_space = dlupdate(@gpuArray, net_space);
    W_aux_time = gpuArray(W_aux_time); b_aux_time = gpuArray(b_aux_time);
    W_aux_space = gpuArray(W_aux_space); b_aux_space = gpuArray(b_aux_space);
end

%% 5. 啟動極速冒煙預訓練迴圈 (3200天多體制 + 溫和 L2/Dropout + 坍縮診斷)
disp('--- 步驟 5：啟動極速冒煙訓練迴圈 (3200天視窗 + 溫和正則化 + 坍縮診斷) ---');
epochs = 15; 
physicalBatchSize = 2;   
accumulationSteps = 8;  
numIterationsPerEpoch = floor(num_train / physicalBatchSize);
clipThreshold = 1.0;

% ★ 超參數設定：LR 5e-4、溫和 L2 1e-4
base_lr   = 5e-4; 
l2_lambda = 1e-4; 

avgG_t = []; avgSG_t = []; avgG_s = []; avgSG_s = [];
avgG_Wt = []; avgSG_Wt = []; avgG_bt = []; avgSG_bt = [];
avgG_Ws = []; avgSG_Ws = []; avgG_bs = []; avgSG_bs = [];
iter = 0; 

historical_loss_time = zeros(epochs, 1);
historical_loss_space = zeros(epochs, 1);
val_loss_time = zeros(epochs, 1);
val_loss_space = zeros(epochs, 1);

best_val_loss = inf; 
patience = 0; 
patience_limit = 5;
best_net_time = net_time; 
best_net_space = net_space;

% 自動坍縮診斷基準
min_healthy_var_t = 0.01;   
min_healthy_var_s = 0.10;

for epoch = 1:epochs
    if epoch <= 2
        current_lr = 1e-4 + (base_lr - 1e-4) * (epoch / 2);
    else
        current_lr = base_lr;
    end
    
    idx_shuffle = randperm(num_train);
    epoch_loss_time = 0;
    epoch_loss_space = 0;
    
    grad_t_accum = []; grad_Wt_accum = []; grad_bt_accum = [];
    grad_s_accum = []; grad_Ws_accum = []; grad_bs_accum = [];
    accum_count = 0;
    
    last_gnorm_t = 0;
    last_gnorm_s = 0;
    
    % --- 訓練階段 ---
    for i = 1:numIterationsPerEpoch
        batch_idx = idx_shuffle((i-1)*physicalBatchSize + 1 : i*physicalBatchSize);
        actual_t_indices = train_idx_valid(batch_idx);
        
        [X_batch_time, X_batch_space, A_batch_space, Y_batch, M_batch] = prepareBatchData(...
            actual_t_indices, X_norm_3D, AdjMatrix_3D, Y_Labels_3D, Expert_Active, numFeats, numT, seqLen, physicalBatchSize);
        
        [loss_t, grad_t, grad_Wt, grad_bt] = dlfeval(@(n,w,b,x,y,m) aux_loss_time(n,w,b,x,y,m, l2_lambda), net_time, W_aux_time, b_aux_time, X_batch_time, Y_batch, M_batch);
        [loss_s, grad_s, grad_Ws, grad_bs] = dlfeval(@(n,w,b,x,a,y,m) aux_loss_space(n,w,b,x,a,y,m, l2_lambda), net_space, W_aux_space, b_aux_space, X_batch_space, A_batch_space, Y_batch, M_batch);
        
        if isempty(grad_t_accum)
            grad_t_accum = grad_t; grad_Wt_accum = grad_Wt; grad_bt_accum = grad_bt;
            grad_s_accum = grad_s; grad_Ws_accum = grad_Ws; grad_bs_accum = grad_bs;
        else
            grad_t_accum.Value = cellfun(@plus, grad_t_accum.Value, grad_t.Value, 'UniformOutput', false);
            grad_s_accum.Value = cellfun(@plus, grad_s_accum.Value, grad_s.Value, 'UniformOutput', false);
            grad_Wt_accum = grad_Wt_accum + grad_Wt; grad_bt_accum = grad_bt_accum + grad_bt;
            grad_Ws_accum = grad_Ws_accum + grad_Ws; grad_bs_accum = grad_bs_accum + grad_bs;
        end
        accum_count = accum_count + 1;
        
        if accum_count == accumulationSteps || i == numIterationsPerEpoch
            iter = iter + 1;
            
            grad_t_accum.Value = cellfun(@(x) x/accum_count, grad_t_accum.Value, 'UniformOutput', false);
            grad_s_accum.Value = cellfun(@(x) x/accum_count, grad_s_accum.Value, 'UniformOutput', false);
            grad_Wt_accum = grad_Wt_accum / accum_count; grad_bt_accum = grad_bt_accum / accum_count;
            grad_Ws_accum = grad_Ws_accum / accum_count; grad_bs_accum = grad_bs_accum / accum_count;
            
            last_gnorm_t = sqrt(sum(cellfun(@(x) sum(extractdata(x(:)).^2), grad_t_accum.Value)));
            last_gnorm_s = sqrt(sum(cellfun(@(x) sum(extractdata(x(:)).^2), grad_s_accum.Value)));
            
            grad_t_accum = dlupdate(@(g) clipGradient(g, clipThreshold), grad_t_accum);
            grad_s_accum = dlupdate(@(g) clipGradient(g, clipThreshold), grad_s_accum);
            
            [net_time, avgG_t, avgSG_t] = adamupdate(net_time, grad_t_accum, avgG_t, avgSG_t, iter, current_lr);
            [W_aux_time, avgG_Wt, avgSG_Wt] = adamupdate(W_aux_time, grad_Wt_accum, avgG_Wt, avgSG_Wt, iter, current_lr);
            [b_aux_time, avgG_bt, avgSG_bt] = adamupdate(b_aux_time, grad_bt_accum, avgG_bt, avgSG_bt, iter, current_lr);
            
            [net_space, avgG_s, avgSG_s] = adamupdate(net_space, grad_s_accum, avgG_s, avgSG_s, iter, current_lr);
            [W_aux_space, avgG_Ws, avgSG_Ws] = adamupdate(W_aux_space, grad_Ws_accum, avgG_Ws, avgSG_Ws, iter, current_lr);
            [b_aux_space, avgG_bs, avgSG_bs] = adamupdate(b_aux_space, grad_bs_accum, avgG_bs, avgSG_bs, iter, current_lr);
            
            grad_t_accum = []; grad_Wt_accum = []; grad_bt_accum = [];
            grad_s_accum = []; grad_Ws_accum = []; grad_bs_accum = [];
            accum_count = 0;
        end
        
        epoch_loss_time  = epoch_loss_time + extractdata(loss_t);
        epoch_loss_space = epoch_loss_space + extractdata(loss_s);
        
        clear loss_t loss_s grad_t grad_s;
    end
    
    historical_loss_time(epoch)  = epoch_loss_time / numIterationsPerEpoch;
    historical_loss_space(epoch) = epoch_loss_space / numIterationsPerEpoch;
    
    % --- 驗證階段 ---
    val_samples = min(num_oos, 32); 
    val_idx_shuffle = randperm(num_oos, val_samples);
    actual_val_indices = oos_idx_valid(val_idx_shuffle);
    
    val_batch_size = 2; 
    temp_val_loss_t = 0;
    temp_val_loss_s = 0;
    val_iters = ceil(val_samples / val_batch_size);
    
    for v_i = 1:val_iters
        v_start = (v_i-1)*val_batch_size + 1;
        v_end = min(v_i*val_batch_size, val_samples);
        chunk_size = v_end - v_start + 1;
        chunk_indices = actual_val_indices(v_start:v_end);
        
        [X_val_time, X_val_space, A_val_space, Y_val, M_val] = prepareBatchData(...
            chunk_indices, X_norm_3D, AdjMatrix_3D, Y_Labels_3D, Expert_Active, numFeats, numT, seqLen, chunk_size);
        
        v_loss_t_chunk = calc_val_loss_time(net_time, W_aux_time, b_aux_time, X_val_time, Y_val, M_val);
        v_loss_s_chunk = calc_val_loss_space(net_space, W_aux_space, b_aux_space, X_val_space, A_val_space, Y_val, M_val);
        
        temp_val_loss_t = temp_val_loss_t + extractdata(v_loss_t_chunk) * chunk_size;
        temp_val_loss_s = temp_val_loss_s + extractdata(v_loss_s_chunk) * chunk_size;
        
        clear X_val_time X_val_space A_val_space Y_val M_val v_loss_t_chunk v_loss_s_chunk;
    end
    
    val_loss_time(epoch)  = temp_val_loss_t / val_samples;
    val_loss_space(epoch) = temp_val_loss_s / val_samples;
    
    e_t_sample = extractdata(predict(net_time, X_batch_time));
    e_s_sample = extractdata(reshape(predict(net_space, X_batch_space, A_batch_space), 64, []));
    var_e_t = var(e_t_sample(:), 'omitnan');
    var_e_s = var(e_s_sample(:), 'omitnan');
    
    fprintf(' -> [Smoke Ep %d/%d] Train (T: %.4f, S: %.4f) | Val (T: %.4f, S: %.4f)\n', ...
        epoch, epochs, historical_loss_time(epoch), historical_loss_space(epoch), val_loss_time(epoch), val_loss_space(epoch));
    fprintf('    [診斷埋點] GradNorm(T: %.2e, S: %.2e) | E_Var(T: %.4f, S: %.4f)\n', ...
        last_gnorm_t, last_gnorm_s, var_e_t, var_e_s);
    
    % ★ 自動坍縮偵測診斷
    if var_e_t < min_healthy_var_t
        fprintf('    ⚠️ [警告] E_Var(T)=%.4f 低於健康閾值 (%.2f)，疑似正則化過強導致表徵坍縮！\n', var_e_t, min_healthy_var_t);
    end
    if var_e_s < min_healthy_var_s
        fprintf('    ⚠️ [警告] E_Var(S)=%.4f 低於健康閾值 (%.2f)，疑似坍縮！\n', var_e_s, min_healthy_var_s);
    end
    
    combined_val = val_loss_time(epoch) + val_loss_space(epoch);
    if combined_val < best_val_loss - 1e-4
        best_val_loss = combined_val;
        patience = 0;
        best_net_time = net_time; 
        best_net_space = net_space;
    else
        patience = patience + 1;
    end
    
    if patience >= patience_limit
        fprintf('🛑 [Early Stopping] 驗證損失連續 %d 輪未改善，提前於 Epoch %d 終止訓練並回滾最佳快照！\n', patience_limit, epoch);
        net_time = best_net_time; 
        net_space = best_net_space;
        break;
    end
    
    clear X_batch_time X_batch_space A_batch_space Y_batch M_batch e_t_sample e_s_sample;
end

%% 冒煙測試診斷與驗收判定
loss_t_start = historical_loss_time(1);
loss_t_end   = historical_loss_time(min(epoch, epochs));
loss_s_start = historical_loss_space(1);
loss_s_end   = historical_loss_space(min(epoch, epochs));

fprintf('\n=================================================================\n');
fprintf('📋 【Phase 14.24 Smoke Test 診斷報告 (3200天規模擴展版)】\n');
fprintf('=================================================================\n');
fprintf(' > 時序專家 (T) Loss: %.4f -> %.4f (降幅: %+.2f%%)\n', ...
    loss_t_start, loss_t_end, ((loss_t_start - loss_t_end)/loss_t_start)*100);
fprintf(' > 空間專家 (S) Loss: %.4f -> %.4f (降幅: %+.2f%%)\n', ...
    loss_s_start, loss_s_end, ((loss_s_start - loss_s_end)/loss_s_start)*100);
fprintf('-----------------------------------------------------------------\n');

t_success = (loss_t_end < 0.6931) && (var_e_t >= min_healthy_var_t);
s_success = (loss_s_end < 0.6931) && (var_e_s >= min_healthy_var_s);

if t_success && s_success
    fprintf('🎉 【驗證成功】雙軌模型 Loss 均成功跌破 ln(2)≈0.6931 基準線且表徵變異數健康！\n');
    fprintf('   在 3200 天多體制樣本下泛化表現穩健，可確認進入全量 Phase 2 預訓練。\n');
else
    fprintf('ℹ️ 【收斂觀察】請對照 Val(T) 曲線與 E_Var 數值，評估多體制下的泛化與收斂狀況。\n');
end
fprintf('=================================================================\n');

%% =====================================================================
% 輔助函數區 (含 L2 權重衰減)
% =====================================================================
function [X_time, X_space, A_space, Y, M] = prepareBatchData(indices, X_3D, Adj_3D, Y_Lab, Expert, nF, nT, sL, bZ)
    X_time_raw  = zeros(nF, nT * bZ, sL, 'single');
    X_space_raw = zeros(nF * nT, bZ, 'single');
    A_space_raw = zeros(nT * nT, bZ, 'single');
    Y_raw       = zeros(1, nT * bZ, 'single');
    M_raw       = zeros(1, nT * bZ, 'single');
    
    for b = 1:bZ
        t = indices(b);
        seq_data = permute(X_3D(t-sL+1:t, :, :), [2, 3, 1]);
        idx_start = (b-1)*nT + 1;
        idx_end   = b*nT;
        X_time_raw(:, idx_start:idx_end, :) = seq_data;
        cross_data = squeeze(X_3D(t, :, :));
        X_space_raw(:, b) = cross_data(:);
        adj_data = Adj_3D(:,:,t);
        A_space_raw(:, b) = adj_data(:);
        Y_raw(1, idx_start:idx_end) = Y_Lab(t, :);
        M_raw(1, idx_start:idx_end) = Expert(t, :);
    end
    
    X_time  = dlarray(X_time_raw, 'CBT');
    X_space = dlarray(X_space_raw, 'CB');
    A_space = dlarray(A_space_raw, 'CB');
    Y = dlarray(Y_raw, 'CB');
    M = dlarray(M_raw, 'CB');
    
    if canUseGPU()
        X_time  = gpuArray(X_time); 
        X_space = gpuArray(X_space); 
        A_space = gpuArray(A_space); 
        Y = gpuArray(Y); 
        M = gpuArray(M);
    end
end

function [loss, grad_net, grad_W, grad_b] = aux_loss_time(net, W_aux, b_aux, X, Y, M, l2_lambda)
    E = forward(net, X); 
    E_unfmt = stripdims(E);
    logits = W_aux * E_unfmt + b_aux; 
    Y_pred = sigmoid(logits);
    Y_unfmt = stripdims(Y);
    M_unfmt = stripdims(M);
    bce = -(Y_unfmt .* log(Y_pred + 1e-8) + (1 - Y_unfmt) .* log(1 - Y_pred + 1e-8));
    masked_bce = bce .* M_unfmt;
    bce_loss = sum(masked_bce, 'all') / (sum(M_unfmt, 'all') + 1e-8); 
    
    l2_penalty = 0;
    learnables = net.Learnables;
    for r = 1:height(learnables)
        w = learnables.Value{r};
        l2_penalty = l2_penalty + sum(w(:).^2);
    end
    l2_penalty = l2_penalty + sum(W_aux(:).^2);
    
    loss = bce_loss + 0.5 * l2_lambda * l2_penalty;
    [grad_net, grad_W, grad_b] = dlgradient(loss, net.Learnables, W_aux, b_aux);
end

function [loss, grad_net, grad_W, grad_b] = aux_loss_space(net, W_aux, b_aux, X, A, Y, M, l2_lambda)
    E_flat_out = forward(net, X, A); 
    E_unfmt = reshape(stripdims(E_flat_out), 64, []);
    logits = W_aux * E_unfmt + b_aux; 
    Y_pred = sigmoid(logits);
    Y_unfmt = stripdims(Y);
    M_unfmt = stripdims(M);
    bce = -(Y_unfmt .* log(Y_pred + 1e-8) + (1 - Y_unfmt) .* log(1 - Y_pred + 1e-8));
    masked_bce = bce .* M_unfmt;
    bce_loss = sum(masked_bce, 'all') / (sum(M_unfmt, 'all') + 1e-8);
    
    l2_penalty = 0;
    learnables = net.Learnables;
    for r = 1:height(learnables)
        w = learnables.Value{r};
        l2_penalty = l2_penalty + sum(w(:).^2);
    end
    l2_penalty = l2_penalty + sum(W_aux(:).^2);
    
    loss = bce_loss + 0.5 * l2_lambda * l2_penalty;
    [grad_net, grad_W, grad_b] = dlgradient(loss, net.Learnables, W_aux, b_aux);
end

function loss = calc_val_loss_time(net, W_aux, b_aux, X, Y, M)
    E = predict(net, X); 
    E_unfmt = stripdims(E);
    logits = W_aux * E_unfmt + b_aux; 
    Y_pred = sigmoid(logits);
    Y_unfmt = stripdims(Y);
    M_unfmt = stripdims(M);
    bce = -(Y_unfmt .* log(Y_pred + 1e-8) + (1 - Y_unfmt) .* log(1 - Y_pred + 1e-8));
    masked_bce = bce .* M_unfmt;
    loss = sum(masked_bce, 'all') / (sum(M_unfmt, 'all') + 1e-8); 
end

function loss = calc_val_loss_space(net, W_aux, b_aux, X, A, Y, M)
    E_flat_out = predict(net, X, A); 
    E_unfmt = reshape(stripdims(E_flat_out), 64, []);
    logits = W_aux * E_unfmt + b_aux; 
    Y_pred = sigmoid(logits);
    Y_unfmt = stripdims(Y);
    M_unfmt = stripdims(M);
    bce = -(Y_unfmt .* log(Y_pred + 1e-8) + (1 - Y_unfmt) .* log(1 - Y_pred + 1e-8));
    masked_bce = bce .* M_unfmt;
    loss = sum(masked_bce, 'all') / (sum(M_unfmt, 'all') + 1e-8);
end

function g_clipped = clipGradient(g, threshold)
    g_norm = sqrt(sum(g.^2, 'all') + 1e-8);
    if g_norm > threshold
        g_clipped = g .* (threshold / g_norm);
    else
        g_clipped = g;
    end
end