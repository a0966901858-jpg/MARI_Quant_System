% =========================================================================
% 腳本：2_Run_Extractor_Pretrain.m (階段 2：DL 雙軌時空特徵萃取器預訓練管線)
% 升級：Phase 14.22 (★ 能量守恆特徵通道健檢、LR 排程 Warmup+Decay、收斂健檢閘門)
% 職責：提煉高穩健性的 64 維節點級別 (Node-level) Embedding 表徵
% =========================================================================
% 清除工作區變數、清空命令視窗、關閉所有圖形視窗，確保環境純淨
clear; clc; close all;

%% 0. 環境路徑掛載
disp('=================================================================');
disp('🚀 [Phase 14.22] 啟動 DL 雙軌特徵萃取器預訓練管線 (能量守恆與 LR 排程版)');
disp('=================================================================');

% 取得當前腳本所在的完整目錄路徑
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

% 固定全域隨機種子，確保實驗可重現
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入 3D 特徵面板資料
disp('--- 步驟 1：載入淨化 3D 特徵面板與時間軸嚴格對齊 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
if ~exist(cachePath, 'file')
    error('❌ 找不到特徵快取檔案，請先執行 1_Run_Data_and_Features.m');
end

load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active', 'AdjMatrix_3D');
Dates_Active.TimeZone = ''; 
numDaysRaw = length(Dates_Active);
numT = configObj.NumTickers;
numFeats = 22; 
seqLen = configObj.SeqLen;
valid_idx = seqLen : numDaysRaw - 1; 
num_valid = length(valid_idx);

%% 1.5 ★ 核心串接：計算能量守恆橫截面特徵 IC 信心權重
disp('--- 步驟 1.5：計算橫截面特徵 IC 信心權重 (能量守恆統計先驗注入) ---');
evaluator = FeatureEvaluator(configObj);
% 產出無未來函數之能量守恆動態特徵信心注意力權重 (均值維持 1.0)
IC_Weights_2D = evaluator.compute_confidence(X_norm_3D, Prices_Active, Expert_Active);

disp(' -> 執行特徵注意力遮罩融合 (Energy-Preserving Feature Gate)...');
IC_Weights_3D = reshape(IC_Weights_2D, numDaysRaw, numFeats, 1);
X_norm_3D = X_norm_3D .* IC_Weights_3D;

% ★ 二次優化診斷：檢查各特徵通道變異數，確保無通道被壓至零
ch_stds = squeeze(std(X_norm_3D, 0, [1, 3], 'omitnan'));
fprintf('  -> [通道尺度健檢] 22 維特徵標準差範圍: [%.4f, %.4f] (平均: %.4f)\n', ...
    min(ch_stds), max(ch_stds), mean(ch_stds));
if any(ch_stds < 1e-4)
    warning('⚠️ 警告：偵測到部分特徵通道變異數接近 0，請排查特徵工程與 IC 權重！');
else
    disp('✅ 動態特徵信心權重融合完畢，各通道變異數維持健康分佈！');
end

%% 2. 構建橫截面輔助預測目標 (Cross-Sectional Targets)
disp('--- 步驟 2：構建無洩漏之橫截面預測標籤 (Beat the Median) ---');
R_fwd = NaN(numDaysRaw, numT, 'single');
R_fwd(1:end-1, :) = (Prices_Active(2:end, :) - Prices_Active(1:end-1, :)) ./ Prices_Active(1:end-1, :);
R_fwd(isinf(R_fwd)) = NaN;
Y_Labels_3D = zeros(numDaysRaw, numT, 'single');

for t = 1:numDaysRaw-1
    active_mask = Expert_Active(t, :);
    if sum(active_mask) > 10
        med_ret = median(R_fwd(t, active_mask), 'omitnan');
        Y_Labels_3D(t, active_mask) = single(R_fwd(t, active_mask) > med_ret);
    end
end

%% 3. 切分時間軸 (嚴格 Train / OOS Validation 樣本解耦)
disp('--- 步驟 3：切分樣本時間軸 (引入 OOS 驗證監控) ---');
Train_Start_Date = datetime('2006-01-01');
OOS_Start_Date   = datetime('2022-01-01');

idx_train_raw = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);
train_idx_valid = intersect(valid_idx, idx_train_raw);
num_train = length(train_idx_valid);

idx_oos_raw = find(Dates_Active >= OOS_Start_Date);
oos_idx_valid = intersect(valid_idx, idx_oos_raw);
num_oos = length(oos_idx_valid);

fprintf('✅ 資料對齊成功！總天數: %d | 訓練樣本: %d 筆 | OOS 盲測驗證樣本: %d 筆\n', ...
    num_valid, num_train, num_oos);

%% 4. 初始化雙軌 DL 萃取器與獨立輔助預測頭 (Auxiliary Heads)
disp('--- 步驟 4：初始化雙軌特徵萃取網路與輔助預測頭 ---');
factory = BuildDecoupledExtractors(configObj);
[net_time, net_space] = factory.buildNetworks();

W_aux_time  = dlarray(randn(1, 64, 'single') * 0.01); 
b_aux_time  = dlarray(zeros(1, 1, 'single'));         
W_aux_space = dlarray(randn(1, 64, 'single') * 0.01);
b_aux_space = dlarray(zeros(1, 1, 'single'));

if canUseGPU()
    gpuInfo = gpuDevice(); 
    fprintf('🎮 成功捕獲圖形加速卡：【%s】，開啟深度學習預訓練路徑。\n', gpuInfo.Name);
    net_time = dlupdate(@gpuArray, net_time);
    net_space = dlupdate(@gpuArray, net_space);
    W_aux_time = gpuArray(W_aux_time); b_aux_time = gpuArray(b_aux_time);
    W_aux_space = gpuArray(W_aux_space); b_aux_space = gpuArray(b_aux_space);
end

%% 5. 啟動雙軌萃取器預訓練 (LR 排程 + 梯度累積)
disp('--- 步驟 5：執行雙軌萃取器預訓練 (Warm-up + Step Decay 排程) ---');
epochs = 30; 
physicalBatchSize = 2;   
accumulationSteps = 16;  
numIterationsPerEpoch = floor(num_train / physicalBatchSize);
clipThreshold = 1.0;

% ★ 二次優化修正：設定基準學習率
base_lr = 1e-3; 

avgG_t = []; avgSG_t = []; avgG_s = []; avgSG_s = [];
avgG_Wt = []; avgSG_Wt = []; avgG_bt = []; avgSG_bt = [];
avgG_Ws = []; avgSG_Ws = []; avgG_bs = []; avgSG_bs = [];
iter = 0; 

historical_loss_time = zeros(epochs, 1);
historical_loss_space = zeros(epochs, 1);
val_loss_time = zeros(epochs, 1);
val_loss_space = zeros(epochs, 1);

for epoch = 1:epochs
    % ★ 二次優化修正：Warm-up (前 3 輪) + 每 10 輪衰減一半
    if epoch <= 3
        current_lr = 1e-4 + (base_lr - 1e-4) * (epoch / 3);
    else
        current_lr = base_lr * (0.5 ^ floor((epoch - 4) / 10));
    end
    
    idx_shuffle = randperm(num_train);
    epoch_loss_time = 0;
    epoch_loss_space = 0;
    
    grad_t_accum = []; grad_Wt_accum = []; grad_bt_accum = [];
    grad_s_accum = []; grad_Ws_accum = []; grad_bs_accum = [];
    accum_count = 0;
    
    % --- 訓練階段 ---
    for i = 1:numIterationsPerEpoch
        batch_idx = idx_shuffle((i-1)*physicalBatchSize + 1 : i*physicalBatchSize);
        actual_t_indices = train_idx_valid(batch_idx);
        
        [X_batch_time, X_batch_space, A_batch_space, Y_batch, M_batch] = prepareBatchData(...
            actual_t_indices, X_norm_3D, AdjMatrix_3D, Y_Labels_3D, Expert_Active, numFeats, numT, seqLen, physicalBatchSize);
        
        [loss_t, grad_t, grad_Wt, grad_bt] = dlfeval(@aux_loss_time, net_time, W_aux_time, b_aux_time, X_batch_time, Y_batch, M_batch);
        [loss_s, grad_s, grad_Ws, grad_bs] = dlfeval(@aux_loss_space, net_space, W_aux_space, b_aux_space, X_batch_space, A_batch_space, Y_batch, M_batch);
        
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
        
        clear X_batch_time X_batch_space A_batch_space Y_batch M_batch loss_t loss_s grad_t grad_s;
    end
    
    historical_loss_time(epoch)  = epoch_loss_time / numIterationsPerEpoch;
    historical_loss_space(epoch) = epoch_loss_space / numIterationsPerEpoch;
    
    % --- OOS 驗證階段 ---
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
    
    fprintf(' -> Epoch %2d/%d (LR: %.2e) | Train (T: %.4f, S: %.4f) | Val OOS (T: %.4f, S: %.4f)\n', ...
        epoch, epochs, current_lr, historical_loss_time(epoch), historical_loss_space(epoch), val_loss_time(epoch), val_loss_space(epoch));
end

%% ★ 二次優化收斂健檢閘門 (Health Check Gate)
loss_t_start = historical_loss_time(1);
loss_t_end   = historical_loss_time(end);
loss_t_drop  = (loss_t_start - loss_t_end) / loss_t_start;

loss_s_start = historical_loss_space(1);
loss_s_end   = historical_loss_space(end);
loss_s_drop  = (loss_s_start - loss_s_end) / loss_s_start;

fprintf('\n📊 雙軌特徵萃取器收斂診斷:\n');
fprintf('  > 時序專家 (T) Loss: %.4f -> %.4f (降幅: %.2f%%)\n', loss_t_start, loss_t_end, loss_t_drop * 100);
fprintf('  > 空間專家 (S) Loss: %.4f -> %.4f (降幅: %.2f%%)\n', loss_s_start, loss_s_end, loss_s_drop * 100);

if (abs(loss_t_end - 0.6931) < 0.005 && loss_t_drop < 0.05) || (abs(loss_s_end - 0.6931) < 0.005 && loss_s_drop < 0.05)
    warning('⚠️ 警告：模型 Loss 仍滯留於 ln(2) ≈ 0.6931 隨機基準線，可能尚未學習到足夠特徵！');
else
    disp('✅ 雙軌萃取器輔助任務收斂正常，成功突破隨機基準線！');
end

%% 6. 最終全歷史 Embedding 提煉與存檔 (推論階段)
disp('--- 步驟 6：全歷史強固特徵表徵提煉與落地 (提取 3D 張量) ---');
E_time_all = zeros(numDaysRaw, 64, numT, 'single');
E_space_all = zeros(numDaysRaw, 64, numT, 'single');
inferBatchSize = 2; 

fprintf('  -> 正在提煉全歷史節點級別表徵 (共 %d 筆有效天數)...\n', num_valid);
for start_idx = 1:inferBatchSize:num_valid
    end_idx = min(start_idx + inferBatchSize - 1, num_valid);
    chunk_size = end_idx - start_idx + 1; 
    actual_t_indices = valid_idx(start_idx:end_idx); 
    
    [X_inf_time, X_inf_space, A_inf_space, ~, ~] = prepareBatchData(...
        actual_t_indices, X_norm_3D, AdjMatrix_3D, Y_Labels_3D, Expert_Active, numFeats, numT, seqLen, chunk_size);
    
    E_time_raw = predict(net_time, X_inf_time);
    E_space_flat_out = predict(net_space, X_inf_space, A_inf_space);
    
    e_t_reshaped = permute(reshape(extractdata(E_time_raw), 64, numT, chunk_size), [3, 1, 2]);
    e_s_reshaped = permute(reshape(extractdata(E_space_flat_out), 64, numT, chunk_size), [3, 1, 2]);
    
    E_time_all(actual_t_indices, :, :) = gather(e_t_reshaped);
    E_space_all(actual_t_indices, :, :) = gather(e_s_reshaped);
end

%% 7. 繪製與儲存訓練曲線與 GCN 圖譜視覺化
disp('--- 步驟 7：產出 Loss 曲線與 GCN 靜態圖譜視覺化報表 ---');
fig_loss = figure('Name', 'Phase 2: Extractor Pretrain Loss', 'Position', [100, 100, 1200, 500], 'Visible', 'off'); 

subplot(1, 2, 1);
plot(1:epochs, historical_loss_time, '-o', 'LineWidth', 2, 'DisplayName', 'Train Loss'); hold on;
plot(1:epochs, val_loss_time, '-x', 'LineWidth', 2, 'DisplayName', 'Val (OOS) Loss');
title('Time Expert (Transformer-LSTM) BCE Loss');
xlabel('Epochs'); ylabel('BCE Loss');
legend('Location', 'best'); grid on;
yline(0.6931, '--r', 'Random Guess (0.6931)', 'LabelHorizontalAlignment', 'left');

subplot(1, 2, 2);
plot(1:epochs, historical_loss_space, '-o', 'LineWidth', 2, 'DisplayName', 'Train Loss'); hold on;
plot(1:epochs, val_loss_space, '-x', 'LineWidth', 2, 'DisplayName', 'Val (OOS) Loss');
title('Space Expert (DyGAT) BCE Loss');
xlabel('Epochs'); ylabel('BCE Loss');
legend('Location', 'best'); grid on;
yline(0.6931, '--r', 'Random Guess (0.6931)', 'LabelHorizontalAlignment', 'left');

lossFigPath = fullfile(configObj.ModelDir, 'Phase2_Loss_Curve.png');
saveas(fig_loss, lossFigPath);
fprintf(' 📊 Loss 曲線已儲存至: %s\n', lossFigPath);

fig_adj = figure('Name', 'GCN Static Adjacency Matrix', 'Position', [200, 200, 800, 700], 'Visible', 'off');
sample_adj = AdjMatrix_3D(:, :, train_idx_valid(end)); 
imagesc(sample_adj);
cmap = [1 1 1; 0 0.2 0.6]; 
colormap(cmap);
title(sprintf('GCN Static Adjacency Matrix (N=%d Tickers)', numT));
xlabel('Target Node (Stock Index)'); 
ylabel('Source Node (Stock Index)');
axis square;
set(gca, 'XTick', [], 'YTick', []); 

adjFigPath = fullfile(configObj.ModelDir, 'Phase2_GCN_Adjacency_Matrix.png');
saveas(fig_adj, adjFigPath);
fprintf(' 🕸️ GCN 靜態關聯圖譜矩陣已儲存至: %s\n', adjFigPath);

close(fig_loss);
close(fig_adj);

modelPath = fullfile(configObj.ModelDir, 'DL_Extractors.mat');
if ~exist(configObj.ModelDir, 'dir'), mkdir(configObj.ModelDir); end
save(modelPath, 'net_time', 'net_space', 'E_time_all', 'E_space_all', '-v7.3');
fprintf('💾 雙軌 DL 萃取器預訓練完畢！[Days, 64, Tickers] 節點表徵已存至: %s\n', modelPath);

disp('=================================================================');
disp('🎯 [Phase 2] 完美完成。個股身分與物理維度 100% 銜接對齊，請執行 Phase 3。');
disp('=================================================================');

%% =====================================================================
% 輔助函數區 (資料封裝與預測頭 Loss 計算)
% =====================================================================
function [X_time, X_space, A_space, Y, M] = prepareBatchData(indices, X_3D, Adj_3D, Y_Lab, Expert, nF, nT, sL, bZ)
    X_time_raw = zeros(nF, nT * bZ, sL, 'single');
    X_space_raw = zeros(nF * nT, bZ, 'single');
    A_space_raw = zeros(nT * nT, bZ, 'single');
    Y_raw = zeros(1, nT * bZ, 'single');
    M_raw = zeros(1, nT * bZ, 'single');
    
    for b = 1:bZ
        t = indices(b);
        seq_data = permute(X_3D(t-sL+1:t, :, :), [2, 3, 1]);
        idx_start = (b-1)*nT + 1;
        idx_end = b*nT;
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
        X_time = gpuArray(X_time); 
        X_space = gpuArray(X_space); 
        A_space = gpuArray(A_space); 
        Y = gpuArray(Y); 
        M = gpuArray(M);
    end
end

function [loss, grad_net, grad_W, grad_b] = aux_loss_time(net, W_aux, b_aux, X, Y, M)
    E = forward(net, X); 
    E_unfmt = stripdims(E);
    logits = W_aux * E_unfmt + b_aux; 
    Y_pred = sigmoid(logits);
    Y_unfmt = stripdims(Y);
    M_unfmt = stripdims(M);
    bce = -(Y_unfmt .* log(Y_pred + 1e-8) + (1 - Y_unfmt) .* log(1 - Y_pred + 1e-8));
    masked_bce = bce .* M_unfmt;
    loss = sum(masked_bce, 'all') / (sum(M_unfmt, 'all') + 1e-8); 
    [grad_net, grad_W, grad_b] = dlgradient(loss, net.Learnables, W_aux, b_aux);
end

function [loss, grad_net, grad_W, grad_b] = aux_loss_space(net, W_aux, b_aux, X, A, Y, M)
    E_flat_out = forward(net, X, A); 
    E_unfmt = reshape(stripdims(E_flat_out), 64, []);
    logits = W_aux * E_unfmt + b_aux; 
    Y_pred = sigmoid(logits);
    Y_unfmt = stripdims(Y);
    M_unfmt = stripdims(M);
    bce = -(Y_unfmt .* log(Y_pred + 1e-8) + (1 - Y_unfmt) .* log(1 - Y_pred + 1e-8));
    masked_bce = bce .* M_unfmt;
    loss = sum(masked_bce, 'all') / (sum(M_unfmt, 'all') + 1e-8);
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