% =========================================================================
% 腳本：2_Run_Extractor_Pretrain.m (階段 2：DL 雙軌時空特徵萃取器預訓練管線)
% 升級：Phase 15.5 Stage 2.5 方案 (★ mrg32k3a 獨立子串流注入、He 權重初始化與
%       Minibatch/驗證集抽樣確定性鎖定、60D 遠期 Horizon、連續標準化超額報酬標籤、
%       Huber + Soft-IC 複合連續迴歸損失、VICReg 變異數保底防坍縮、60天動態 Embargo 隔離)
% 職責：使用 18 維個股微觀/相對特徵提煉 64 維 Embedding，打破二元中位數硬切天花板
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 DL 雙軌特徵萃取器預訓練管線 (mrg32k3a 確定性子串流版)');
disp('=================================================================');

%% 0. 環境路徑掛載 (規範化階層回溯解析與路徑重新整理)
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
        error('❌ 找不到 MARI_Quant_System 專案根目錄 (包含 configs/)，請確認執行路徑！');
    end
    projectRoot = parentDir;
end

addpath(genpath(fullfile(projectRoot, 'configs')));
addpath(genpath(fullfile(projectRoot, 'data')));
addpath(genpath(fullfile(projectRoot, 'agents')));
addpath(genpath(fullfile(projectRoot, 'models')));
addpath(genpath(fullfile(projectRoot, 'envs')));
addpath(genpath(fullfile(projectRoot, 'utils')));

rehash path;
rehash;

if exist('Config', 'class') ~= 8
    error('❌ 已掛載路徑但仍找不到 Config 類別，請檢查 configs/Config.m 權限或語法錯誤！');
end

configObj = Config();

% ★ 核心修復 1：由 Config 統一生產 mrg32k3a 隨機數引擎，並設為全域主串流 (Substream = 1)
stream = configObj.getRandStream(1);
RandStream.setGlobalStream(stream);
disp('🔒 已成功掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定預訓練確定性。');

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
numFeats_All = size(X_norm_3D, 2); 
seqLen = configObj.SeqLen;

% ★ Direction 3 升級：遠期預測視窗延長為 60 日 (一季動能趨勢)
horizon = 60; 
valid_idx = seqLen : (numDaysRaw - horizon); 
num_valid = length(valid_idx);

fprintf('  -> 宇宙規模: %d 檔 | 交易天數: %d 天 | 原始特徵維度: %d 維 | 預測視窗: %d 日\n', ...
    numT, numDaysRaw, numFeats_All, horizon);

%% 1.5 核心串接：18 維特徵切片、1D/60D 雙軌 HAC-ICIR 健檢與特徵注意力閘門
disp('--- 步驟 1.5：特徵切片 (剝離 Macro)、1D/60D HAC-ICIR 健檢與特徵注意力加權 ---');
numExtractorFeats = 3 + configObj.NumMicroFeatures; % 18 維 (Rel 3 + Micro 15)
X_norm_3D_extractor_raw = X_norm_3D(:, 1:numExtractorFeats, :);

evaluator = FeatureEvaluator(configObj);
feat_names_18d = [{'Beta', 'Corr', 'RelStrength'}, ...
    {'R1', 'R5', 'R20', 'Vol20', 'IdioVol20', 'VolRatio', 'Amihud20', 'SMA20', 'SMA60', ...
     'MACD_Hist', 'RSI', 'OBV20', 'HL_Spread', 'Dist_H20', 'Dist_H252'}];

% 1 日 Horizon 微結構訊號
[IC_Weights_2D, Raw_IC_Weight, Daily_IC_1D] = evaluator.compute_confidence(X_norm_3D_extractor_raw, Prices_Active, Expert_Active, 1);
evaluator.report_icir_ranking(Daily_IC_1D, feat_names_18d, 0.05, '1D (Micro-Structure)');
evaluator.report_ic_ranking(Raw_IC_Weight, feat_names_18d);

% ★ 60 日 Horizon HAC-ICIR 健檢 (與當前萃取器目標嚴格對齊)
[~, ~, Daily_IC_60D] = evaluator.compute_confidence(X_norm_3D_extractor_raw, Prices_Active, Expert_Active, horizon);
evaluator.report_icir_ranking(Daily_IC_60D, feat_names_18d, 0.05, sprintf('%dD (Aligned with Extractor Target)', horizon));

disp(' -> 執行特徵注意力遮罩融合 (Energy-Preserving Feature Gate)...');
IC_Weights_3D = reshape(IC_Weights_2D, numDaysRaw, numExtractorFeats, 1);
X_norm_3D_extractor = X_norm_3D_extractor_raw .* IC_Weights_3D;

ch_stds = squeeze(std(X_norm_3D_extractor, 0, [1, 3], 'omitnan'));
fprintf('  -> [萃取器通道健檢] 18 維特徵標準差範圍: [%.4f, %.4f] (平均: %.4f)\n', ...
    min(ch_stds), max(ch_stds), mean(ch_stds));

%% 2. 構建橫截面連續預測目標 (60 日遠期標準化超額報酬 Z-Score)
disp('--- 步驟 2：構建橫截面連續超額報酬標籤 (Direction 2: 60D Continuous Z-Score) ---');
R_fwd = NaN(numDaysRaw, numT, 'single');
R_fwd(1:end-horizon, :) = (Prices_Active(1+horizon:end, :) - Prices_Active(1:end-horizon, :)) ...
                          ./ (Prices_Active(1:end-horizon, :) + 1e-8);
R_fwd(isnan(R_fwd) | isinf(R_fwd)) = NaN;

Y_Labels_3D = zeros(numDaysRaw, numT, 'single');
for t = 1:numDaysRaw-horizon
    active_mask = Expert_Active(t, :) & ~isnan(R_fwd(t, :)) & ~isinf(R_fwd(t, :));
    if sum(active_mask) >= 10
        r_t = R_fwd(t, active_mask);
        mu_t  = mean(r_t, 'omitnan');
        std_t = std(r_t, 0, 'omitnan') + 1e-6;
        % ★ 橫截面 Z-Score 標準化：保留連續幅度資訊
        Y_Labels_3D(t, active_mask) = (r_t - mu_t) ./ std_t;
    end
end
Y_Labels_3D(isnan(Y_Labels_3D) | isinf(Y_Labels_3D)) = 0;

%% 3. 切分時間軸 (嚴格防洩漏：60 天 Embargo 隔離)
disp('--- 步驟 3：切分時間軸 (嚴格 IS 內 Purged Embargo 跨體制驗證集) ---');
Train_Start_Date = datetime('2006-01-01');
OOS_Start_Date   = datetime('2022-01-01');
idx_train_raw = find(Dates_Active >= Train_Start_Date & Dates_Active < OOS_Start_Date);
is_idx_valid = intersect(valid_idx, idx_train_raw);

regime_windows = { ...
    struct('name','2008 金融海嘯', 'start', datetime('2007-09-01'), 'end', datetime('2009-06-01')), ...
    struct('name','2015-16 盤整修正', 'start', datetime('2015-06-01'), 'end', datetime('2016-06-01')), ...
    struct('name','2020 COVID崩盤', 'start', datetime('2020-01-01'), 'end', datetime('2020-12-01')) ...
};

val_idx_valid = [];
embargo_val_idx = [];
% ★ 隔離期必須至少等於標籤視窗 H (60 天)
embargo = max(20, horizon); 

for i = 1:length(regime_windows)
    w = regime_windows{i};
    idx_w = find(Dates_Active >= w.start & Dates_Active <= w.end);
    val_idx_valid = [val_idx_valid; intersect(valid_idx, idx_w)];
    
    idx_embargo = find(Dates_Active >= (w.start - caldays(embargo)) & Dates_Active <= (w.end + caldays(embargo)));
    embargo_val_idx = [embargo_val_idx; intersect(valid_idx, idx_embargo)];
end

val_idx_valid = unique(val_idx_valid);
embargo_val_idx = unique(embargo_val_idx);
train_idx_valid = setdiff(is_idx_valid, embargo_val_idx);
num_train = length(train_idx_valid);
num_val = length(val_idx_valid);

fprintf('✅ Purged 多體制時間軸劃分成功！訓練樣本: %d 筆 | 跨體制驗證樣本: %d 筆 (嚴格 Embargo: %d 天)\n', ...
    num_train, num_val, embargo);

%% 4. 初始化雙軌 DL 萃取器與連續迴歸線性預測頭
disp('--- 步驟 4：初始化雙軌特徵萃取網路 (連續預測頭 - mrg32k3a 確定性初值) ---');
factory = BuildDecoupledExtractors(configObj, numExtractorFeats);
[net_time, net_space] = factory.buildNetworks();

% ★ 核心修復 2：使用 stream 產生確定性 He 初值權重
W_aux_time  = dlarray(randn(stream, 1, 64, 'single') * 0.01); 
b_aux_time  = dlarray(zeros(1, 1, 'single'));         
W_aux_space = dlarray(randn(stream, 1, 64, 'single') * 0.01);
b_aux_space = dlarray(zeros(1, 1, 'single'));

if canUseGPU()
    gpuInfo = gpuDevice(); 
    fprintf('🎮 成功捕獲圖形加速卡：【%s】，開啟深度學習全量預訓練。\n', gpuInfo.Name);
    net_time = dlupdate(@gpuArray, net_time);
    net_space = dlupdate(@gpuArray, net_space);
    W_aux_time = gpuArray(W_aux_time); b_aux_time = gpuArray(b_aux_time);
    W_aux_space = gpuArray(W_aux_space); b_aux_space = gpuArray(b_aux_space);
end

%% 5. 啟動雙軌萃取器預訓練 (Huber + Soft-IC 複合損失 + VICReg 保底)
disp('--- 步驟 5：執行雙軌萃取器預訓練 (Soft-IC 連續損失 + 跨體制早停) ---');
epochs = 30; 
physicalBatchSize = 2;   
accumulationSteps = 16;  
numIterationsPerEpoch = floor(num_train / physicalBatchSize);
clipThreshold = 1.0;
base_lr        = 1e-3; 
l2_lambda      = configObj.DL_L2_Regularization;      % 1e-5
var_lambda     = configObj.DL_VarianceFloorLambda;    % 0.05
var_target     = configObj.DL_VarianceFloorTarget;    % 1.0
patience_limit = configObj.DL_EarlyStoppingPatience;
ic_loss_weight = 0.5;

avgG_t = []; avgSG_t = []; avgG_s = []; avgSG_s = [];
avgG_Wt = []; avgSG_Wt = []; avgG_bt = []; avgSG_bt = [];
avgG_Ws = []; avgSG_Ws = []; avgG_bs = []; avgSG_bs = [];
iter = 0; 

historical_loss_time  = zeros(epochs, 1);
historical_loss_space = zeros(epochs, 1);
val_loss_time         = zeros(epochs, 1);
val_loss_space        = zeros(epochs, 1);
best_val_loss = inf; 
patience = 0; 
best_net_time  = net_time; 
best_net_space = net_space;
min_healthy_var_t = 0.05;
min_healthy_var_s = 0.10;

for epoch = 1:epochs
    if epoch <= 3
        current_lr = 1e-4 + (base_lr - 1e-4) * (epoch / 3);
    else
        current_lr = base_lr * (0.5 ^ floor((epoch - 4) / 10));
    end
    
    % ★ 核心修復 3：使用 stream 進行 Epoch 訓練樣本隨機置換
    idx_shuffle = randperm(stream, num_train);
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
            actual_t_indices, X_norm_3D_extractor, AdjMatrix_3D, Y_Labels_3D, Expert_Active, numExtractorFeats, numT, seqLen, physicalBatchSize);
        
        % 計算時序專家 Huber + Soft-IC 連續損失與梯度
        [loss_t, grad_t, grad_Wt, grad_bt] = dlfeval(@(n,w,b,x,y,m) aux_loss_time(...
            n, w, b, x, y, m, l2_lambda, var_lambda, var_target, numT, physicalBatchSize, ic_loss_weight), ...
            net_time, W_aux_time, b_aux_time, X_batch_time, Y_batch, M_batch);
        
        % 計算空間專家 Huber + Soft-IC 連續損失與梯度
        [loss_s, grad_s, grad_Ws, grad_bs] = dlfeval(@(n,w,b,x,a,y,m) aux_loss_space(...
            n, w, b, x, a, y, m, l2_lambda, numT, physicalBatchSize, ic_loss_weight), ...
            net_space, W_aux_space, b_aux_space, X_batch_space, A_batch_space, Y_batch, M_batch);
        
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
    
    % --- 多體制連續迴歸驗證階段 ---
    % ★ 核心修復 4：使用 stream 進行多體制驗證樣本隨機抽樣
    val_samples = min(num_val, 64); 
    val_idx_shuffle = randperm(stream, num_val, val_samples);
    actual_val_indices = val_idx_valid(val_idx_shuffle);
    
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
            chunk_indices, X_norm_3D_extractor, AdjMatrix_3D, Y_Labels_3D, Expert_Active, numExtractorFeats, numT, seqLen, chunk_size);
        
        v_loss_t_chunk = calc_val_loss_time(net_time, W_aux_time, b_aux_time, X_val_time, Y_val, M_val, numT, chunk_size, ic_loss_weight);
        v_loss_s_chunk = calc_val_loss_space(net_space, W_aux_space, b_aux_space, X_val_space, A_val_space, Y_val, M_val, numT, chunk_size, ic_loss_weight);
        
        temp_val_loss_t = temp_val_loss_t + extractdata(v_loss_t_chunk) * chunk_size;
        temp_val_loss_s = temp_val_loss_s + extractdata(v_loss_s_chunk) * chunk_size;
        
        clear X_val_time X_val_space A_val_space Y_val M_val v_loss_t_chunk v_loss_s_chunk;
    end
    
    val_loss_time(epoch)  = temp_val_loss_t / val_samples;
    val_loss_space(epoch) = temp_val_loss_s / val_samples;
    
    % 診斷埋點：確保未發生表徵坍塌
    e_t_sample = extractdata(predict(net_time, X_batch_time));
    e_s_sample = extractdata(reshape(predict(net_space, X_batch_space, A_batch_space), 64, []));
    var_e_t = var(e_t_sample(:), 'omitnan');
    var_e_s = var(e_s_sample(:), 'omitnan');
    
    fprintf(' -> Epoch %2d/%d (LR: %.2e) | Train ContLoss (T: %.4f, S: %.4f) | Val Multi-Regime (T: %.4f, S: %.4f)\n', ...
        epoch, epochs, current_lr, historical_loss_time(epoch), historical_loss_space(epoch), val_loss_time(epoch), val_loss_space(epoch));
    fprintf('    [診斷] GradNorm(T: %.2e, S: %.2e) | E_Var(T: %.4f, S: %.4f)\n', ...
        last_gnorm_t, last_gnorm_s, var_e_t, var_e_s);
    
    if var_e_t < min_healthy_var_t
        fprintf('    ⚠️ [警告] E_Var(T)=%.4f 低於健康閾值 (%.2f)，疑似發生表徵坍縮！\n', var_e_t, min_healthy_var_t);
    end
    if var_e_s < min_healthy_var_s
        fprintf('    ⚠️ [警告] E_Var(S)=%.4f 低於健康閾值 (%.2f)，疑似發生表徵坍縮！\n', var_e_s, min_healthy_var_s);
    end
    
    % 快照保存健康變異數閘門
    combined_val = val_loss_time(epoch) + val_loss_space(epoch);
    is_healthy = (var_e_t >= min_healthy_var_t) && (var_e_s >= min_healthy_var_s);
    
    if is_healthy && (combined_val < best_val_loss - 1e-4)
        best_val_loss = combined_val;
        patience = 0;
        best_net_time = net_time; 
        best_net_space = net_space;
    else
        patience = patience + 1;
        if ~is_healthy
            fprintf('    ⏭️  [跳過快照] 本輪因表徵變異數未達健康門檻不列入候選！\n');
        end
    end
    
    if patience >= patience_limit
        fprintf('🛑 [Early Stopping] 多體制驗證損失連續 %d 輪未改善，提前於 Epoch %d 終止訓練並回滾最佳快照！\n', patience_limit, epoch);
        net_time = best_net_time; 
        net_space = best_net_space;
        break;
    end
    
    clear X_batch_time X_batch_space A_batch_space Y_batch M_batch e_t_sample e_s_sample;
end

%% ★ 收斂健檢閘門 (連續損失與表徵健全度)
loss_t_end = val_loss_time(min(epoch, epochs)); 
loss_s_end = val_loss_space(min(epoch, epochs));
fprintf('\n📊 全量雙軌萃取器收斂診斷 (跨體制驗證集基準):\n');
fprintf('  > 時序專家 (T) 最終 Val Continuous Loss: %.4f\n', loss_t_end);
fprintf('  > 空間專家 (S) 最終 Val Continuous Loss: %.4f\n', loss_s_end);

if isnan(loss_t_end) || isinf(loss_t_end) || isnan(loss_s_end) || isinf(loss_s_end)
    warning('❌ 警告：模型最終 Val Loss 存在 NaN 或 Inf，訓練異常！');
else
    disp('✅ 全量雙軌萃取器連續目標訓練完畢，成功建立連續超額報酬梯度流！');
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
        actual_t_indices, X_norm_3D_extractor, AdjMatrix_3D, Y_Labels_3D, Expert_Active, numExtractorFeats, numT, seqLen, chunk_size);
    
    E_time_raw = predict(net_time, X_inf_time);
    E_space_flat_out = predict(net_space, X_inf_space, A_inf_space);
    
    e_t_reshaped = permute(reshape(extractdata(E_time_raw), 64, numT, chunk_size), [3, 1, 2]);
    e_s_reshaped = permute(reshape(extractdata(E_space_flat_out), 64, numT, chunk_size), [3, 1, 2]);
    
    E_time_all(actual_t_indices, :, :) = gather(e_t_reshaped);
    E_space_all(actual_t_indices, :, :) = gather(e_s_reshaped);
end

%% 7. 繪製與儲存訓練曲線與 GCN 圖譜視覺化 (標準白底黑字學術格式)
disp('--- 步驟 7：產出 Loss 曲線與 GCN 靜態圖譜視覺化報表 (白底黑字) ---');
fig_loss = figure('Name', 'Phase 2: Extractor Pretrain Continuous Loss', ...
    'Position', [100, 100, 1200, 500], 'Color', 'w', 'Visible', 'off'); 
set(fig_loss, 'InvertHardcopy', 'off');

% 子圖 1：時序專家
subplot(1, 2, 1);
plot(1:epochs, historical_loss_time, '-o', 'LineWidth', 1.8, 'Color', '#D95319', 'DisplayName', 'Train Continuous Loss'); hold on;
plot(1:epochs, val_loss_time, '-x', 'LineWidth', 1.8, 'Color', '#0072BD', 'DisplayName', 'Val (Multi-Regime) Loss');
title('Time Expert (Continuous Soft-IC + Huber)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epochs', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Continuous Loss', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 10);
grid on; box on;

% 子圖 2：空間專家
subplot(1, 2, 2);
plot(1:epochs, historical_loss_space, '-o', 'LineWidth', 1.8, 'Color', '#D95319', 'DisplayName', 'Train Continuous Loss'); hold on;
plot(1:epochs, val_loss_space, '-x', 'LineWidth', 1.8, 'Color', '#0072BD', 'DisplayName', 'Val (Multi-Regime) Loss');
title('Space Expert (Continuous Soft-IC + Huber)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epochs', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Continuous Loss', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 10);
grid on; box on;

lossFigPath = fullfile(configObj.ModelDir, 'Phase2_Loss_Curve.png');
exportgraphics(fig_loss, lossFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf(' 📊 Loss 曲線 (白底黑字) 已儲存至: %s\n', lossFigPath);

% 圖 2：GCN 靜態關聯圖譜矩陣
fig_adj = figure('Name', 'GCN Static Adjacency Matrix', ...
    'Position', [200, 200, 750, 650], 'Color', 'w', 'Visible', 'off');
set(fig_adj, 'InvertHardcopy', 'off');
sample_adj = AdjMatrix_3D(:, :, train_idx_valid(end)); 
imagesc(sample_adj);
cmap = [1.0 1.0 1.0; 0.0 0.15 0.45]; 
colormap(fig_adj, cmap);
title(sprintf('GCN Static Adjacency Matrix (N = %d Tickers)', numT), ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Target Node (Stock Index)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k'); 
ylabel('Source Node (Stock Index)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
axis square;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'XTick', [], 'YTick', [], 'FontName', 'Helvetica');
box on;

adjFigPath = fullfile(configObj.ModelDir, 'Phase2_GCN_Adjacency_Matrix.png');
exportgraphics(fig_adj, adjFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf(' 🕸️ GCN 靜態關聯圖譜矩陣 (白底黑字) 已儲存至: %s\n', adjFigPath);
close(fig_loss);
close(fig_adj);

% 儲存模型權重與表徵
modelPath = fullfile(configObj.ModelDir, 'DL_Extractors.mat');
if ~exist(configObj.ModelDir, 'dir'), mkdir(configObj.ModelDir); end
save(modelPath, 'net_time', 'net_space', 'E_time_all', 'E_space_all', '-v7.3');
fprintf('💾 雙軌 DL 萃取器預訓練完畢！[Days, 64, Tickers] 節點表徵已存至: %s\n', modelPath);

disp('=================================================================');
disp('🎯 [Phase 2] 連續迴歸萃取器預訓練完成！請執行 Phase 3。');
disp('=================================================================');

%% =====================================================================
% 輔助函數區 (Huber + Soft-IC 連續迴歸、VICReg 變異數保底與 L2 解耦)
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

function [loss, grad_net, grad_W, grad_b] = aux_loss_time(net, W_aux, b_aux, X, Y, M, l2_lambda, var_lambda, var_target, nT, bZ, ic_w)
    E = forward(net, X); 
    E_unfmt = stripdims(E);
    Y_pred = W_aux * E_unfmt + b_aux; 
    
    cont_loss = compute_continuous_loss_internal(Y_pred, Y, M, nT, bZ, ic_w);
    
    % 變異數保底正則化 (VICReg 風格)
    mu_dim = mean(E_unfmt, 2);
    var_per_dim = mean((E_unfmt - mu_dim).^2, 2);
    std_per_dim = sqrt(var_per_dim + 1e-4);
    var_penalty = mean(max(0, var_target - std_per_dim));
    
    % L2 懲罰排除最終表徵輸出層 ('E_time')
    l2_penalty = 0;
    learnables = net.Learnables;
    for r = 1:height(learnables)
        if ~contains(learnables.Layer{r}, 'E_time')
            w = learnables.Value{r};
            l2_penalty = l2_penalty + sum(w(:).^2);
        end
    end
    l2_penalty = l2_penalty + sum(W_aux(:).^2);
    
    loss = cont_loss + 0.5 * l2_lambda * l2_penalty + var_lambda * var_penalty;
    [grad_net, grad_W, grad_b] = dlgradient(loss, net.Learnables, W_aux, b_aux);
end

function [loss, grad_net, grad_W, grad_b] = aux_loss_space(net, W_aux, b_aux, X, A, Y, M, l2_lambda, nT, bZ, ic_w)
    E_flat_out = forward(net, X, A); 
    E_unfmt = reshape(stripdims(E_flat_out), 64, []);
    Y_pred = W_aux * E_unfmt + b_aux; 
    
    cont_loss = compute_continuous_loss_internal(Y_pred, Y, M, nT, bZ, ic_w);
    
    l2_penalty = 0;
    learnables = net.Learnables;
    for r = 1:height(learnables)
        w = learnables.Value{r};
        l2_penalty = l2_penalty + sum(w(:).^2);
    end
    l2_penalty = l2_penalty + sum(W_aux(:).^2);
    
    loss = cont_loss + 0.5 * l2_lambda * l2_penalty;
    [grad_net, grad_W, grad_b] = dlgradient(loss, net.Learnables, W_aux, b_aux);
end

function loss = calc_val_loss_time(net, W_aux, b_aux, X, Y, M, nT, bZ, ic_w)
    E = predict(net, X); 
    E_unfmt = stripdims(E);
    Y_pred = W_aux * E_unfmt + b_aux; 
    loss = compute_continuous_loss_internal(Y_pred, Y, M, nT, bZ, ic_w);
end

function loss = calc_val_loss_space(net, W_aux, b_aux, X, A, Y, M, nT, bZ, ic_w)
    E_flat_out = predict(net, X, A); 
    E_unfmt = reshape(stripdims(E_flat_out), 64, []);
    Y_pred = W_aux * E_unfmt + b_aux; 
    loss = compute_continuous_loss_internal(Y_pred, Y, M, nT, bZ, ic_w);
end

function loss = compute_continuous_loss_internal(y_pred, y_true, mask, nT, bZ, ic_weight)
    yp = stripdims(y_pred); yp = yp(:);
    yt = stripdims(y_true); yt = yt(:);
    m  = logical(stripdims(mask)); m = m(:);
    
    % 1. 元素級 Huber 迴歸損失
    yp_m = yp(m);
    yt_m = yt(m);
    delta = 0.1;
    err = abs(yp_m - yt_m);
    is_small = err <= delta;
    huber = mean(is_small .* (0.5 * err.^2) + (~is_small) .* (delta * (err - 0.5 * delta)), 'all');
    
    % 2. 每日橫截面 Soft-IC 損失
    yp_mat = reshape(yp, nT, bZ);
    yt_mat = reshape(yt, nT, bZ);
    m_mat  = reshape(m,  nT, bZ);
    
    ic_sum = 0;
    valid_cnt = 0;
    for b = 1:bZ
        mb = m_mat(:, b);
        if sum(mb) >= 5
            yp_b = yp_mat(mb, b);
            yt_b = yt_mat(mb, b);
            
            yp_c = yp_b - mean(yp_b);
            yt_c = yt_b - mean(yt_b);
            
            cov_xy  = sum(yp_c .* yt_c);
            norm_xy = sqrt(sum(yp_c.^2) * sum(yt_c.^2) + 1e-6);
            
            ic_sum = ic_sum + cov_xy / norm_xy;
            valid_cnt = valid_cnt + 1;
        end
    end
    
    if valid_cnt > 0
        ic_loss = -(ic_sum / valid_cnt);
    else
        ic_loss = sum(yp, 'all') * 0;
    end
    
    loss = huber + ic_weight * (1.0 + ic_loss);
end

function g_clipped = clipGradient(g, threshold)
    g_norm = sqrt(sum(g.^2, 'all') + 1e-8);
    if g_norm > threshold
        g_clipped = g .* (threshold / g_norm);
    else
        g_clipped = g;
    end
end