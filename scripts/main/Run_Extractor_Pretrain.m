% =========================================================================
% 腳本：2_Run_Extractor_Pretrain.m (階段 2：DL 特徵萃取器預訓練管線)
% 升級：Phase 15.5 深度表徵容量增強 ＋ 200 輪極限平坦收斂與解耦退火生產基準版
% 核心升級：
%   1. 【解耦退火週期 (Decoupled Cosine Decay)】：餘弦退火基數固定為 120 輪，
%      不隨總訓練輪數縮放，保證前 1～120 輪學習率、梯度與損失數值絕對一致 (前綴不變性)；
%      超期 (第 121～200 輪) 鎖定 min_lr (1e-5)，以微步長爬行展示絕對水平漸近線[cite: 8, 9]。
%   2. 【權重指數移動平均 (EMA Shadow Weights)】：維護平滑影子權重 (Beta=0.995)，
%      驗證評估與歷史快照全面採用 EMA 網絡，徹底撫平 Adam 動量超調引發的鋸齒回彈[cite: 6]。
%   3. 【預熱保護期 (Warmup Immunity)】：前 5 輪禁止鎖定快照與累積早停，杜絕未成熟隨機初值誤殺[cite: 1, 3]。
%   4. 【多體制驗證取樣擴增】：驗證抽樣擴大至 256 筆 (覆蓋率 > 27%)，消除小樣本方差雜訊[cite: 1]。
%   5. 【物理批次吞吐優化】：physicalBatchSize 提升至 8、累積步數縮至 4，提升 GPU 利用率[cite: 1, 3]。
%   6. 【特徵直通殘差與剪枝相容】：配合 BuildDecoupledExtractors Highway 通路強化容量，
%      並以實際完成輪次精準截斷 Loss 圖表，杜絕斷崖暴跌繪圖 Bug[cite: 1, 2]。
% 職責：使用 18 維個股微觀/相對特徵提煉 64 維 Embedding，學習橫截面連續排序表徵
% =========================================================================
clear; clc; close all;

% 啟動全域總計時器
t_total_start = tic;
disp('=================================================================');
disp('🚀 [Phase 15.5] 啟動 DL 特徵萃取器預訓練管線 (權重 EMA ＋ 200輪極限收斂版)');
disp('=================================================================');

%% 0. 環境路徑掛載 (規範化階層回溯解析與路徑重新整理)
t_step0 = tic;
disp('--- 步驟 0：環境路徑掛載、Config 載入與隨機串流鎖定 ---');
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

% 由 Config 統一生產 mrg32k3a 隨機數引擎，並設為全域主串流 (Substream = 1)
stream = configObj.getRandStream(1);
RandStream.setGlobalStream(stream);
disp('🔒 已成功掛載 mrg32k3a 主隨機串流 (Substream=1)，鎖定預訓練確定性。');
time_step0 = toc(t_step0);
fprintf('⏱️ [步驟 0 完成] 耗時: %.2f 秒\n\n', time_step0);

%% 1. 載入 3D 特徵面板資料
t_step1 = tic;
disp('--- 步驟 1：載入淨化 3D 特徵面板與時間軸嚴格對齊 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
if ~exist(cachePath, 'file')
    error('❌ 找不到特徵快取檔案，請先執行 1_Run_Data_and_Features.m');
end

load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active', 'AdjMatrix_3D');
Dates_Active.TimeZone = ''; 
numDaysRaw   = length(Dates_Active);
numT         = configObj.NumTickers;
numFeats_All = size(X_norm_3D, 2); 
seqLen       = configObj.SeqLen;

% 優先繼承 Config.m 的全域 Horizon，消除硬編碼
if isprop(configObj, 'Horizon') && ~isempty(configObj.Horizon)
    horizon = configObj.Horizon;
else
    horizon = 20;
end
valid_idx = seqLen : (numDaysRaw - horizon); 
num_valid = length(valid_idx);
fprintf('  -> 宇宙規模: %d 檔 | 交易天數: %d 天 | 原始特徵維度: %d 維 | 預測視窗: %d 日\n', ...
    numT, numDaysRaw, numFeats_All, horizon);
time_step1 = toc(t_step1);
fprintf('⏱️ [步驟 1 完成] 耗時: %.2f 秒\n\n', time_step1);

%% 1.5 核心串接：18 維特徵切片、1D/Horizon 雙軌 HAC-ICIR 健檢與特徵注意力閘門
t_step1_5 = tic;
disp('--- 步驟 1.5：特徵切片 (剝離 Macro)、1D/HAC-ICIR 健檢與特徵注意力加權 ---');
numExtractorFeats = 3 + configObj.NumMicroFeatures; % 18 維 (Rel 3 + Micro 15)
X_norm_3D_extractor_raw = X_norm_3D(:, 1:numExtractorFeats, :);
evaluator = FeatureEvaluator(configObj);
feat_names_18d = [{'Beta', 'Corr', 'RelStrength'}, ...
    {'R1', 'R5', 'R20', 'Vol20', 'IdioVol20', 'VolRatio', 'Amihud20', 'SMA20', 'SMA60', ...
     'MACD_Hist', 'RSI', 'OBV20', 'HL_Spread', 'Dist_H20', 'Dist_H252'}];

% 1 日 Horizon 微結構訊號健檢
[IC_Weights_2D, Raw_IC_Weight, Daily_IC_1D] = evaluator.compute_confidence(X_norm_3D_extractor_raw, Prices_Active, Expert_Active, 1);
evaluator.report_icir_ranking(Daily_IC_1D, feat_names_18d, 0.05, '1D (Micro-Structure)');
evaluator.report_ic_ranking(Raw_IC_Weight, feat_names_18d);

% 與當前萃取器目標視窗嚴格對齊之 HAC-ICIR 健檢
[~, ~, Daily_IC_Target] = evaluator.compute_confidence(X_norm_3D_extractor_raw, Prices_Active, Expert_Active, horizon);
evaluator.report_icir_ranking(Daily_IC_Target, feat_names_18d, 0.05, sprintf('%dD (Aligned with Extractor Target)', horizon));

disp(' -> 執行特徵注意力遮罩融合 (Energy-Preserving Feature Gate)...');
IC_Weights_3D = reshape(IC_Weights_2D, numDaysRaw, numExtractorFeats, 1);
X_norm_3D_extractor = X_norm_3D_extractor_raw .* IC_Weights_3D;
ch_stds = squeeze(std(X_norm_3D_extractor, 0, [1, 3], 'omitnan'));
fprintf('  -> [萃取器通道健檢] 18 維特徵標準差範圍: [%.4f, %.4f] (平均: %.4f)\n', ...
    min(ch_stds), max(ch_stds), mean(ch_stds));
time_step1_5 = toc(t_step1_5);
fprintf('⏱️ [步驟 1.5 完成] 耗時: %.2f 秒 (%.2f 分鐘)\n\n', time_step1_5, time_step1_5 / 60);

%% 2. 構建橫截面連續預測目標 (動態 Horizon 連續報酬 Z-Score)
t_step2 = tic;
fprintf('--- 步驟 2：構建橫截面連續超額報酬標籤 (Direction 2: %dD Continuous Z-Score) ---\n', horizon);
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
        % 橫截面 Z-Score 標準化：保留連續排序與幅度資訊
        Y_Labels_3D(t, active_mask) = (r_t - mu_t) ./ std_t;
    end
end
Y_Labels_3D(isnan(Y_Labels_3D) | isinf(Y_Labels_3D)) = 0;
time_step2 = toc(t_step2);
fprintf('⏱️ [步驟 2 完成] 耗時: %.2f 秒\n\n', time_step2);

%% 3. 切分時間軸 (嚴格 IS 內 Purged Embargo 跨體制驗證集)
t_step3 = tic;
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

if isprop(configObj, 'PurgeEmbargo') && ~isempty(configObj.PurgeEmbargo)
    embargo = configObj.PurgeEmbargo;
else
    embargo = horizon;
end

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
num_val   = length(val_idx_valid);
fprintf('✅ Purged 多體制時間軸劃分成功！訓練樣本: %d 筆 | 跨體制驗證樣本: %d 筆 (嚴格 Embargo: %d 天)\n', ...
    num_train, num_val, embargo);
time_step3 = toc(t_step3);
fprintf('⏱️ [步驟 3 完成] 耗時: %.2f 秒\n\n', time_step3);

%% 4. 初始化 DL 萃取器與連續迴歸線性預測頭
t_step4 = tic;
disp('--- 步驟 4：初始化特徵萃取網路 (連續預測頭 - mrg32k3a 確定性初值) ---');
factory = BuildDecoupledExtractors(configObj, numExtractorFeats);
[net_time, net_space] = factory.buildNetworks();

enable_space = ~isempty(net_space);

W_aux_time = dlarray(randn(stream, 1, 64, 'single') * 0.01); 
b_aux_time = dlarray(zeros(1, 1, 'single'));         
if enable_space
    W_aux_space = dlarray(randn(stream, 1, 64, 'single') * 0.01);
    b_aux_space = dlarray(zeros(1, 1, 'single'));
else
    W_aux_space = [];
    b_aux_space = [];
    disp('⏩ [Pure-Time 模式] 空間專家已剪枝，已略過空間線性頭初始化。');
end

if canUseGPU()
    gpuInfo = gpuDevice(); 
    fprintf('🎮 成功捕獲圖形加速卡：【%s】，開啟深度學習加速。\n', gpuInfo.Name);
    net_time = dlupdate(@gpuArray, net_time);
    W_aux_time = gpuArray(W_aux_time); 
    b_aux_time = gpuArray(b_aux_time);
    
    if enable_space
        net_space = dlupdate(@gpuArray, net_space);
        W_aux_space = gpuArray(W_aux_space); 
        b_aux_space = gpuArray(b_aux_space);
    end
end

% =========================================================================
% ★ 初始化權重指數移動平均 (EMA Shadow Weights)
% =========================================================================
net_time_ema = net_time;
W_aux_time_ema = W_aux_time;
b_aux_time_ema = b_aux_time;

if enable_space
    net_space_ema = net_space;
    W_aux_space_ema = W_aux_space;
    b_aux_space_ema = b_aux_space;
else
    net_space_ema = [];
    W_aux_space_ema = [];
    b_aux_space_ema = [];
end

% EMA 衰減率基礎參數 (優先讀取 Config.m，SSOT 統一預設為 0.995)
if isprop(configObj, 'DL_EMA_Decay') && ~isempty(configObj.DL_EMA_Decay)
    base_beta_ema = configObj.DL_EMA_Decay;
else
    base_beta_ema = 0.995;
end
fprintf('🪞 [EMA 引擎啟用] 影子權重已初始化完成 (基礎衰減率 Beta = %.3f，具備動態偏差校正)。\n', base_beta_ema);

time_step4 = toc(t_step4);
fprintf('⏱️ [步驟 4 完成] 耗時: %.2f 秒\n\n', time_step4);

%% 5. 啟動萃取器預訓練 (Warmup保護 + 解耦餘弦退火 + 權重 EMA 平滑)
t_step5 = tic;
disp('--- 步驟 5：執行萃取器預訓練 (Warmup保護 + 解耦餘弦退火排程 + 權重 EMA 平滑) ---');

% 動態讀取訓練輪數與早停耐心值 (SSOT 統一對齊 200 輪)
if isprop(configObj, 'DL_MaxEpochs') && ~isempty(configObj.DL_MaxEpochs)
    epochs = configObj.DL_MaxEpochs;
else
    epochs = 200; 
end

if isprop(configObj, 'DL_EarlyStoppingPatience') && ~isempty(configObj.DL_EarlyStoppingPatience)
    patience_limit = configObj.DL_EarlyStoppingPatience;
else
    patience_limit = 200;
end

% =========================================================================
% ★ 核心改進：解耦退火週期參數 (Decoupled Annealing Schedule)
% 始終以 fixed_decay_epochs 作為餘弦衰減基數 (SSOT 統一為 120 輪)。
% 保證前段 (1 ~ 120 輪) 的學習率與數值完全一致，第 121~200 輪鎖定 min_lr (1e-5)。
% =========================================================================
if isprop(configObj, 'DL_DecoupledDecayEpochs') && ~isempty(configObj.DL_DecoupledDecayEpochs)
    fixed_decay_epochs = configObj.DL_DecoupledDecayEpochs;
else
    fixed_decay_epochs = 120; 
end

% 讀取學習率基礎排程 (SSOT 統一)
if isprop(configObj, 'DL_BaseLR') && ~isempty(configObj.DL_BaseLR)
    base_lr = configObj.DL_BaseLR;
else
    base_lr = 1e-3; 
end

if isprop(configObj, 'DL_MinLR') && ~isempty(configObj.DL_MinLR)
    min_lr = configObj.DL_MinLR;
else
    min_lr = 1e-5; % 調降至 1e-5，保證尾段徹底走出水平收斂漸近線
end

fprintf('🎯 [退火排程解耦] 餘弦退火週期固定為 %d 輪 (總輪數: %d 輪 | 超期將鎖定於 min_lr=%.1e 平滑搜尋)。\n', ...
    fixed_decay_epochs, epochs, min_lr);

physicalBatchSize = 8;    
accumulationSteps = 4;    
numIterationsPerEpoch = floor(num_train / physicalBatchSize);
clipThreshold = 1.0;

% 讀取正則化超參數
l2_lambda      = configObj.DL_L2_Regularization;      
var_lambda     = configObj.DL_VarianceFloorLambda;    
var_target     = configObj.DL_VarianceFloorTarget;    
cfg_feat_drop  = configObj.FeatureDropoutRate;       
cfg_noise_std  = configObj.InputNoiseStd;            
cfg_var_drop   = configObj.VariationalDropRate;      
huber_delta    = configObj.DL_HuberDelta;            
ic_loss_weight = configObj.DL_ICLossWeight;          

avgG_t = []; avgSG_t = []; avgG_Wt = []; avgSG_Wt = []; avgG_bt = []; avgSG_bt = [];
avgG_s = []; avgSG_s = []; avgG_Ws = []; avgSG_Ws = []; avgG_bs = []; avgSG_bs = [];
iter = 0; 
historical_loss_time  = zeros(epochs, 1);
historical_loss_space = zeros(epochs, 1);
val_loss_time         = zeros(epochs, 1);
val_loss_space        = zeros(epochs, 1);
best_val_loss = inf; 
patience = 0; 

% 歷史最佳快照預設綁定初始 EMA 結構
best_net_time  = net_time_ema; 
best_net_space = net_space_ema;
min_healthy_var_t = 0.05;
min_healthy_var_s = 0.10;
actual_completed_epochs = 0;
warmup_epochs = 5; % 預熱保護期 5 輪

for epoch = 1:epochs
    actual_completed_epochs = epoch;
    
    % --- ★ 學習率平滑預熱與解耦餘弦退火排程 (前 5 輪預熱 + 120 輪退火 + 尾段鎖底) ---
    warmup_lr_epochs = 5;
    if epoch <= warmup_lr_epochs
        % 前 5 輪線性預熱
        current_lr = min_lr + (base_lr - min_lr) * (epoch / warmup_lr_epochs);
    elseif epoch <= fixed_decay_epochs
        % 第 6 輪至固定退火週期 (第 120 輪)：平滑餘弦衰減至 min_lr (1e-5)
        decay_ratio = (epoch - warmup_lr_epochs) / (fixed_decay_epochs - warmup_lr_epochs);
        current_lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * decay_ratio));
    else
        % 第 121 輪至第 200 輪：強制鎖定於最低學習率 min_lr (1e-5)，以微步長爬行展示走平
        current_lr = min_lr;
    end
    
    % --- 階梯式動態正則化排程 ---
    if epoch <= warmup_epochs
        feat_drop_rate = cfg_feat_drop * 0.50; 
        noise_std      = cfg_noise_std * 0.33; 
        var_drop_rate  = cfg_var_drop * 0.50;
    else
        feat_drop_rate = cfg_feat_drop;
        noise_std      = cfg_noise_std;
        var_drop_rate  = cfg_var_drop;
    end
    
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
            actual_t_indices, X_norm_3D_extractor, AdjMatrix_3D, Y_Labels_3D, Expert_Active, ...
            numExtractorFeats, numT, seqLen, physicalBatchSize, enable_space);
        
        % 1. 時序專家梯度計算
        [loss_t, grad_t, grad_Wt, grad_bt] = dlfeval(@(n,w,b,x,y,m) aux_loss_time(...
            n, w, b, x, y, m, l2_lambda, var_lambda, var_target, numT, physicalBatchSize, ...
            ic_loss_weight, huber_delta, feat_drop_rate, noise_std, var_drop_rate), ...
            net_time, W_aux_time, b_aux_time, X_batch_time, Y_batch, M_batch);
        
        if isempty(grad_t_accum)
            grad_t_accum = grad_t; grad_Wt_accum = grad_Wt; grad_bt_accum = grad_bt;
        else
            grad_t_accum.Value = cellfun(@plus, grad_t_accum.Value, grad_t.Value, 'UniformOutput', false);
            grad_Wt_accum = grad_Wt_accum + grad_Wt; 
            grad_bt_accum = grad_bt_accum + grad_bt;
        end
        
        % 2. 空間專家梯度計算 (僅在未剪枝時)
        if enable_space
            [loss_s, grad_s, grad_Ws, grad_bs] = dlfeval(@(n,w,b,x,a,y,m) aux_loss_space(...
                n, w, b, x, a, y, m, l2_lambda, numT, physicalBatchSize, ...
                ic_loss_weight, huber_delta, feat_drop_rate, noise_std), ...
                net_space, W_aux_space, b_aux_space, X_batch_space, A_batch_space, Y_batch, M_batch);
            
            if isempty(grad_s_accum)
                grad_s_accum = grad_s; grad_Ws_accum = grad_Ws; grad_bs_accum = grad_bs;
            else
                grad_s_accum.Value = cellfun(@plus, grad_s_accum.Value, grad_s.Value, 'UniformOutput', false);
                grad_Ws_accum = grad_Ws_accum + grad_Ws; 
                grad_bs_accum = grad_bs_accum + grad_bs;
            end
            epoch_loss_space = epoch_loss_space + extractdata(loss_s);
            clear loss_s grad_s;
        end
        
        accum_count = accum_count + 1;
        
        % --- 梯度累積更新 ＋ EMA 影子權重平滑更新 ---
        if accum_count == accumulationSteps || i == numIterationsPerEpoch
            iter = iter + 1;
            
            % 時序專家 Adam 更新
            grad_t_accum.Value = cellfun(@(x) x/accum_count, grad_t_accum.Value, 'UniformOutput', false);
            grad_Wt_accum = grad_Wt_accum / accum_count; 
            grad_bt_accum = grad_bt_accum / accum_count;
            
            last_gnorm_t = sqrt(sum(cellfun(@(x) sum(extractdata(x(:)).^2), grad_t_accum.Value)));
            grad_t_accum = dlupdate(@(g) clipGradient(g, clipThreshold), grad_t_accum);
            
            [net_time, avgG_t, avgSG_t] = adamupdate(net_time, grad_t_accum, avgG_t, avgSG_t, iter, current_lr);
            [W_aux_time, avgG_Wt, avgSG_Wt] = adamupdate(W_aux_time, grad_Wt_accum, avgG_Wt, avgSG_Wt, iter, current_lr);
            [b_aux_time, avgG_bt, avgSG_bt] = adamupdate(b_aux_time, grad_bt_accum, avgG_bt, avgSG_bt, iter, current_lr);
            
            % 空間專家 Adam 更新
            if enable_space
                grad_s_accum.Value = cellfun(@(x) x/accum_count, grad_s_accum.Value, 'UniformOutput', false);
                grad_Ws_accum = grad_Ws_accum / accum_count; 
                grad_bs_accum = grad_bs_accum / accum_count;
                
                last_gnorm_s = sqrt(sum(cellfun(@(x) sum(extractdata(x(:)).^2), grad_s_accum.Value)));
                grad_s_accum = dlupdate(@(g) clipGradient(g, clipThreshold), grad_s_accum);
                
                [net_space, avgG_s, avgSG_s] = adamupdate(net_space, grad_s_accum, avgG_s, avgSG_s, iter, current_lr);
                [W_aux_space, avgG_Ws, avgSG_Ws] = adamupdate(W_aux_space, grad_Ws_accum, avgG_Ws, avgSG_Ws, iter, current_lr);
                [b_aux_space, avgG_bs, avgSG_bs] = adamupdate(b_aux_space, grad_bs_accum, avgG_bs, avgSG_bs, iter, current_lr);
                
                grad_s_accum = []; grad_Ws_accum = []; grad_bs_accum = [];
            end
            
            grad_t_accum = []; grad_Wt_accum = []; grad_bt_accum = [];
            accum_count = 0;
            
            % ★ EMA 影子權重更新 (動態偏差校正)
            cur_beta = min(base_beta_ema, (1.0 + iter) / (10.0 + iter));
            net_time_ema.Learnables.Value = cellfun(@(we, w) cur_beta * we + (1.0 - cur_beta) * w, ...
                net_time_ema.Learnables.Value, net_time.Learnables.Value, 'UniformOutput', false);
            W_aux_time_ema = cur_beta * W_aux_time_ema + (1.0 - cur_beta) * W_aux_time;
            b_aux_time_ema = cur_beta * b_aux_time_ema + (1.0 - cur_beta) * b_aux_time;
            
            if enable_space
                net_space_ema.Learnables.Value = cellfun(@(we, w) cur_beta * we + (1.0 - cur_beta) * w, ...
                    net_space_ema.Learnables.Value, net_space.Learnables.Value, 'UniformOutput', false);
                W_aux_space_ema = cur_beta * W_aux_space_ema + (1.0 - cur_beta) * W_aux_space;
                b_aux_space_ema = cur_beta * b_aux_space_ema + (1.0 - cur_beta) * b_aux_space;
            end
        end
        
        epoch_loss_time = epoch_loss_time + extractdata(loss_t);
        clear loss_t grad_t;
    end
    
    historical_loss_time(epoch) = epoch_loss_time / numIterationsPerEpoch;
    if enable_space
        historical_loss_space(epoch) = epoch_loss_space / numIterationsPerEpoch;
    end
    
    % --- 多體制連續迴歸驗證階段 (使用 EMA 影子權重推論評估) ---
    val_samples = min(num_val, 256); 
    val_idx_shuffle = randperm(stream, num_val, val_samples);
    actual_val_indices = val_idx_valid(val_idx_shuffle);
    
    val_batch_size = 8; 
    temp_val_loss_t = 0;
    temp_val_loss_s = 0;
    val_iters = ceil(val_samples / val_batch_size);
    
    for v_i = 1:val_iters
        v_start = (v_i-1)*val_batch_size + 1;
        v_end = min(v_i*val_batch_size, val_samples);
        chunk_size = v_end - v_start + 1;
        chunk_indices = actual_val_indices(v_start:v_end);
        
        [X_val_time, X_val_space, A_val_space, Y_val, M_val] = prepareBatchData(...
            chunk_indices, X_norm_3D_extractor, AdjMatrix_3D, Y_Labels_3D, Expert_Active, ...
            numExtractorFeats, numT, seqLen, chunk_size, enable_space);
        
        v_loss_t_chunk = calc_val_loss_time(net_time_ema, W_aux_time_ema, b_aux_time_ema, ...
            X_val_time, Y_val, M_val, numT, chunk_size, ic_loss_weight, huber_delta);
        temp_val_loss_t = temp_val_loss_t + extractdata(v_loss_t_chunk) * chunk_size;
        
        if enable_space
            v_loss_s_chunk = calc_val_loss_space(net_space_ema, W_aux_space_ema, b_aux_space_ema, ...
                X_val_space, A_val_space, Y_val, M_val, numT, chunk_size, ic_loss_weight, huber_delta);
            temp_val_loss_s = temp_val_loss_s + extractdata(v_loss_s_chunk) * chunk_size;
        end
        
        clear X_val_time X_val_space A_val_space Y_val M_val v_loss_t_chunk;
    end
    
    val_loss_time(epoch) = temp_val_loss_t / val_samples;
    if enable_space
        val_loss_space(epoch) = temp_val_loss_s / val_samples;
    end
    
    % 表徵健康度抽樣檢查 (採用 EMA 網絡)
    e_t_sample = extractdata(predict(net_time_ema, X_batch_time));
    var_e_t = var(e_t_sample(:), 'omitnan');
    
    if enable_space
        e_s_sample = extractdata(reshape(predict(net_space_ema, X_batch_space, A_batch_space), 64, []));
        var_e_s = var(e_s_sample(:), 'omitnan');
        
        fprintf(' -> Epoch %3d/%d (LR: %.2e | FeatDrop: %.2f) | Train (T: %.4f, S: %.4f) | Val_EMA (T: %.4f, S: %.4f)\n', ...
            epoch, epochs, current_lr, feat_drop_rate, historical_loss_time(epoch), historical_loss_space(epoch), val_loss_time(epoch), val_loss_space(epoch));
        fprintf('    [診斷] GradNorm(T: %.2e, S: %.2e) | E_Var_EMA(T: %.4f, S: %.4f)\n', ...
            last_gnorm_t, last_gnorm_s, var_e_t, var_e_s);
        
        combined_val = val_loss_time(epoch) + val_loss_space(epoch);
        is_healthy = (var_e_t >= min_healthy_var_t) && (var_e_s >= min_healthy_var_s);
    else
        fprintf(' -> Epoch %3d/%d (LR: %.2e | FeatDrop: %.2f) | Train ContLoss (T: %.4f) | Val_EMA Multi-Regime (T: %.4f)\n', ...
            epoch, epochs, current_lr, feat_drop_rate, historical_loss_time(epoch), val_loss_time(epoch));
        fprintf('    [診斷] GradNorm(T: %.2e) | E_Var_EMA(T: %.4f)\n', last_gnorm_t, var_e_t);
        
        combined_val = val_loss_time(epoch);
        is_healthy = (var_e_t >= min_healthy_var_t);
    end
    
    if var_e_t < min_healthy_var_t
        fprintf('    ⚠️ [警告] E_Var_EMA(T)=%.4f 低於健康閾值 (%.2f)，疑似發生時序表徵坍縮！\n', var_e_t, min_healthy_var_t);
    end
    
    % ★ 早停與快照裁決 (鎖定 EMA 平滑網絡)：前 warmup_epochs 輪禁止鎖定最佳值
    if epoch > warmup_epochs
        if is_healthy && (combined_val < best_val_loss - 1e-4)
            best_val_loss = combined_val;
            patience = 0;
            % 捕捉經過 EMA 平滑後的最高品質泛化權重
            best_net_time  = net_time_ema; 
            best_net_space = net_space_ema;
            fprintf('    🌟 [EMA 最佳快照更新] 於正式訓練期鎖定平滑新低點 Val Loss = %.4f！\n', best_val_loss);
        else
            patience = patience + 1;
            if ~is_healthy
                fprintf('    ⏭️  [跳過快照] 本輪因表徵變異數未達健康門檻不列入候選！\n');
            end
        end
    else
        fprintf('    ⏳ [預熱保護期] Epoch %3d <= %d，暫不鎖定最佳快照與早停計數。\n', epoch, warmup_epochs);
    end
    
    % 早停中斷裁決 (Patience 設定為 200 時，迴圈將完整跑完)
    if patience >= patience_limit && epoch >= 10
        fprintf('🛑 [Early Stopping] EMA 驗證損失連續 %d 輪未改善，提前於 Epoch %d 終止訓練並回滾最佳快照！\n', patience_limit, epoch);
        break;
    end
    
    clear X_batch_time X_batch_space A_batch_space Y_batch M_batch e_t_sample;
end

% ★ 訓練全面結束：確定將模型回滾至全歷史最佳 EMA 快照 (絕不拿過擬合末端權重存檔)
fprintf('\n💾 正在將全域模型權重切換至歷史最佳 EMA 快照 (Val Loss = %.4f)...\n', best_val_loss);
net_time  = best_net_time; 
net_space = best_net_space;

%% 收斂健檢閘門 (連續損失與表徵健全度)
loss_t_end = val_loss_time(actual_completed_epochs); 
fprintf('\n📊 萃取器收斂診斷 (跨體制 EMA 驗證基準):\n');
fprintf('  > 時序專家 (T) 最終 Val_EMA Continuous Loss: %.4f\n', loss_t_end);
fprintf('  > 時序專家 (T) 歷史最佳快照 Val Loss      : %.4f\n', best_val_loss);

if enable_space
    loss_s_end = val_loss_space(actual_completed_epochs);
    fprintf('  > 空間專家 (S) 最終 Val_EMA Continuous Loss: %.4f\n', loss_s_end);
else
    disp('  > 空間專家 (S) 狀態: ⏩ 已依實驗結論成功剪枝 (Pure-Time 模式)');
end

if isnan(loss_t_end) || isinf(loss_t_end)
    warning('❌ 警告：模型最終 Val Loss 存在 NaN 或 Inf，訓練異常！');
else
    disp('✅ 萃取器連續目標訓練完畢，成功建立抗過擬合連續超額報酬特徵表徵！');
end
time_step5 = toc(t_step5);
fprintf('⏱️ [步驟 5 完成] 耗時: %.2f 秒 (%.2f 分鐘)\n\n', time_step5, time_step5 / 60);

%% 6. 最終全歷史 Embedding 提煉與存檔 (推論階段：純淨無隨機擾動)
t_step6 = tic;
disp('--- 步驟 6：全歷史強固特徵表徵提煉與落地 (使用 EMA 最佳權重提取 3D 張量) ---');
E_time_all  = zeros(numDaysRaw, 64, numT, 'single');
E_space_all = zeros(numDaysRaw, 64, numT, 'single');
inferBatchSize = 16; 
fprintf('  -> 正在提煉全歷史節點級別表徵 (共 %d 筆有效天數，批次步長: %d)...\n', num_valid, inferBatchSize);

for start_idx = 1:inferBatchSize:num_valid
    end_idx = min(start_idx + inferBatchSize - 1, num_valid);
    chunk_size = end_idx - start_idx + 1; 
    actual_t_indices = valid_idx(start_idx:end_idx); 
    
    [X_inf_time, X_inf_space, A_inf_space, ~, ~] = prepareBatchData(...
        actual_t_indices, X_norm_3D_extractor, AdjMatrix_3D, Y_Labels_3D, Expert_Active, ...
        numExtractorFeats, numT, seqLen, chunk_size, enable_space);
    
    % 以鎖定之最佳 EMA net_time 推論，原生關閉 Dropout 與隨機噪聲
    E_time_raw = predict(net_time, X_inf_time);
    e_t_reshaped = permute(reshape(extractdata(E_time_raw), 64, numT, chunk_size), [3, 1, 2]);
    E_time_all(actual_t_indices, :, :) = gather(e_t_reshaped);
    
    if enable_space
        E_space_flat_out = predict(net_space, X_inf_space, A_inf_space);
        e_s_reshaped = permute(reshape(extractdata(E_space_flat_out), 64, numT, chunk_size), [3, 1, 2]);
        E_space_all(actual_t_indices, :, :) = gather(e_s_reshaped);
    end
end
time_step6 = toc(t_step6);
fprintf('⏱️ [步驟 6 完成] 耗時: %.2f 秒 (%.2f 分鐘)\n\n', time_step6, time_step6 / 60);

%% 7. 繪製與儲存訓練曲線 (標準白底黑字格式，去除密集節點圖示)
t_step7 = tic;
disp('--- 步驟 7：產出 Loss 曲線視覺化報表與模型檔案存檔 ---');
if ~exist(configObj.ModelDir, 'dir')
    mkdir(configObj.ModelDir); 
end
actual_epochs = 1:actual_completed_epochs;
if enable_space
    fig_loss = figure('Name', 'Phase 2: Extractor Pretrain Continuous Loss', ...
        'Position', [100, 100, 1200, 500], 'Color', 'w', 'Visible', 'off'); 
    set(fig_loss, 'InvertHardcopy', 'off');
    
    subplot(1, 2, 1);
    plot(actual_epochs, historical_loss_time(actual_epochs), '-', 'LineWidth', 1.8, 'Color', '#D95319', 'DisplayName', 'Train Continuous Loss'); hold on;
    plot(actual_epochs, val_loss_time(actual_epochs), '-', 'LineWidth', 1.8, 'Color', '#0072BD', 'DisplayName', 'Val (Multi-Regime EMA) Loss');
    title(sprintf('Time Expert (%dD Continuous Soft-IC + Huber)', horizon), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    xlabel('Epochs', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('Continuous Loss', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
    set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
        'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 10);
    grid on; box on;
    
    subplot(1, 2, 2);
    plot(actual_epochs, historical_loss_space(actual_epochs), '-', 'LineWidth', 1.8, 'Color', '#D95319', 'DisplayName', 'Train Continuous Loss'); hold on;
    plot(actual_epochs, val_loss_space(actual_epochs), '-', 'LineWidth', 1.8, 'Color', '#0072BD', 'DisplayName', 'Val (Multi-Regime EMA) Loss');
    title(sprintf('Space Expert (%dD Continuous Soft-IC + Huber)', horizon), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    xlabel('Epochs', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('Continuous Loss', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
    set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
        'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 10);
    grid on; box on;
else
    fig_loss = figure('Name', 'Phase 2: Pure-Time Extractor Continuous Loss', ...
        'Position', [100, 100, 700, 500], 'Color', 'w', 'Visible', 'off'); 
    set(fig_loss, 'InvertHardcopy', 'off');
    
    plot(actual_epochs, historical_loss_time(actual_epochs), '-', 'LineWidth', 1.8, 'Color', '#D95319', 'DisplayName', 'Train Continuous Loss'); hold on;
    plot(actual_epochs, val_loss_time(actual_epochs), '-', 'LineWidth', 1.8, 'Color', '#0072BD', 'DisplayName', 'Val (Multi-Regime EMA) Loss');
    title(sprintf('Pure-Time Expert (%dD Continuous Soft-IC + Huber)', horizon), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    xlabel('Epochs', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    ylabel('Continuous Loss', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
    set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
        'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica', 'FontSize', 10);
    grid on; box on;
end

lossFigPath = fullfile(configObj.ModelDir, 'Phase2_Loss_Curve.png');
exportgraphics(fig_loss, lossFigPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf(' 📊 Loss 曲線 (白底黑字，已截斷真實輪次) 已儲存至: %s\n', lossFigPath);
close(fig_loss);

% 儲存最佳 EMA 模型實體與 Embedding 矩陣 (相容下游 GBDT 與回測讀取介面)
modelPath = fullfile(configObj.ModelDir, 'DL_Extractors.mat');
save(modelPath, 'net_time', 'net_space', 'E_time_all', 'E_space_all', '-v7.3');
fprintf('💾 DL 萃取器預訓練完畢！[Days, 64, Tickers] 節點表徵已存至: %s\n', modelPath);
time_step7 = toc(t_step7);
fprintf('⏱️ [步驟 7 完成] 耗時: %.2f 秒\n\n', time_step7);

%% =========================================================================
% 結算全流程執行時長審計報告
% =========================================================================
total_elapsed_sec = toc(t_total_start);
tot_hours = floor(total_elapsed_sec / 3600);
tot_mins  = floor(mod(total_elapsed_sec, 3600) / 60);
tot_secs  = mod(total_elapsed_sec, 60);

fprintf('=================================================================\n');
fprintf('📊 【Phase 2 各階段耗時明細與總時長審計報告】\n');
fprintf('=================================================================\n');
fprintf(' 步驟 0：環境掛載與隨機串流鎖定   : %8.2f 秒 (%5.1f%%)\n', time_step0, (time_step0 / total_elapsed_sec) * 100);
fprintf(' 步驟 1：3D 特徵快取載入與對齊   : %8.2f 秒 (%5.1f%%)\n', time_step1, (time_step1 / total_elapsed_sec) * 100);
fprintf(' 步驟 1.5：HAC-ICIR 健檢與閘門融合: %8.2f 秒 (%5.1f%%)\n', time_step1_5, (time_step1_5 / total_elapsed_sec) * 100);
fprintf(' 步驟 2：連續超額報酬 Z-Score 構建: %8.2f 秒 (%5.1f%%)\n', time_step2, (time_step2 / total_elapsed_sec) * 100);
fprintf(' 步驟 3：Purged 多體制時間軸劃分  : %8.2f 秒 (%5.1f%%)\n', time_step3, (time_step3 / total_elapsed_sec) * 100);
fprintf(' 步驟 4：網路拓撲與線性預測頭初始化: %8.2f 秒 (%5.1f%%)\n', time_step4, (time_step4 / total_elapsed_sec) * 100);
fprintf(' 步驟 5：萃取器預訓練與跨體制早停 : %8.2f 秒 (%5.1f%%)\n', time_step5, (time_step5 / total_elapsed_sec) * 100);
fprintf(' 步驟 6：全歷史 64D 表徵提煉落地 : %8.2f 秒 (%5.1f%%)\n', time_step6, (time_step6 / total_elapsed_sec) * 100);
fprintf(' 步驟 7：曲線繪製與模型權重存檔   : %8.2f 秒 (%5.1f%%)\n', time_step7, (time_step7 / total_elapsed_sec) * 100);
fprintf('-----------------------------------------------------------------\n');
if tot_hours > 0
    fprintf('⏱️ 【總執行時長】: %d 小時 %d 分 %.2f 秒 (共 %.2f 秒)\n', tot_hours, tot_mins, tot_secs, total_elapsed_sec);
else
    fprintf('⏱️ 【總執行時長】: %d 分 %.2f 秒 (共 %.2f 秒)\n', tot_mins, tot_secs, total_elapsed_sec);
end
fprintf('=================================================================\n');
disp('🎯 [Phase 2] 連續迴歸萃取器預訓練完成！請推進至 Phase 3。');
disp('=================================================================');

%% =====================================================================
% 輔助函數區 (批次切片、正則化迴歸損失、VICReg 保底與梯度裁剪)
% =====================================================================
function [X_time, X_space, A_space, Y, M] = prepareBatchData(indices, X_3D, Adj_3D, Y_Lab, Expert, ...
    nF, nT, sL, bZ, enable_space)
    
    X_time_raw = zeros(nF, nT * bZ, sL, 'single');
    Y_raw      = zeros(1, nT * bZ, 'single');
    M_raw      = zeros(1, nT * bZ, 'single');
    
    if enable_space
        X_space_raw = zeros(nF * nT, bZ, 'single');
        A_space_raw = zeros(nT * nT, bZ, 'single');
    end
    
    for b = 1:bZ
        t = indices(b);
        seq_data = permute(X_3D(t-sL+1:t, :, :), [2, 3, 1]);
        idx_start = (b-1)*nT + 1;
        idx_end   = b*nT;
        X_time_raw(:, idx_start:idx_end, :) = seq_data;
        
        if enable_space
            cross_data = squeeze(X_3D(t, :, :));
            X_space_raw(:, b) = cross_data(:);
            adj_data = Adj_3D(:,:,t);
            A_space_raw(:, b) = adj_data(:);
        end
        
        Y_raw(1, idx_start:idx_end) = Y_Lab(t, :);
        M_raw(1, idx_start:idx_end) = Expert(t, :);
    end
    
    X_time = dlarray(X_time_raw, 'CBT');
    Y      = dlarray(Y_raw, 'CB');
    M      = dlarray(M_raw, 'CB');
    
    if enable_space
        X_space = dlarray(X_space_raw, 'CB');
        A_space = dlarray(A_space_raw, 'CB');
    else
        X_space = dlarray([]);
        A_space = dlarray([]);
    end
    
    if canUseGPU()
        X_time = gpuArray(X_time); 
        Y      = gpuArray(Y); 
        M      = gpuArray(M);
        if enable_space
            X_space = gpuArray(X_space); 
            A_space = gpuArray(A_space); 
        end
    end
end

function [loss, grad_net, grad_W, grad_b] = aux_loss_time(net, W_aux, b_aux, X, Y, M, ...
    l2_lambda, var_lambda, var_target, nT, bZ, ic_w, delta, feat_drop, noise_std, var_drop)
    
    X_aug = BuildDecoupledExtractors.apply_input_regularization(X, feat_drop, noise_std, true);
    X_aug = BuildDecoupledExtractors.apply_variational_dropout(X_aug, var_drop, true);
    
    E = forward(net, X_aug); 
    E_unfmt = stripdims(E);
    Y_pred = W_aux * E_unfmt + b_aux; 
    
    cont_loss = compute_continuous_loss_internal(Y_pred, Y, M, nT, bZ, ic_w, delta);
    var_penalty = BuildDecoupledExtractors.compute_vicreg_penalty(E_unfmt, var_target);
    
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

function [loss, grad_net, grad_W, grad_b] = aux_loss_space(net, W_aux, b_aux, X, A, Y, M, ...
    l2_lambda, nT, bZ, ic_w, delta, feat_drop, noise_std)
    
    X_aug = BuildDecoupledExtractors.apply_input_regularization(X, feat_drop, noise_std, true);
    
    E_flat_out = forward(net, X_aug, A); 
    E_unfmt = reshape(stripdims(E_flat_out), 64, []);
    Y_pred = W_aux * E_unfmt + b_aux; 
    
    cont_loss = compute_continuous_loss_internal(Y_pred, Y, M, nT, bZ, ic_w, delta);
    
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

function loss = calc_val_loss_time(net, W_aux, b_aux, X, Y, M, nT, bZ, ic_w, delta)
    E = predict(net, X); 
    E_unfmt = stripdims(E);
    Y_pred = W_aux * E_unfmt + b_aux; 
    loss = compute_continuous_loss_internal(Y_pred, Y, M, nT, bZ, ic_w, delta);
end

function loss = calc_val_loss_space(net, W_aux, b_aux, X, A, Y, M, nT, bZ, ic_w, delta)
    E_flat_out = predict(net, X, A); 
    E_unfmt = reshape(stripdims(E_flat_out), 64, []);
    Y_pred = W_aux * E_unfmt + b_aux; 
    loss = compute_continuous_loss_internal(Y_pred, Y, M, nT, bZ, ic_w, delta);
end

function loss = compute_continuous_loss_internal(y_pred, y_true, mask, nT, bZ, ic_weight, delta)
    if nargin < 7 || isempty(delta)
        delta = 0.10;
    end
    
    yp = stripdims(y_pred); yp = yp(:);
    yt = stripdims(y_true); yt = yt(:);
    m  = logical(stripdims(mask)); m = m(:);
    
    yp_m = yp(m);
    yt_m = yt(m);
    err = abs(yp_m - yt_m);
    is_small = err <= delta;
    huber = mean(is_small .* (0.5 * err.^2) + (~is_small) .* (delta * (err - 0.5 * delta)), 'all');
    
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
