% =========================================================================
% 腳本：Stage2_5_MultiSeed_DirectionCompare.m (Task B: 多種子/多子集穩健性驗證)
% 依據：《Stage 2.5 – 4 Code Review 計劃書》 §2 規範
% 職責：在 5 組隨機種子 (n=5) 下評估 Direction 2 (Soft-IC) 與 Direction 3 (60D)，
%       並以 Wilcoxon 符號秩檢定 (Wilcoxon Signed-Rank Test) 檢驗跨子集穩健性
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🧪 [Stage 2.5 Task B] 啟動多種子/多子集穩健性交叉檢驗 (5 Seeds x 10 Epochs)');
disp('=================================================================');

%% 0. 環境路徑掛載與全域設定
currentFile = mfilename('fullpath');
if isempty(currentFile), currentPath = pwd; else, currentPath = fileparts(currentFile); end
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

%% 1. 載入特徵快取與候選標的池篩選 (§2.2 規範)
disp('--- 步驟 1：載入特徵快取並建立高活躍度候選標的池 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
if ~exist(cachePath, 'file')
    error('❌ 找不到特徵快取檔案，請先確認已執行 1_Run_Data_and_Features.m！');
end

load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC';
numDaysRaw_full = length(Dates_Active);
numT_full = configObj.NumTickers;

% 限制候選池：活躍天數佔比 >= 50%，避免隨機抽樣抽到全 NaN 標的

active_sums = sum(Expert_Active, 1);
candidate_pool = find(active_sums / numDaysRaw_full >= 0.5);
candidate_pool = candidate_pool(:)'; % 強制鎖定為列向量
fprintf('  📊 總標的數: %d 檔 | 活躍佔比 >= 50%% 候選池: %d 檔\n', numT_full, length(candidate_pool));

% 鎖定 3200 天跨體制時序區間
start_day = max(1, round(numDaysRaw_full * 0.4));
end_day   = min(numDaysRaw_full, start_day + 3200 - 1);
sub_days  = start_day : end_day;

numExtractorFeats = 3 + configObj.NumMicroFeatures; % 18 維
seqLen = configObj.SeqLen;
numDays = length(sub_days);

% 載入 GICS 產業別對應表
universePath = fullfile(projectRoot, 'data', 'crawlers', 'us_universe.csv');
all_tickers = configObj.IdxTickers;
full_sector_map = repmat({'Unknown'}, 1, numT_full);
if exist(universePath, 'file')
    u_tbl = readtable(universePath, 'TextType', 'string');
    if ismember('GICS_Sector', u_tbl.Properties.VariableNames)
        [lia, loc] = ismember(string(all_tickers), string(u_tbl.Ticker));
        valid_loc = loc(lia);
        valid_idx = find(lia);
        for k = 1:length(valid_idx)
            s_val = char(u_tbl.GICS_Sector(valid_loc(k)));
            if ~isempty(strtrim(s_val)), full_sector_map{valid_idx(k)} = strtrim(s_val); end
        end
    end
end

use_gpu = canUseGPU();
if use_gpu
    gpu_dev = gpuDevice();
    fprintf('  🎮 偵測到硬體加速卡: 【%s】，啟用 GPU 向量化並行訓練。\n', gpu_dev.Name);
end

%% 2. 多種子實驗參數設定 (§2.2 規範)
seeds = [101, 102, 103, 104, 105]; % 5 組固定獨立種子
n_seeds = length(seeds);
sample_T = 60;
epochs = 10; % 10 Epochs 足以捕捉收斂趨勢並節省計算預算
lr = 1e-3;
batch_size = 64;

% 結果收集矩陣
res_g1_loss = zeros(n_seeds, 1);
res_g3_loss = zeros(n_seeds, 1);
res_g3_ic   = zeros(n_seeds, 1);
res_g3_phac = zeros(n_seeds, 1);
res_g4_loss = zeros(n_seeds, 1);
res_g4_ic   = zeros(n_seeds, 1);
res_g4_phac = zeros(n_seeds, 1);

%% 3. 啟動 5 組種子多子集訓練迴圈
disp('--- 步驟 2：開始執行 5 組種子多子集對照訓練 ---');
total_tic = tic;

for s = 1:n_seeds
    curr_seed = seeds(s);
    fprintf('\n=================================================================\n');
    fprintf('🌱 [Seed %d/%d: %d] 抽樣 60 檔標的並構建對照特徵與標籤...\n', s, n_seeds, curr_seed);
    fprintf('=================================================================\n');
    
    rng(curr_seed, 'twister');
    sub_tickers = randsample(candidate_pool, sample_T); % 獨立隨機抽樣
    sub_tickers = sub_tickers(:)'; % ★ 強制鎖定為 1 x sample_T 列向量
    
    % 截取當前子集資料
    Prices_s    = single(Prices_Active(sub_days, sub_tickers));
    Expert_s    = logical(Expert_Active(sub_days, sub_tickers));
    X_raw_18D_s = single(X_norm_3D(sub_days, 1:numExtractorFeats, sub_tickers));
    
    % 構建當前子集 GICS 產業中性化特徵矩陣
    sectors_cell = full_sector_map(sub_tickers);
    sectors_cat = categorical(sectors_cell(:)'); % ★ 強制保證 1 x sample_T 列向量
    unique_sectors = categories(sectors_cat);
    X_gics_18D_s = X_raw_18D_s;
    
    for t = 1:numDays
        act_m = Expert_s(t, :);
        act_m = act_m(:)'; % 強制 1 x sample_T
        
        if sum(act_m) >= 10
            vals_all = X_raw_18D_s(t, :, act_m);
            mu_all = mean(vals_all, 3, 'omitnan');
            std_all = std(vals_all, 0, 3, 'omitnan') + 1e-8;
            X_gics_18D_s(t, :, act_m) = (vals_all - mu_all) ./ std_all;
            
            for sec_i = 1:length(unique_sectors)
                s_name = unique_sectors{sec_i};
                if ismember(s_name, {'Unknown', 'Macro', 'Safe Haven', 'Broad Market Index'}), continue; end
                % ★ 核心修復：統一為 1 x sample_T 進行 element-wise &，杜絕廣播為 60x60
                sec_m = act_m & (sectors_cat == s_name);
                if sum(sec_m) >= 4
                    vals_sec = X_raw_18D_s(t, :, sec_m);
                    mu_sec = mean(vals_sec, 3, 'omitnan');
                    std_sec = std(vals_sec, 0, 3, 'omitnan') + 1e-8;
                    X_gics_18D_s(t, :, sec_m) = (vals_sec - mu_sec) ./ std_sec;
                end
            end
        end
    end
    X_gics_18D_s(isnan(X_gics_18D_s) | isinf(X_gics_18D_s)) = 0;
    
    % 構建 5D 與 60D 標籤 (防禦 NaN)
    R_5D = (Prices_s(6:end, :) - Prices_s(1:end-5, :)) ./ (Prices_s(1:end-5, :) + 1e-8);
    R_5D(isnan(R_5D) | isinf(R_5D)) = NaN;
    Y_bin_5D = false(numDays, sample_T);
    Y_cont_5D = zeros(numDays, sample_T, 'single');
    for t = 1:numDays-5
        act_m = Expert_s(t, :) & ~isnan(R_5D(t, :)) & ~isinf(R_5D(t, :));
        if sum(act_m) >= 5
            r_t = R_5D(t, act_m);
            Y_bin_5D(t, act_m) = (r_t > median(r_t, 'omitnan'));
            Y_cont_5D(t, act_m) = (r_t - mean(r_t, 'omitnan')) ./ (std(r_t, 0, 'omitnan') + 1e-6);
        end
    end
    Y_cont_5D(isnan(Y_cont_5D) | isinf(Y_cont_5D)) = 0;
    
    R_60D = (Prices_s(61:end, :) - Prices_s(1:end-60, :)) ./ (Prices_s(1:end-60, :) + 1e-8);
    R_60D(isnan(R_60D) | isinf(R_60D)) = NaN;
    Y_cont_60D = zeros(numDays, sample_T, 'single');
    for t = 1:numDays-60
        act_m = Expert_s(t, :) & ~isnan(R_60D(t, :)) & ~isinf(R_60D(t, :));
        if sum(act_m) >= 5
            r_t = R_60D(t, act_m);
            Y_cont_60D(t, act_m) = (r_t - mean(r_t, 'omitnan')) ./ (std(r_t, 0, 'omitnan') + 1e-6);
        end
    end
    Y_cont_60D(isnan(Y_cont_60D) | isinf(Y_cont_60D)) = 0;
    
    % 定義當前 seed 需測試的三組體制
    groups_seed = { ...
        struct('name', 'G1: Base BCE',     'X', X_raw_18D_s,  'Y', Y_bin_5D,  'H', 5,  'type', 'binary',     'embargo', 20), ...
        struct('name', 'G3: Soft-IC Loss', 'X', X_gics_18D_s, 'Y', Y_cont_5D, 'H', 5,  'type', 'continuous', 'embargo', 20), ...
        struct('name', 'G4: 60D Horizon',  'X', X_gics_18D_s, 'Y', Y_cont_60D,'H', 60, 'type', 'continuous', 'embargo', 60) ...
    };

    % 逐組執行訓練
    for g_idx = 1:3
        grp = groups_seed{g_idx};
        H_val = grp.H;
        emb_val = grp.embargo;
        
        split_pt = round((numDays - H_val) * 0.75);
        train_days = seqLen : (split_pt - emb_val);
        val_days   = (split_pt + emb_val) : (numDays - H_val);
        batches_per_ep = floor(length(train_days) / batch_size);
        
        smokeConfig = configObj;
        smokeConfig.NumTickers = sample_T;
        factory = BuildDecoupledExtractors(smokeConfig, numExtractorFeats, 'pure_lstm');
        [net_t, ~] = factory.buildNetworks();
        
        W_head = dlarray(randn(64, 1, 'single') * 0.01);
        b_head = dlarray(zeros(1, 1, 'single'));
        avgG = []; avgsqG = [];
        
        X_in = grp.X;
        Y_in = grp.Y;
        
        for ep = 1:epochs
            shuffled = train_days(randperm(length(train_days)));
            for b = 1:batches_per_ep
                iter_idx = (ep - 1) * batches_per_ep + b;
                b_days = shuffled((b - 1) * batch_size + 1 : b * batch_size);
                B = length(b_days);
                
                X_batch = zeros(numExtractorFeats, sample_T * B, seqLen, 'single');
                Y_batch = zeros(sample_T * B, 1, 'single');
                Act_batch = false(sample_T * B, 1);
                
                for i = 1:B
                    t_c = b_days(i);
                    t_seq = (t_c - seqLen + 1) : t_c;
                    cols = (i - 1) * sample_T + 1 : i * sample_T;
                    X_batch(:, cols, :) = permute(X_in(t_seq, :, :), [2, 3, 1]);
                    Y_batch(cols)       = single(Y_in(t_c, :)');
                    Act_batch(cols)     = (Expert_s(t_c, :) & ~isnan(Y_in(t_c, :)))';
                end
                
                dl_x = dlarray(X_batch, 'CBT');
                if use_gpu, dl_x = gpuArray(dl_x); end
                
                if strcmp(grp.type, 'binary')
                    [l_b, g_net, g_w, g_b] = dlfeval(@compute_bce_gradient, net_t, W_head, b_head, dl_x, Y_batch, Act_batch);
                else
                    [l_b, g_net, g_w, g_b] = dlfeval(@compute_soft_ic_gradient, net_t, W_head, b_head, dl_x, Y_batch, Act_batch, sample_T, B);
                end
                
                [net_t, avgG, avgsqG] = adamupdate(net_t, g_net, avgG, avgsqG, iter_idx, lr);
                W_head = W_head - lr * g_w;
                b_head = b_head - lr * g_b;
            end
        end
        
        % 驗證損失與探針評估 (貫穿 max_lag = H)
        val_loss_final = evaluate_val_chunked(net_t, W_head, b_head, X_in, Y_in, Expert_s, val_days, seqLen, sample_T, grp.type, use_gpu);
        [probe_metric, p_hac_corr] = evaluate_probe_hac_fast(net_t, X_in, Y_in, Expert_s, val_days, seqLen, grp.type, use_gpu, H_val);
        
        if g_idx == 1
            res_g1_loss(s) = val_loss_final;
            fprintf('  [Seed %d] G1 Base BCE     -> Val Loss: %.4f | Probe AUC: %.4f\n', curr_seed, val_loss_final, probe_metric);
        elseif g_idx == 2
            res_g3_loss(s) = val_loss_final;
            res_g3_ic(s)   = probe_metric;
            res_g3_phac(s) = p_hac_corr;
            fprintf('  [Seed %d] G3 Soft-IC Loss -> Val Loss: %.4f | Rank IC: %+.4f (p=%.4f)\n', curr_seed, val_loss_final, probe_metric, p_hac_corr);
        else
            res_g4_loss(s) = val_loss_final;
            res_g4_ic(s)   = probe_metric;
            res_g4_phac(s) = p_hac_corr;
            fprintf('  [Seed %d] G4 60D Horizon  -> Val Loss: %.4f | Rank IC: %+.4f (p=%.4f)\n', curr_seed, val_loss_final, probe_metric, p_hac_corr);
        end
    end
end

%% 4. 統計推論：跨種子非參數檢定 (Wilcoxon Signed-Rank Test) [§2.2 規範]
disp(' ');
disp('====================================================================================================');
disp('📋 【Task B 核心產出表：5 組隨機種子獨立子集穩健性全量對照表】');
disp('====================================================================================================');
fprintf(' Seed | G1 Base Val Loss | G3 Soft-IC Val Loss | G1 - G3 損失差 | G3 OOF Rank IC | G4 60D OOF Rank IC (p-val)\n');
fprintf('----------------------------------------------------------------------------------------------------\n');

loss_diff_g1_g3 = res_g1_loss - res_g3_loss;

for s = 1:n_seeds
    fprintf('  %3d |      %.4f      |       %.4f        |    %+.4f    |    %+.4f     |    %+.4f (p=%.4f)\n', ...
        seeds(s), res_g1_loss(s), res_g3_loss(s), loss_diff_g1_g3(s), res_g3_ic(s), res_g4_ic(s), res_g4_phac(s));
end
fprintf('----------------------------------------------------------------------------------------------------\n');
fprintf(' 均值 |      %.4f      |       %.4f        |    %+.4f    |    %+.4f     |    %+.4f\n', ...
    mean(res_g1_loss), mean(res_g3_loss), mean(loss_diff_g1_g3), mean(res_g3_ic), mean(res_g4_ic));
fprintf('====================================================================================================\n\n');

% Wilcoxon Signed-Rank 統計檢定 (種子彼此獨立，無需 HAC 校正)
p_wilcoxon_g1_g3 = signrank(res_g1_loss, res_g3_loss, 'tail', 'right'); %
p_wilcoxon_g3_ic = signrank(res_g3_ic, 0, 'tail', 'right');             %
p_wilcoxon_g4_ic = signrank(res_g4_ic, 0, 'tail', 'right');             %

fprintf('📊 【Task B 非參數統計推論報告 (Wilcoxon Signed-Rank Test, n=5)】\n');
fprintf('  > 假說 1 (Direction 2 穩健性): G1 Loss > G3 Loss (Soft-IC 能否持續突破 BCE)?\n');
fprintf('    - Wilcoxon 統計量 p-value = %.4f\n', p_wilcoxon_g1_g3);
if p_wilcoxon_g1_g3 < 0.05
    fprintf('    - 判定: ⭐ DIRECTION 2 ROBUST CONFIRMED (跨 5 組種子損失均顯著改善)！\n');
else
    fprintf('    - 判定: ⚠️ DIRECTION 2 MARGINAL (跨種子差異未達顯著)。\n');
end

fprintf('  > 假說 2 (Direction 3 穩健性): G4 (60D) 預測 Rank IC 是否跨種子顯著大於 0?\n');
fprintf('    - 跨種子平均 IC = %+.4f | Wilcoxon p-value = %.4f\n', mean(res_g4_ic), p_wilcoxon_g4_ic);
if p_wilcoxon_g4_ic < 0.05 && mean(res_g4_ic) > 0.02
    fprintf('    - 判定: ⭐ DIRECTION 3 ROBUST CONFIRMED (60D 動態隔離具備穩定正向預測力)！\n');
else
    fprintf('    - 判定: ✅ DIRECTION 3 FALSIFIED (跨種子 IC 邊際薄弱或不穩定)。\n');
end

fprintf('\n📢 【誠實聲明（Stage 3 規範）】\n');
fprintf('   n=5 組種子樣本仍屬於小樣本檢定，此結果旨在將結論由「n=1 無法排除抽樣運氣」\n');
fprintf('   提升至「具備初步子集穩健性證據」，並非宣稱已達大數絕對證明。\n');
fprintf('====================================================================================================\n');

%% 5. 儲存多種子檢定結果
summary_taskB = table(seeds', res_g1_loss, res_g3_loss, loss_diff_g1_g3, res_g3_ic, res_g4_ic, res_g4_phac, ...
    'VariableNames', {'Seed', 'G1_Loss', 'G3_Loss', 'Loss_Diff', 'G3_Rank_IC', 'G4_Rank_IC', 'G4_HAC_P'});
if ~exist(configObj.ResultDir, 'dir'), mkdir(configObj.ResultDir); end
savePath = fullfile(configObj.ResultDir, 'Stage2_5_MultiSeed_Summary.mat');
save(savePath, 'summary_taskB', 'p_wilcoxon_g1_g3', 'p_wilcoxon_g3_ic', 'p_wilcoxon_g4_ic');
fprintf('\n💾 多種子檢定匯總已儲存至: %s\n', savePath);

%% =====================================================================
% 輔助函數區
% =====================================================================
function [loss, grad_net, grad_w, grad_b] = compute_bce_gradient(net, W, b, dl_x, y_true, act_m)
    emb = forward(net, dl_x);
    emb = stripdims(emb);
    logits = W' * emb + b;
    probs = sigmoid(logits);
    y_t = y_true(act_m); y_t = y_t(:);
    p_t = probs(act_m); p_t = p_t(:);
    loss = -mean(y_t .* log(p_t + 1e-8) + (1 - y_t) .* log(1 - p_t + 1e-8));
    [grad_net, grad_w, grad_b] = dlgradient(loss, net.Learnables, W, b);
end

function [loss, grad_net, grad_w, grad_b] = compute_soft_ic_gradient(net, W, b, dl_x, y_true, act_m, numT, B)
    emb = forward(net, dl_x);
    emb = stripdims(emb);
    pred = W' * emb + b;
    loss = BuildDecoupledExtractors.compute_continuous_return_loss(pred, y_true, act_m, numT, B, 0.5);
    [grad_net, grad_w, grad_b] = dlgradient(loss, net.Learnables, W, b);
end

function val_loss = evaluate_val_chunked(net, W, b, X_in, Y_in, Expert, val_days, seqLen, sample_T, loss_type, use_gpu)
    V = length(val_days);
    chunk_size = 32;
    loss_sum = 0;
    count = 0;
    numFeats = size(X_in, 2);
    
    for c = 1:ceil(V / chunk_size)
        c_idx = (c - 1) * chunk_size + 1 : min(c * chunk_size, V);
        c_days = val_days(c_idx);
        C = length(c_days);
        
        X_chunk   = zeros(numFeats, sample_T * C, seqLen, 'single');
        Y_chunk   = zeros(sample_T * C, 1, 'single');
        Act_chunk = false(sample_T * C, 1);
        
        for i = 1:C
            t_c = c_days(i);
            t_seq = (t_c - seqLen + 1) : t_c;
            cols = (i - 1) * sample_T + 1 : i * sample_T;
            X_chunk(:, cols, :) = permute(X_in(t_seq, :, :), [2, 3, 1]);
            Y_chunk(cols)       = single(Y_in(t_c, :)');
            Act_chunk(cols)     = (Expert(t_c, :) & ~isnan(Y_in(t_c, :)))';
        end
        
        dl_x = dlarray(X_chunk, 'CBT');
        if use_gpu, dl_x = gpuArray(dl_x); end
        
        emb = forward(net, dl_x);
        emb = stripdims(emb);
        logits = W' * emb + b;
        
        if strcmp(loss_type, 'binary')
            p_t = sigmoid(logits(Act_chunk));
            y_t = Y_chunk(Act_chunk);
            bce_c = -sum(y_t(:) .* log(p_t(:) + 1e-8) + (1 - y_t(:)) .* log(1 - p_t(:) + 1e-8), 'all');
            loss_sum = loss_sum + double(extractdata(gather(bce_c)));
        else
            pred_t = logits(Act_chunk);
            y_t = Y_chunk(Act_chunk);
            huber_c = sum(abs(pred_t(:) - y_t(:)), 'all');
            loss_sum = loss_sum + double(extractdata(gather(huber_c)));
        end
        count = count + sum(Act_chunk, 'all');
    end
    val_loss = loss_sum / max(1, count);
end

function [probe_metric, p_hac_corr] = evaluate_probe_hac_fast(net, X_in, Y_in, Expert, val_days, seqLen, target_type, use_gpu, H_horizon)
    V = length(val_days);
    total_val = sum(Expert(val_days, :), 'all');
    E_val = zeros(total_val, 64, 'single');
    Y_val = zeros(total_val, 1, 'single');
    cur_row = 1;
    
    for i = 1:V
        t_c = val_days(i);
        t_seq = (t_c - seqLen + 1) : t_c;
        act_m = Expert(t_c, :) & ~isnan(Y_in(t_c, :));
        n_act = sum(act_m);
        
        X_day = permute(X_in(t_seq, :, :), [2, 3, 1]);
        dl_x = dlarray(X_day, 'CBT');
        if use_gpu, dl_x = gpuArray(dl_x); end
        
        emb_day = extractdata(gather(forward(net, dl_x)));
        y_day   = single(Y_in(t_c, :)');
        
        if n_act > 0
            cols = find(act_m);
            E_val(cur_row : cur_row + n_act - 1, :) = emb_day(:, cols)';
            Y_val(cur_row : cur_row + n_act - 1)    = y_day(cols);
            cur_row = cur_row + n_act;
        end
    end
    
    E_val = E_val(1:cur_row-1, :);
    Y_val = Y_val(1:cur_row-1);
    valid_samples = ~isnan(Y_val) & ~isinf(Y_val);
    E_val = E_val(valid_samples, :);
    Y_val = Y_val(valid_samples);
    
    if strcmp(target_type, 'binary')
        mdl = fitclinear(E_val, Y_val, 'Learner', 'logistic');
        [~, scores] = predict(mdl, E_val);
        [~, ~, ~, probe_metric] = perfcurve(Y_val, scores(:, 2), 1);
        p_hac_corr = 1.0;
    else
        mdl = fitrlinear(E_val, Y_val, 'Learner', 'leastsquares', 'Regularization', 'ridge');
        preds = predict(mdl, E_val);
        
        daily_ic = zeros(V, 1);
        v_idx = 0;
        row_c = 1;
        for i = 1:V
            act_m_day = Expert(val_days(i), :) & ~isnan(Y_in(val_days(i), :));
            n_act = sum(act_m_day);
            if n_act >= 10 && (row_c + n_act - 1) <= length(preds)
                v_idx = v_idx + 1;
                yp = preds(row_c : row_c + n_act - 1);
                yt = Y_val(row_c : row_c + n_act - 1);
                daily_ic(v_idx) = corr(yp, yt, 'Type', 'Spearman', 'Rows', 'complete');
            end
            row_c = row_c + n_act;
        end
        daily_ic = daily_ic(1:v_idx);
        probe_metric = mean(daily_ic, 'omitnan');
        
        if length(daily_ic) >= 20
            hac_lag = max(H_horizon, floor(4 * (length(daily_ic)/100)^(2/9))); %
            [~, p_hac_corr] = hac_significance_test(daily_ic, hac_lag);         %
        else
            p_hac_corr = 1.0;
        end
    end
end