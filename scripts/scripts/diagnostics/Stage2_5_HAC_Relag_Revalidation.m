% =========================================================================
% 腳本：Stage2_5_HAC_Relag_Revalidation.m (Task A: Newey-West HAC 頻寬修正重驗證)
% 依據：《Stage 2.5 – 4 Code Review 計劃書》 §1.5 規範
% 職責：重跑 Group 1 (Base)、Group 3 (Soft-IC) 與 Group 4 (60D)；
%       比較自動頻寬與強制頻寬 (max_lag = H) 之 HAC 檢定顯著性變化，
%       直接裁決 Direction 3 (60D 中長週期) 的統計真確性。
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🔬 [Stage 2.5 Task A] 啟動 HAC max_lag 頻寬下限貫穿重驗證');
disp('=================================================================');

%% 0. 環境路徑掛載與隨機種子鎖定
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
if isprop(configObj, 'RNG_Seed')
    rng(configObj.RNG_Seed, 'twister');
else
    rng(42, 'twister');
end

%% 1. 載入特徵快取與子集抽樣 (60 檔 x 3200 天)
disp('--- 步驟 1：載入特徵快取並進行跨週期抽樣 (60 檔 x 3200 天) ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
if ~exist(cachePath, 'file')
    error('❌ 找不到特徵快取檔案，請先確認已執行 1_Run_Data_and_Features.m！');
end

load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC';
numDaysRaw_full = length(Dates_Active);
numT_full = configObj.NumTickers;

sample_T = min(60, numT_full);
active_sums = sum(Expert_Active, 1);
[~, top_active] = sort(active_sums, 'descend');
sub_tickers = top_active(1:sample_T);

start_day = max(1, round(numDaysRaw_full * 0.4));
end_day   = min(numDaysRaw_full, start_day + 3200 - 1);
sub_days  = start_day : end_day;

numExtractorFeats = 3 + configObj.NumMicroFeatures; % 18 維

Prices_sub    = single(Prices_Active(sub_days, sub_tickers));
Expert_sub    = logical(Expert_Active(sub_days, sub_tickers));
Dates_sub     = Dates_Active(sub_days);
X_sub_raw_18D = single(X_norm_3D(sub_days, 1:numExtractorFeats, sub_tickers));
numDays       = length(Dates_sub);
seqLen        = configObj.SeqLen;

clear X_norm_3D Prices_Active Expert_Active Dates_Active;

fprintf('  📊 樣本規模：標的 %d 檔 | 交易天數 %d 天 | 特徵維度 %d 維\n', ...
    sample_T, numDays, numExtractorFeats);

%% 2. 構建 GICS 產業內中性化特徵矩陣
disp('--- 步驟 2：構建 GICS 產業內橫截面 Z-Score 特徵 ---');
universePath = fullfile(projectRoot, 'data', 'crawlers', 'us_universe.csv');
ticker_list = configObj.IdxTickers(sub_tickers);
sector_map = repmat({'Unknown'}, 1, sample_T);

if exist(universePath, 'file')
    u_tbl = readtable(universePath, 'TextType', 'string');
    if ismember('GICS_Sector', u_tbl.Properties.VariableNames)
        [lia, loc] = ismember(string(ticker_list), string(u_tbl.Ticker));
        valid_loc = loc(lia);
        valid_idx = find(lia);
        for k = 1:length(valid_idx)
            s_val = char(u_tbl.GICS_Sector(valid_loc(k)));
            if ~isempty(strtrim(s_val)), sector_map{valid_idx(k)} = strtrim(s_val); end
        end
    end
end
sectors_cat = categorical(sector_map);
unique_sectors = categories(sectors_cat);

X_sub_gics_18D = X_sub_raw_18D;
for t = 1:numDays
    act_m = Expert_sub(t, :);
    if sum(act_m) >= 10
        vals_all = X_sub_raw_18D(t, :, act_m);
        mu_all = mean(vals_all, 3, 'omitnan');
        std_all = std(vals_all, 0, 3, 'omitnan') + 1e-8;
        X_sub_gics_18D(t, :, act_m) = (vals_all - mu_all) ./ std_all;
        
        for s = 1:length(unique_sectors)
            s_name = unique_sectors{s};
            if ismember(s_name, {'Unknown', 'Macro', 'Safe Haven', 'Broad Market Index'}), continue; end
            sec_m = act_m & (sectors_cat == s_name);
            if sum(sec_m) >= 4
                vals_sec = X_sub_raw_18D(t, :, sec_m);
                mu_sec = mean(vals_sec, 3, 'omitnan');
                std_sec = std(vals_sec, 0, 3, 'omitnan') + 1e-8;
                X_sub_gics_18D(t, :, sec_m) = (vals_sec - mu_sec) ./ std_sec;
            end
        end
    end
end
X_sub_gics_18D(isnan(X_sub_gics_18D) | isinf(X_sub_gics_18D)) = 0;

%% 3. 構建 5D 與 60D 標籤 (防禦 NaN)
disp('--- 步驟 3：構建 5D 與 60D 目標標籤 ---');
R_fwd_5D = (Prices_sub(6:end, :) - Prices_sub(1:end-5, :)) ./ (Prices_sub(1:end-5, :) + 1e-8);
R_fwd_5D(isnan(R_fwd_5D) | isinf(R_fwd_5D)) = NaN;
Y_bin_5D  = false(numDays, sample_T);
Y_cont_5D = zeros(numDays, sample_T, 'single');

for t = 1:numDays-5
    act_m = Expert_sub(t, :) & ~isnan(R_fwd_5D(t, :)) & ~isinf(R_fwd_5D(t, :));
    if sum(act_m) >= 5
        r_t = R_fwd_5D(t, act_m);
        med_r = median(r_t, 'omitnan');
        Y_bin_5D(t, act_m) = (r_t > med_r);
        Y_cont_5D(t, act_m) = (r_t - mean(r_t, 'omitnan')) ./ (std(r_t, 0, 'omitnan') + 1e-6);
    end
end
Y_cont_5D(isnan(Y_cont_5D) | isinf(Y_cont_5D)) = 0;

R_fwd_60D = (Prices_sub(61:end, :) - Prices_sub(1:end-60, :)) ./ (Prices_sub(1:end-60, :) + 1e-8);
R_fwd_60D(isnan(R_fwd_60D) | isinf(R_fwd_60D)) = NaN;
Y_cont_60D = zeros(numDays, sample_T, 'single');

for t = 1:numDays-60
    act_m = Expert_sub(t, :) & ~isnan(R_fwd_60D(t, :)) & ~isinf(R_fwd_60D(t, :));
    if sum(act_m) >= 5
        r_t = R_fwd_60D(t, act_m);
        Y_cont_60D(t, act_m) = (r_t - mean(r_t, 'omitnan')) ./ (std(r_t, 0, 'omitnan') + 1e-6);
    end
end
Y_cont_60D(isnan(Y_cont_60D) | isinf(Y_cont_60D)) = 0;
clear R_fwd_5D R_fwd_60D Prices_sub;

%% 4. 定義 Task A 核心重驗證組別 (略過 Group 2 以節省算力)
exp_groups = { ...
    struct('id', 1, 'name', 'Group 1: Raw 18D + BCE (Base)',       'X', X_sub_raw_18D,  'Y', Y_bin_5D,  'H', 5,  'type', 'binary',     'embargo', 20), ...
    struct('id', 3, 'name', 'Group 3: GICS-Neutral + Soft-IC Loss','X', X_sub_gics_18D, 'Y', Y_cont_5D, 'H', 5,  'type', 'continuous', 'embargo', 20), ...
    struct('id', 4, 'name', 'Group 4: 60D Horizon (Embargo=60D)',  'X', X_sub_gics_18D, 'Y', Y_cont_60D,'H', 60, 'type', 'continuous', 'embargo', 60) ...
};

num_groups = length(exp_groups);
history_results = struct();

use_gpu = canUseGPU();
if use_gpu
    gpu_dev = gpuDevice();
    fprintf('  🎮 圖形加速卡: 【%s】，啟用 GPU 批次訓練。\n', gpu_dev.Name);
end

epochs = 15;
lr = 1e-3;
batch_size = 64;

%% 5. 執行模型訓練與日層級 IC 萃取
disp('--- 步驟 4：執行 Group 1、Group 3 與 Group 4 模型訓練 ---');

for g = 1:num_groups
    grp = exp_groups{g};
    H_val = grp.H;
    emb_val = grp.embargo;
    
    fprintf('\n▶ 正在執行 [%d/%d] %s ...\n', g, num_groups, grp.name);
    
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
    
    tic;
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
                Act_batch(cols)     = (Expert_sub(t_c, :) & ~isnan(Y_in(t_c, :)))';
            end
            
            dl_x = dlarray(X_batch, 'CBT');
            if use_gpu, dl_x = gpuArray(dl_x); end
            
            if strcmp(grp.type, 'binary')
                [~, g_net, g_w, g_b] = dlfeval(@compute_bce_gradient, ...
                    net_t, W_head, b_head, dl_x, Y_batch, Act_batch);
            else
                [~, g_net, g_w, g_b] = dlfeval(@compute_soft_ic_gradient, ...
                    net_t, W_head, b_head, dl_x, Y_batch, Act_batch, sample_T, B);
            end
            
            [net_t, avgG, avgsqG] = adamupdate(net_t, g_net, avgG, avgsqG, iter_idx, lr);
            W_head = W_head - lr * g_w;
            b_head = b_head - lr * g_b;
        end
    end
    t_sec = toc;
    fprintf('  ⚡ %s 訓練完成！耗時: %.2f 秒\n', grp.name, t_sec);
    
    % 提取日層級損失與完整 IC 序列
    [daily_losses, probe_metric, daily_ic, eval_days] = extract_daily_eval_stats(net_t, ...
        X_in, Y_in, Expert_sub, val_days, seqLen, grp.type, use_gpu);
    
    history_results(g).name         = grp.name;
    history_results(g).H            = grp.H;
    history_results(g).type         = grp.type;
    history_results(g).daily_loss   = daily_losses;
    history_results(g).probe_metric = probe_metric;
    history_results(g).daily_ic     = daily_ic;
    history_results(g).eval_days    = eval_days;
end

%% 6. Task A 核心產出表：HAC 頻寬敏感度與顯著性轉移分析
disp(' ');
disp('========================================================================================================================');
disp('📋 【Task A 核心產出表：Newey-West HAC max_lag 頻寬敏感度檢定分析】');
disp('========================================================================================================================');
fprintf(' Group                  | H  | N    | Auto Lag | Auto p-val | Forced Lag (H) | Corrected p-val | 結論是否改變\n');
fprintf('------------------------------------------------------------------------------------------------------------------------\n');

revalidation_summary = cell(num_groups, 8);

for g = 1:num_groups
    res = history_results(g);
    grp_name = res.name;
    H_val = res.H;
    
    if strcmp(res.type, 'binary')
        fprintf(' %-22s | %2d | %4d |   N/A    |   N/A      |      N/A       |     N/A         | 基準對照組 (不適用)\n', ...
            grp_name, H_val, length(res.daily_loss));
        revalidation_summary(g, :) = {grp_name, H_val, length(res.daily_loss), NaN, NaN, NaN, NaN, "Baseline"};
    else
        ic_vec = res.daily_ic;
        n_obs = length(ic_vec);
        
        % 1. 自動頻寬估計
        auto_lag = max(1, floor(4 * (n_obs / 100)^(2/9)));
        [~, p_auto] = hac_significance_test(ic_vec, auto_lag);
        
        % 2. 強制頻寬下限 (max_lag = max(H, auto_lag))
        forced_lag = max(H_val, auto_lag);
        [~, p_forced] = hac_significance_test(ic_vec, forced_lag);
        
        % 3. 判斷顯著性是否反轉
        sig_auto = (p_auto < 0.05);
        sig_forced = (p_forced < 0.05);
        
        if sig_auto == sig_forced
            conclusion_str = "保持不變 (穩健)";
        else
            conclusion_str = "⚠️ 顯著轉為不顯著 (偽訊號)";
        end
        
        fprintf(' %-22s | %2d | %4d |    %2d    |   %.4f   |       %2d       |     %.4f      | %s\n', ...
            grp_name, H_val, n_obs, auto_lag, p_auto, forced_lag, p_forced, conclusion_str);
        
        revalidation_summary(g, :) = {grp_name, H_val, n_obs, auto_lag, p_auto, forced_lag, p_forced, conclusion_str};
    end
end
fprintf('========================================================================================================================\n\n');

%% 7. 對比 Base 之配對損失差檢定 (同步強制 max_lag = H)
disp('--- 步驟 5：對比 Base 每日損失差 HAC 頻寬修正檢定 ---');
base_daily = history_results(1).daily_loss;

for g = 2:num_groups
    res = history_results(g);
    min_len = min(length(res.daily_loss), length(base_daily));
    diff_d = base_daily(1:min_len) - res.daily_loss(1:min_len);
    
    n_diff = length(diff_d);
    auto_lag_diff = max(1, floor(4 * (n_diff / 100)^(2/9)));
    forced_lag_diff = max(res.H, auto_lag_diff);
    
    [~, p_pair_auto]   = hac_significance_test(diff_d, auto_lag_diff);
    [~, p_pair_forced] = hac_significance_test(diff_d, forced_lag_diff);
    
    fprintf('  > %s vs Base:\n', res.name);
    fprintf('     - Auto Lag   = %2d | p-value = %.4f\n', auto_lag_diff, p_pair_auto);
    fprintf('     - Forced Lag = %2d | p-value = %.4f\n', forced_lag_diff, p_pair_forced);
end
disp('========================================================================================================================');

%% 8. 儲存檢定數據與日層級序列
if ~exist(configObj.ResultDir, 'dir'), mkdir(configObj.ResultDir); end
savePath = fullfile(configObj.ResultDir, 'Stage2_5_HAC_Relag_Summary.mat');
save(savePath, 'history_results', 'revalidation_summary');
fprintf('💾 檢定對照數據已成功匯出至: %s\n', savePath);

disp('=================================================================');
disp('🎯 [Stage 2.5 Task A] 執行完畢！請查閱上方核心產出表以決定 Stage 3 定稿措辭。');
disp('=================================================================');

%% =====================================================================
% 輔助函數：二元交叉熵 (BCE) 損失與梯度
% =====================================================================
function [loss, grad_net, grad_w, grad_b] = compute_bce_gradient(net, W, b, dl_x, y_true, act_m)
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
% 輔助函數：連續回歸 Huber + 截面 Soft-IC 損失與梯度
% =====================================================================
function [loss, grad_net, grad_w, grad_b] = compute_soft_ic_gradient(net, W, b, dl_x, y_true, act_m, numT, B)
    emb = forward(net, dl_x);
    emb = stripdims(emb);
    pred = W' * emb + b;
    
    loss = BuildDecoupledExtractors.compute_continuous_return_loss(pred, y_true, act_m, numT, B, 0.5);
    [grad_net, grad_w, grad_b] = dlgradient(loss, net.Learnables, W, b);
end

%% =====================================================================
% 輔助函數：提取日層級損失與完整 IC 序列 (支援 Task A 統計重檢定)
% =====================================================================
function [daily_losses, probe_metric, daily_ic, val_days_clean] = extract_daily_eval_stats(net, ...
    X_in, Y_in, Expert, val_days, seqLen, target_type, use_gpu)

    V = length(val_days);
    daily_losses = zeros(V, 1);
    
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
        
        if strcmp(target_type, 'binary')
            daily_losses(i) = log(2);
        else
            daily_losses(i) = mean(abs(y_day(act_m)), 'omitnan');
        end
    end
    
    E_val = E_val(1:cur_row-1, :);
    Y_val = Y_val(1:cur_row-1);
    
    valid_samples = ~isnan(Y_val) & ~isinf(Y_val);
    E_val = E_val(valid_samples, :);
    Y_val = Y_val(valid_samples);
    
    val_days_clean = val_days;
    
    if strcmp(target_type, 'binary')
        mdl = fitclinear(E_val, Y_val, 'Learner', 'logistic');
        [~, scores] = predict(mdl, E_val);
        [~, ~, ~, probe_metric] = perfcurve(Y_val, scores(:, 2), 1);
        daily_ic = [];
    else
        mdl = fitrlinear(E_val, Y_val, 'Learner', 'leastsquares', 'Regularization', 'ridge');
        preds = predict(mdl, E_val);
        
        daily_ic_temp = zeros(V, 1);
        v_idx = 0;
        row_c = 1;
        
        for i = 1:V
            act_m_day = Expert(val_days(i), :) & ~isnan(Y_in(val_days(i), :));
            n_act = sum(act_m_day);
            if n_act >= 10 && (row_c + n_act - 1) <= length(preds)
                v_idx = v_idx + 1;
                yp = preds(row_c : row_c + n_act - 1);
                yt = Y_val(row_c : row_c + n_act - 1);
                daily_ic_temp(v_idx) = corr(yp, yt, 'Type', 'Spearman', 'Rows', 'complete');
            end
            row_c = row_c + n_act;
        end
        
        daily_ic = daily_ic_temp(1:v_idx);
        probe_metric = mean(daily_ic, 'omitnan');
    end
end
    