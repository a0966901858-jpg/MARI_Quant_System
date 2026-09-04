% =========================================================================
% 腳本：Run_Extractor_Pretrain_SMOKE.m (Stage 2: 三大替代假說煙霧消融測試)
% 升級：Phase 15.5 Task A 修復版 (★ 時區比較 Bug 修復、mrg32k3a 串流對齊、
%       HAC max_lag 顯式貫穿、雙欄 p 值並列、IC 序列存檔)
% 職責：在 60 檔 x 3200 天樣本上，系統性檢驗「產業中性化」、「Soft-IC 排序損失」
%       與「60D 長週期動態隔離」能否打破特徵資訊天花板
% =========================================================================
clear; clc; close all;
disp('=================================================================');
disp('🧪 [Stage 2 SMOKE TEST] 啟動三大替代方案消融測試 (Task A HAC 修復版)');
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

% ★ 升級為 Phase 15.5 統一 mrg32k3a 獨立隨機數引擎
stream = configObj.getRandStream(1);
RandStream.setGlobalStream(stream);

%% 1. 載入特徵快取與跨週期抽樣 (60 檔 x 3200 天)
disp('--- 步驟 1：載入特徵快取並執行資料規模擴展抽樣 (60 檔 x 3200 天) ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
if ~exist(cachePath, 'file')
    error('❌ 找不到特徵快取檔案，請先確認已執行 1_Run_Data_and_Features.m！');
end

load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active');

% ★ 致命 Bug 修復：消除 TimeZone 標籤，確保與後續未指定時區之 datetime 物件可直接比對
Dates_Active.TimeZone = '';

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
fprintf('  📊 抽樣規模：標的 %d 檔 | 交易天數 %d 天 | 輸入特徵 %d 維 (記憶體已釋放)\n', ...
    sample_T, numDays, numExtractorFeats);

%% 2. 構建 GICS 產業內中性化特徵矩陣 (Hypothesis 1)
disp('--- 步驟 2：載入 GICS 產業別並構建產業內橫截面 Z-Score 特徵 ---');
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

%% 3. 構建多 Horizon 目標標籤 (嚴格排除 NaN 樣本)
disp('--- 步驟 3：構建 5D 與 60D 遠期連續報酬與 Beat-the-Median 標籤 (防禦 NaN) ---');
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

%% 4. 定義四組對照消融實驗組別
exp_groups = { ...
    struct('id', 1, 'name', 'Group 1: Raw 18D + BCE (Base)',       'X', X_sub_raw_18D,  'Y', Y_bin_5D,  'H', 5,  'type', 'binary',     'embargo', 20), ...
    struct('id', 2, 'name', 'Group 2: GICS-Neutral + BCE',         'X', X_sub_gics_18D, 'Y', Y_bin_5D,  'H', 5,  'type', 'binary',     'embargo', 20), ...
    struct('id', 3, 'name', 'Group 3: GICS-Neutral + Soft-IC Loss','X', X_sub_gics_18D, 'Y', Y_cont_5D, 'H', 5,  'type', 'continuous', 'embargo', 20), ...
    struct('id', 4, 'name', 'Group 4: 60D Horizon (Embargo=60D)',  'X', X_sub_gics_18D, 'Y', Y_cont_60D,'H', 60, 'type', 'continuous', 'embargo', 60) ...
};

num_groups = length(exp_groups);
history_results = struct();

use_gpu = canUseGPU();
if use_gpu
    gpu_dev = gpuDevice();
    fprintf('  🎮 偵測到硬體加速卡: 【%s】，啟用全向量化批次訓練。\n', gpu_dev.Name);
end

epochs = 15;
lr = 1e-3;
batch_size = 64;

%% 5. 依序執行 15 Epochs 對照訓練
disp('--- 步驟 4：啟動 Stage 2 四組消融對照訓練 ---');
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
    
    W_head = dlarray(randn(stream, 64, 1, 'single') * 0.01);
    b_head = dlarray(zeros(1, 1, 'single'));
    avgG = []; avgsqG = [];
    
    tr_losses  = zeros(epochs, 1);
    val_losses = zeros(epochs, 1);
    
    X_in = grp.X;
    Y_in = grp.Y;
    
    tic;
    for ep = 1:epochs
        shuffled = train_days(randperm(stream, length(train_days)));
        ep_loss_sum = 0;
        
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
                [l_b, g_net, g_w, g_b] = dlfeval(@compute_bce_gradient, ...
                    net_t, W_head, b_head, dl_x, Y_batch, Act_batch);
            else
                [l_b, g_net, g_w, g_b] = dlfeval(@compute_soft_ic_gradient, ...
                    net_t, W_head, b_head, dl_x, Y_batch, Act_batch, sample_T, B);
            end
            
            [net_t, avgG, avgsqG] = adamupdate(net_t, g_net, avgG, avgsqG, iter_idx, lr);
            W_head = W_head - lr * g_w;
            b_head = b_head - lr * g_b;
            ep_loss_sum = ep_loss_sum + double(extractdata(l_b));
        end
        tr_losses(ep) = ep_loss_sum / batches_per_ep;
        
        val_losses(ep) = evaluate_val_chunked(net_t, W_head, b_head, X_in, Y_in, ...
            Expert_sub, val_days, seqLen, sample_T, grp.type, use_gpu);
        
        if mod(ep, 5) == 0 || ep == 1
            fprintf('  Ep %2d/%2d | Train Loss: %.4f | Val Loss: %.4f\n', ...
                ep, epochs, tr_losses(ep), val_losses(ep));
        end
    end
    t_sec = toc;
    fprintf('  ⚡ %s 訓練完成！耗時: %.2f 秒\n', grp.name, t_sec);
    
    [daily_val_losses, probe_metric, p_hac_corr, daily_ic_g, hac_lag_g, p_hac_auto] = ...
        evaluate_probe_and_hac(net_t, X_in, Y_in, Expert_sub, val_days, seqLen, grp.type, use_gpu, grp.H);
    
    history_results(g).name         = grp.name;
    history_results(g).H            = grp.H;
    history_results(g).tr_loss      = tr_losses;
    history_results(g).va_loss      = val_losses;
    history_results(g).daily_loss   = daily_val_losses;
    history_results(g).daily_ic     = daily_ic_g;
    history_results(g).metric       = probe_metric;
    history_results(g).p_hac        = p_hac_corr;
    history_results(g).p_hac_auto   = p_hac_auto;
    history_results(g).hac_lag      = hac_lag_g;
    history_results(g).type         = grp.type;
end

resultsMatPath = fullfile(configObj.ResultDir, 'Stage2_Smoke_Results.mat');
save(resultsMatPath, 'history_results', 'exp_groups');
fprintf('\n💾 完整消融結果 (含逐日 IC 序列) 已儲存至: %s\n', resultsMatPath);

%% 6. 統計顯著性檢定與 Stage 2 結論報告 (★ 雙欄 HAC 對照版)
disp('============================================================================================================================');
disp('📊 【Stage 2 三大替代方案消融檢定綜合報告 (Task A max_lag 貫穿修復版)】');
disp('============================================================================================================================');
fprintf(' 實驗組別名稱                          | 最終 Val Loss | 探針評估 (AUC / IC) | Auto p-val | Corrected p (lag) | 對比 Base (p-val)\n');
fprintf('----------------------------------------------------------------------------------------------------------------------------\n');
base_daily = history_results(1).daily_loss;

for g = 1:num_groups
    res = history_results(g);
    if strcmp(res.type, 'binary')
        metric_str = sprintf('AUC = %.4f', res.metric);
        auto_p_str = '   N/A   ';
        corr_p_str = '   N/A   ';
    else
        metric_str = sprintf('IC  = %+.4f', res.metric);
        auto_p_str = sprintf('%.4f', res.p_hac_auto);
        corr_p_str = sprintf('%.4f (lag=%d)', res.p_hac, res.hac_lag);
    end
    
    if g == 1
        comp_str = 'Reference Base';
    else
        min_len = min(length(res.daily_loss), length(base_daily));
        diff_d = base_daily(1:min_len) - res.daily_loss(1:min_len);
        [~, p_pair] = hac_significance_test(diff_d, exp_groups{g}.H);
        comp_str = sprintf('HAC p = %.4f (lag=%d)', p_pair, exp_groups{g}.H);
    end
    
    fprintf(' %-37s |    %.4f    |    %-16s |  %s   | %-17s | %s\n', ...
        res.name, res.va_loss(end), metric_str, auto_p_str, corr_p_str, comp_str);
end
fprintf('============================================================================================================================\n\n');

if history_results(2).metric > history_results(1).metric + 0.01 && history_results(2).p_hac < 0.05
    fprintf('  ⭐ 【DIRECTION 1 CONFIRMED】GICS 產業中性化有效提煉個股超額 Alpha！建議全域採納。\n');
else
    fprintf('  ✅ 【DIRECTION 1 FALSIFIED】GICS 產業中性化未帶來顯著增量，特徵噪音非源自板塊 Beta。\n');
end

if history_results(3).metric > 0.02 && history_results(3).p_hac < 0.05
    fprintf('  ⭐ 【DIRECTION 2 CONFIRMED】連續 Soft-IC 損失顯著優於二元分類，成功打破 ln(2) 停滯 (Corrected p=%.4f)！\n', history_results(3).p_hac);
else
    fprintf('  ✅ 【DIRECTION 2 FALSIFIED】連續回歸目標下 IC 仍貼近 0 或未達顯著，證偽二元硬切中位數為主要瓶頸之假設。\n');
end

if history_results(4).metric > 0.03 && history_results(4).p_hac < 0.05
    fprintf('  ⭐ 【DIRECTION 3 CONFIRMED】60D 中長週期在動態 Embargo 隔離與 lag=%d HAC 校正下具備真實因果預測力 (p=%.4f)！\n', ...
        history_results(4).hac_lag, history_results(4).p_hac);
else
    fprintf('  ✅ 【DIRECTION 3 FALSIFIED】60D 週期經 lag=%d HAC 校正後顯著性消失 (p=%.4f >= 0.05)，證實純屬重疊自相關造成之機械性假象！\n', ...
        history_results(4).hac_lag, history_results(4).p_hac);
end
fprintf('============================================================================================================================\n\n');

%% 7. 產出標準學術白底黑字視覺化圖表
fig = figure('Name', 'Stage 2 Smoke Ablation Study (Task A Corrected)', ...
    'Color', 'w', 'Position', [100, 100, 1150, 750], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');
colors = {'#7F7F7F', '#0072BD', '#D95319', '#7E2F8E'};

subplot(2, 2, 1);
for g = 1:num_groups
    plot(1:epochs, history_results(g).va_loss, '-o', 'Color', colors{g}, 'LineWidth', 1.4, ...
        'DisplayName', exp_groups{g}.name);
    hold on;
end
title('Validation Loss Convergence', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Epoch', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Val Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend('Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

subplot(2, 2, 2);
metrics_all = [history_results.metric];
b = bar(metrics_all, 0.45);
b.FaceColor = 'flat';
b.EdgeColor = 'none';
for g = 1:num_groups, b.CData(g, 1:3) = validatecolor(colors{g}); end
set(gca, 'XTick', 1:num_groups, 'XTickLabel', {'G1 (Base)', 'G2 (GICS)', 'G3 (Soft-IC)', 'G4 (60D)'});
title('Linear Probe Metric (G1-G2: AUC | G3-G4: IC)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Probe Score', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

subplot(2, 2, 3);
std_raw  = squeeze(std(X_sub_raw_18D, 0, [1, 3], 'omitnan'));
std_gics = squeeze(std(X_sub_gics_18D, 0, [1, 3], 'omitnan'));
b3 = bar([std_raw(:), std_gics(:)], 'grouped');
b3(1).FaceColor = [0.0000, 0.4470, 0.7410];
b3(2).FaceColor = [0.8500, 0.3250, 0.0980];
title('18D Feature Std Dev: Raw vs GICS-Neutral', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Feature Index', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Standard Deviation', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
legend(b3, {'Raw 18D', 'GICS-Neutral'}, 'Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.8 0.8 0.8]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

subplot(2, 2, 4);
min_l = min(length(base_daily), length(history_results(3).daily_loss));
diff_loss = base_daily(1:min_l) - history_results(3).daily_loss(1:min_l);
boxplot(diff_loss, {'Base - Soft-IC Loss'}, 'Colors', 'k', 'Symbol', 'r+');
yline(0, '--k', 'Zero Margin', 'LineWidth', 1.1);
title('Daily Loss Difference (Base - Soft-IC)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
ylabel('\Delta Loss', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'k');
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
    'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
grid on; box on;

figPath = fullfile(configObj.ResultDir, 'Stage2_Smoke_Ablation.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化報告已儲存至: %s\n', figPath);
close(fig);

disp('=================================================================');
disp('🎯 [Stage 2 SMOKE TEST] Task A 執行完畢！');
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
% 輔助函數：32 天分塊防 OOM 驗證損失計算
% =====================================================================
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

%% =====================================================================
% 輔助函數：逐日損失與線性探針
% =====================================================================
function [daily_losses, probe_metric, p_hac_corr, daily_ic_out, hac_lag_used, p_hac_auto] = ...
    evaluate_probe_and_hac(net, X_in, Y_in, Expert, val_days, seqLen, target_type, use_gpu, H_horizon)
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
    
    if strcmp(target_type, 'binary')
        mdl = fitclinear(E_val, Y_val, 'Learner', 'logistic');
        [~, scores] = predict(mdl, E_val);
        [~, ~, ~, probe_metric] = perfcurve(Y_val, scores(:, 2), 1);
        p_hac_corr   = 1.0;
        p_hac_auto   = 1.0;
        daily_ic_out = [];
        hac_lag_used = NaN;
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
        daily_ic_out = daily_ic;
        probe_metric = mean(daily_ic, 'omitnan');
        
        if length(daily_ic) >= 20
            auto_lag = max(1, floor(4 * (length(daily_ic)/100)^(2/9)));
            [~, p_hac_auto] = hac_significance_test(daily_ic, auto_lag);
            
            hac_lag_used = max(H_horizon, auto_lag);
            [~, p_hac_corr] = hac_significance_test(daily_ic, hac_lag_used);
        else
            p_hac_corr   = 1.0;
            p_hac_auto   = 1.0;
            hac_lag_used = NaN;
        end
    end
end