% =========================================================================
% 腳本：Stage2_5_GBDT_SoftIC_Validation.m (Task B': GBDT 連續迴歸與 Soft-IC 獨立交叉驗證)
% 依據：《Stage 2.5 – 4 Code Review 計劃書》 §3.4 規範
% 職責：
%   1. 在全 504 檔標的大宇宙 (Full Universe) 上提取 18 維微觀/相對特徵 (無 GICS 中性化)
%   2. 構建 5D Beat-the-Median 二元標籤與連續標準化超額報酬標籤
%   3. 透過 RawBaselineTrainer 執行完全同構的 5-Fold Purged Expanding-Window 驗證
%   4. 同步輸出 OOF AUC 與 OOF Rank IC、Day-Level Bootstrap 95% CI 與 HAC 檢定
%   5. 提供獨立於神經網路 (DL) 管線之外的 Direction 2 跨模型方法論真確性驗證
% =========================================================================
clear; clc; close all;

disp('=================================================================');
disp('🌲 [Stage 2.5 Task B''] 啟動 GBDT 連續迴歸交叉驗證 (全 504 檔宇宙)');
disp('=================================================================');

%% 0. 環境路徑掛載與平行池管理
currentFile = mfilename('fullpath');
if isempty(currentFile), currentPath = pwd; else, currentPath = fileparts(currentFile); end
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

% 喚醒多核心運算池
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('Processes');
end

%% 1. 載入特徵快取與記憶體型態壓縮 (504 檔全宇宙)
disp('--- 步驟 1：載入特徵快取 (504 檔全宇宙) 並進行記憶體瘦身 ---');
cachePath = fullfile(configObj.CacheDir, 'features_denoised.mat');
if ~exist(cachePath, 'file')
    error('❌ 找不到特徵快取檔案，請先執行 1_Run_Data_and_Features.m！');
end

% 載入全域資料 (不載入未使用的 AdjMatrix_3D 以節省記憶體)[cite: 1]
load(cachePath, 'X_norm_3D', 'Prices_Active', 'Expert_Active', 'Dates_Active');
Dates_Active.TimeZone = 'UTC';

seqLen = configObj.SeqLen;
numDaysRaw = length(Dates_Active);
valid_idx = seqLen : numDaysRaw;

% 資料型態轉換：double -> single / logical
Prices_Active = single(Prices_Active(valid_idx, :));
Expert_Active = logical(Expert_Active(valid_idx, :));
Dates_Active  = Dates_Active(valid_idx);

% 提取 18 維微觀與相對特徵 (排除宏觀特徵與 GICS 中性化，維持基準線乾淨)[cite: 1]
numExtractorFeats = 3 + configObj.NumMicroFeatures; % 18 維
X_raw_18D     = single(X_norm_3D(valid_idx, 1:numExtractorFeats, :));
numDays       = length(Dates_Active);
numTickers    = size(Prices_Active, 2);

clear X_norm_3D;

fprintf('  📊 驗證宇宙：有效天數 %d 天 | 宇宙規模 %d 檔 | 原始特徵維度 %d 維\n', ...
    numDays, numTickers, numExtractorFeats);

%% 2. 構建 5 日遠期預測標籤 (二元分類 vs 連續標準化超額報酬)
disp('--- 步驟 2：構建 5 日遠期標籤 (Beat-the-Median vs Continuous Z-Score) ---');
H_horizon = 5;
valid_label_days = numDays - H_horizon;

R_fwd_5D = (Prices_Active(1+H_horizon:end, :) - Prices_Active(1:end-H_horizon, :)) ...
           ./ (Prices_Active(1:end-H_horizon, :) + 1e-8);
R_fwd_5D(isnan(R_fwd_5D) | isinf(R_fwd_5D)) = NaN;

Y_bin_5D  = false(valid_label_days, numTickers);
Y_cont_5D = NaN(valid_label_days, numTickers, 'single');

for t = 1:valid_label_days
    act_m = Expert_Active(t, :) & ~isnan(R_fwd_5D(t, :)) & ~isinf(R_fwd_5D(t, :));
    if sum(act_m) >= 10
        r_t = R_fwd_5D(t, act_m);
        med_r = median(r_t, 'omitnan');
        mu_r  = mean(r_t, 'omitnan');
        std_r = std(r_t, 0, 'omitnan') + 1e-6;
        
        Y_bin_5D(t, act_m)  = (r_t > med_r);
        Y_cont_5D(t, act_m) = (r_t - mu_r) ./ std_r; % 橫截面標準化連續超額報酬[cite: 1]
    end
end

clear R_fwd_5D Prices_Active;

%% 3. 展平特徵面板為 2D 樣本矩陣 (含時間陣列映射)
disp('--- 步驟 3：展平特徵面板並進行缺失值防禦過濾 ---');
active_counts = sum(Expert_Active(1:valid_label_days, :), 'all');

X_flat      = zeros(active_counts, numExtractorFeats, 'single');
Y_bin_flat  = false(active_counts, 1);
Y_cont_flat = zeros(active_counts, 1, 'single');

% ★ 核心修復：使用 double 替代 uint16，徹底解決 linspace 型態衝突與下溢截斷
day_arr     = zeros(active_counts, 1, 'double');

row_cur = 1;
for t = 1:valid_label_days
    act_ids = find(Expert_Active(t, :));
    n_t = length(act_ids);
    if n_t > 0
        x_slice = permute(X_raw_18D(t, :, act_ids), [3, 2, 1]);
        X_flat(row_cur : row_cur + n_t - 1, :)   = x_slice;
        Y_bin_flat(row_cur : row_cur + n_t - 1)  = Y_bin_5D(t, act_ids)';
        Y_cont_flat(row_cur : row_cur + n_t - 1) = Y_cont_5D(t, act_ids)';
        day_arr(row_cur : row_cur + n_t - 1)     = double(t);
        row_cur = row_cur + n_t;
    end
end

clear X_raw_18D Y_bin_5D Y_cont_5D Expert_Active;

% 顯式過濾無效樣本 (去除 NaN 與 Inf)[cite: 1]
valid_mask = ~isnan(Y_cont_flat) & ~isinf(Y_cont_flat) & ...
             ~isnan(Y_bin_flat) & ...
             all(~isnan(X_flat) & ~isinf(X_flat), 2);

X_flat      = X_flat(valid_mask, :);
Y_bin_flat  = Y_bin_flat(valid_mask);
Y_cont_flat = Y_cont_flat(valid_mask);
day_arr     = day_arr(valid_mask);

total_samples = length(Y_cont_flat);
fprintf('  -> 展平後有效橫截面觀測樣本數: %d 筆\n', total_samples);

%% 4. 實例化 RawBaselineTrainer 並執行雙軌對照評估[cite: 1]
disp('--- 步驟 4：啟動 GBDT 5-Fold 時序 Purged CV 雙軌評估 (同構切分) ---');
trainer = RawBaselineTrainer(configObj);
trainer.NumFolds = 5;
trainer.EmbargoDays = 20;
trainer.MaxTrainSamples = 150000; % 防 OOM 上限

% 4A. 執行二元分類基準評估 (LogitBoost)[cite: 1]
fprintf('\n▶ [1/2] 正在執行 GBDT 二元分類基準模型 (LogitBoost 5D Beat-the-Median)...\n');
tic;
[metrics_cls, ~, ~] = trainer.train_and_eval(X_flat, Y_bin_flat, day_arr, 1000);
t_cls = toc;
fprintf('  ⚡ 分類基準評估完畢！耗時: %.2f 秒\n', t_cls);

% 4B. 執行連續迴歸評估 (LSBoost + Rank IC)[cite: 1]
fprintf('\n▶ [2/2] 正在執行 GBDT 連續迴歸模型 (LSBoost 5D Continuous Return)...\n');
tic;
[metrics_reg, ~, ~] = trainer.train_and_eval_regression(X_flat, Y_cont_flat, day_arr, H_horizon, 1000);
t_reg = toc;
fprintf('  ⚡ 連續迴歸評估完畢！耗時: %.2f 秒\n', t_reg);

%% 5. 統計顯著性對照報告與方法論判讀
disp(' ');
disp('========================================================================================================================');
disp('📊 【Task B'' 核心產出表：GBDT 樹模型二元分類 vs 連續迴歸資訊天花板對照報告】');
disp('========================================================================================================================');
fprintf(' 評估體制                  | 核心指標點估計 | 95%% 信賴區間 (Day-Level) | HAC 顯著性檢定 (lag=%d) | 資訊邊際判定\n', metrics_reg.HAC_MaxLag);
fprintf('------------------------------------------------------------------------------------------------------------------------\n');

% 分類體制判定
if metrics_cls.AUC_Point >= 0.5200 && metrics_cls.CI_Lower > 0.5050
    dec_cls = "GO (Strong Signal)";
elseif metrics_cls.AUC_Point >= 0.5050 && metrics_cls.CI_Lower > 0.5000
    dec_cls = "MARGINAL (Weak)";
else
    dec_cls = "NO-GO (Noise Ceiling)";
end

% 迴歸體制判定
if metrics_reg.OOF_IC >= 0.0200 && metrics_reg.CI_Lower > 0.0000 && metrics_reg.HAC_p < 0.05
    dec_reg = "⭐ CONFIRMED (Viable Alpha)";
elseif metrics_reg.OOF_IC >= 0.0100 && metrics_reg.HAC_p < 0.05
    dec_reg = "MARGINAL (Weak)";
else
    dec_reg = "NO-GO (Noise/Falsified)";
end

fprintf(' 二元分類 (LogitBoost BCE) | AUC = %.4f   | [%.4f, %.4f]           |           N/A           | %s\n', ...
    metrics_cls.AUC_Point, metrics_cls.CI_Lower, metrics_cls.CI_Upper, dec_cls);
fprintf(' 連續迴歸 (LSBoost Soft-IC)| IC  = %+.4f  | [%+.4f, %+.4f]         | p = %.4f (HAC)          | %s\n', ...
    metrics_reg.OOF_IC, metrics_reg.CI_Lower, metrics_reg.CI_Upper, metrics_reg.HAC_p, dec_reg);
fprintf('========================================================================================================================\n\n');

%% 6. 學術推論與跨模型因果鏈判讀[cite: 1]
fprintf('💡 【Task B'' 跨模型獨立驗證因果鏈判讀結論】\n');
if metrics_reg.OOF_IC > 0.015 && metrics_reg.HAC_p < 0.05 && metrics_reg.CI_Lower > 0.0
    fprintf('  ⭐ 【DIRECTION 2 INDEPENDENTLY CONFIRMED】\n');
    fprintf('     在全 504 檔真實大宇宙且排除神經網路 (DL) 架構下，GBDT 於連續回歸目標之 OOF Rank IC\n');
    fprintf('     成功達到 %+.4f (95%% CI: [%+.4f, %+.4f]，HAC p = %.4f < 0.05)！\n', ...
        metrics_reg.OOF_IC, metrics_reg.CI_Lower, metrics_reg.CI_Upper, metrics_reg.HAC_p);
    fprintf('     這獨立證實了：先前分類任務下 AUC 卡在 0.501~0.510 的天花板瓶頸，\n');
    fprintf('     本質上確實源自於「硬切中位數 (Beat-the-Median) 造成的標籤噪聲與橫截面梯度損失」，\n');
    fprintf('     而非 18 維微觀特徵本身毫無預測價值！\n');
else
    fprintf('  ✅ 【DIRECTION 2 FALSIFIED BY TREE MODEL】\n');
    fprintf('     即使換用非線性 GBDT 迴歸，OOF Rank IC 仍未達統計顯著邊際 (IC = %+.4f, p = %.4f)。\n', ...
        metrics_reg.OOF_IC, metrics_reg.HAC_p);
    fprintf('     說明目標函數變更在樹模型上未產生超額收益，瓶頸仍可能在特徵本身。\n');
end
fprintf('========================================================================================================================\n\n');

%% 7. 儲存驗證成果與產出白底黑字視覺化圖表
if ~exist(configObj.ResultDir, 'dir'), mkdir(configObj.ResultDir); end
savePath = fullfile(configObj.ResultDir, 'Stage2_5_GBDT_SoftIC_Validation.mat');
save(savePath, 'metrics_cls', 'metrics_reg');
fprintf('💾 交叉驗證統計成果已成功儲存至: %s\n', savePath);

% 視覺化報表
fig = figure('Name', 'Stage 2.5 Task B'' GBDT Validation', ...
    'Color', 'w', 'Position', [100, 100, 1000, 450], 'Visible', 'off');
set(fig, 'InvertHardcopy', 'off');

% 子圖 1：分類 AUC 與信賴區間
subplot(1, 2, 1);
errorbar(1, metrics_cls.AUC_Point, metrics_cls.AUC_Point - metrics_cls.CI_Lower, ...
    metrics_cls.CI_Upper - metrics_cls.AUC_Point, 'o', 'Color', [0.0000, 0.4470, 0.7410], ...
    'MarkerFaceColor', [0.0000, 0.4470, 0.7410], 'LineWidth', 1.5, 'CapSize', 8, 'MarkerSize', 7);
hold on;
yline(0.50, '--k', 'Random Guess (0.50)', 'LineWidth', 1.1);
yline(0.52, ':r', 'Signal Threshold (0.52)', 'LineWidth', 1.1);
xlim([0.5, 1.5]);
ylim([0.48, max(0.54, metrics_cls.CI_Upper + 0.01)]);
set(gca, 'XTick', 1, 'XTickLabel', {'LogitBoost (5D BCE)'}, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', ...
    'LineWidth', 1.0, 'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
ylabel('OOF AUC', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
title('Binary Beat-the-Median (Classification Ceiling)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
grid on; box on;

% 子圖 2：連續迴歸 Rank IC 與信賴區間
subplot(1, 2, 2);
errorbar(1, metrics_reg.OOF_IC, metrics_reg.OOF_IC - metrics_reg.CI_Lower, ...
    metrics_reg.CI_Upper - metrics_reg.OOF_IC, 's', 'Color', [0.8500, 0.3250, 0.0980], ...
    'MarkerFaceColor', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5, 'CapSize', 8, 'MarkerSize', 7);
hold on;
yline(0.00, '--k', 'Zero IC', 'LineWidth', 1.1);
yline(0.02, ':r', 'Alpha Viability (IC = 0.02)', 'LineWidth', 1.1);
xlim([0.5, 1.5]);
set(gca, 'XTick', 1, 'XTickLabel', {'LSBoost (Continuous Return)'}, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', ...
    'LineWidth', 1.0, 'GridColor', [0.85, 0.85, 0.85], 'GridAlpha', 0.8, 'FontName', 'Helvetica');
ylabel('OOF Spearman Rank IC', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
title(sprintf('Continuous Return (HAC p = %.4f, Lag = %d)', metrics_reg.HAC_p, metrics_reg.HAC_MaxLag), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
grid on; box on;

figPath = fullfile(configObj.ResultDir, 'Stage2_5_GBDT_SoftIC_Validation.png');
exportgraphics(fig, figPath, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('📊 視覺化圖表已成功儲存至: %s\n', figPath);
close(fig);

disp('=================================================================');
disp('🎯 [Stage 2.5 Task B''] 交叉驗證完畢！');
disp('=================================================================');