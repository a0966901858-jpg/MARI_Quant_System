classdef RawBaselineTrainer < handle
    % =========================================================================
    % 類別：RawBaselineTrainer (原始特徵 GBDT 基準評估引擎)
    % 升級：Phase 15.5 Task B' 規格 (★ 支援 LSBoost 連續迴歸評估、
    %       共用 Purged Expanding-Window 切分、日層級橫截面 Rank IC、
    %       HAC max_lag=H 頻寬檢定與 Day-Level Bootstrap 信賴區間)
    % 職責：在排除深度學習架構的情況下，以 GBDT 嚴格評估特徵的分類與迴歸資訊天花板
    % =========================================================================
    
    properties
        ConfigObj           % 全域設定檔參考
        NumFolds            % 時序交叉驗證折數 (預設 = 5)
        EmbargoDays         % 訓練集與驗證集隔離間隔 (預設 = 20)
        MaxTrainSamples     % 單一 Fold 最大訓練抽樣上限 (防 OOM)
        NumLearningCycles   % GBDT 迭代次數 (預設 = 30)
        LearnRate           % 學習率 (預設 = 0.1)
        MaxNumSplits        % 決策樹最大分裂數 (預設 = 20)
        MinLeafSize         % 葉節點最小樣本數 (預設 = 50)
    end
    
    methods
        % ---------------------------------------------------------
        % 建構子：初始化超參數
        % ---------------------------------------------------------
        function obj = RawBaselineTrainer(configObj)
            obj.ConfigObj = configObj;
            obj.NumFolds = 5;
            obj.EmbargoDays = 20;
            obj.MaxTrainSamples = 150000;
            obj.NumLearningCycles = 30;
            obj.LearnRate = 0.1;
            obj.MaxNumSplits = 20;
            obj.MinLeafSize = 50;
        end
        
        % =========================================================
        % 1. 二元分類評估 (Beat-the-Median AUC 與 Day-Block Bootstrap)
        % =========================================================
        function [metrics, oof_probs, eval_mask] = train_and_eval(obj, X_flat, Y_bin_flat, day_array, n_boot)
            if nargin < 5 || isempty(n_boot)
                n_boot = 1000;
            end
            
            % 清洗無效樣本
            valid_mask = ~isnan(Y_bin_flat) & ~isinf(Y_bin_flat) & all(~isnan(X_flat) & ~isinf(X_flat), 2);
            X_clean = X_flat(valid_mask, :);
            Y_clean = logical(Y_bin_flat(valid_mask));
            day_clean = day_array(valid_mask);
            
            total_samples = length(Y_clean);
            oof_scores = zeros(total_samples, 1, 'single');
            eval_mask_clean = false(total_samples, 1);
            
            % 呼叫共用 Purged CV 切分核心
            [u_days, fold_edges, K] = obj.compute_purged_folds(day_clean);
            t_tree = templateTree('MaxNumSplits', obj.MaxNumSplits, 'MinLeafSize', obj.MinLeafSize);
            Y_cat = categorical(Y_clean);
            
            for k = 2:K
                test_days = u_days(fold_edges(k) : fold_edges(k+1) - 1);
                min_d = min(test_days);
                max_d = max(test_days);
                
                train_mask = (day_clean < (min_d - obj.EmbargoDays));
                test_mask  = (day_clean >= min_d) & (day_clean <= max_d);
                
                tr_idx = find(train_mask);
                va_idx = find(test_mask);
                
                if length(tr_idx) > obj.MaxTrainSamples
                    tr_idx = randsample(tr_idx, obj.MaxTrainSamples);
                end
                
                if length(tr_idx) > 500 && length(va_idx) > 50 && numel(unique(Y_clean(tr_idx))) > 1
                    mdl = fitcensemble(X_clean(tr_idx, :), Y_cat(tr_idx), ...
                        'Method', 'LogitBoost', 'Learners', t_tree, ...
                        'NumLearningCycles', obj.NumLearningCycles, 'LearnRate', obj.LearnRate);
                    
                    [~, score] = predict(mdl, X_clean(va_idx, :));
                    oof_scores(va_idx) = single(score(:, 2));
                    eval_mask_clean(va_idx) = true;
                end
            end
            
            oof_probs = zeros(size(Y_bin_flat), 'single');
            oof_probs(valid_mask) = 1 ./ (1 + exp(-oof_scores));
            
            eval_mask = false(size(Y_bin_flat));
            eval_mask(valid_mask) = eval_mask_clean;
            
            y_eval = Y_clean(eval_mask_clean);
            p_eval = oof_probs(eval_mask);
            
            if numel(unique(y_eval)) > 1
                [~, ~, ~, auc_point] = perfcurve(y_eval, p_eval, true);
                
                % Day-Block Bootstrap AUC 信賴區間
                eval_days = day_clean(eval_mask_clean);
                u_eval_days = unique(eval_days);
                n_eval_days = length(u_eval_days);
                boot_aucs = zeros(n_boot, 1);
                
                for b = 1:n_boot
                    s_days = randsample(u_eval_days, n_eval_days, true);
                    b_mask = ismember(eval_days, s_days(1:min(50, n_eval_days)));
                    if sum(y_eval(b_mask) == 1) > 5 && sum(y_eval(b_mask) == 0) > 5
                        [~, ~, ~, boot_aucs(b)] = perfcurve(y_eval(b_mask), p_eval(b_mask), true);
                    else
                        boot_aucs(b) = auc_point;
                    end
                end
                ci_lower = prctile(boot_aucs, 2.5);
                ci_upper = prctile(boot_aucs, 97.5);
            else
                auc_point = 0.5; ci_lower = 0.5; ci_upper = 0.5;
            end
            
            metrics = struct('AUC_Point', auc_point, 'CI_Lower', ci_lower, ...
                             'CI_Upper', ci_upper, 'PosClassRatio', mean(Y_clean));
        end
        
        % =========================================================
        % 2. 連續迴歸評估 (★ Task B': LSBoost 連續超額報酬與 Rank IC)
        % =========================================================
        function [metrics, oof_preds, eval_mask] = train_and_eval_regression(obj, X_flat, Y_cont_flat, day_array, H_horizon, n_boot)
            if nargin < 6 || isempty(n_boot)
                n_boot = 1000;[cite: 1]
            end
            if nargin < 5 || isempty(H_horizon)
                H_horizon = 5;[cite: 1]
            end
            
            % 清洗無效樣本 (去除 NaN 與 Inf)
            valid_mask = ~isnan(Y_cont_flat) & ~isinf(Y_cont_flat) & all(~isnan(X_flat) & ~isinf(X_flat), 2);
            X_clean = X_flat(valid_mask, :);
            Y_clean = single(Y_cont_flat(valid_mask));
            day_clean = day_array(valid_mask);
            
            total_samples = length(Y_clean);
            oof_preds_clean = zeros(total_samples, 1, 'single');
            eval_mask_clean = false(total_samples, 1);
            
            % 呼叫共用 Purged CV 切分核心[cite: 1]
            [u_days, fold_edges, K] = obj.compute_purged_folds(day_clean);
            t_tree = templateTree('MaxNumSplits', obj.MaxNumSplits, 'MinLeafSize', obj.MinLeafSize);
            
            % 動態調整 Embargo：不得小於預測視窗 H
            effective_embargo = max(obj.EmbargoDays, H_horizon);
            
            for k = 2:K
                test_days = u_days(fold_edges(k) : fold_edges(k+1) - 1);
                min_d = min(test_days);
                max_d = max(test_days);
                
                train_mask = (day_clean < (min_d - effective_embargo)) | (day_clean > (max_d + effective_embargo));
                test_mask  = (day_clean >= min_d) & (day_clean <= max_d);
                
                tr_idx = find(train_mask);
                va_idx = find(test_mask);
                
                if length(tr_idx) > obj.MaxTrainSamples
                    tr_idx = randsample(tr_idx, obj.MaxTrainSamples);
                end
                
                if length(tr_idx) > 500 && length(va_idx) > 50
                    % 訓練連續目標迴歸樹 (LSBoost)[cite: 1]
                    mdl = fitrensemble(X_clean(tr_idx, :), Y_clean(tr_idx), ...
                        'Method', 'LSBoost', 'Learners', t_tree, ...
                        'NumLearningCycles', obj.NumLearningCycles, 'LearnRate', obj.LearnRate);[cite: 1]
                    
                    oof_preds_clean(va_idx) = single(predict(mdl, X_clean(va_idx, :)));[cite: 1]
                    eval_mask_clean(va_idx) = true;
                end
            end
            
            oof_preds = zeros(size(Y_cont_flat), 'single');
            oof_preds(valid_mask) = oof_preds_clean;
            
            eval_mask = false(size(Y_cont_flat));
            eval_mask(valid_mask) = eval_mask_clean;
            
            % 逐日橫截面 Spearman IC 計算[cite: 1]
            daily_ic = obj.compute_daily_cross_sectional_ic(...
                oof_preds_clean(eval_mask_clean), Y_clean(eval_mask_clean), day_clean(eval_mask_clean));[cite: 1]
            
            % 顯式貫穿 Newey-West HAC 檢定 (強制 max_lag = max(H, auto_lag))[cite: 1]
            hac_lag = max(H_horizon, floor(4 * (length(daily_ic) / 100)^(2/9)));[cite: 1]
            if length(daily_ic) >= 20
                [~, p_hac] = hac_significance_test(daily_ic, hac_lag);[cite: 1]
            else
                p_hac = 1.0;
            end
            
            % 呼叫日層級 Bootstrap 信賴區間估計[cite: 1]
            [ci_lower, ci_upper, ic_point] = block_bootstrap_ic_by_day(daily_ic, n_boot, 0.95);[cite: 1]
            
            metrics = struct('OOF_IC', ic_point, 'CI_Lower', ci_lower, 'CI_Upper', ci_upper, ...
                'HAC_p', p_hac, 'HAC_MaxLag', hac_lag, 'NumDays', length(daily_ic));[cite: 1]
        end
    end
    
    %% =====================================================================
    % 私有輔助方法：共用 Purged 切分與橫截面 IC 運算
    % =====================================================================
    methods (Access = private)
        % ---------------------------------------------------------
        % 共用 Purged 時序折數切分 (消除程式碼重複)[cite: 1]
        % ---------------------------------------------------------
        function [u_days, fold_edges, K] = compute_purged_folds(obj, day_array)
            u_days = unique(day_array);
            n_days = length(u_days);
            K = obj.NumFolds;
            fold_edges = round(linspace(1, n_days + 1, K + 1));
        end
        
        % ---------------------------------------------------------
        % 逐日橫截面 Spearman Rank IC 向量運算[cite: 1]
        % ---------------------------------------------------------
        function daily_ic = compute_daily_cross_sectional_ic(~, preds, y_true, day_arr)
            u_eval_days = unique(day_arr);
            n_days = length(u_eval_days);
            daily_ic_temp = zeros(n_days, 1);
            valid_cnt = 0;
            
            for d = 1:n_days
                day_val = u_eval_days(d);
                idx_d = (day_arr == day_val);
                if sum(idx_d) >= 10
                    valid_cnt = valid_cnt + 1;
                    daily_ic_temp(valid_cnt) = corr(preds(idx_d), y_true(idx_d), ...
                        'Type', 'Spearman', 'Rows', 'complete');
                end
            end
            daily_ic = daily_ic_temp(1:valid_cnt);
        end
    end
end