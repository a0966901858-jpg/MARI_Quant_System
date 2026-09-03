classdef RawBaselineTrainer < handle
    % =========================================================================
    % 類別：RawBaselineTrainer (原始特徵 GBDT 基準評估引擎)
    % 升級：Phase 15.5 Task B' (★ 支援外部注入 RandStream / mrg32k3a 獨立子串流、
    %       randsample 訓練抽樣與 Day-Block Bootstrap 完全確定性重現、
    %       修復 uint16 傳入 linspace 之型態相容性、防禦整數下溢截斷)
    % =========================================================================
    
    properties
        ConfigObj           % 全域設定檔參考
        RandStream          % 隨機數串流物件 (支援 mrg32k3a 獨立子串流)
        NumFolds            % 時序交叉驗證折數 (預設 = 5)
        EmbargoDays         % 訓練集與驗證集隔離間隔 (預設 = 20)
        MaxTrainSamples     % 單一 Fold 最大訓練抽樣上限 (防 OOM)
        NumLearningCycles   % GBDT 迭代次數 (預設 = 30)
        LearnRate           % 學習率 (預設 = 0.1)
        MaxNumSplits        % 決策樹最大分裂數 (預設 = 20)
        MinLeafSize         % 葉節點最小樣本數 (預設 = 50)
    end
    
    methods
        function obj = RawBaselineTrainer(configObj, stream)
            obj.ConfigObj = configObj;
            obj.NumFolds = 5;
            obj.EmbargoDays = 20;
            obj.MaxTrainSamples = 150000;
            obj.NumLearningCycles = 30;
            obj.LearnRate = 0.1;
            obj.MaxNumSplits = 20;
            obj.MinLeafSize = 50;
            
            % 綁定隨機數串流
            if nargin >= 2 && ~isempty(stream)
                obj.RandStream = stream;
            elseif ~isempty(configObj) && ismethod(configObj, 'getRandStream')
                obj.RandStream = configObj.getRandStream(1);
            else
                obj.RandStream = [];
            end
        end
        
        % =========================================================
        % 1. 二元分類評估 (Beat-the-Median AUC 與 Day-Block Bootstrap)
        % =========================================================
        function [metrics, oof_probs, eval_mask] = train_and_eval(obj, X_flat, Y_bin_flat, day_array, n_boot, stream)
            if nargin < 5 || isempty(n_boot), n_boot = 1000; end
            if nargin < 6 || isempty(stream)
                s = obj.resolveStream();
            else
                s = obj.resolveStream(stream);
            end
            
            % 同步全域串流以防下游第三方工具箱隱式依賴
            old_stream = obj.RandStream.setGlobalStream(s);
            cleanupObj = onCleanup(@() obj.RandStream.setGlobalStream(old_stream));
            
            valid_mask = ~isnan(Y_bin_flat) & ~isinf(Y_bin_flat) & all(~isnan(X_flat) & ~isinf(X_flat), 2);
            X_clean = X_flat(valid_mask, :);
            Y_clean = logical(Y_bin_flat(valid_mask));
            
            % 強制轉為 double 杜絕型別衝突與整數下溢
            day_clean = double(day_array(valid_mask));
            
            total_samples = length(Y_clean);
            oof_scores = zeros(total_samples, 1, 'single');
            eval_mask_clean = false(total_samples, 1);
            
            K = obj.NumFolds;
            u_days = unique(day_clean);
            n_days = length(u_days);
            fold_edges = round(linspace(1, n_days + 1, K + 1));
            
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
                
                % ★ 注入串流進行無放回訓練抽樣
                if length(tr_idx) > obj.MaxTrainSamples
                    tr_idx = randsample(s, tr_idx, obj.MaxTrainSamples, false);
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
                
                eval_days = day_clean(eval_mask_clean);
                u_eval_days = unique(eval_days);
                n_eval_days = length(u_eval_days);
                boot_aucs = zeros(n_boot, 1);
                
                % ★ 注入串流進行 Day-Block Bootstrap 重抽樣
                for b = 1:n_boot
                    s_days = randsample(s, u_eval_days, n_eval_days, true);
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
        % 2. 連續迴歸評估 (LSBoost 連續超額報酬與 Rank IC)
        % =========================================================
        function [metrics, oof_preds, eval_mask] = train_and_eval_regression(obj, X_flat, Y_cont_flat, day_array, H_horizon, n_boot, stream)
            if nargin < 6 || isempty(n_boot), n_boot = 1000; end
            if nargin < 5 || isempty(H_horizon), H_horizon = 5; end
            if nargin < 7 || isempty(stream)
                s = obj.resolveStream();
            else
                s = obj.resolveStream(stream);
            end
            
            old_stream = obj.RandStream.setGlobalStream(s);
            cleanupObj = onCleanup(@() obj.RandStream.setGlobalStream(old_stream));
            
            valid_mask = ~isnan(Y_cont_flat) & ~isinf(Y_cont_flat) & all(~isnan(X_flat) & ~isinf(X_flat), 2);
            X_clean = X_flat(valid_mask, :);
            Y_clean = single(Y_cont_flat(valid_mask));
            
            day_clean = double(day_array(valid_mask));
            
            total_samples = length(Y_clean);
            oof_preds_clean = zeros(total_samples, 1, 'single');
            eval_mask_clean = false(total_samples, 1);
            
            K = obj.NumFolds;
            u_days = unique(day_clean);
            n_days = length(u_days);
            fold_edges = round(linspace(1, n_days + 1, K + 1));
            
            t_tree = templateTree('MaxNumSplits', obj.MaxNumSplits, 'MinLeafSize', obj.MinLeafSize);
            effective_embargo = max(obj.EmbargoDays, H_horizon);
            
            for k = 2:K
                test_days = u_days(fold_edges(k) : fold_edges(k+1) - 1);
                min_d = min(test_days);
                max_d = max(test_days);
                
                train_mask = (day_clean < (min_d - effective_embargo)) | (day_clean > (max_d + effective_embargo));
                test_mask  = (day_clean >= min_d) & (day_clean <= max_d);
                
                tr_idx = find(train_mask);
                va_idx = find(test_mask);
                
                % ★ 注入串流進行無放回訓練抽樣
                if length(tr_idx) > obj.MaxTrainSamples
                    tr_idx = randsample(s, tr_idx, obj.MaxTrainSamples, false);
                end
                
                if length(tr_idx) > 500 && length(va_idx) > 50
                    mdl = fitrensemble(X_clean(tr_idx, :), Y_clean(tr_idx), ...
                        'Method', 'LSBoost', 'Learners', t_tree, ...
                        'NumLearningCycles', obj.NumLearningCycles, 'LearnRate', obj.LearnRate);
                    
                    oof_preds_clean(va_idx) = single(predict(mdl, X_clean(va_idx, :)));
                    eval_mask_clean(va_idx) = true;
                end
            end
            
            oof_preds = zeros(size(Y_cont_flat), 'single');
            oof_preds(valid_mask) = oof_preds_clean;
            
            eval_mask = false(size(Y_cont_flat));
            eval_mask(valid_mask) = eval_mask_clean;
            
            % 逐日橫截面 Spearman IC 計算
            eval_days = day_clean(eval_mask_clean);
            u_eval_days = unique(eval_days);
            n_eval_days = length(u_eval_days);
            daily_ic_temp = zeros(n_eval_days, 1);
            valid_cnt = 0;
            
            for d = 1:n_eval_days
                day_val = u_eval_days(d);
                idx_d = (eval_days == day_val);
                if sum(idx_d) >= 10
                    valid_cnt = valid_cnt + 1;
                    daily_ic_temp(valid_cnt) = corr(oof_preds_clean(idx_d), Y_clean(idx_d), ...
                        'Type', 'Spearman', 'Rows', 'complete');
                end
            end
            daily_ic = daily_ic_temp(1:valid_cnt);
            
            hac_lag = max(H_horizon, floor(4 * (length(daily_ic) / 100)^(2/9)));
            if length(daily_ic) >= 20
                [~, p_hac] = hac_significance_test(daily_ic, hac_lag);
            else
                p_hac = 1.0;
            end
            
            % block_bootstrap_ic_by_day 會自動繼承當前設定之全域 s 串流
            [ci_lower, ci_upper, ic_point] = block_bootstrap_ic_by_day(daily_ic, n_boot, 0.95);
            
            metrics = struct('OOF_IC', ic_point, 'CI_Lower', ci_lower, 'CI_Upper', ci_upper, ...
                'HAC_p', p_hac, 'HAC_MaxLag', hac_lag, 'NumDays', length(daily_ic));
        end
    end
    
    methods (Access = private)
        % =========================================================
        % 私有輔助函數：解析優先級 (外部傳入 > 成員變數 > Config > Global)
        % =========================================================
        function s = resolveStream(obj, stream_in)
            if nargin >= 2 && ~isempty(stream_in)
                s = stream_in;
            elseif ~isempty(obj.RandStream)
                s = obj.RandStream;
            elseif ~isempty(obj.ConfigObj) && ismethod(obj.ConfigObj, 'getRandStream')
                s = obj.ConfigObj.getRandStream(1);
            else
                s = obj.RandStream.getGlobalStream();
            end
        end
    end
end