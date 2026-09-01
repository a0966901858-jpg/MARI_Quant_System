classdef RawBaselineTrainer < handle
    % =========================================================================
    % 類別：RawBaselineTrainer
    % 職責：獨立於 DL 與 GBDT 代理人本體，專門執行 Raw 特徵面板的 5-Fold Purged & Embargoed CV 基準評估
    % 用途：
    %   1. 支援 Workstream A 多 Horizon 天花板掃描 (18D / 10D / 28D 特徵子集)[cite: 1]
    %   2. 採用 LogitBoost + 決策樹基學習器進行非線性 OOF 預測[cite: 1]
    %   3. 整合 block_bootstrap_auc_by_day 計算 Day-Level Block Bootstrap 95% 信賴區間[cite: 1]
    % =========================================================================

    properties
        ConfigObj
        NumFolds        = 5
        EmbargoDays     = 20
        MaxTrainSamples = 120000
        NumLearningCycles = 30
        LearnRate       = 0.1
        MaxNumSplits    = 20
        MinLeafSize     = 50
    end

    methods
        % -----------------------------------------------------------------
        % 建構子
        % -----------------------------------------------------------------
        function obj = RawBaselineTrainer(configObj)
            if nargin > 0 && ~isempty(configObj)
                obj.ConfigObj = configObj;
            end
        end

        % =================================================================
        % 核心方法：執行 5-Fold Purged CV 訓練、OOF 預測與統計檢定[cite: 1]
        % =================================================================
        function [metrics, oof_probs, oof_scores, eval_mask] = train_and_eval(obj, X_flat, Y_flat, day_array, n_boot)
            % -------------------------------------------------------------
            % 輸入：
            %   X_flat      : [N x D] 特徵矩陣
            %   Y_flat      : [N x 1] 二元標籤 (0 或 1)[cite: 1]
            %   day_array   : [N x 1] 各樣本對應的交易日 ID
            %   n_boot      : 自助重抽樣次數 (預設 = 1000)[cite: 1]
            % 輸出：
            %   metrics     : 包含 AUC 點估計、Day-Block 95% CI、正類比例之結構體[cite: 1]
            %   oof_probs   : [N x 1] OOF 預測機率 (Sigmoid 校準後)
            %   oof_scores  : [N x 1] OOF 原始對數勝率 (Logits)
            %   eval_mask   : [N x 1] 參與 OOF 驗證之有效樣本遮罩 (Fold >= 2)
            % -------------------------------------------------------------
            if nargin < 5 || isempty(n_boot)
                n_boot = 1000;
            end

            total_samples = size(X_flat, 1);
            day_array = day_array(:);
            Y_flat = Y_flat(:);

            % 1. Purged & Embargoed 時序切分 (以交易日為劃分邊界)[cite: 1]
            K = obj.NumFolds;
            embargo = obj.EmbargoDays;
            max_day = max(day_array);
            min_day = min(day_array);

            edges = linspace(min_day, max_day + 0.1, K + 1);
            fold_indices = zeros(total_samples, 1);
            for k = 1:K
                mask = (day_array >= edges(k)) & (day_array < edges(k+1));
                fold_indices(mask) = k;
            end

            oof_scores = NaN(total_samples, 1, 'single');
            Y_cat = categorical(Y_flat);
            t_tree = templateTree('MaxNumSplits', obj.MaxNumSplits, 'MinLeafSize', obj.MinLeafSize);

            % 2. 依序對 Fold 2 ~ K 進行前向推論 (Expanding Window)[cite: 1]
            for k = 2:K
                val_mask   = (fold_indices == k);
                train_mask = (day_array < (edges(k) - embargo));

                tr_idx = find(train_mask);
                va_idx = find(val_mask);

                if isempty(tr_idx) || isempty(va_idx)
                    continue;
                end

                % 採樣加速與防止記憶體溢出
                if length(tr_idx) > obj.MaxTrainSamples
                    tr_idx = randsample(tr_idx, obj.MaxTrainSamples);
                end

                % LogitBoost 訓練[cite: 1]
                mdl = fitcensemble(X_flat(tr_idx, :), Y_cat(tr_idx), ...
                    'Method', 'LogitBoost', ...
                    'Learners', t_tree, ...
                    'NumLearningCycles', obj.NumLearningCycles, ...
                    'LearnRate', obj.LearnRate);

                [~, score] = predict(mdl, X_flat(va_idx, :));
                oof_scores(va_idx) = single(score(:, 2));
            end

            % 3. 提取有效 OOF 評估集合 (Fold >= 2)[cite: 1]
            eval_mask = (fold_indices >= 2) & ~isnan(oof_scores);
            val_y_true = double(Y_flat(eval_mask));
            val_scores = double(oof_scores(eval_mask));
            val_probs  = 1 ./ (1 + exp(-val_scores)); % Sigmoid 轉換為機率
            val_days   = day_array(eval_mask);

            oof_probs = NaN(total_samples, 1, 'single');
            oof_probs(eval_mask) = single(val_probs);

            % 4. 調用 Day-Level Block Bootstrap 計算 95% 信賴區間[cite: 1]
            [ci_lower, ci_upper, auc_point, boot_aucs] = block_bootstrap_auc_by_day(...
                val_y_true, val_probs, val_days, n_boot, 0.95);

            % 5. 封裝結果結構體
            pos_ratio = mean(val_y_true == 1);
            metrics = struct();
            metrics.AUC_Point     = auc_point;
            metrics.CI_Lower      = ci_lower;
            metrics.CI_Upper      = ci_upper;
            metrics.PosClassRatio = pos_ratio;
            metrics.NumSamples    = length(val_y_true);
            metrics.NumDays       = length(unique(val_days));
            metrics.BootAUCs      = boot_aucs;
        end
    end
end