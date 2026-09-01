classdef GBDTExpertAgent < handle
    % =========================================================================
    % 類別：GBDTExpertAgent (顯性風險預測流與 SHAP 解釋器)
    % 升級：Phase 15 (★ 崩盤正樣本分層時序 CV、Bootstrap 95% CI、Platt 機率校準)
    % 職責：接收 DL 萃取之 Embedding 與擴充宏觀特徵，輸出 OOF 校準機率與 SHAP 歸因
    % =========================================================================
    
    properties
        ConfigObj           % 全域設定檔參考
        MdlTime             % 最終訓練完成之時序專家 GBDT 模型
        MdlSpace            % 最終訓練完成之空間專家 GBDT 模型
        MdlCrash            % 最終訓練完成之崩盤護欄 GBDT 模型
        MdlPlatt            % 崩盤護欄之 Platt Scaling 機率校準模型
        
        FeatureNamesTime    % 時序特徵名稱對應表 (供 SHAP 解釋使用)
        FeatureNamesSpace   % 空間特徵名稱對應表 (供 SHAP 解釋使用)
        FeatureNamesMacro   % 宏觀特徵名稱對應表 (供 SHAP 解釋使用)
    end
    
    methods
        % ---------------------------------------------------------
        % 建構子：初始化代理人與載入設定
        % ---------------------------------------------------------
        function obj = GBDTExpertAgent(configObj)
            obj.ConfigObj = configObj; 
            fprintf(' ⚙️ [GBDTExpertAgent] 實例化完成。已啟動正樣本分層 CV、Bootstrap 95%% CI 與 Platt 校準。\n');
        end
        
        % =========================================================
        % 1. 橫截面專家訓練與 OOF (Out-of-Fold) 機率推論
        % =========================================================
        function [P_time_oof, P_space_oof] = train_and_predict_oof_cross_sectional(obj, E_time_3D, E_space_3D, X_norm_18D, Macro_2D, Y_Labels_3D, Expert_Active)
            disp('--- 啟動 GBDT 雙軌橫截面專家訓練 (Purged 時序 OOF 與 AUC 即時監控) ---');
            
            [numDays, embedDim, numTickers] = size(E_time_3D);
            numMacro = size(Macro_2D, 2); 
            numRawFeats = size(X_norm_18D, 2);
            
            totalActive = sum(Expert_Active, 'all');
            fprintf('  -> 總有效橫截面樣本數: %d 筆 | 宏觀特徵維度: %d 維\n', totalActive, numMacro);
            
            X_time_flat = zeros(totalActive, embedDim + numMacro, 'single');
            X_space_flat = zeros(totalActive, embedDim + numMacro, 'single');
            X_raw_flat = zeros(totalActive, numRawFeats + numMacro, 'single'); % Baseline 專用矩陣
            Y_flat = zeros(totalActive, 1, 'single');
            row_mapping = zeros(totalActive, 2); 
            
            idx = 1;
            for t = 1:numDays
                active_idx = find(Expert_Active(t, :));
                n_active = length(active_idx);
                
                if n_active > 0
                    e_t = permute(E_time_3D(t, :, active_idx), [3, 2, 1]);
                    e_s = permute(E_space_3D(t, :, active_idx), [3, 2, 1]);
                    x_r = permute(X_norm_18D(t, :, active_idx), [3, 2, 1]);
                    mac = repmat(Macro_2D(t, :), n_active, 1);
                    
                    X_time_flat(idx : idx+n_active-1, :) = [e_t, mac];
                    X_space_flat(idx : idx+n_active-1, :) = [e_s, mac];
                    X_raw_flat(idx : idx+n_active-1, :) = [x_r, mac];
                    Y_flat(idx : idx+n_active-1) = Y_Labels_3D(t, active_idx)';
                    
                    row_mapping(idx : idx+n_active-1, 1) = t; 
                    row_mapping(idx : idx+n_active-1, 2) = active_idx(:);
                    idx = idx + n_active;
                end
            end
            
            macro_names = obj.get_macro_names(numMacro);
            obj.FeatureNamesMacro = macro_names;
            
            obj.FeatureNamesTime = [arrayfun(@(x) sprintf('T_Emb_%d', x), 1:embedDim, 'UniformOutput', false), macro_names];
            obj.FeatureNamesSpace = [arrayfun(@(x) sprintf('S_Emb_%d', x), 1:embedDim, 'UniformOutput', false), macro_names];
            
            Y_categorical = categorical(Y_flat);
            t_tree = templateTree('MaxNumSplits', 20, 'MinLeafSize', 50); 
            
            oof_scores_time = zeros(totalActive, 1, 'single');
            oof_scores_space = zeros(totalActive, 1, 'single');
            oof_scores_raw = zeros(totalActive, 1, 'single');
            
            K = 5; 
            embargo = 20; % 20 天滾動特徵隔離期
            fprintf('  -> 啟動手動 %d-Fold 區塊時序 (Purged Expanding Window, Embargo=%d天) 交叉驗證...\n', K, embargo);
            
            day_array = row_mapping(:, 1);
            edges = linspace(1, numDays + 0.1, K + 1);
            fold_indices = zeros(totalActive, 1);
            for k = 1:K
                mask = (day_array >= edges(k)) & (day_array < edges(k+1));
                fold_indices(mask) = k;
            end
            
            max_train_samples = 150000;
            
            for k = 2:K
                fprintf('     - 正在訓練時間區塊 Fold %d/%d (嚴格防洩漏) ... ', k, K);
                
                val_mask = (fold_indices == k);
                train_mask = (day_array < (edges(k) - embargo));
                
                train_idx = find(train_mask);
                val_idx = find(val_mask);
                
                if length(train_idx) > max_train_samples
                    train_idx = randsample(train_idx, max_train_samples);
                end
                
                X_T_train = X_time_flat(train_idx, :);
                X_S_train = X_space_flat(train_idx, :);
                X_R_train = X_raw_flat(train_idx, :);
                Y_train = Y_categorical(train_idx);
                
                mdl_t = fitcensemble(X_T_train, Y_train, 'Method', 'LogitBoost', ...
                    'Learners', t_tree, 'NumLearningCycles', 30, 'LearnRate', 0.1, ...
                    'PredictorNames', obj.FeatureNamesTime);
                    
                mdl_s = fitcensemble(X_S_train, Y_train, 'Method', 'LogitBoost', ...
                    'Learners', t_tree, 'NumLearningCycles', 30, 'LearnRate', 0.1, ...
                    'PredictorNames', obj.FeatureNamesSpace);
                    
                % 訓練純原始特徵 Baseline (不經 DL 萃取)
                mdl_r = fitcensemble(X_R_train, Y_train, 'Method', 'LogitBoost', ...
                    'Learners', t_tree, 'NumLearningCycles', 30, 'LearnRate', 0.1);
                
                chunk_size = 100000;
                for start_idx = 1:chunk_size:length(val_idx)
                    end_idx = min(start_idx + chunk_size - 1, length(val_idx));
                    curr_val_idx = val_idx(start_idx:end_idx);
                    
                    [~, score_t] = predict(mdl_t, X_time_flat(curr_val_idx, :));
                    [~, score_s] = predict(mdl_s, X_space_flat(curr_val_idx, :));
                    [~, score_r] = predict(mdl_r, X_raw_flat(curr_val_idx, :));
                    
                    oof_scores_time(curr_val_idx) = score_t(:, 2);
                    oof_scores_space(curr_val_idx) = score_s(:, 2);
                    oof_scores_raw(curr_val_idx) = score_r(:, 2);
                end
                
                val_y = double(Y_flat(val_idx));
                p_t_fold = 1 ./ (1 + exp(-oof_scores_time(val_idx)));
                p_s_fold = 1 ./ (1 + exp(-oof_scores_space(val_idx)));
                p_r_fold = 1 ./ (1 + exp(-oof_scores_raw(val_idx)));
                
                auc_t = 0.5; auc_s = 0.5; auc_r = 0.5;
                if length(unique(val_y)) > 1
                    try
                        [~,~,~,auc_t] = perfcurve(val_y, p_t_fold, 1);
                        [~,~,~,auc_s] = perfcurve(val_y, p_s_fold, 1);
                        [~,~,~,auc_r] = perfcurve(val_y, p_r_fold, 1);
                    catch
                    end
                end
                fprintf('完成！(Fold %d OOF AUC -> Raw Baseline: %.4f | Time: %.4f | Space: %.4f)\n', k, auc_r, auc_t, auc_s);
            end
            
            oof_p_time = 1 ./ (1 + exp(-oof_scores_time));
            oof_p_space = 1 ./ (1 + exp(-oof_scores_space));
            oof_p_raw = 1 ./ (1 + exp(-oof_scores_raw));
            
            eval_idx_all = find(fold_indices >= 2);
            val_y_all = double(Y_flat(eval_idx_all));
            overall_auc_t = 0.5; overall_auc_s = 0.5; overall_auc_raw = 0.5;
            if length(unique(val_y_all)) > 1
                try
                    [~,~,~,overall_auc_t] = perfcurve(val_y_all, oof_p_time(eval_idx_all), 1);
                    [~,~,~,overall_auc_s] = perfcurve(val_y_all, oof_p_space(eval_idx_all), 1);
                    [~,~,~,overall_auc_raw] = perfcurve(val_y_all, oof_p_raw(eval_idx_all), 1);
                catch
                end
            end
            fprintf('\n 📊 [GBDT 交叉驗證總評] 全域 OOF AUC:\n');
            fprintf('    > 原始 18D 基準線 : %.4f\n', overall_auc_raw);
            fprintf('    > 時序專家 (Time) : %.4f\n', overall_auc_t);
            fprintf('    > 空間專家 (Space): %.4f\n', overall_auc_s);
            
            if (overall_auc_t - overall_auc_raw) <= 0.005 && (overall_auc_s - overall_auc_raw) <= 0.005
                warning('⚠️ 警告：DL Embedding 的 AUC 未能顯著超越「原始未萃取特徵」，表示特徵萃取器可能未有效增值！');
            else
                disp('✅ DL Embedding 成功打敗原始特徵基準，證明時空特徵萃取器確實捕捉到額外 Alpha 訊號！');
            end
            
            % ---------------------------------------------------------
            % 訓練最終全域模型 (時間衰減加權抽樣版)
            % ---------------------------------------------------------
            fprintf('  -> 訓練最終全域橫截面模型 (時間衰減加權抽樣版)...\n');
            
            time_idx_of_sample = row_mapping(:, 1);
            half_life_days = 756; 
            sample_weights = 0.5 .^ ((numDays - time_idx_of_sample) / half_life_days);
            final_train_idx = randsample(totalActive, min(totalActive, max_train_samples), true, sample_weights);
            
            obj.MdlTime = fitcensemble(X_time_flat(final_train_idx, :), Y_categorical(final_train_idx), ...
                'Method', 'LogitBoost', 'Learners', t_tree, 'NumLearningCycles', 50, ...
                'LearnRate', 0.1, 'PredictorNames', obj.FeatureNamesTime);
                
            obj.MdlSpace = fitcensemble(X_space_flat(final_train_idx, :), Y_categorical(final_train_idx), ...
                'Method', 'LogitBoost', 'Learners', t_tree, 'NumLearningCycles', 50, ...
                'LearnRate', 0.1, 'PredictorNames', obj.FeatureNamesSpace);
            
            P_time_oof = zeros(numDays, numTickers, 'single');
            P_space_oof = zeros(numDays, numTickers, 'single');
            
            for i = 1:totalActive
                t_idx = row_mapping(i, 1);
                tic_idx = row_mapping(i, 2);
                P_time_oof(t_idx, tic_idx) = oof_p_time(i);
                P_space_oof(t_idx, tic_idx) = oof_p_space(i);
            end
            
            P_time_oof(P_time_oof == 0) = 0.5;
            P_space_oof(P_space_oof == 0) = 0.5;
            
            disp('✅ 個股橫截面 GBDT 訓練與 OOF 機率重組完畢！');
        end
        
        % =========================================================
        % 2. 崩盤護欄訓練與 OOF 機率推論 (★ Phase 15 正樣本分層時序 CV + Bootstrap 95% CI)
        % =========================================================
        function P_crash_oof = train_and_predict_oof_crash(obj, Macro_2D, Y_Crash_1D)
            disp('--- 啟動崩盤護欄 GBDT 訓練 (正樣本分層時序 OOF、Bootstrap 95% CI 與 Platt 校準) ---');
            
            numDays = size(Macro_2D, 1);
            numMacro = size(Macro_2D, 2);
            Y_categorical = categorical(Y_Crash_1D);
            
            t_tree = templateTree('MaxNumSplits', 16, 'MinLeafSize', 25);
            obj.FeatureNamesMacro = obj.get_macro_names(numMacro);
            
            K = 5;
            embargo = 20; % 20 天滾動特徵隔離期
            P_crash_oof_scores = zeros(numDays, 1, 'single');
            
            % ★ Phase 15 (P1-3)：依據崩盤正樣本累積密度決定時序切分邊界 (Stratified Chronological Folds)
            pos_crash_indices = find(Y_Crash_1D == 1);
            num_pos_crash = length(pos_crash_indices);
            
            if num_pos_crash >= (K * 5)
                % 確保每個驗證 Fold 均勻分配到等量的崩盤歷史日
                pos_segment_idx = round(linspace(1, num_pos_crash, K + 1));
                edges = pos_crash_indices(pos_segment_idx);
                edges(1) = 1;
                edges(end) = numDays + 1;
                edges = unique(edges);
                if length(edges) < (K + 1)
                    edges = round(linspace(1, numDays + 0.1, K + 1));
                end
            else
                edges = round(linspace(1, numDays + 0.1, K + 1));
            end
            
            for k = 2:K
                val_mask = ((1:numDays)' >= edges(k)) & ((1:numDays)' < edges(k+1));
                train_mask = ((1:numDays)' < (edges(k) - embargo));
                
                num_val_pos = sum(Y_Crash_1D(val_mask) == 1);
                fprintf('     - 正在訓練崩盤護欄 Fold %d/%d (驗證崩盤樣本: %d 天) ... ', k, K, num_val_pos);
                
                % 防呆：若訓練集正樣本過少則退回基礎權重
                if sum(Y_Crash_1D(train_mask) == 1) < 5
                    mdl_crash_fold = fitcensemble(Macro_2D(train_mask, :), Y_categorical(train_mask), ...
                        'Method', 'LogitBoost', 'Learners', t_tree, 'NumLearningCycles', 30, ...
                        'LearnRate', 0.1, 'PredictorNames', obj.FeatureNamesMacro);
                else
                    mdl_crash_fold = fitcensemble(Macro_2D(train_mask, :), Y_categorical(train_mask), ...
                        'Method', 'RUSBoost', 'Learners', t_tree, 'NumLearningCycles', 30, ...
                        'LearnRate', 0.1, 'PredictorNames', obj.FeatureNamesMacro);
                end
                
                [~, score_c] = predict(mdl_crash_fold, Macro_2D(val_mask, :));
                P_crash_oof_scores(val_mask) = score_c(:, 2);
                
                val_crash_y = double(Y_Crash_1D(val_mask));
                p_c_fold = 1 ./ (1 + exp(-score_c(:, 2)));
                
                auc_c = 0.5;
                ci_lower = 0.5; ci_upper = 0.5;
                
                if length(unique(val_crash_y)) > 1
                    try
                        [~,~,~,auc_c] = perfcurve(val_crash_y, p_c_fold, 1);
                        % ★ Phase 15 (P1-3)：Bootstrap 95% 信賴區間計算
                        n_boot = 1000;
                        boot_aucs = zeros(n_boot, 1);
                        n_val = length(val_crash_y);
                        for b = 1:n_boot
                            b_idx = randsample(n_val, n_val, true);
                            if length(unique(val_crash_y(b_idx))) > 1
                                [~,~,~,boot_aucs(b)] = perfcurve(val_crash_y(b_idx), p_c_fold(b_idx), 1);
                            else
                                boot_aucs(b) = 0.5;
                            end
                        end
                        ci_lower = prctile(boot_aucs, 2.5);
                        ci_upper = prctile(boot_aucs, 97.5);
                    catch
                    end
                end
                fprintf('完成！(Fold %d OOF Crash AUC: %.4f [95%% CI: %.4f, %.4f])\n', k, auc_c, ci_lower, ci_upper);
            end
            
            P_crash_oof_uncalibrated = 1 ./ (1 + exp(-P_crash_oof_scores));
            
            eval_crash_mask = ((1:numDays)' >= edges(2));
            val_crash_true = double(Y_Crash_1D(eval_crash_mask));
            uncalibrated_pred = P_crash_oof_uncalibrated(eval_crash_mask);
            
            % ★ Platt Scaling 事後機率校準
            fprintf('  -> 執行 Platt Scaling 事後機率校準 (Logistic Regression)...\n');
            try
                obj.MdlPlatt = fitglm(uncalibrated_pred, val_crash_true, 'Distribution', 'binomial');
                P_crash_oof = predict(obj.MdlPlatt, P_crash_oof_uncalibrated);
                val_crash_pred_calibrated = P_crash_oof(eval_crash_mask);
            catch ME
                warning('⚠️ Platt Scaling 失敗，退回未校準機率: %s', ME.message);
                obj.MdlPlatt = [];
                P_crash_oof = P_crash_oof_uncalibrated;
                val_crash_pred_calibrated = uncalibrated_pred;
            end
            
            overall_auc_c = 0.5;
            overall_ci_lower = 0.5; overall_ci_upper = 0.5;
            
            if length(unique(val_crash_true)) > 1
                try
                    [~,~,~,overall_auc_c] = perfcurve(val_crash_true, val_crash_pred_calibrated, 1);
                    n_boot = 1000;
                    boot_aucs_all = zeros(n_boot, 1);
                    n_all = length(val_crash_true);
                    for b = 1:n_boot
                        b_idx = randsample(n_all, n_all, true);
                        if length(unique(val_crash_true(b_idx))) > 1
                            [~,~,~,boot_aucs_all(b)] = perfcurve(val_crash_true(b_idx), val_crash_pred_calibrated(b_idx), 1);
                        else
                            boot_aucs_all(b) = 0.5;
                        end
                    end
                    overall_ci_lower = prctile(boot_aucs_all, 2.5);
                    overall_ci_upper = prctile(boot_aucs_all, 97.5);
                catch
                end
            end
            
            brier_uncalibrated = mean((uncalibrated_pred - val_crash_true).^2);
            brier_calibrated = mean((val_crash_pred_calibrated - val_crash_true).^2);
            
            fprintf('\n=================================================================\n');
            fprintf('📊 【Phase 15 崩盤護欄總評與 Platt Scaling 校準報告】\n');
            fprintf('=================================================================\n');
            fprintf(' > 全域 OOF Crash AUC   : %.4f [95%% CI: %.4f, %.4f]\n', overall_auc_c, overall_ci_lower, overall_ci_upper);
            fprintf(' > 校準前 Brier Score   : %.4f\n', brier_uncalibrated);
            fprintf(' > 校準後 Brier Score   : %.4f (改善: %+.2f%%)\n', brier_calibrated, (brier_uncalibrated - brier_calibrated)/brier_uncalibrated*100);
            fprintf('-----------------------------------------------------------------\n\n');
            
            obj.MdlCrash = fitcensemble(Macro_2D, Y_categorical, 'Method', 'RUSBoost', ...
                'Learners', t_tree, 'NumLearningCycles', 50, ...
                'LearnRate', 0.1, 'PredictorNames', obj.FeatureNamesMacro);
                
            disp('✅ 崩盤護欄 GBDT 訓練與 OOF Platt 校準機率產出完畢！');
        end
        
        % =========================================================
        % 3. 全域盲測期 (OOS) 真實前向推論
        % =========================================================
        function [P_time_oos, P_space_oos, P_crash_oos] = predict_oos(obj, E_time_3D, E_space_3D, Macro_2D, Expert_Active)
            [numDays, ~, numTickers] = size(E_time_3D);
            
            P_time_oos = zeros(numDays, numTickers, 'single') + 0.5;
            P_space_oos = zeros(numDays, numTickers, 'single') + 0.5;
            
            for t = 1:numDays
                active_idx = find(Expert_Active(t, :));
                n_active = length(active_idx);
                if n_active > 0
                    e_t = permute(E_time_3D(t, :, active_idx), [3, 2, 1]);
                    e_s = permute(E_space_3D(t, :, active_idx), [3, 2, 1]);
                    mac = repmat(Macro_2D(t, :), n_active, 1);
                    
                    X_t = [e_t, mac];
                    X_s = [e_s, mac];
                    
                    [~, s_time] = predict(obj.MdlTime, X_t);
                    [~, s_space] = predict(obj.MdlSpace, X_s);
                    
                    P_time_oos(t, active_idx) = 1 ./ (1 + exp(-s_time(:, 2)));
                    P_space_oos(t, active_idx) = 1 ./ (1 + exp(-s_space(:, 2)));
                end
            end
            
            [~, s_crash] = predict(obj.MdlCrash, Macro_2D);
            uncalibrated_p = 1 ./ (1 + exp(-s_crash(:, 2)));
            
            if ~isempty(obj.MdlPlatt)
                P_crash_oos = predict(obj.MdlPlatt, uncalibrated_p);
            else
                P_crash_oos = uncalibrated_p;
            end
        end
        
        % =========================================================
        % 4. 決策可解釋性分析 (SHAP Value Attribution)
        % =========================================================
        function explain_shapley(obj, X_query_raw, X_bg_raw, mode)
            fprintf('--- 啟動 %s 專家 SHAP 歸因分析 (K-Means 加速) ---\n', upper(mode));
            
            if strcmp(mode, 'time')
                target_mdl = obj.MdlTime;
                var_names = obj.FeatureNamesTime;
            elseif strcmp(mode, 'space')
                target_mdl = obj.MdlSpace;
                var_names = obj.FeatureNamesSpace;
            else
                target_mdl = obj.MdlCrash;
                var_names = obj.FeatureNamesMacro;
            end
            
            num_centroids = min(100, size(X_bg_raw, 1));
            fprintf('  -> 執行 K-Means 壓縮背景樣本至 %d 個質心...\n', num_centroids);
            [~, bg_centroids] = kmeans(double(X_bg_raw), num_centroids, 'MaxIter', 100);
            
            X_query_tbl = array2table(double(X_query_raw), 'VariableNames', var_names);
            X_bg_tbl = array2table(bg_centroids, 'VariableNames', var_names);
            
            explainer = shapley(target_mdl, X_bg_tbl);
            shap_results = fit(explainer, X_query_tbl, 'UseParallel', true);
            
            fig_name = sprintf('SHAP 歸因分析 - %s', upper(mode));
            figure('Name', fig_name, 'Color', 'w', 'Position', [100, 100, 900, 600]);
            plot(shap_results);
            title(sprintf('特徵對預測的邊際貢獻度 (Mode: %s)', upper(mode)), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            
            disp('✅ SHAP 圖表已生成！');
        end
    end
    
    % =========================================================
    % 私有輔助函數：動態解析宏觀特徵名稱
    % =========================================================
    methods (Access = private)
        function names = get_macro_names(~, numMacro)
            base_names = {'VIX_Proxy', 'SPY_R20', 'SPY_R60', 'Market_Breadth', ...
                          'Real_VIX', 'VRP_Spread', 'T10Y2Y_YieldCurve', 'HY_CreditSpread', 'DGS10_Rate', 'UNRATE_Macro'};
            if numMacro <= length(base_names)
                names = base_names(1:numMacro);
            else
                extra_names = arrayfun(@(x) sprintf('Macro_%d', x), (length(base_names)+1):numMacro, 'UniformOutput', false);
                names = [base_names, extra_names];
            end
        end
    end
end
