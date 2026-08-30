classdef GBDTExpertAgent < handle
    % =========================================================================
    % 類別：GBDTExpertAgent (顯性風險預測流與 SHAP 解釋器)
    % 升級：Phase 14.22 (★ 補齊 OOF AUC 即時可觀測性、時間衰減加權抽樣、RUSBoost 崩盤防護)
    % 職責：接收 DL 萃取之 Embedding，輸出 OOF 校準機率與 SHAP 解釋
    % =========================================================================
    
    properties
        ConfigObj           % 全域設定檔參考
        MdlTime             % 最終訓練完成之時序專家 GBDT 模型
        MdlSpace            % 最終訓練完成之空間專家 GBDT 模型
        MdlCrash            % 最終訓練完成之崩盤護欄 GBDT 模型
        
        FeatureNamesTime    % 時序特徵名稱對應表 (供 SHAP 解釋使用)
        FeatureNamesSpace   % 空間特徵名稱對應表 (供 SHAP 解釋使用)
        FeatureNamesMacro   % 宏觀特徵名稱對應表 (供 SHAP 解釋使用)
    end
    
    methods
        % ---------------------------------------------------------
        % 建構子：初始化代理人與載入設定
        % ---------------------------------------------------------
        function obj = GBDTExpertAgent(configObj)
            obj.ConfigObj = configObj; % 綁定全域設定
            fprintf(' ⚙️ [GBDTExpertAgent] 實例化完成。已啟動時序防洩漏 OOF 與即時可觀測性監控。\n');
        end
        
        % =========================================================
        % 1. 橫截面專家訓練與 OOF (Out-of-Fold) 機率推論
        % =========================================================
        function [P_time_oof, P_space_oof] = train_and_predict_oof_cross_sectional(obj, E_time_3D, E_space_3D, Macro_2D, Y_Labels_3D, Expert_Active)
            disp('--- 啟動 GBDT 雙軌橫截面專家訓練 (嚴格區塊時序 OOF 與 AUC 即時監控) ---');
            
            % 提取張量維度 [交易天數, 嵌入維度, 標的總數]
            [numDays, embedDim, numTickers] = size(E_time_3D);
            numMacro = size(Macro_2D, 2); % 提取宏觀特徵維度 (4 維)
            
            % 計算全域活躍樣本總數，預分配記憶體防爆
            totalActive = sum(Expert_Active, 'all');
            fprintf('  -> 總有效橫截面樣本數: %d 筆\n', totalActive);
            
            X_time_flat = zeros(totalActive, embedDim + numMacro, 'single');
            X_space_flat = zeros(totalActive, embedDim + numMacro, 'single');
            Y_flat = zeros(totalActive, 1, 'single');
            row_mapping = zeros(totalActive, 2); % 記錄 [TimeIndex, TickerIndex]
            
            idx = 1;
            for t = 1:numDays
                active_idx = find(Expert_Active(t, :));
                n_active = length(active_idx);
                
                if n_active > 0
                    e_t = permute(E_time_3D(t, :, active_idx), [3, 2, 1]);
                    e_s = permute(E_space_3D(t, :, active_idx), [3, 2, 1]);
                    mac = repmat(Macro_2D(t, :), n_active, 1);
                    
                    X_time_flat(idx : idx+n_active-1, :) = [e_t, mac];
                    X_space_flat(idx : idx+n_active-1, :) = [e_s, mac];
                    Y_flat(idx : idx+n_active-1) = Y_Labels_3D(t, active_idx)';
                    
                    row_mapping(idx : idx+n_active-1, 1) = t; 
                    row_mapping(idx : idx+n_active-1, 2) = active_idx(:);
                    idx = idx + n_active;
                end
            end
            
            obj.FeatureNamesTime = [arrayfun(@(x) sprintf('T_Emb_%d', x), 1:embedDim, 'UniformOutput', false), ...
                                    {'VIX_Proxy', 'SPY_R20', 'SPY_R60', 'Market_Breadth'}];
            obj.FeatureNamesSpace = [arrayfun(@(x) sprintf('S_Emb_%d', x), 1:embedDim, 'UniformOutput', false), ...
                                    {'VIX_Proxy', 'SPY_R20', 'SPY_R60', 'Market_Breadth'}];
            
            Y_categorical = categorical(Y_flat);
            t_tree = templateTree('MaxNumSplits', 20, 'MinLeafSize', 50); 
            
            oof_scores_time = zeros(totalActive, 1, 'single');
            oof_scores_space = zeros(totalActive, 1, 'single');
            
            K = 5; 
            fprintf('  -> 啟動手動 %d-Fold 區塊時序 (Expanding Window) 交叉驗證...\n', K);
            
            day_array = row_mapping(:, 1);
            edges = linspace(1, numDays + 0.1, K + 1);
            fold_indices = zeros(totalActive, 1);
            for k = 1:K
                mask = (day_array >= edges(k)) & (day_array < edges(k+1));
                fold_indices(mask) = k;
            end
            
            max_train_samples = 150000;
            
            % Expanding Window (遞增視窗) 交叉驗證
            for k = 2:K
                fprintf('     - 正在訓練時間區塊 Fold %d/%d (嚴格防洩漏) ... ', k, K);
                
                val_mask = (fold_indices == k);
                train_mask = (fold_indices < k);
                
                train_idx = find(train_mask);
                val_idx = find(val_mask);
                
                if length(train_idx) > max_train_samples
                    train_idx = randsample(train_idx, max_train_samples);
                end
                
                X_T_train = X_time_flat(train_idx, :);
                X_S_train = X_space_flat(train_idx, :);
                Y_train = Y_categorical(train_idx);
                
                mdl_t = fitcensemble(X_T_train, Y_train, 'Method', 'LogitBoost', ...
                    'Learners', t_tree, 'NumLearningCycles', 30, 'LearnRate', 0.1, ...
                    'PredictorNames', obj.FeatureNamesTime);
                    
                mdl_s = fitcensemble(X_S_train, Y_train, 'Method', 'LogitBoost', ...
                    'Learners', t_tree, 'NumLearningCycles', 30, 'LearnRate', 0.1, ...
                    'PredictorNames', obj.FeatureNamesSpace);
                
                chunk_size = 100000;
                for start_idx = 1:chunk_size:length(val_idx)
                    end_idx = min(start_idx + chunk_size - 1, length(val_idx));
                    curr_val_idx = val_idx(start_idx:end_idx);
                    
                    [~, score_t] = predict(mdl_t, X_time_flat(curr_val_idx, :));
                    [~, score_s] = predict(mdl_s, X_space_flat(curr_val_idx, :));
                    
                    oof_scores_time(curr_val_idx) = score_t(:, 2);
                    oof_scores_space(curr_val_idx) = score_s(:, 2);
                end
                
                % ★ 二次優化修正 1：計算並即時輸出該 Fold 的 OOF AUC
                val_y = double(Y_flat(val_idx));
                p_t_fold = 1 ./ (1 + exp(-oof_scores_time(val_idx)));
                p_s_fold = 1 ./ (1 + exp(-oof_scores_space(val_idx)));
                
                auc_t = 0.5; auc_s = 0.5;
                if length(unique(val_y)) > 1
                    try
                        [~,~,~,auc_t] = perfcurve(val_y, p_t_fold, 1);
                        [~,~,~,auc_s] = perfcurve(val_y, p_s_fold, 1);
                    catch
                    end
                end
                fprintf('完成！(Fold %d OOF AUC -> Time: %.4f | Space: %.4f)\n', k, auc_t, auc_s);
            end
            
            % Sigmoid 轉換為真實機率 (Fold 1 預設為 0.5 中立機率)
            oof_p_time = 1 ./ (1 + exp(-oof_scores_time));
            oof_p_space = 1 ./ (1 + exp(-oof_scores_space));
            
            % ★ 二次優化修正 2：計算全域總體 OOF AUC 診斷指標
            eval_idx_all = find(fold_indices >= 2);
            val_y_all = double(Y_flat(eval_idx_all));
            overall_auc_t = 0.5; overall_auc_s = 0.5;
            if length(unique(val_y_all)) > 1
                try
                    [~,~,~,overall_auc_t] = perfcurve(val_y_all, oof_p_time(eval_idx_all), 1);
                    [~,~,~,overall_auc_s] = perfcurve(val_y_all, oof_p_space(eval_idx_all), 1);
                catch
                end
            end
            fprintf(' 📊 [GBDT 交叉驗證總評] 全域 OOF AUC -> 時序專家: %.4f | 空間專家: %.4f\n', ...
                overall_auc_t, overall_auc_s);
            
            if overall_auc_t < 0.51 && overall_auc_s < 0.51
                warning('⚠️ 警告：GBDT 專家 OOF AUC 接近 0.50 隨機基準線，代表選股訊號較弱，請排查特徵與 Embedding！');
            else
                disp('✅ GBDT 專家成功從特徵表徵中學得具備統計意義的選股排序能力！');
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
        % 2. 崩盤護欄訓練與 OOF 機率推論
        % =========================================================
        function P_crash_oof = train_and_predict_oof_crash(obj, Macro_2D, Y_Crash_1D)
            disp('--- 啟動崩盤護欄 GBDT 訓練 (Block 時序 OOF 與 RUSBoost 不平衡防護) ---');
            
            numDays = size(Macro_2D, 1);
            Y_categorical = categorical(Y_Crash_1D);
            
            t_tree = templateTree('MaxNumSplits', 16, 'MinLeafSize', 25);
            obj.FeatureNamesMacro = {'VIX_Proxy', 'SPY_R20', 'SPY_R60', 'Market_Breadth'};
            
            K = 5;
            edges = linspace(1, numDays + 0.1, K + 1);
            P_crash_oof_scores = zeros(numDays, 1, 'single');
            
            for k = 2:K
                fprintf('     - 正在訓練崩盤護欄 Fold %d/%d ... ', k, K);
                
                val_mask = ((1:numDays)' >= edges(k)) & ((1:numDays)' < edges(k+1));
                train_mask = ((1:numDays)' < edges(k));
                
                mdl_crash_fold = fitcensemble(Macro_2D(train_mask, :), Y_categorical(train_mask), ...
                    'Method', 'RUSBoost', 'Learners', t_tree, 'NumLearningCycles', 30, ...
                    'LearnRate', 0.1, 'PredictorNames', obj.FeatureNamesMacro);
                
                [~, score_c] = predict(mdl_crash_fold, Macro_2D(val_mask, :));
                P_crash_oof_scores(val_mask) = score_c(:, 2);
                
                % 計算各 Fold 崩盤預測 AUC
                val_crash_y = double(Y_Crash_1D(val_mask));
                p_c_fold = 1 ./ (1 + exp(-score_c(:, 2)));
                auc_c = 0.5;
                if length(unique(val_crash_y)) > 1
                    try
                        [~,~,~,auc_c] = perfcurve(val_crash_y, p_c_fold, 1);
                    catch
                    end
                end
                fprintf('完成！(Fold %d OOF Crash AUC: %.4f)\n', k, auc_c);
            end
            
            P_crash_oof = 1 ./ (1 + exp(-P_crash_oof_scores));
            
            % 評估總體崩盤護欄 OOF AUC
            eval_crash_mask = ((1:numDays)' >= edges(2));
            overall_auc_c = 0.5;
            if length(unique(Y_Crash_1D(eval_crash_mask))) > 1
                try
                    [~,~,~,overall_auc_c] = perfcurve(double(Y_Crash_1D(eval_crash_mask)), P_crash_oof(eval_crash_mask), 1);
                catch
                end
            end
            fprintf(' 📊 [崩盤護欄總評] 全域 OOF Crash AUC: %.4f\n', overall_auc_c);
            
            obj.MdlCrash = fitcensemble(Macro_2D, Y_categorical, 'Method', 'RUSBoost', ...
                'Learners', t_tree, 'NumLearningCycles', 50, ...
                'LearnRate', 0.1, 'PredictorNames', obj.FeatureNamesMacro);
                
            disp('✅ 崩盤護欄 GBDT 訓練 (RUSBoost) 與 OOF 機率產出完畢！');
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
            P_crash_oos = 1 ./ (1 + exp(-s_crash(:, 2)));
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
end