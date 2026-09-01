classdef FeatureEvaluator < handle
    % =========================================================================
    % 類別：FeatureEvaluator (特徵信心評估器、ICIR 穩定性健檢與共線性診斷中心)
    % 升級：Phase 15.5 (★ 導入 ICIR 方向穩定性健檢、能量守恆特徵注意力、共線性診斷)
    % 職責：
    %   1. 計算每日橫截面 Spearman Rank IC，產出能量守恆特徵注意力權重
    %   2. 評估特徵長周期 ICIR = mean(signed IC) / std(signed IC)，檢驗方向可交易邊際
    %   3. 評估全特徵相關係數矩陣，自動標記 |corr| > 0.85 的高度共線性特徵對
    % =========================================================================
    
    properties
        ConfigObj % 全域設定檔參考
    end
    
    methods
        % ---------------------------------------------------------
        % 建構子：初始化評估器並綁定全域設定
        % ---------------------------------------------------------
        function obj = FeatureEvaluator(configObj)
            obj.ConfigObj = configObj;
            fprintf(' ⚙️ [FeatureEvaluator] 初始化完成。已載入橫截面 IC/ICIR 評估與共線性診斷引擎。\n');
        end
        
        % =========================================================
        % 核心運算：計算無未來函數的動態特徵權重 (回傳注意力權重、滾動|IC|與每日IC)
        % =========================================================
        function [IC_Weights_2D, Raw_IC_Weight, Daily_IC] = compute_confidence(obj, X_denoised_3D, Prices_Active, Expert_Active)
            % -----------------------------------------------------------------
            % 函數輸入維度：
            % X_denoised_3D: [Days, Feats, NumTickers]
            % Prices_Active: [Days, NumTickers]
            % Expert_Active: [Days, NumTickers] (活躍標的遮罩)
            % 輸出維度：
            % IC_Weights_2D: [Days, Feats] (每日各特徵的能量守恆注意力權重，均值維持 1.0)
            % Raw_IC_Weight: [Days, Feats] (各特徵歷史滾動平均 |IC|，供特徵閘門聚焦使用)
            % Daily_IC:      [Days, Feats] (每日橫截面帶符號 Spearman Rank IC)
            % -----------------------------------------------------------------
            
            disp('--- 啟動極速多核心橫截面特徵信心評估 (Cross-Sectional Spearman IC) ---');
            
            [numDays, numFeats, numTickers] = size(X_denoised_3D);
            
            if isprop(obj.ConfigObj, 'Lookback')
                lookback = obj.ConfigObj.Lookback;
            else
                lookback = 60; 
            end
            
            %% 1. 預先計算 T 日至 T+1 日的 Forward Returns (未來報酬)
            R_fwd = NaN(numDays, numTickers, 'single');
            R_fwd(1:end-1, :) = (Prices_Active(2:end, :) - Prices_Active(1:end-1, :)) ./ Prices_Active(1:end-1, :);
            R_fwd(isinf(R_fwd)) = NaN;
            
            %% 2. 核心引擎：計算每日「橫截面」帶符號 Rank IC
            Daily_IC = zeros(numDays, numFeats, 'single');
            min_valid_samples = max(10, floor(numTickers * 0.05));
            
            fprintf('  -> 啟動 Parpool 平行運算，評估 %d 維特徵的橫截面預測力 (動態門檻: %d 檔)...\n', ...
                numFeats, min_valid_samples);
            
            parfor k = 1:numFeats
                feat_ic_col = zeros(numDays, 1, 'single');
                X_k = reshape(X_denoised_3D(:, k, :), numDays, numTickers);
                
                for tau = 1:numDays-1
                    active_mask = Expert_Active(tau, :) & Expert_Active(tau+1, :);
                    
                    if sum(active_mask) < min_valid_samples
                        continue;
                    end
                    
                    x_cross = X_k(tau, active_mask)';
                    y_cross = R_fwd(tau, active_mask)';
                    
                    valid_data = ~isnan(x_cross) & ~isnan(y_cross);
                    x_valid = x_cross(valid_data);
                    y_valid = y_cross(valid_data);
                    
                    if sum(valid_data) < min_valid_samples
                        continue;
                    end
                    
                    rank_x = tiedrank(x_valid);
                    rank_y = tiedrank(y_valid);
                    
                    rx_mean = mean(rank_x);
                    ry_mean = mean(rank_y);
                    
                    num = sum((rank_x - rx_mean) .* (rank_y - ry_mean));
                    den = sqrt(sum((rank_x - rx_mean).^2) * sum((rank_y - ry_mean).^2));
                    
                    if den > 1e-8
                        feat_ic_col(tau) = num / den;
                    end
                end
                
                Daily_IC(:, k) = feat_ic_col;
            end
            
            %% 3. ★ 零洩漏時間平移 (Zero-Leakage Time-Shift 特徵閘門強度)
            disp('  -> 執行時間平移與滾動平滑 (生成無未來函數之 IC 強度)...');
            Raw_IC_Weight = zeros(numDays, numFeats, 'single');
            
            for t = lookback+1 : numDays
                window_ic = Daily_IC(t-lookback : t-1, :);
                mean_abs_ic = mean(abs(window_ic), 1, 'omitnan');
                mean_abs_ic(isnan(mean_abs_ic)) = 0;
                Raw_IC_Weight(t, :) = mean_abs_ic;
            end
            
            %% 4. ★ 能量守恆 Softmax 注意力縮放 (維持特徵尺度均值為 1.0)
            disp('  -> 執行能量守恆 Softmax 注意力縮放 (Energy-Preserving Normalization)...');
            temperature = 1.0; 
            IC_Weights_2D = ones(numDays, numFeats, 'single');
            
            for t = lookback+1 : numDays
                raw_ic = Raw_IC_Weight(t, :);
                
                if sum(raw_ic) > 0
                    mu_ic = mean(raw_ic);
                    std_ic = std(raw_ic) + 1e-8;
                    norm_ic = (raw_ic - mu_ic) ./ std_ic;
                    
                    max_z = max(norm_ic);
                    exp_vals = exp((norm_ic - max_z) / temperature);
                    softmax_prob = exp_vals ./ (sum(exp_vals) + 1e-8);
                    
                    % 保能量乘法縮放：sum(weights) = numFeats，平均權重 = 1.0
                    IC_Weights_2D(t, :) = softmax_prob .* numFeats;
                else
                    IC_Weights_2D(t, :) = 1.0;
                end
            end
            
            disp('✅ 橫截面 Spearman IC 評估與能量守恆注意力權重生成完畢！');
        end
        
        % =====================================================================
        % ★ Phase 15.5 新增：特徵訊號方向穩定性 ICIR 排行榜健檢報告
        % =====================================================================
        function [icir_table, max_icir] = report_icir_ranking(obj, Daily_IC, feat_names, min_icir_threshold)
            if nargin < 4 || isempty(min_icir_threshold)
                min_icir_threshold = 0.05; 
            end
            
            if isprop(obj.ConfigObj, 'Lookback')
                lookback = obj.ConfigObj.Lookback;
            else
                lookback = 60; 
            end
            
            numFeats = size(Daily_IC, 2);
            if nargin < 3 || isempty(feat_names) || length(feat_names) ~= numFeats
                feat_names = arrayfun(@(x) sprintf('Feat_%d', x), 1:numFeats, 'UniformOutput', false);
            end
            
            valid_ic_window = Daily_IC(lookback+1:end, :);
            mean_ic = mean(valid_ic_window, 1, 'omitnan');
            std_ic  = std(valid_ic_window, 0, 1, 'omitnan');
            icir    = mean_ic ./ (std_ic + 1e-8);
            
            % 橫截面有效樣本數檢定 (初篩 t-stat 與 p-val)
            n_eff  = sum(~isnan(valid_ic_window), 1);
            t_stat = icir .* sqrt(n_eff);
            p_val  = 2 * (1 - normcdf(abs(t_stat)));
            
            [sorted_icir, sort_idx] = sort(abs(icir), 'descend');
            
            fprintf('\n=================================================================\n');
            fprintf('📊 【Phase 15.5 特徵訊號方向穩定性健檢：ICIR 排行榜】\n');
            fprintf('=================================================================\n');
            for i = 1:length(sorted_icir)
                j = sort_idx(i);
                sig_flag = '';
                if p_val(j) < 0.05
                    sig_flag = '⭐ p<0.05';
                end
                fprintf('  [%2d] %-16s | mean(IC)=%+.4f  std(IC)=%.4f  ICIR=%+.4f  %s\n', ...
                    i, feat_names{j}, mean_ic(j), std_ic(j), icir(j), sig_flag);
            end
            fprintf('-----------------------------------------------------------------\n');
            
            max_icir = max(abs(icir));
            if max_icir < min_icir_threshold
                warning(['⚠️ 警告：所有特徵 |ICIR| 均低於 %.4f！\n' ...
                         '   代表特徵瞬時相關性不小，但方向長期頻繁反轉，不具可靠交易邊際。\n' ...
                         '   此診斷與 GBDT AUC≈0.50、DSR=0.0000 的結論完全吻合！'], min_icir_threshold);
            else
                fprintf('  ✅ 通過穩定性健檢：最高特徵 |ICIR| = %.4f (>= %.4f)，具備方向預測邊際。\n', ...
                    max_icir, min_icir_threshold);
            end
            fprintf('=================================================================\n\n');
            
            icir_table = table(feat_names(:), mean_ic(:), std_ic(:), icir(:), p_val(:), ...
                'VariableNames', {'Feature', 'Mean_IC', 'Std_IC', 'ICIR', 'P_Value'});
        end
        
        % =====================================================================
        % 特徵絕對相關性強度報告 (mean(|IC|)，供特徵閘門參考)
        % =====================================================================
        function report_ic_ranking(obj, Raw_IC_Weight, feat_names)
            if isprop(obj.ConfigObj, 'Lookback')
                lookback = obj.ConfigObj.Lookback;
            else
                lookback = 60; 
            end
            
            numFeats = size(Raw_IC_Weight, 2);
            if nargin < 3 || isempty(feat_names) || length(feat_names) ~= numFeats
                feat_names = arrayfun(@(x) sprintf('Feat_%d', x), 1:numFeats, 'UniformOutput', false);
            end
            
            mean_abs_ic_per_feat = mean(abs(Raw_IC_Weight(lookback+1:end, :)), 1, 'omitnan');
            [sorted_ic, sort_idx] = sort(mean_abs_ic_per_feat, 'descend');
            
            fprintf('\n=================================================================\n');
            fprintf('📊 【特徵閘門注意力強度：平均絕對 |IC| 排行榜】\n');
            fprintf('=================================================================\n');
            for i = 1:length(sorted_ic)
                fprintf('  [%2d] %-16s | 滾動 mean(|IC|) = %.4f\n', i, feat_names{sort_idx(i)}, sorted_ic(i));
            end
            fprintf('=================================================================\n\n');
        end
        
        % =====================================================================
        % 全特徵 Pairwise 相關係數共線性診斷
        % =====================================================================
        function [corr_matrix, high_corr_pairs] = diagnose_collinearity(~, X_3D, Expert_Active, feat_names)
            disp('--- 啟動全特徵 Pairwise 相關性共線性診斷 (Collinearity Audit) ---');
            
            [numDays, numFeats, numTickers] = size(X_3D);
            
            if nargin < 4 || isempty(feat_names)
                feat_names = arrayfun(@(x) sprintf('Feat_%d', x), 1:numFeats, 'UniformOutput', false);
            end
            
            active_mask_flat = reshape(Expert_Active, [numDays * numTickers, 1]);
            X_flat = zeros(numDays * numTickers, numFeats, 'single');
            for k = 1:numFeats
                slice_k = X_3D(:, k, :);
                X_flat(:, k) = slice_k(:);
            end
            
            valid_rows = active_mask_flat & all(~isnan(X_flat) & ~isinf(X_flat), 2);
            X_valid = double(X_flat(valid_rows, :));
            
            if size(X_valid, 1) > 200000
                sample_idx = randsample(size(X_valid, 1), 200000);
                X_valid = X_valid(sample_idx, :);
            end
            
            corr_matrix = corr(X_valid, 'Type', 'Spearman');
            corr_matrix(isnan(corr_matrix)) = 0;
            
            threshold = 0.85;
            [row_idx, col_idx] = find(triu(abs(corr_matrix) > threshold, 1));
            
            high_corr_pairs = cell(length(row_idx), 3);
            fprintf('\n=================================================================\n');
            fprintf('📊 【特徵共線性診斷報告 (|Corr| > %.2f)】\n', threshold);
            fprintf('=================================================================\n');
            
            if isempty(row_idx)
                fprintf('  ✅ 優秀！所有特徵間之 Spearman 相關係數均低於 %.2f，無嚴重共線性！\n', threshold);
            else
                fprintf('  ⚠️ 警告：偵測到 %d 對高度共線性特徵，建議在 FeatureEngineer 中評估合併或剔除：\n\n', length(row_idx));
                for p = 1:length(row_idx)
                    f1 = feat_names{row_idx(p)};
                    f2 = feat_names{col_idx(p)};
                    c_val = corr_matrix(row_idx(p), col_idx(p));
                    high_corr_pairs{p, 1} = f1;
                    high_corr_pairs{p, 2} = f2;
                    high_corr_pairs{p, 3} = c_val;
                    fprintf('  [%2d] %-18s <---> %-18s | Spearman Corr = %+.4f\n', p, f1, f2, c_val);
                end
            end
            fprintf('=================================================================\n\n');
        end
    end
end
