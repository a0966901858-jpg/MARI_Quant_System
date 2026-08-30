classdef FeatureEvaluator < handle
    % =========================================================================
    % 類別：FeatureEvaluator (特徵信心評估器)
    % 升級：Phase 14.22 (★ 能量守恆特徵注意力、軟化 Softmax 溫度、防禦特徵通道尺度坍塌)
    % 職責：計算各維度特徵的動態預測能力 (Spearman Rank IC)，產出特徵信心注意力權重矩陣
    % 對接：產出之 [Days, Feats] 權重可直接注入 Phase 2 作為雙軌萃取器的特徵動態遮罩
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
            fprintf(' ⚙️ [FeatureEvaluator] 初始化完成。準備執行橫截面 IC 檢定與信心注意力生成。\n');
        end
        
        % =========================================================
        % 核心運算：計算無未來函數的動態特徵權重
        % =========================================================
        function IC_Weights_2D = compute_confidence(obj, X_denoised_3D, Prices_Active, Expert_Active)
            % -----------------------------------------------------------------
            % 函數輸入維度斷言：
            % X_denoised_3D: [Days, Feats, NumTickers] (由 FeatureEngineer 重構產出)
            % Prices_Active: [Days, NumTickers]
            % Expert_Active: [Days, NumTickers] (活躍標的遮罩)
            % 輸出維度：
            % IC_Weights_2D: [Days, Feats] (每日各特徵的能量守恆注意力權重，均值維持 1.0)
            % -----------------------------------------------------------------
            
            disp('--- 啟動極速多核心橫截面特徵信心評估 (Cross-Sectional Spearman IC) ---');
            
            [numDays, numFeats, numTickers] = size(X_denoised_3D);
            
            % 動態獲取 Config 中的回溯視窗長度，若無則預設為 60 (約一季)
            if isprop(obj.ConfigObj, 'Lookback')
                lookback = obj.ConfigObj.Lookback;
            else
                lookback = 60; 
            end
            
            %% 1. 預先計算 T 日至 T+1 日的 Forward Returns (未來報酬)
            % 嚴格物理定義：R_fwd(t) 是從 t 日收盤買進，持有至 t+1 日收盤的報酬
            R_fwd = NaN(numDays, numTickers, 'single');
            R_fwd(1:end-1, :) = (Prices_Active(2:end, :) - Prices_Active(1:end-1, :)) ./ Prices_Active(1:end-1, :);
            R_fwd(isinf(R_fwd)) = NaN;
            
            %% 2. 核心引擎：計算每日「橫截面 (Cross-Sectional)」Rank IC
            Daily_IC = zeros(numDays, numFeats, 'single');
            
            % 統一橫截面最小樣本門檻，與 FeatureEngineer.m 完全一致
            min_valid_samples = max(10, floor(numTickers * 0.05));
            
            fprintf('  -> 啟動 Parpool 平行運算，評估 %d 維特徵的橫截面預測力 (動態門檻: %d 檔)...\n', ...
                numFeats, min_valid_samples);
            
            % 使用 parfor 平行運算各維度特徵的 IC
            parfor k = 1:numFeats
                feat_ic_col = zeros(numDays, 1, 'single');
                
                % 鎖定 [Days, NumTickers] 2D 矩陣
                X_k = reshape(X_denoised_3D(:, k, :), numDays, numTickers);
                
                for tau = 1:numDays-1
                    % 僅針對當日與次日皆為活躍的標的進行截面比較
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
                    
                    % 計算 Spearman Rank Correlation (等級相關係數)
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
            
            %% 3. ★ 零洩漏時間平移 (Zero-Leakage Time-Shift)
            disp('  -> 執行時間平移與滾動平滑 (生成無未來函數之 IC 強度)...');
            Raw_IC_Weight = zeros(numDays, numFeats, 'single');
            
            for t = lookback+1 : numDays
                % ☢️ 絕對防禦：在 t 日推論時，嚴格只使用歷史 Daily_IC(t-lookback : t-1)
                window_ic = Daily_IC(t-lookback : t-1, :);
                
                % 取絕對值：捕獲預測能力強度
                mean_abs_ic = mean(abs(window_ic), 1, 'omitnan');
                mean_abs_ic(isnan(mean_abs_ic)) = 0;
                
                Raw_IC_Weight(t, :) = mean_abs_ic;
            end
            
            %% 4. ★ 能量守恆 Softmax 注意力縮放 (防止特徵通道變異數歸零)
            disp('  -> 執行能量守恆 Softmax 注意力縮放 (Energy-Preserving Normalization)...');
            
            % ★ 二次優化修正 1：軟化 Softmax 溫度至 1.0 (避免過度尖銳導致非主力通道權重歸零)
            temperature = 1.0; 
            
            % ★ 二次優化修正 2：全域預設值設為 1.0 (確保暖機期特徵尺度不被縮減)
            IC_Weights_2D = ones(numDays, numFeats, 'single');
            
            for t = lookback+1 : numDays
                raw_ic = Raw_IC_Weight(t, :);
                
                if sum(raw_ic) > 0
                    % 橫截面 Z-score 標準化
                    mu_ic = mean(raw_ic);
                    std_ic = std(raw_ic) + 1e-8;
                    norm_ic = (raw_ic - mu_ic) ./ std_ic;
                    
                    % Softmax 數值穩定運算
                    max_z = max(norm_ic);
                    exp_vals = exp((norm_ic - max_z) / temperature);
                    softmax_prob = exp_vals ./ (sum(exp_vals) + 1e-8);
                    
                    % ★ 二次優化修正 3：保能量乘法縮放 (乘以 numFeats)
                    % 使得 sum(weights) = numFeats，平均權重 = 1.0
                    % 既保留相對 IC 信心加權，又徹底防禦 DL 輸入特徵尺度坍塌
                    IC_Weights_2D(t, :) = softmax_prob .* numFeats;
                else
                    IC_Weights_2D(t, :) = 1.0;
                end
            end
            
            disp('✅ 橫截面 Spearman IC 評估與能量守恆注意力權重生成完畢！');
        end
    end
end