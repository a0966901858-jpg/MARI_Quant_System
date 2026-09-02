% 定義 FeatureEngineer 類別，繼承自 handle (傳址參考)，確保物件傳遞時不會消耗額外記憶體
classdef FeatureEngineer < handle
    % =========================================================================
    % 模組：FeatureEngineer.m 
    % 升級：Phase 15.5 Stage 2 (★ GICS 產業內橫截面 Z-Score 中性化 + 退縮保護版)
    % 職責：計算無未來函數之相對/微觀/宏觀特徵、產業中性化特徵標準化與 DyGAT 時變共整合圖譜
    % =========================================================================
    
    properties
        ConfigObj       
        NumMicro        
        NumMacro        
        NumRel          
        NumTickers      
        TotalNodeFeats  
    end
    
    methods
        function obj = FeatureEngineer(config)
            obj.ConfigObj = config; 
            obj.NumMicro = config.NumMicroFeatures; % 預設 15
            
            if isprop(config, 'NumMacroFeatures') && config.NumMacroFeatures >= 10
                obj.NumMacro = config.NumMacroFeatures;
            else
                obj.NumMacro = 10; 
            end
            
            obj.NumTickers = config.NumTickers;     
            obj.NumRel = 3;                         
            
            obj.TotalNodeFeats = obj.NumRel + obj.NumMicro + obj.NumMacro; % 3 + 15 + 10 = 28
            
            fprintf(' ⚙️ [FeatureEngineer] 初始化。準備產出 3D 面板資料 [Days, %d, %d]\n', ...
                obj.TotalNodeFeats, obj.NumTickers);
        end
        
        function [X_norm_3D, Prices_Active, Expert_Active, Dates_Active, AdjMatrix_3D] = process(obj, dataStruct)
            disp(' 🚀 啟動機構級大宇宙特徵工程 (True PiT 橫截面無洩漏版)...');
            
            Dates_Active   = dataStruct.Dates; 
            Opens_Active   = dataStruct.Opens; 
            Prices_Active  = dataStruct.Prices;     
            Highs_Active   = dataStruct.Highs; 
            Lows_Active    = dataStruct.Lows; 
            Volumes_Active = dataStruct.Volumes; 
            IsConst_Active = dataStruct.IsConst;    
            numDays = length(Dates_Active);         
            
            ticker_list = obj.ConfigObj.IdxTickers; 
            
            disp(' -> 構建 True PiT 流動性濾網 (排除 IPO前/停牌/非成分股/雞蛋水餃股)...');
            Expert_Active = false(numDays, obj.NumTickers); 
            
            vol_20d = movmean(Volumes_Active, [19 0], 1, 'omitnan');
            valid_condition = (Prices_Active > 1.0) & (vol_20d > 50000) & IsConst_Active & (Volumes_Active > 0);
            Expert_Active(valid_condition) = true; 
            
            spy_idx = find(strcmp(ticker_list, 'SPY'));
            if isempty(spy_idx)
                error('❌ 找不到基準標的 SPY，無法計算相對大盤與宏觀特徵！'); 
            end
            
            % -------------------------------------------------------------
            % 載入 GICS 產業別映射表 (Stage 2: 產業中性化前置)
            % -------------------------------------------------------------
            disp(' -> 載入 GICS 產業別映射表以支援產業內橫截面中性化...');
            universePath = fullfile(obj.ConfigObj.ProjectDir, 'data', 'crawlers', 'us_universe.csv');
            if ~exist(universePath, 'file')
                potentialPaths = {
                    fullfile(fileparts(mfilename('fullpath')), 'crawlers', 'us_universe.csv'), ...
                    fullfile(fileparts(obj.ConfigObj.DataLakeDir), 'crawlers', 'us_universe.csv')
                };
                for p = 1:length(potentialPaths)
                    if exist(potentialPaths{p}, 'file')
                        universePath = potentialPaths{p};
                        break;
                    end
                end
            end
            
            sector_map = repmat({'Unknown'}, 1, obj.NumTickers);
            if exist(universePath, 'file')
                try
                    u_tbl = readtable(universePath, 'TextType', 'string');
                    if ismember('GICS_Sector', u_tbl.Properties.VariableNames)
                        [lia, loc] = ismember(string(ticker_list), string(u_tbl.Ticker));
                        valid_loc = loc(lia);
                        valid_idx = find(lia);
                        for k = 1:length(valid_idx)
                            s_val = char(u_tbl.GICS_Sector(valid_loc(k)));
                            if ~isempty(strtrim(s_val))
                                sector_map{valid_idx(k)} = strtrim(s_val);
                            end
                        end
                        fprintf('    🏷️ 成功識別 %d 檔標的之 GICS 板塊標籤。\n', sum(~strcmp(sector_map, 'Unknown')));
                    end
                catch ME
                    warning('⚠️ 讀取 GICS 產業表失敗: %s，將回退至全截面基準。', ME.message);
                end
            else
                warning('⚠️ 找不到 us_universe.csv，將回退至全截面標準化。');
            end
            
            sectors_cat = categorical(sector_map);
            unique_sectors = categories(sectors_cat);
            
            % 讀取並對齊 FRED 總經快取 (VIX, 殖利率, 利差, 失業率)
            disp(' -> 讀取並對齊 FRED 總經領先指標快取 (fred_macro.parquet)...');
            fredPath = fullfile(obj.ConfigObj.DataLakeDir, 'fred_macro.parquet');
            fred_aligned = NaN(numDays, 5, 'single'); % [VIXCLS, T10Y2Y, BAMLH0A0HYM2, DGS10, UNRATE]
            
            if exist(fredPath, 'file')
                try
                    fred_tbl = parquetread(fredPath);
                    f_dates = datetime(fred_tbl.Date);
                    f_dates.TimeZone = Dates_Active.TimeZone; 
                    
                    [Lia, Locb] = ismember(Dates_Active, f_dates);
                    
                    valid_loc = Locb(Lia);
                    valid_idx = find(Lia);
                    
                    if ismember('VIXCLS', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 1) = fred_tbl.VIXCLS(valid_loc);
                    end
                    if ismember('T10Y2Y', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 2) = fred_tbl.T10Y2Y(valid_loc);
                    end
                    if ismember('BAMLH0A0HYM2', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 3) = fred_tbl.BAMLH0A0HYM2(valid_loc);
                    end
                    if ismember('DGS10', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 4) = fred_tbl.DGS10(valid_loc);
                    end
                    if ismember('UNRATE', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 5) = fred_tbl.UNRATE(valid_loc);
                    end
                    
                    fred_aligned = fillmissing(fred_aligned, 'previous', 1);
                catch ME
                    warning('⚠️ 解析 FRED 數據失敗: %s', ME.message);
                end
            else
                warning('⚠️ 找不到 fred_macro.parquet，總經特徵將以 0 填充！請確認是否已執行 fred_crawler.py');
            end
            
            disp(' -> 計算 15 維微觀特徵 (含特質波動度、Amihud衝擊與52週新高)...');
            Micro_3D = obj.calc_micro_features(Opens_Active, Highs_Active, Lows_Active, Prices_Active, Volumes_Active, spy_idx, numDays);
            
            disp(' -> 計算 3 維相對大盤特徵 (Beta, Correlation, Relative Strength)...');
            Rel_3D   = obj.calc_relative_features(Prices_Active, spy_idx, numDays);
            
            fprintf(' -> 計算 %d 維宏觀總經特徵...\n', obj.NumMacro);
            Macro_2D = obj.calc_macro_features(Prices_Active, Expert_Active, spy_idx, fred_aligned, numDays);
            
            disp(' -> 組合特徵為 3D 面板資料...');
            X_raw_3D = NaN(numDays, obj.TotalNodeFeats, obj.NumTickers, 'single'); 
            Macro_3D = repmat(reshape(Macro_2D, [numDays, obj.NumMacro, 1]), [1, 1, obj.NumTickers]);
            
            idx_rel = 1:obj.NumRel;
            idx_micro = (obj.NumRel + 1):(obj.NumRel + obj.NumMicro);
            idx_macro = (obj.NumRel + obj.NumMicro + 1):obj.TotalNodeFeats;
            
            X_raw_3D(:, idx_rel, :)   = Rel_3D; 
            X_raw_3D(:, idx_micro, :) = Micro_3D; 
            X_raw_3D(:, idx_macro, :) = Macro_3D; 
            
            % -------------------------------------------------------------
            % 特徵標準化 (★ Stage 2: GICS 產業內 Z-Score + 退縮保護)
            % -------------------------------------------------------------
            disp(' -> 執行特徵標準化 (★ GICS 產業內橫截面 Z-Score + 巨集滾動標準化)...');
            X_norm_3D = zeros(size(X_raw_3D), 'single'); 
            
            % 1. 宏觀指標維持時序滾動 Z-Score
            macro_raw = X_raw_3D(:, idx_macro, 1);
            mu_macro = movmean(macro_raw, [251 0], 1, 'omitnan');
            std_macro = movstd(macro_raw, [251 0], 1, 'omitnan') + 1e-8;
            macro_norm = (macro_raw - mu_macro) ./ std_macro;
            macro_norm(isnan(macro_norm)) = 0;
            
            min_cs_samples = max(10, floor(obj.NumTickers * 0.05));
            min_sector_samples = 4; % 產業內至少需 4 檔活躍標的，否則退縮回退至全市場
            cs_feat_indices = [idx_rel, idx_micro];
            
            for t = 1:numDays
                act_mask = Expert_Active(t, :);
                n_act = sum(act_mask);
                
                if n_act >= min_cs_samples
                    % 計算當日全市場基準 (作為小樣本產業回退的 Fallback)
                    vals_all = X_raw_3D(t, cs_feat_indices, act_mask);
                    mu_all = mean(vals_all, 3, 'omitnan');
                    std_all = std(vals_all, 0, 3, 'omitnan') + 1e-8;
                    
                    % 預設以全市場基準填入
                    X_norm_3D(t, cs_feat_indices, act_mask) = (vals_all - mu_all) ./ std_all;
                    
                    % 逐板塊執行產業內中性化
                    for s = 1:length(unique_sectors)
                        s_name = unique_sectors{s};
                        % 排除非股票類別或未知標的
                        if strcmp(s_name, 'Unknown') || strcmp(s_name, 'Macro') || strcmp(s_name, 'Safe Haven')
                            continue;
                        end
                        
                        sec_mask = act_mask & (sectors_cat == s_name);
                        n_sec = sum(sec_mask);
                        
                        if n_sec >= min_sector_samples
                            % 樣本充足：計算產業內局部相對強度 Z-Score
                            vals_sec = X_raw_3D(t, cs_feat_indices, sec_mask);
                            mu_sec = mean(vals_sec, 3, 'omitnan');
                            std_sec = std(vals_sec, 0, 3, 'omitnan') + 1e-8;
                            X_norm_3D(t, cs_feat_indices, sec_mask) = (vals_sec - mu_sec) ./ std_sec;
                        end
                    end
                else
                    X_norm_3D(t, cs_feat_indices, act_mask) = 0;
                end
                
                X_norm_3D(t, idx_macro, :) = repmat(macro_norm(t, :), [1, 1, obj.NumTickers]);
            end
            
            X_norm_3D(isnan(X_norm_3D) | isinf(X_norm_3D)) = 0;
            inactive_mask = repmat(reshape(~Expert_Active, [numDays, 1, obj.NumTickers]), [1, obj.TotalNodeFeats, 1]);
            X_norm_3D(inactive_mask) = 0; 
            
            disp(' -> 構建 DyGAT 時變圖譜矩陣 (BH-FDR 多重比較校正防偽陽性邊)...');
            
            n_tickers = obj.NumTickers;
            daily_rets = NaN(numDays, n_tickers, 'single');
            daily_rets(2:end, :) = (Prices_Active(2:end,:) - Prices_Active(1:end-1,:)) ./ Prices_Active(1:end-1,:);
            log_prices = log(Prices_Active);
            
            lookback = 60;          
            coint_lookback = 252;   
            
            calc_days = (lookback + 1) : 5 : numDays;
            if calc_days(1) ~= (lookback + 1)
                calc_days = [lookback + 1, calc_days];
            end
            num_calc = length(calc_days);
            Anchor_Adj = cell(num_calc, 1);
            
            fprintf(' -> 預計計算 %d 個圖譜錨點，任務已發配至運算池...\n', num_calc);
            
            dq = parallel.pool.DataQueue;
            hWait = waitbar(0, '啟動多核運算池...', 'Name', 'DyGAT 空間圖譜運算進度');
            hWait.UserData = 0; 
            afterEach(dq, @(~) obj.update_progress(hWait, num_calc));
            
            parfor i = 1:num_calc
                warn_state_1 = warning('off', 'econ:egcitest:LeftYVarColinear');
                warn_state_2 = warning('off', 'stats:corr:Ties');
                
                t = calc_days(i);
                
                window_rets = daily_rets(t-lookback : t-1, :); 
                
                if t > coint_lookback
                    window_logP = log_prices(t-coint_lookback : t-1, :);
                else
                    window_logP = log_prices(t-lookback : t-1, :);
                end
                
                current_active_mask = Expert_Active(t, :);
                valid_nodes = all(~isnan(window_rets), 1) & all(~isnan(window_logP), 1) & current_active_mask;
                valid_idx = find(valid_nodes);
                num_valid = length(valid_idx);
                
                bin_adj = false(n_tickers, n_tickers);
                bin_adj(1:n_tickers+1:end) = true; 
                
                if num_valid > 2
                    clean_corr = corr(double(window_rets(:, valid_nodes)), 'Type', 'Spearman');
                    clean_corr(isnan(clean_corr)) = 0;
                    
                    [row_idx, col_idx] = find(triu(clean_corr > 0.5, 1));
                    num_pairs = length(row_idx);
                    
                    max_eg_tests = 2000; 
                    if num_pairs > max_eg_tests
                        linear_idx = sub2ind(size(clean_corr), row_idx, col_idx);
                        pair_correlations = clean_corr(linear_idx);
                        
                        [~, sort_idx] = sort(pair_correlations, 'descend');
                        top_k_idx = sort_idx(1:max_eg_tests);
                        
                        row_idx = row_idx(top_k_idx);
                        col_idx = col_idx(top_k_idx);
                        num_pairs = max_eg_tests;
                    end
                    
                    % ★ Phase 15 (P1-2)：先收集所有候選配對 p-value
                    p_values = NaN(num_pairs, 1);
                    for k = 1:num_pairs
                        idx_A = valid_idx(row_idx(k));
                        idx_B = valid_idx(col_idx(k));
                        
                        pA = window_logP(:, idx_A);
                        pB = window_logP(:, idx_B);
                        
                        [~, pVal] = egcitest(double([pA, pB]), 'Alpha', 0.05);
                        p_values(k) = pVal;
                    end
                    
                    % ★ Phase 15 (P1-2)：執行 Benjamini-Hochberg FDR 多重比較校正 (q = 0.05)
                    fdr_q = 0.05;
                    [sorted_p, ~] = sort(p_values);
                    m = length(sorted_p);
                    bh_critical = ((1:m)' / m) * fdr_q;
                    below = sorted_p <= bh_critical;
                    if any(below)
                        p_threshold = sorted_p(find(below, 1, 'last'));
                    else
                        p_threshold = 0; % 無配對通過校正
                    end
                    significant_mask = p_values <= p_threshold;
                    
                    for k = 1:num_pairs
                        if significant_mask(k)
                            idx_A = valid_idx(row_idx(k));
                            idx_B = valid_idx(col_idx(k));
                            bin_adj(idx_A, idx_B) = true;
                            bin_adj(idx_B, idx_A) = true; 
                        end
                    end
                end
                Anchor_Adj{i} = bin_adj;
                
                warning(warn_state_1);
                warning(warn_state_2);
                
                send(dq, 1);
            end
            
            if isgraphics(hWait), close(hWait); end
            
            disp(' -> 錨點運算全數完成，正在進行時間平移與 3D 矩陣對齊...');
            AdjMatrix_3D = false(n_tickers, n_tickers, numDays);
            
            for t = lookback + 1 : numDays
                idx = find(calc_days <= t, 1, 'last');
                if ~isempty(idx)
                    AdjMatrix_3D(:, :, t) = Anchor_Adj{idx};
                end
            end
            
            disp('✅ 3D 特徵面板 (產業中性化) 與 BH-FDR 降噪圖譜提取完畢！');
        end  
        
        %% --- 內部特徵計算函數 ---
        
        function Micro = calc_micro_features(obj, ~, H, L, P, V, spy_idx, numDays)
            obj.NumTickers = size(P, 2);
            Micro = NaN(numDays, 15, obj.NumTickers, 'single'); 
            
            % 1. 基本動能與波動度
            R1 = NaN(numDays, obj.NumTickers, 'single');
            R1(2:end,:) = (P(2:end,:) - P(1:end-1,:)) ./ P(1:end-1,:);
            R5 = NaN(numDays, obj.NumTickers, 'single');
            R5(6:end,:) = (P(6:end,:) - P(1:end-5,:)) ./ P(1:end-5,:);
            R20 = NaN(numDays, obj.NumTickers, 'single');
            R20(21:end,:) = (P(21:end,:) - P(1:end-20,:)) ./ P(1:end-20,:);
            
            Vol20 = movstd(R1, [19 0], 1, 'omitnan');
            
            % 特質波動度 (Idiosyncratic Volatility 20D)
            spy_R1 = R1(:, spy_idx);
            spy_Var20 = movvar(spy_R1, [19 0], 1, 'omitnan') + 1e-8;
            IdioVol20 = NaN(numDays, obj.NumTickers, 'single');
            for i = 1:obj.NumTickers
                cov_i = movmean(R1(:,i) .* spy_R1, [19 0], 1, 'omitnan') - ...
                        (movmean(R1(:,i), [19 0], 1, 'omitnan') .* movmean(spy_R1, [19 0], 1, 'omitnan'));
                beta_i = cov_i ./ spy_Var20;
                res_i = R1(:,i) - beta_i .* spy_R1;
                IdioVol20(:, i) = movstd(res_i, [19 0], 1, 'omitnan');
            end
            
            % 2. 成交量與流動性
            V5 = movmean(V, [4 0], 1, 'omitnan');
            V20 = movmean(V, [19 0], 1, 'omitnan');
            VolRatio = V5 ./ (V20 + 1e-8);
            
            % Amihud 20D 流動性衝擊係數 (|R1| / DollarVolume)
            dollar_vol = P .* V + 1e-8;
            amihud_daily = abs(R1) ./ dollar_vol;
            Amihud_20 = movmean(amihud_daily, [19 0], 1, 'omitnan') * 1e6;
            
            % 3. 趨勢與震盪
            SMA20 = movmean(P, [19 0], 1, 'omitnan') ./ P;
            SMA60 = movmean(P, [59 0], 1, 'omitnan') ./ P;
            
            % MACD Histogram (消除 Line 與 Signal 之高度共線)
            EMA12 = obj.calc_ema(P, 12);
            EMA26 = obj.calc_ema(P, 26);
            MACD_Line = (EMA12 - EMA26) ./ P;
            MACD_Sig = obj.calc_ema(MACD_Line, 9);
            MACD_Hist = MACD_Line - MACD_Sig;
            
            % RSI
            diff_P = NaN(numDays, obj.NumTickers, 'single');
            diff_P(2:end,:) = diff(P);
            U = max(diff_P, 0);
            D = max(-diff_P, 0);
            EMA_U = obj.calc_ema(U, 27);
            EMA_D = obj.calc_ema(D, 27);
            RS = EMA_U ./ (EMA_D + 1e-8); 
            RSI = 100 - (100 ./ (1 + RS)); 
            
            % OBV_20
            SignR = sign(R1);
            OBV_diff = SignR .* V;
            OBV_20 = movsum(OBV_diff, [19 0], 1, 'omitnan') ./ (V20 * 20 + 1e-8);
            
            % 4. 價格區間與新高/新低距離
            H20 = movmax(H, [19 0], 1, 'omitnan');
            L20 = movmin(L, [19 0], 1, 'omitnan');
            HL_Spread = (H20 - L20) ./ P;
            Dist_H20 = (P - H20) ./ H20;
            
            % 52 週 (252日) 新高距離 (Dist_H252)
            H252 = movmax(H, [251 0], 1, 'omitnan');
            Dist_H252 = (P - H252) ./ (H252 + 1e-8);
            
            % 組合 15 維微觀特徵矩陣
            Micro(:, 1, :)  = R1;          Micro(:, 2, :)  = R5;         Micro(:, 3, :)  = R20;
            Micro(:, 4, :)  = Vol20;       Micro(:, 5, :)  = IdioVol20;  Micro(:, 6, :)  = VolRatio;
            Micro(:, 7, :)  = Amihud_20;   Micro(:, 8, :)  = SMA20;      Micro(:, 9, :)  = SMA60;
            Micro(:, 10, :) = MACD_Hist;   Micro(:, 11, :) = RSI;        Micro(:, 12, :) = OBV_20;
            Micro(:, 13, :) = HL_Spread;   Micro(:, 14, :) = Dist_H20;   Micro(:, 15, :) = Dist_H252;
        end
        
        function Rel = calc_relative_features(~, P, spy_idx, numDays)
            n_tickers = size(P, 2);
            Rel = NaN(numDays, 3, n_tickers, 'single'); 
            
            R1 = NaN(numDays, n_tickers, 'single');
            R1(2:end,:) = (P(2:end,:) - P(1:end-1,:)) ./ P(1:end-1,:); 
            spy_R1 = R1(:, spy_idx); 
            spy_Var20 = movvar(spy_R1, [19 0], 1, 'omitnan') + 1e-8;
            
            parfor i = 1:n_tickers
                cov_val = movmean(R1(:,i) .* spy_R1, [19 0], 1, 'omitnan') - ...
                          (movmean(R1(:,i), [19 0], 1, 'omitnan') .* movmean(spy_R1, [19 0], 1, 'omitnan'));
                          
                beta_i = cov_val ./ spy_Var20; 
                std_i = movstd(R1(:,i), [19 0], 1, 'omitnan') + 1e-8;
                corr_i = beta_i .* (sqrt(spy_Var20) ./ std_i); 
                
                RS_i = (P(:, i) ./ movmean(P(:, i), [19 0], 1, 'omitnan')) - ...
                       (P(:, spy_idx) ./ movmean(P(:, spy_idx), [19 0], 1, 'omitnan'));
                
                temp_rel = NaN(numDays, 3, 'single');
                temp_rel(:, 1) = beta_i;
                temp_rel(:, 2) = corr_i;
                temp_rel(:, 3) = RS_i;
                Rel(:, :, i) = temp_rel; 
            end
        end
        
        function Macro = calc_macro_features(obj, P, Expert, spy_idx, fred_aligned, numDays)
            Macro = NaN(numDays, obj.NumMacro, 'single');
            
            R1_spy = NaN(numDays, 1, 'single');
            R1_spy(2:end) = (P(2:end, spy_idx) - P(1:end-1, spy_idx)) ./ P(1:end-1, spy_idx);
            
            vix_proxy = movstd(R1_spy, [19 0], 1, 'omitnan') * sqrt(252);
            
            spy_r20 = NaN(numDays, 1, 'single');
            spy_r20(21:end) = (P(21:end, spy_idx) - P(1:end-20, spy_idx)) ./ P(1:end-20, spy_idx);
            
            spy_r60 = NaN(numDays, 1, 'single');
            spy_r60(61:end) = (P(61:end, spy_idx) - P(1:end-60, spy_idx)) ./ P(1:end-60, spy_idx);
            
            MA20_all = movmean(P, [19 0], 1, 'omitnan');
            is_above = (P > MA20_all) & Expert; 
            active_counts = sum(Expert, 2);
            breadth = sum(is_above, 2) ./ (active_counts + 1e-8);
            
            real_vix = fred_aligned(:, 1);
            invalid_vix = isnan(real_vix) | isinf(real_vix) | (real_vix <= 0);
            real_vix(invalid_vix) = vix_proxy(invalid_vix) * 100;
            
            vrp = real_vix - (vix_proxy * 100);
            
            Macro(:, 1) = vix_proxy;
            Macro(:, 2) = spy_r20;
            Macro(:, 3) = spy_r60;
            Macro(:, 4) = breadth;
            Macro(:, 5) = real_vix;           % 真實 VIX (CBOE)
            Macro(:, 6) = vrp;                % VRP 波動率風險溢酬
            Macro(:, 7) = fred_aligned(:, 2); % T10Y2Y (殖利率曲線倒掛)
            Macro(:, 8) = fred_aligned(:, 3); % BAMLH0A0HYM2 (高收益債信用利差)
            Macro(:, 9) = fred_aligned(:, 4); % DGS10 (10年期公債殖利率)
            Macro(:, 10)= fred_aligned(:, 5); % UNRATE (官方失業率)
            
            Macro = fillmissing(Macro, 'previous', 1);
            Macro(isnan(Macro)) = 0;
        end
        
        function ema_data = calc_ema(~, data, window)
            alpha = 2 / (window + 1);
            ema_data = NaN(size(data), 'single');
            
            parfor i = 1:size(data, 2)
                col = data(:, i);
                first_valid = find(~isnan(col), 1, 'first');
                if isempty(first_valid)
                    continue;
                end
                
                ema_col = NaN(size(col), 'single');
                ema_col(first_valid) = col(first_valid);
                
                for t = first_valid + 1 : length(col)
                    if isnan(col(t))
                        ema_col(t) = ema_col(t-1);
                    else
                        ema_col(t) = alpha * col(t) + (1 - alpha) * ema_col(t-1); 
                    end
                end
                ema_data(:, i) = ema_col; 
            end
        end
    end
    
    methods (Access = private)
        function update_progress(~, hWait, total)
            if isgraphics(hWait)
                count = hWait.UserData + 1;
                hWait.UserData = count;
                
                waitbar(count/total, hWait, sprintf('多核平行運算中: 已完成 %d / %d', count, total));
                
                if mod(count, 50) == 0 || count == total
                    fprintf('  [系統總控] 成功回收訊號：已確實完成 %d / %d 個時變圖譜...\n', count, total);
                end
            end
        end
    end
end