% 定義 FeatureEngineer 類別，繼承自 handle (傳址參考)，確保物件傳遞時不會消耗額外記憶體
classdef FeatureEngineer < handle
    % =========================================================================
    % 模組：FeatureEngineer.m 
    % 升級：Phase 15.5 生產基準版 (★ 支援空間圖譜開關剪枝、純向量化極速微觀計算、
    %       FRED 總經缺失防呆與品質稽核、GICS 產業遮罩預快取中性化、Headless 伺服器防禦)
    % 職責：計算無未來函數之相對/微觀/宏觀特徵、產業中性化特徵標準化與時變共整合圖譜
    % =========================================================================
    
    properties
        ConfigObj               % 全域配置物件 (Config 實例)
        NumMicro                % 微觀特徵維度 (預設 15)
        NumMacro                % 宏觀特徵維度 (預設 10)
        NumRel                  % 相對大盤特徵維度 (預設 3)
        NumTickers              % 橫截面股票標的數量
        TotalNodeFeats          % 總節點特徵數 (3 + 15 + 10 = 28)
        EnableGraphConstruction % 空間協整圖譜建構開關 (布林值)
        RandStream              % 獨立子串流隨機數引擎
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
            obj.TotalNodeFeats = obj.NumRel + obj.NumMicro + obj.NumMacro; % 28 維
            
            % ★ 空間專家剪枝連動：若全域關閉空間專家，則自動關閉繁重的圖譜協整運算
            if isprop(config, 'EnableSpaceExpertTraining') && ~config.EnableSpaceExpertTraining
                obj.EnableGraphConstruction = false;
            elseif isprop(config, 'EnableGraphConstruction')
                obj.EnableGraphConstruction = config.EnableGraphConstruction;
            else
                obj.EnableGraphConstruction = true;
            end
            
            fprintf(' ⚙️ [FeatureEngineer] 初始化。準備產出 3D 面板資料 [Days, %d, %d] (圖譜協整開關: %d)\n', ...
                obj.TotalNodeFeats, obj.NumTickers, obj.EnableGraphConstruction);
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
            numDays        = length(Dates_Active);         
            ticker_list    = obj.ConfigObj.IdxTickers; 
            
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
            % 1. 載入 GICS 產業別映射表
            % -------------------------------------------------------------
            disp(' -> 載入 GICS 產業別映射表以支援產業內橫截面中性化...');
            currentClassDir = fileparts(mfilename('fullpath'));
            
            potentialPaths = { ...
                fullfile(currentClassDir, 'crawlers', 'us_universe.csv'), ...
                fullfile(fileparts(currentClassDir), 'data', 'crawlers', 'us_universe.csv'), ...
                fullfile(fileparts(obj.ConfigObj.DataLakeDir), 'crawlers', 'us_universe.csv'), ...
                fullfile(obj.ConfigObj.DataLakeDir, '..', 'crawlers', 'us_universe.csv'), ...
                fullfile(pwd, 'data', 'crawlers', 'us_universe.csv') ...
            };
            
            if isprop(obj.ConfigObj, 'ProjectDir') && ~isempty(obj.ConfigObj.ProjectDir)
                potentialPaths = [fullfile(obj.ConfigObj.ProjectDir, 'data', 'crawlers', 'us_universe.csv'), potentialPaths];
            end
            
            universePath = '';
            for p = 1:length(potentialPaths)
                if exist(potentialPaths{p}, 'file')
                    universePath = potentialPaths{p};
                    break;
                end
            end
            
            sector_map = repmat({'Unknown'}, 1, obj.NumTickers);
            if ~isempty(universePath) && exist(universePath, 'file')
                try
                    u_tbl = readtable(universePath, 'TextType', 'string');
                    if ismember('GICS_Sector', u_tbl.Properties.VariableNames)
                        cfg_tickers = replace(string(ticker_list), '.', '-');
                        csv_tickers = replace(string(u_tbl.Ticker), '.', '-');
                        [lia, loc] = ismember(cfg_tickers, csv_tickers);
                        
                        valid_loc = loc(lia);
                        valid_idx = find(lia);
                        for k = 1:length(valid_idx)
                            sec_str = u_tbl.GICS_Sector(valid_loc(k));
                            if ~ismissing(sec_str) && strlength(strtrim(sec_str)) > 0
                                sector_map{valid_idx(k)} = char(strtrim(sec_str));
                            end
                        end
                        fprintf('    🏷️ 成功識別 %d 檔標的之 GICS 板塊標籤 (來源: %s)。\n', ...
                            sum(~strcmp(sector_map, 'Unknown')), universePath);
                    else
                        warning('⚠️ us_universe.csv 中未找到 GICS_Sector 欄位。');
                    end
                catch ME
                    warning('⚠️ 讀取 GICS 產業表失敗: %s，將回退至全截面基準。', ME.message);
                end
            else
                warning('⚠️ 未能定位 us_universe.csv，將回退至全截面標準化。');
            end
            
            sectors_cat = categorical(sector_map);
            unique_sectors = categories(sectors_cat);
            
            % 預先快取板塊遮罩索引，避免在逐日迴圈中重複計算字串比較
            valid_sector_masks = cell(length(unique_sectors), 1);
            valid_sector_names = cell(length(unique_sectors), 1);
            sec_count = 0;
            for s = 1:length(unique_sectors)
                s_name = unique_sectors{s};
                if ~ismember(s_name, {'Unknown', 'Macro', 'Safe Haven', 'Broad Market Index'})
                    sec_count = sec_count + 1;
                    valid_sector_masks{sec_count} = (sectors_cat == s_name);
                    valid_sector_names{sec_count} = s_name;
                end
            end
            valid_sector_masks = valid_sector_masks(1:sec_count);
            valid_sector_names = valid_sector_names(1:sec_count);
            
            % ★ 方案 B：調用 valid_sector_names 輸出日誌，消除未使用變數警告並提高除錯可見性
            fprintf('    🏢 已建立 %d 個有效 GICS 產業中性化遮罩: %s\n', ...
                sec_count, strjoin(valid_sector_names, ', '));
            
            % -------------------------------------------------------------
            % 2. 讀取並對齊 FRED 總經快取 (Parquet / CSV 雙相容 + 品質稽核)
            % -------------------------------------------------------------
            disp(' -> 讀取並對齊 FRED 總經領先指標快取...');
            fredParquet = fullfile(obj.ConfigObj.DataLakeDir, 'fred_macro.parquet');
            fredCsv     = fullfile(obj.ConfigObj.DataLakeDir, 'fred_macro.csv');
            fred_aligned = NaN(numDays, 5, 'single'); % [VIXCLS, T10Y2Y, BAMLH0A0HYM2, DGS10, UNRATE]
            
            fred_loaded = false;
            fred_tbl = table();
            
            if exist(fredParquet, 'file')
                try
                    fred_tbl = parquetread(fredParquet);
                    fred_loaded = true;
                catch
                    fred_loaded = false;
                end
            end
            
            if ~fred_loaded && exist(fredCsv, 'file')
                try
                    fred_tbl = readtable(fredCsv);
                    fred_loaded = true;
                catch ME
                    warning('⚠️ 讀取 fred_macro.csv 失敗: %s', ME.message);
                end
            end
            
            if fred_loaded
                try
                    f_dates = datetime(fred_tbl.Date);
                    if ~isempty(Dates_Active.TimeZone)
                        f_dates.TimeZone = Dates_Active.TimeZone;
                    else
                        f_dates.TimeZone = '';
                    end
                    
                    [Lia, Locb] = ismember(Dates_Active, f_dates);
                    valid_loc = Locb(Lia);
                    valid_idx = find(Lia);
                    
                    if ismember('VIXCLS', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 1) = single(fred_tbl.VIXCLS(valid_loc));
                    end
                    if ismember('T10Y2Y', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 2) = single(fred_tbl.T10Y2Y(valid_loc));
                    end
                    
                    % 信用利差支援原生 BAMLH0A0HYM2 或備用代理指標 BAA10Y
                    if ismember('BAMLH0A0HYM2', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 3) = single(fred_tbl.BAMLH0A0HYM2(valid_loc));
                    elseif ismember('BAA10Y', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 3) = single(fred_tbl.BAA10Y(valid_loc));
                    end
                    
                    if ismember('DGS10', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 4) = single(fred_tbl.DGS10(valid_loc));
                    end
                    if ismember('UNRATE', fred_tbl.Properties.VariableNames)
                        fred_aligned(valid_idx, 5) = single(fred_tbl.UNRATE(valid_loc));
                    end
                    
                    % 雙向對齊：向下填充防前視，開頭缺失處以首筆真實值平移回填，防止常態 0 污染
                    fred_aligned = fillmissing(fred_aligned, 'previous', 1);
                    fred_aligned = fillmissing(fred_aligned, 'nearest', 1);
                    
                    % 資料品質即時審計報告
                    hy_zeros = sum(fred_aligned(:, 3) == 0 | isnan(fred_aligned(:, 3)));
                    if hy_zeros > (0.10 * numDays)
                        warning('⚠️ [FRED 數據審計] BAMLH0A0HYM2 存在 %d 天 (%.1f%%) 零值或空值！請確認 fred_crawler.py 是否已回填。', ...
                            hy_zeros, (hy_zeros / numDays) * 100);
                    else
                        disp('    ✅ FRED 總經指標對齊完畢，信用利差 (HY Spread) 具備完整歷史覆蓋。');
                    end
                catch ME
                    warning('⚠️ 解析 FRED 數據失敗: %s，總經指標將啟用統計代理值。', ME.message);
                end
            else
                warning('⚠️ 找不到 fred_macro 快取檔，將以價格序列與局部統計進行替代。');
            end
            
            % -------------------------------------------------------------
            % 3. 純矩陣向量化計算特徵 (消滅迴圈開銷)
            % -------------------------------------------------------------
            disp(' -> 計算 15 維微觀結構特徵 (純向量化極速模式)...');
            Micro_3D = obj.calc_micro_features(Opens_Active, Highs_Active, Lows_Active, Prices_Active, Volumes_Active, spy_idx, numDays);
            
            disp(' -> 計算 3 維相對大盤特徵 (純向量化極速模式)...');
            Rel_3D   = obj.calc_relative_features(Prices_Active, spy_idx, numDays);
            
            fprintf(' -> 計算 %d 維宏觀總經特徵...\n', obj.NumMacro);
            Macro_2D = obj.calc_macro_features(Prices_Active, Expert_Active, spy_idx, fred_aligned, numDays);
            
            disp(' -> 組合特徵為 3D 面板資料...');
            X_raw_3D = NaN(numDays, obj.TotalNodeFeats, obj.NumTickers, 'single'); 
            Macro_3D = repmat(reshape(Macro_2D, [numDays, obj.NumMacro, 1]), [1, 1, obj.NumTickers]);
            
            idx_rel   = 1:obj.NumRel;
            idx_micro = (obj.NumRel + 1):(obj.NumRel + obj.NumMicro);
            idx_macro = (obj.NumRel + obj.NumMicro + 1):obj.TotalNodeFeats;
            
            X_raw_3D(:, idx_rel, :)   = Rel_3D; 
            X_raw_3D(:, idx_micro, :) = Micro_3D; 
            X_raw_3D(:, idx_macro, :) = Macro_3D; 
            
            % -------------------------------------------------------------
            % 4. 特徵標準化 (GICS 產業內 Z-Score + 宏觀時序滾動 Z-Score)
            % -------------------------------------------------------------
            disp(' -> 執行特徵標準化 (GICS 產業內橫截面 Z-Score + 退縮保護)...');
            X_norm_3D = zeros(size(X_raw_3D), 'single'); 
            
            macro_raw = X_raw_3D(:, idx_macro, 1);
            mu_macro = movmean(macro_raw, [251 0], 1, 'omitnan');
            std_macro = movstd(macro_raw, [251 0], 1, 'omitnan') + 1e-8;
            macro_norm = (macro_raw - mu_macro) ./ std_macro;
            macro_norm(isnan(macro_norm)) = 0;
            
            min_cs_samples = max(10, floor(obj.NumTickers * 0.05));
            min_sector_samples = 4; 
            cs_feat_indices = [idx_rel, idx_micro];
            
            for t = 1:numDays
                act_mask = Expert_Active(t, :);
                n_act = sum(act_mask);
                
                if n_act >= min_cs_samples
                    % 預設全市場基準填入
                    vals_all = X_raw_3D(t, cs_feat_indices, act_mask);
                    mu_all = mean(vals_all, 3, 'omitnan');
                    std_all = std(vals_all, 0, 3, 'omitnan') + 1e-8;
                    X_norm_3D(t, cs_feat_indices, act_mask) = (vals_all - mu_all) ./ std_all;
                    
                    % 產業板塊中性化 (使用預快取之邏輯遮罩加速)
                    for s = 1:sec_count
                        sec_mask = act_mask & valid_sector_masks{s};
                        n_sec = sum(sec_mask);
                        
                        if n_sec >= min_sector_samples
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
            
            % -------------------------------------------------------------
            % 5. 構建圖譜拓撲 (支援開關式跳過，徹底解除 Phase 1 算力負擔)
            % -------------------------------------------------------------
            n_tickers = obj.NumTickers;
            AdjMatrix_3D = false(n_tickers, n_tickers, numDays);
            
            if ~obj.EnableGraphConstruction
                fprintf(' ⏩ [FeatureEngineer] 依實驗結論，已剪枝空間圖譜 (DyGAT) 協整計算，自動配置對角自環圖譜矩陣。\n');
                eye_mask = repmat(eye(n_tickers, 'logical'), [1, 1, numDays]);
                AdjMatrix_3D = eye_mask;
                disp('✅ 3D 特徵面板 (GICS 產業中性化) 提取完畢 (圖譜運算已旁路)！');
                return;
            end
            
            disp(' -> 構建 DyGAT 時變圖譜矩陣 (BH-FDR 多重比較校正防偽陽性邊)...');
            daily_rets = NaN(numDays, n_tickers, 'single');
            daily_rets(2:end, :) = (Prices_Active(2:end,:) - Prices_Active(1:end-1,:)) ./ (Prices_Active(1:end-1,:) + 1e-8);
            log_prices = log(max(Prices_Active, 1e-4));
            
            lookback = 60;          
            coint_lookback = 252;   
            calc_days = (lookback + 1) : 5 : numDays;
            if isempty(calc_days) || calc_days(1) ~= (lookback + 1)
                calc_days = [lookback + 1, calc_days];
            end
            num_calc = length(calc_days);
            Anchor_Adj = cell(num_calc, 1);
            
            fprintf(' -> 預計計算 %d 個圖譜錨點，任務已發配至運算池...\n', num_calc);
            
            % Headless 環境防呆判定
            show_gui = usejava('desktop') && usejava('awt');
            hWait = [];
            if show_gui
                hWait = waitbar(0, '啟動多核運算池...', 'Name', 'DyGAT 空間圖譜運算進度');
                hWait.UserData = 0;
            end
            
            dq = parallel.pool.DataQueue;
            afterEach(dq, @(~) obj.update_progress(hWait, num_calc, show_gui));
            
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
                bin_adj(1:n_tickers+1:end) = true; % 自環保底
                
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
                    
                    p_values = NaN(num_pairs, 1);
                    for k = 1:num_pairs
                        idx_A = valid_idx(row_idx(k));
                        idx_B = valid_idx(col_idx(k));
                        pA = window_logP(:, idx_A);
                        pB = window_logP(:, idx_B);
                        [~, pVal] = egcitest(double([pA, pB]), 'Alpha', 0.05);
                        p_values(k) = pVal;
                    end
                    
                    % Benjamini-Hochberg FDR 校正
                    fdr_q = 0.05;
                    [sorted_p, ~] = sort(p_values);
                    m = length(sorted_p);
                    bh_critical = ((1:m)' / m) * fdr_q;
                    below = sorted_p <= bh_critical;
                    if any(below)
                        p_threshold = sorted_p(find(below, 1, 'last'));
                    else
                        p_threshold = 0;
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
            
            if show_gui && isgraphics(hWait), close(hWait); end
            
            disp(' -> 錨點運算完成，正在進行時間軸前向對齊...');
            for t = (lookback + 1) : numDays
                idx = find(calc_days <= t, 1, 'last');
                if ~isempty(idx)
                    AdjMatrix_3D(:, :, t) = Anchor_Adj{idx};
                else
                    AdjMatrix_3D(1:n_tickers+1:end, t) = true;
                end
            end
            
            disp('✅ 3D 特徵面板 (GICS 產業中性化) 與 BH-FDR 降噪圖譜提取完畢！');
        end  
        
        %% ====================================================================
        % 內部特徵計算函數 (全面矩陣向量化)
        % ====================================================================
        
        function Micro = calc_micro_features(obj, ~, H, L, P, V, spy_idx, numDays)
            obj.NumTickers = size(P, 2);
            Micro = NaN(numDays, 15, obj.NumTickers, 'single'); 
            
            % 1. 基本動能與波動度
            R1 = NaN(numDays, obj.NumTickers, 'single');
            R1(2:end,:) = (P(2:end,:) - P(1:end-1,:)) ./ (P(1:end-1,:) + 1e-8);
            
            R5 = NaN(numDays, obj.NumTickers, 'single');
            R5(6:end,:) = (P(6:end,:) - P(1:end-5,:)) ./ (P(1:end-5,:) + 1e-8);
            
            R20 = NaN(numDays, obj.NumTickers, 'single');
            R20(21:end,:) = (P(21:end,:) - P(1:end-20,:)) ./ (P(1:end-20,:) + 1e-8);
            
            Vol20 = movstd(R1, [19 0], 1, 'omitnan');
            
            % 特質波動度 (Idiosyncratic Volatility 20D - 全矩陣向量化)
            spy_R1 = R1(:, spy_idx);
            spy_Var20 = movvar(spy_R1, [19 0], 1, 'omitnan') + 1e-8;
            cov_all = movmean(R1 .* spy_R1, [19 0], 1, 'omitnan') - ...
                     (movmean(R1, [19 0], 1, 'omitnan') .* movmean(spy_R1, [19 0], 1, 'omitnan'));
            beta_all = cov_all ./ spy_Var20;
            res_all = R1 - (beta_all .* spy_R1);
            IdioVol20 = movstd(res_all, [19 0], 1, 'omitnan');
            
            % 2. 成交量與流動性
            V5 = movmean(V, [4 0], 1, 'omitnan');
            V20 = movmean(V, [19 0], 1, 'omitnan');
            VolRatio = V5 ./ (V20 + 1e-8);
            
            dollar_vol = P .* V + 1e-8;
            amihud_daily = abs(R1) ./ dollar_vol;
            Amihud_20 = movmean(amihud_daily, [19 0], 1, 'omitnan') * 1e6;
            
            % 3. 趨勢與震盪
            SMA20 = movmean(P, [19 0], 1, 'omitnan') ./ (P + 1e-8);
            SMA60 = movmean(P, [59 0], 1, 'omitnan') ./ (P + 1e-8);
            
            EMA12 = obj.calc_ema(P, 12);
            EMA26 = obj.calc_ema(P, 26);
            MACD_Line = (EMA12 - EMA26) ./ (P + 1e-8);
            MACD_Sig  = obj.calc_ema(MACD_Line, 9);
            MACD_Hist = MACD_Line - MACD_Sig;
            
            diff_P = NaN(numDays, obj.NumTickers, 'single');
            diff_P(2:end,:) = diff(P);
            U = max(diff_P, 0);
            D = max(-diff_P, 0);
            EMA_U = obj.calc_ema(U, 27);
            EMA_D = obj.calc_ema(D, 27);
            RS = EMA_U ./ (EMA_D + 1e-8); 
            RSI = 100 - (100 ./ (1 + RS)); 
            
            SignR = sign(R1);
            OBV_diff = SignR .* V;
            OBV_20 = movsum(OBV_diff, [19 0], 1, 'omitnan') ./ (V20 * 20 + 1e-8);
            
            % 4. 價格通道與極值距離
            H20 = movmax(H, [19 0], 1, 'omitnan');
            L20 = movmin(L, [19 0], 1, 'omitnan');
            HL_Spread = (H20 - L20) ./ (P + 1e-8);
            Dist_H20 = (P - H20) ./ (H20 + 1e-8);
            
            H252 = movmax(H, [251 0], 1, 'omitnan');
            Dist_H252 = (P - H252) ./ (H252 + 1e-8);
            
            % 填入微觀特徵矩陣
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
            R1(2:end,:) = (P(2:end,:) - P(1:end-1,:)) ./ (P(1:end-1,:) + 1e-8); 
            spy_R1 = R1(:, spy_idx); 
            spy_Var20 = movvar(spy_R1, [19 0], 1, 'omitnan') + 1e-8;
            
            % 全矩陣向量化計算 Beta 與 Correlation
            cov_all = movmean(R1 .* spy_R1, [19 0], 1, 'omitnan') - ...
                     (movmean(R1, [19 0], 1, 'omitnan') .* movmean(spy_R1, [19 0], 1, 'omitnan'));
            beta_all = cov_all ./ spy_Var20;
            std_all  = movstd(R1, [19 0], 1, 'omitnan') + 1e-8;
            corr_all = beta_all .* (sqrt(spy_Var20) ./ std_all);
            
            P_spy = P(:, spy_idx);
            RS_all = (P ./ movmean(P, [19 0], 1, 'omitnan')) - ...
                     (P_spy ./ movmean(P_spy, [19 0], 1, 'omitnan'));
            
            Rel(:, 1, :) = beta_all;
            Rel(:, 2, :) = corr_all;
            Rel(:, 3, :) = RS_all;
        end
        
        function Macro = calc_macro_features(obj, P, Expert, spy_idx, fred_aligned, numDays)
            Macro = NaN(numDays, obj.NumMacro, 'single');
            
            R1_spy = NaN(numDays, 1, 'single');
            R1_spy(2:end) = (P(2:end, spy_idx) - P(1:end-1, spy_idx)) ./ (P(1:end-1, spy_idx) + 1e-8);
            vix_proxy = movstd(R1_spy, [19 0], 1, 'omitnan') * sqrt(252);
            
            spy_r20 = NaN(numDays, 1, 'single');
            spy_r20(21:end) = (P(21:end, spy_idx) - P(1:end-20, spy_idx)) ./ (P(1:end-20, spy_idx) + 1e-8);
            
            spy_r60 = NaN(numDays, 1, 'single');
            spy_r60(61:end) = (P(61:end, spy_idx) - P(1:end-60, spy_idx)) ./ (P(1:end-60, spy_idx) + 1e-8);
            
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
            Macro(:, 6) = vrp;                % 波動率風險溢酬 VRP
            Macro(:, 7) = fred_aligned(:, 2); % T10Y2Y (殖利率曲線倒掛)
            Macro(:, 8) = fred_aligned(:, 3); % BAMLH0A0HYM2 (高收益債信用利差)
            Macro(:, 9) = fred_aligned(:, 4); % DGS10 (10年期公債殖利率)
            Macro(:, 10)= fred_aligned(:, 5); % UNRATE (官方失業率)
            
            Macro = fillmissing(Macro, 'previous', 1);
            Macro = fillmissing(Macro, 'nearest', 1);
            Macro(isnan(Macro)) = 0;
        end
        
        function ema_data = calc_ema(~, data, window)
            alpha = 2 / (window + 1);
            ema_data = NaN(size(data), 'single');
            n_cols = size(data, 2);
            n_rows = size(data, 1);
            
            % MATLAB JIT 對單維度向量循環具備高度優化，消除 parfor 排程負擔
            for i = 1:n_cols
                col = data(:, i);
                first_valid = find(~isnan(col), 1, 'first');
                if isempty(first_valid)
                    continue;
                end
                
                ema_col = NaN(n_rows, 1, 'single');
                ema_col(first_valid) = col(first_valid);
                for t = (first_valid + 1) : n_rows
                    if isnan(col(t))
                        ema_col(t) = ema_col(t-1);
                    else
                        ema_col(t) = alpha * col(t) + (1.0 - alpha) * ema_col(t-1); 
                    end
                end
                ema_data(:, i) = ema_col; 
            end
        end
    end
    
    methods (Access = private)
        function update_progress(~, hWait, total, show_gui)
            persistent counter;
            if isempty(counter) || counter >= total
                counter = 0;
            end
            counter = counter + 1;
            
            if show_gui && ~isempty(hWait) && isgraphics(hWait)
                waitbar(counter / total, hWait, sprintf('空間圖譜運算中: 已完成 %d / %d', counter, total));
            end
            
            if mod(counter, 100) == 0 || counter == total
                fprintf('  [進度監控] 已完成 %d / %d 個時變圖譜錨點 (%.1f%%)...\n', ...
                    counter, total, (counter / total) * 100);
            end
        end
    end
end
