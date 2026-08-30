% 定義 FeatureEngineer 類別，繼承自 handle (傳址參考)，確保物件傳遞時不會消耗額外記憶體
classdef FeatureEngineer < handle
    % =========================================================================
    % 模組：FeatureEngineer.m (MARI 特徵工程與時變圖譜引擎)
    % 升級：Phase 14.20 (★ 解耦 252 日共整合長視窗、統一橫截面標準化門檻、Top-K 優化)
    % 職責：從 DataFetcher 接收純淨矩陣，計算並產出無未來函數的 3D 面板張量
    % =========================================================================
    
    % 定義類別的屬性 (Properties)，儲存系統全域設定與各類特徵的維度大小
    properties
        ConfigObj       % 儲存全域設定檔物件 (Config)
        NumMicro        % 微觀特徵的數量 (如：RSI, MACD 等個股技術指標)
        NumMacro        % 宏觀總經特徵的數量 (如：大盤波動率、市場寬度)
        NumRel          % 相對特徵的數量 (與大盤的相對關係，如：Beta, 相關係數)
        NumTickers      % 全市場的股票總檔數 (節點數量)
        TotalNodeFeats  % 每個節點 (股票) 最終的特徵總維度 (Micro + Macro + Rel)
    end
    
    methods
        % 建構子 (Constructor)：實例化物件時初始化各項維度參數
        function obj = FeatureEngineer(config)
            obj.ConfigObj = config; % 綁定設定檔
            obj.NumMicro = config.NumMicroFeatures; % 載入微觀特徵數 (預設應為 15)
            obj.NumMacro = config.NumMacroFeatures; % 載入宏觀特徵數 (預設應為 4)
            obj.NumTickers = config.NumTickers;     % 載入股票總檔數
            obj.NumRel = 3;                         % 鎖定相對大盤特徵為 3 維
            
            % 計算單一節點(股票)的特徵總維度 (3 + 15 + 4 = 22 維)
            obj.TotalNodeFeats = obj.NumRel + obj.NumMicro + obj.NumMacro; 
            
            % 印出初始化完成的提示訊息，顯示預期產出的 3D 張量維度
            fprintf(' ⚙️ [FeatureEngineer] 初始化。準備產出 3D 面板資料 [Days, %d, %d]\n', ...
                obj.TotalNodeFeats, obj.NumTickers);
        end
        
        % 核心處理函數：接收 DataFetcher 輸出的原始矩陣，產出標準化特徵與時變圖譜
        function [X_norm_3D, Prices_Active, Expert_Active, Dates_Active, AdjMatrix_3D] = process(obj, dataStruct)
            disp(' 🚀 啟動機構級大宇宙特徵工程 (True PiT 橫截面無洩漏版)...');
            
            % 1. 從輸入的資料結構 (dataStruct) 中解構出各項原始價格與成交量矩陣
            Dates_Active   = dataStruct.Dates; 
            Opens_Active   = dataStruct.Opens; 
            Prices_Active  = dataStruct.Prices;     % 收盤價
            Highs_Active   = dataStruct.Highs; 
            Lows_Active    = dataStruct.Lows; 
            Volumes_Active = dataStruct.Volumes; 
            IsConst_Active = dataStruct.IsConst;    % 歷史成分股標籤 (Point-in-Time)
            numDays = length(Dates_Active);         % 取得總交易天數
            
            ticker_list = obj.ConfigObj.IdxTickers; % 取得股票代號清單
            
            % 2. 構建 Expert_Active 活躍遮罩 (動態投資池)
            disp(' -> 構建 True PiT 流動性濾網 (排除 IPO前/停牌/非成分股/雞蛋水餃股)...');
            Expert_Active = false(numDays, obj.NumTickers); % 預先配置全為 false 的布林矩陣
            
            % 計算 20 日移動平均成交量 (使用 'omitnan' 忽略缺失值)
            vol_20d = movmean(Volumes_Active, [19 0], 1, 'omitnan');
            % 設立嚴格的實體市場可交易條件：股價大於 1 元、20日均量大於 5 萬、當日為成分股、當日有成交量
            valid_condition = (Prices_Active > 1.0) & (vol_20d > 50000) & IsConst_Active & (Volumes_Active > 0);
            Expert_Active(valid_condition) = true; % 將符合條件的節點標記為 true (活躍可交易)
            
            % 找出基準大盤 ETF (SPY) 在矩陣中的索引位置
            spy_idx = find(strcmp(ticker_list, 'SPY'));
            if isempty(spy_idx)
                error('❌ 找不到基準標的 SPY，無法計算相對大盤與宏觀特徵！'); 
            end
            
            % 3. 計算核心特徵 (呼叫內部的特徵計算函數)
            disp(' -> 計算 15 維微觀特徵與 3 維相對大盤特徵...');
            % 計算微觀特徵 (產出 3D 矩陣：[Days, 15, Tickers])
            Micro_3D = obj.calc_micro_features(Opens_Active, Highs_Active, Lows_Active, Prices_Active, Volumes_Active, numDays);
            % 計算相對大盤特徵 (產出 3D 矩陣：[Days, 3, Tickers])
            Rel_3D   = obj.calc_relative_features(Prices_Active, spy_idx, numDays);
            
            disp(' -> 計算 4 維宏觀總經特徵...');
            % 計算宏觀特徵 (產出 2D 矩陣：[Days, 4]，因為全市場共用一組宏觀數據)
            Macro_2D = obj.calc_macro_features(Prices_Active, Expert_Active, spy_idx, numDays);
            
            % 4. 組合為最終的 3D 原始特徵張量
            disp(' -> 組合特徵為 3D 面板資料...');
            % 預配置 3D 原始特徵矩陣 [Days, 特徵總數, Tickers]
            X_raw_3D = NaN(numDays, obj.TotalNodeFeats, obj.NumTickers, 'single'); 
            
            % 將 2D 的宏觀特徵複製並擴展為 3D，讓每一檔股票都能對應到當天的宏觀數據
            Macro_3D = repmat(reshape(Macro_2D, [numDays, obj.NumMacro, 1]), [1, 1, obj.NumTickers]);
            
            % 依序將相對特徵、微觀特徵、宏觀特徵填入指定的維度通道中
            X_raw_3D(:, 1:3, :)   = Rel_3D; 
            X_raw_3D(:, 4:18, :)  = Micro_3D; 
            X_raw_3D(:, 19:22, :) = Macro_3D; 
            
            % 5. 執行特徵標準化 (消除不同特徵間的絕對數值差異，利於神經網路收斂)
            disp(' -> 執行特徵標準化 (Cross-Sectional Z-Score + Macro 保留絕對水位)...');
            X_norm_3D = zeros(size(X_raw_3D), 'single'); % 預配置標準化後的 3D 矩陣
            
            % 宏觀特徵標準化 (使用時序滾動 Z-Score，視窗為 251 天，約一個交易年)
            macro_raw = X_raw_3D(:, 19:22, 1); % 取出任一檔股票的宏觀特徵 (皆相同)
            mu_macro = movmean(macro_raw, [251 0], 1, 'omitnan'); % 滾動平均
            std_macro = movstd(macro_raw, [251 0], 1, 'omitnan') + 1e-8; % 滾動標準差 (加 1e-8 防除以零)
            macro_norm = (macro_raw - mu_macro) ./ std_macro; % 計算 Z-Score
            macro_norm(isnan(macro_norm)) = 0; % 將 NaN 補 0
            
            % ★ 計畫書修正 (問題 3-2 🟡 P2)：統一橫截面標準化門檻 (消除小樣本噪聲)
            min_cs_samples = max(10, floor(obj.NumTickers * 0.05));
            for t = 1:numDays
                active_idx = Expert_Active(t, :); % 取出當日活躍的股票遮罩
                if sum(active_idx) >= min_cs_samples
                    vals = X_raw_3D(t, 1:18, active_idx); % 擷取當日活躍股票的 1~18 維特徵
                    mu_cross = mean(vals, 3, 'omitnan');  % 計算橫截面平均
                    std_cross = std(vals, 0, 3, 'omitnan') + 1e-8; % 計算橫截面標準差
                    % 填入橫截面標準化後的值
                    X_norm_3D(t, 1:18, active_idx) = (vals - mu_cross) ./ std_cross; 
                else
                    % 樣本不足時特徵直接置零，不硬算 Z-score
                    X_norm_3D(t, 1:18, active_idx) = 0;
                end
                % 將當天的宏觀特徵標準化值廣播填入所有股票中
                X_norm_3D(t, 19:22, :) = repmat(macro_norm(t, :), [1, 1, obj.NumTickers]);
            end
            
            % 極端值清理：將可能產生的 NaN 或 Inf 強制設為 0 (均值)
            X_norm_3D(isnan(X_norm_3D) | isinf(X_norm_3D)) = 0;
            % 根據活躍遮罩，將「非活躍」股票的特徵強制清零，避免模型學習到無效節點的雜訊
            inactive_mask = repmat(reshape(~Expert_Active, [numDays, 1, obj.NumTickers]), [1, obj.TotalNodeFeats, 1]);
            X_norm_3D(inactive_mask) = 0; 
            
            % 6. 動態圖譜矩陣 (DyGAT Adjacency Matrix) 構建
            disp(' -> 構建 DyGAT 時變圖譜矩陣 (啟動 12 執行緒平行運算與進度追蹤)...');
            
            n_tickers = obj.NumTickers;
            daily_rets = NaN(numDays, n_tickers, 'single');
            % 計算每日簡單報酬率 (今日收盤 - 昨日收盤) / 昨日收盤
            daily_rets(2:end, :) = (Prices_Active(2:end,:) - Prices_Active(1:end-1,:)) ./ Prices_Active(1:end-1,:);
            log_prices = log(Prices_Active); % 計算對數價格，用於後續共整合檢定
            
            lookback = 60;          % 短期 Spearman 相關性回溯視窗 (60 天)
            coint_lookback = 252;   % ★ 計畫書修正 (問題 3-1 🟡 P2)：長天期共整合檢定獨立視窗 (1 年)
            
            % 設定計算節點 (每 5 天計算一次錨點以節省算力，其他天數沿用前一個錨點)
            calc_days = (lookback + 1) : 5 : numDays;
            if calc_days(1) ~= (lookback + 1)
                calc_days = [lookback + 1, calc_days]; % 確保第一天有被計算到
            end
            num_calc = length(calc_days);
            Anchor_Adj = cell(num_calc, 1); % 用 Cell 陣列儲存計算出來的稀疏圖譜錨點
            
            fprintf(' -> 預計計算 %d 個圖譜錨點，任務已發配至運算池...\n', num_calc);
            
            % 建立平行運算池的資料佇列，用於在 parfor 執行中回傳進度條訊號
            dq = parallel.pool.DataQueue;
            hWait = waitbar(0, '啟動多核運算池...', 'Name', 'DyGAT 空間圖譜運算進度');
            hWait.UserData = 0; 
            
            % 註冊回呼函數，當 parfor 每次完成一個迴圈時更新進度條
            afterEach(dq, @(~) obj.update_progress(hWait, num_calc));
            
            % 啟動平行迴圈計算圖譜 (空間拓樸)
            parfor i = 1:num_calc
                % 關閉系統 I/O 警告
                warn_state_1 = warning('off', 'econ:egcitest:LeftYVarColinear');
                warn_state_2 = warning('off', 'stats:corr:Ties');
                
                t = calc_days(i); % 取得當前要計算的絕對天數
                
                % 短期 60 天報酬率視窗 (用於計算近期動態相關性)
                window_rets = daily_rets(t-lookback : t-1, :); 
                
                % ★ 計畫書修正 (問題 3-1 🟡 P2)：動態切換長視窗對數價格進行嚴謹 EG 共整合
                if t > coint_lookback
                    window_logP = log_prices(t-coint_lookback : t-1, :);
                else
                    window_logP = log_prices(t-lookback : t-1, :);
                end
                
                % 過濾掉在視窗內有任何 NaN (停牌或未上市) 的不合法節點
                valid_nodes = all(~isnan(window_rets), 1) & all(~isnan(window_logP), 1);
                valid_idx = find(valid_nodes);
                num_valid = length(valid_idx);
                
                % 預配置二元鄰接矩陣 (Adjacency Matrix)，並將對角線設為 true (自我連接)
                bin_adj = false(n_tickers, n_tickers);
                bin_adj(1:n_tickers+1:end) = true; 
                
                % 至少需要有兩檔以上的合法股票才能進行關聯性計算
                if num_valid > 2
                    % 第一階段篩選：計算短期 Spearman 秩相關係數
                    clean_corr = corr(double(window_rets(:, valid_nodes)), 'Type', 'Spearman');
                    clean_corr(isnan(clean_corr)) = 0;
                    
                    % 尋找相關係數大於 0.5 的配對 (取上三角矩陣避免重複計算)
                    [row_idx, col_idx] = find(triu(clean_corr > 0.5, 1));
                    num_pairs = length(row_idx);
                    
                    % Top-K 強相關截斷機制 (防止高相關市場環境下的 O(N^2) 運算災難)
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
                    
                    % 第二階段篩選：使用長天期視窗進行 Engle-Granger 兩步法共整合檢定
                    for k = 1:num_pairs
                        idx_A = valid_idx(row_idx(k));
                        idx_B = valid_idx(col_idx(k));
                        
                        pA = window_logP(:, idx_A);
                        pB = window_logP(:, idx_B);
                        
                        % 執行 EG 檢定，Alpha 設為 0.05 (95% 信心水準)
                        [h, ~] = egcitest(double([pA, pB]), 'Alpha', 0.05);
                        
                        % 若拒絕虛無假說 (存在共整合)，則在圖譜建立雙向連接邊
                        if h == 1 
                            bin_adj(idx_A, idx_B) = true;
                            bin_adj(idx_B, idx_A) = true; 
                        end
                    end
                end
                Anchor_Adj{i} = bin_adj; % 將計算完的圖譜存入 Cell 陣列中
                
                % 恢復警告狀態
                warning(warn_state_1);
                warning(warn_state_2);
                
                send(dq, 1); % 更新進度條
            end
            
            if isgraphics(hWait), close(hWait); end
            
            disp(' -> 錨點運算全數完成，正在進行時間平移與 3D 矩陣對齊...');
            % 將離散的圖譜錨點平移映射回每一天的絕對時間軸上 (前向填充 Forward Fill)
            AdjMatrix_3D = false(n_tickers, n_tickers, numDays);
            
            for t = lookback + 1 : numDays
                idx = find(calc_days <= t, 1, 'last');
                if ~isempty(idx)
                    AdjMatrix_3D(:, :, t) = Anchor_Adj{idx};
                end
            end
            
            disp('✅ 3D 特徵面板與共整合圖譜提取完畢！維度與物理意義已絕對對齊。');
        end  
        
        %% --- 內部特徵計算函數 ---
        
        % 計算微觀特徵 (技術指標與價格動能)
        function Micro = calc_micro_features(obj, ~, H, L, P, V, numDays)
            obj.NumTickers = size(P, 2);
            Micro = NaN(numDays, 15, obj.NumTickers, 'single'); 
            
            % 計算動能：1日、5日、20日報酬率
            R1 = NaN(numDays, obj.NumTickers, 'single');
            R1(2:end,:) = (P(2:end,:) - P(1:end-1,:)) ./ P(1:end-1,:);
            R5 = NaN(numDays, obj.NumTickers, 'single');
            R5(6:end,:) = (P(6:end,:) - P(1:end-5,:)) ./ P(1:end-5,:);
            R20 = NaN(numDays, obj.NumTickers, 'single');
            R20(21:end,:) = (P(21:end,:) - P(1:end-20,:)) ./ P(1:end-20,:);
            
            % 計算 20 日波動率
            Vol20 = movstd(R1, [19 0], 1, 'omitnan');
            
            % 計算成交量比率 (5日均量 / 20日均量)
            V5 = movmean(V, [4 0], 1, 'omitnan');
            V20 = movmean(V, [19 0], 1, 'omitnan');
            VolRatio = V5 ./ (V20 + 1e-8);
            
            % 計算簡單移動平均線乖離率 (均線值 / 價格)
            SMA20 = movmean(P, [19 0], 1, 'omitnan') ./ P;
            SMA60 = movmean(P, [59 0], 1, 'omitnan') ./ P;
            
            % 計算 MACD (指數移動平均)
            EMA12 = obj.calc_ema(P, 12);
            EMA26 = obj.calc_ema(P, 26);
            MACD_Line = (EMA12 - EMA26) ./ P; % 標準化 MACD 線
            MACD_Sig = obj.calc_ema(MACD_Line, 9); % 訊號線
            
            % 計算 RSI (相對強弱指標)
            diff_P = NaN(numDays, obj.NumTickers, 'single');
            diff_P(2:end,:) = diff(P); % 價格變動量
            U = max(diff_P, 0); % 上漲量
            D = max(-diff_P, 0); % 下跌量
            EMA_U = obj.calc_ema(U, 27); % 使用 EMA 平滑
            EMA_D = obj.calc_ema(D, 27);
            RS = EMA_U ./ (EMA_D + 1e-8); 
            RSI = 100 - (100 ./ (1 + RS)); 
            
            % 計算 OBV (能量潮指標) 變化率
            SignR = sign(R1);
            OBV_diff = SignR .* V;
            OBV_20 = movsum(OBV_diff, [19 0], 1, 'omitnan') ./ (V20 * 20 + 1e-8);
            
            % 計算 MFI (資金流量指標)
            Typical_P = (H + L + P) / 3; % 典型價格
            MF = Typical_P .* V; % 資金流量
            PosMF = MF .* (R1 > 0); 
            NegMF = MF .* (R1 < 0);
            MFRatio = movsum(PosMF, [13 0], 1, 'omitnan') ./ (movsum(NegMF, [13 0], 1, 'omitnan') + 1e-8);
            MFI = 100 - (100 ./ (1 + MFRatio));
            
            % 計算 VPT (量價趨勢指標)
            VPT_diff = V .* R1;
            VPT_20 = movsum(VPT_diff, [19 0], 1, 'omitnan') ./ (V20 * 20 + 1e-8);
            
            % 計算高低價區間特徵
            H20 = movmax(H, [19 0], 1, 'omitnan'); % 20日最高價
            L20 = movmin(L, [19 0], 1, 'omitnan'); % 20日最低價
            HL_Spread = (H20 - L20) ./ P; % 高低價差比例
            Dist_H20 = (P - H20) ./ H20;  % 距離 20日最高價的幅度
            
            % 將 15 維特徵寫入 Micro 張量對應的通道中
            Micro(:, 1, :)  = R1;        Micro(:, 2, :)  = R5;        Micro(:, 3, :)  = R20;
            Micro(:, 4, :)  = Vol20;     Micro(:, 5, :)  = VolRatio;  Micro(:, 6, :)  = SMA20;
            Micro(:, 7, :)  = SMA60;     Micro(:, 8, :)  = MACD_Line; Micro(:, 9, :)  = MACD_Sig;
            Micro(:, 10, :) = RSI;      Micro(:, 11, :) = OBV_20;   Micro(:, 12, :) = MFI;
            Micro(:, 13, :) = VPT_20;   Micro(:, 14, :) = HL_Spread;Micro(:, 15, :) = Dist_H20;
        end
        
        % 計算個股相對於大盤 (SPY) 的特徵
        function Rel = calc_relative_features(~, P, spy_idx, numDays)
            n_tickers = size(P, 2);
            Rel = NaN(numDays, 3, n_tickers, 'single'); 
            
            % 計算全市場每日報酬與大盤每日報酬
            R1 = NaN(numDays, n_tickers, 'single');
            R1(2:end,:) = (P(2:end,:) - P(1:end-1,:)) ./ P(1:end-1,:); 
            spy_R1 = R1(:, spy_idx); 
            spy_Var20 = movvar(spy_R1, [19 0], 1, 'omitnan') + 1e-8; % 大盤 20 日變異數
            
            % 使用 parfor 加速每檔股票與大盤的滾動共變異數計算
            parfor i = 1:n_tickers
                % 計算個股與大盤的 20 日滾動共變異數 Cov(X,Y) = E(XY) - E(X)E(Y)
                cov_val = movmean(R1(:,i) .* spy_R1, [19 0], 1, 'omitnan') - ...
                          (movmean(R1(:,i), [19 0], 1, 'omitnan') .* movmean(spy_R1, [19 0], 1, 'omitnan'));
                          
                % 特徵 1：計算 Beta 值 (Cov(i, SPY) / Var(SPY))
                beta_i = cov_val ./ spy_Var20; 
                
                % 特徵 2：計算滾動相關係數 (Beta * (Std(SPY) / Std(i)))
                std_i = movstd(R1(:,i), [19 0], 1, 'omitnan') + 1e-8;
                corr_i = beta_i .* (sqrt(spy_Var20) ./ std_i); 
                
                % 特徵 3：計算相對強度 (RS) 差異
                RS_i = (P(:, i) ./ movmean(P(:, i), [19 0], 1, 'omitnan')) - ...
                       (P(:, spy_idx) ./ movmean(P(:, spy_idx), [19 0], 1, 'omitnan'));
                
                % 封裝並寫回 Rel 矩陣
                temp_rel = NaN(numDays, 3, 'single');
                temp_rel(:, 1) = beta_i;
                temp_rel(:, 2) = corr_i;
                temp_rel(:, 3) = RS_i;
                Rel(:, :, i) = temp_rel; 
            end
        end
        
        % 計算系統總體/宏觀特徵 (全市場股票共用這 4 維數據)
        function Macro = calc_macro_features(~, P, Expert, spy_idx, numDays)
            Macro = NaN(numDays, 4, 'single');
            
            % 計算大盤每日報酬
            R1_spy = NaN(numDays, 1, 'single');
            R1_spy(2:end) = (P(2:end, spy_idx) - P(1:end-1, spy_idx)) ./ P(1:end-1, spy_idx);
            
            % 宏觀特徵 1：使用 SPY 的 20 日年化波動率作為 VIX 的代理指標
            vix_proxy = movstd(R1_spy, [19 0], 1, 'omitnan') * sqrt(252);
            
            % 宏觀特徵 2：大盤 20 日報酬 (中期趨勢)
            spy_r20 = NaN(numDays, 1, 'single');
            spy_r20(21:end) = (P(21:end, spy_idx) - P(1:end-20, spy_idx)) ./ P(1:end-20, spy_idx);
            
            % 宏觀特徵 3：大盤 60 日報酬 (長期趨勢)
            spy_r60 = NaN(numDays, 1, 'single');
            spy_r60(61:end) = (P(61:end, spy_idx) - P(1:end-60, spy_idx)) ./ P(1:end-60, spy_idx);
            
            % 宏觀特徵 4：市場寬度 (Market Breadth)，計算活躍股票中「股價站上 20 日均線」的比例
            MA20_all = movmean(P, [19 0], 1, 'omitnan');
            is_above = (P > MA20_all) & Expert; 
            active_counts = sum(Expert, 2); % 每日的活躍總股數
            breadth = sum(is_above, 2) ./ (active_counts + 1e-8);
            
            Macro(:, 1) = vix_proxy;
            Macro(:, 2) = spy_r20;
            Macro(:, 3) = spy_r60;
            Macro(:, 4) = breadth;
        end
        
        % 計算指數平滑移動平均 (EMA) 的自訂函數
        function ema_data = calc_ema(~, data, window)
            alpha = 2 / (window + 1); % 計算平滑因子
            ema_data = NaN(size(data), 'single');
            
            % 針對每檔股票獨立平行計算 EMA
            parfor i = 1:size(data, 2)
                col = data(:, i);
                first_valid = find(~isnan(col), 1, 'first'); % 找出上市後第一筆非 NaN 的資料
                if isempty(first_valid)
                    continue; % 若全是 NaN 則跳過
                end
                
                ema_col = NaN(size(col), 'single');
                ema_col(first_valid) = col(first_valid); % 初始化起點
                
                % 由於 EMA 具備時序依賴性 (今日依賴昨日)，故需使用 for 迴圈逐日遞迴計算
                for t = first_valid + 1 : length(col)
                    if isnan(col(t))
                        ema_col(t) = ema_col(t-1);  % 遇到 NaN 則沿用昨日的 EMA 值
                    else
                        ema_col(t) = alpha * col(t) + (1 - alpha) * ema_col(t-1); 
                    end
                end
                ema_data(:, i) = ema_col; 
            end
        end
    end
    
    % 定義私有方法 (僅限類別內部呼叫)
    methods (Access = private)
        % 用於平行運算 (parfor) 的進度條更新機制
        function update_progress(~, hWait, total)
            if isgraphics(hWait)
                count = hWait.UserData + 1; % 讀取並遞增已完成的任務數
                hWait.UserData = count;
                
                % 更新 waitbar 視窗的進度與文字
                waitbar(count/total, hWait, sprintf('多核平行運算中: 已完成 %d / %d', count, total));
                
                % 每完成 50 個任務或全部完成時，在命令列輸出系統總控訊息，方便監控
                if mod(count, 50) == 0 || count == total
                    fprintf('  [系統總控] 成功回收訊號：已確實完成 %d / %d 個時變圖譜...\n', count, total);
                end
            end
        end
    end
end