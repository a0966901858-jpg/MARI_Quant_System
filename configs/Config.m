classdef Config < handle
    % =========================================================================
    % 類別: Config (系統全域超參數與路徑配置中心)
    % 升級: Phase 15.5 深度表徵增強與生產剪枝基準版
    % 核心演進:
    %   1. 【空間剪枝鎖定】全面關閉 DyGAT 空間圖神經網路與協整建圖，鎖定時序專家權重 100% (防訊號稀釋)[cite: 1]
    %   2. 【表徵容量強化】新增特徵直通殘差開關 (Highway Residual)，解決 Trans-LSTM 略遜於 Raw 18D 瓶頸[cite: 1]
    %   3. 【解鎖早停早夭】放寬 Early Stopping 容忍度至 10 輪，保證學習率完整進入退火衰減期[cite: 1]
    %   4. 【週期嚴格同構】統一 20D 月步進全域基準線 (Horizon=20, Embargo=20, HACLag=20, Stride=20)[cite: 1]
    %   5. 【決策架構簡化】啟用單軌穩健 CIO 模式，消除低訊噪比下三軌 PPO 動態統合之無效震盪[cite: 1]
    % 職責: 作為 MARI 量化系統的超參數與組態單一真理來源 (Single Source of Truth, SSOT)
    % =========================================================================
    
    % ---------------------------------------------------------
    % 系統路徑 (對外唯讀 SetAccess = private，防止腳本執行中途意外覆寫)
    % ---------------------------------------------------------
    properties (SetAccess = private)
        ProjectRoot      % 專案根目錄路徑
        ProjectDir       % 專案根目錄相容別名
        DataDir          % 靜態與基礎資料夾路徑
        CacheDir         % 存放降噪與特徵工程後的 .mat 快取目錄
        DataLakeDir      % 存放 Python 爬蟲微服務輸出之 Parquet 原始長表目錄
        ModelDir         % 存放 VQ-VAE, GBDT, RL 大腦權重檔目錄
        ResultDir        % 存放回測績效報表與 SHAP 視覺化圖表目錄
    end
    
    % ---------------------------------------------------------
    % 動態大宇宙定義 (Dynamic Universe)
    % ---------------------------------------------------------
    properties
        IdxNames         % 標的名稱陣列 (供 UI 顯示或報表生成)
        IdxTickers       % 標的代號陣列 (如 'AAPL', 'SPY' 等純 Equity 標的)
        NumTickers       % 橫截面有效股票總數 (由 us_universe.csv 自動解析)
    end
    
    % ---------------------------------------------------------
    % 系統全域超參數 (Hyperparameters - 集中控管杜絕 Hardcoding)
    % ---------------------------------------------------------
    properties
        % =================================================================
        % 0. 剪枝計畫與架構簡化開關 (Pruning Plan Switches)
        % =================================================================
        % 依據 P2-1 消融實驗 (空間模型夏普為負) 與 Round 8a-v3 結構性缺陷診斷：
        % 圖卷積存在不可逆之過度平滑 (Over-smoothing) 且注意力梯度衰減至 10^-5，故全面剪枝[cite: 1]。
        EnableSpaceExpertTraining = false % ★ 空間專家 (DyGAT) 訓練開關 (設為 false 徹底略過空間訓練)[cite: 1]
        EnableSpaceExpert         = false % ★ 空間專家建構別名 (支援下游 BuildDecoupledExtractors 工廠)[cite: 1]
        EnableGraphConstruction   = false % ★ 時變協整圖譜建構開關 (設為 false 旁路 Phase 1 數千次協整檢定)[cite: 1]
        SpaceExpertMixMode        = 'gcn_only' % 空間專家混合模式 ('gcn_only' | 'dynamic')
        
        % 依據 P2-3 強化學習消融結論：三軌動態集成與單一決策在統計上無顯著差異 (p=0.9563)，
        % 故全面簡化為單軌穩健決策，大幅節省算力並阻斷策略梯度在極低訊噪比下的無效震盪[cite: 1]。
        EnableSingleTrackCIO      = true  % ★ 啟用單軌穩健 CIO 決策模式 (避免三軌負報酬發散)[cite: 1]
        RolloutSteps              = 40    % 向量化模擬環境訓練回合步長 (動態對齊 Horizon*2)[cite: 1]
        
        % =================================================================
        % 0.5 全域訊號與預測週期基準 (Unified 20D Configuration)
        % =================================================================
        % 實證支持：20D 連續超額報酬排序具備顯著 Rank IC (+0.0264 ~ +0.0290, p=0.0000)，
        % 且 20 日調倉能有效抑制短週期高頻換手帶來的滑價與手續費摩擦吞噬[cite: 1]。
        Horizon         = 20     % 全域超額報酬預測跨度 (20 日月度動能連續排序目標)[cite: 1]
        PurgeEmbargo    = 20     % 時序交叉驗證隔離期 (Embargo >= Horizon，阻斷標籤自相關洩漏)[cite: 1]
        HACLag          = 20     % Newey-West HAC 滯後階數 (校正時序重疊收益之統計檢定顯著性)[cite: 1]
        RebalanceStride = 20     % 前向回測調倉步進天數 (每 20 個交易日定期換手，與目標週期嚴格同構)[cite: 1]
        
        % =================================================================
        % 1. 特徵工程與雙軌萃取器 (Phase 1 & Phase 2)
        % =================================================================
        NumMacroFeatures = 10    % 宏觀總經特徵維度 (VIX, 殖利率, 信用利差, 失業率等)
        NumMicroFeatures = 15    % 個股微觀技術特徵維度 (動能, 波動度, Amihud流動性衝擊, 52週新高距離等)
        NumCointFeatures = 3     % 相對大盤特徵維度 (Beta, Correlation, Relative Strength)
        % 總節點特徵數 = 10 + 15 + 3 = 28 維；抽取器輸入維度 = 15 + 3 = 18 維
        
        SeqLen   = 60            % LSTM 時序專家歷史回溯視窗長度 (保留一季 60 個交易日微結構動態)[cite: 1]
        Lookback = 60            % 滾動視窗計算基礎天數
        
        % --- 深度學習模型容量與表徵增強 ---
        % ★ 核心新增：特徵直通殘差通路 (Linear Highway Connection)
        % 將最新一期 (t日) 18 維微觀特徵直接投影並加權融合成 64 維 Embedding，
        % 在數學架構上保證表徵排序力不低於 Raw 18D 基準線 (+0.0290)，打破過度平滑瓶頸[cite: 1]。
        EnableHighwayResidual    = true   % 啟用特徵直通殘差連接[cite: 1]
        
        % --- 早停機制與變異數保底 (VICReg 風格) ---
        % 最新主線執行診斷：原設 5 輪導致模型在 Epoch 8 早夭，尚未進入學習率衰減退火階段，
        % 故放寬至 10 輪，容忍多體制跨體驗證集的天然波動[cite: 1]。
        DL_MaxEpochs             = 50     % 將訓練輪數改為所需回數 (例如 50 回)
        DL_EarlyStoppingPatience = 50     % ★ 早停容忍輪數 (由 5 放寬至 10，防止假性早停)[cite: 1]
        DL_DecoupledDecayEpochs  = 35     % 解耦退火週期參數保證前段的學習率與數值完全一致，超期鎖定 min_lr。
        DL_DropoutRate           = 0.20   % 密集層/全連接層標準 Dropout 比率
        DL_L2_Regularization     = 1e-5   % 輕量 L2 權重衰減，避免壓制弱特徵梯度
        DL_VarianceFloorLambda   = 0.05   % 表徵變異數保底正則化係數
        DL_VarianceFloorTarget   = 1.00   % 各 Embedding 維度標準差下限 (防止幾何表徵坍縮)
        
        % --- 深度學習輸入層與時序防過擬合正則化 ---
        FeatureDropoutRate       = 0.12   % 特徵欄位隨機遮蔽率 (微調至 0.12，平衡去噪與微弱訊號保留)[cite: 1]
        InputNoiseStd            = 0.015  % 輸入層高斯動態噪聲標準差 (模擬真實盤面滑價與微結構微抖動)[cite: 1]
        VariationalDropRate      = 0.15   % 時序循環 Dropout 率 (整條 Sequence 共享同一個遮蔽 Mask)[cite: 1]
        AttentionDropRate        = 0.10   % 自注意力權重丟棄率 (抑制模型對特定歷史 K 線形態死記硬背)
        
        % --- 複合損失函數超參數 (Huber Robust Loss + Soft-IC) ---
        DL_HuberDelta            = 0.10   % Huber Loss 線性過渡門檻 (抗金融厚尾極端值拉扯)[cite: 1]
        DL_ICLossWeight          = 0.50   % Continuous Soft-IC 損失權重 (強化橫截面連續排序能力)
        
        % =================================================================
        % 2. VQ-VAE 向量量化降噪器 (Phase 1)
        % =================================================================
        VQ_DLatent   = 3         % 潛在空間維度 (高頻動能壓縮空間)
        VQ_KCodebook = 256       % 離散碼簿大小 (EMA 字典容量)
        VQ_Gamma     = 0.99      % 字典更新指數移動平均衰減率
        VQ_DHidden   = 128       % 編碼器/解碼器隱藏層神經元數量
        
        % =================================================================
        % 3. 顯性風險預測流 (GBDT Experts - Phase 3)
        % =================================================================
        GBDT_NumCycles = 50      % 決策樹森林迭代次數 (基學習器數量)
        GBDT_LearnRate = 0.10    % LSBoost / LogitBoost 學習率
        GBDT_MaxDepth  = 5       % 單一決策樹最大分裂深度
        
        % =================================================================
        % 4. HRL 總管狀態與決策空間 (Phase 5)
        % =================================================================
        CIO_SeqLen    = 10       % 總管決策大腦的歷史回看步長
        CIO_StateDim  = 5        % 總管宏觀狀態維度 [P_crash, SPY_Ret20, Vol20, MDD252, PrevCash]
        CIO_ActionDim = 3        % 動作輸出維度 [w_time, w_space, target_cash]
        
        % =================================================================
        % 5. 交易摩擦、成本與風控護欄 (實盤物理環境同構)
        % =================================================================
        HRL_LR           = 0.0005 % 強化學習基礎學習率
        MoE_FrictionMask = 0.0050 % 機構級慣性摩擦力過濾門檻 (換手變動小於 0.5% 不執行調倉)
        
        % 全域統一交易成本模型 (基於每日市場波動率之動態滑價方程)
        BaseFrictionFee  = 0.0005 % 基礎固定手續費 (單邊 0.05%)
        SlippageVolCoeff = 0.10   % 波動率衝擊滑價係數 (滑價成本隨日波幅線性上升)
        
        % 生產基準超參數 (★ 剪枝模式下 Time_W 嚴格鎖定 1.0000 防止訊號折半)
        Guardrail_CrashProb = 0.0978 % 崩盤護欄硬熔斷閾值 (經 Phase 4 BO 通過 DSR=1.0000 檢定)[cite: 1]
        Expert_Time_Weight  = 1.0000 % ★ 時序專家權重鎖定 100% (空間圖模型剪枝歸零)[cite: 1]
        Top_K_Assets        = 10     % 最佳持股集中度 (經 Phase 4 BO 最佳化判定)[cite: 1]
        
        % =================================================================
        % 6. 強化學習演算法參數 (RL Hyperparameters)
        % =================================================================
        HRL_Epochs = 500         % 強化學習最大訓練輪數
        HRL_Gamma  = 0.96        % 折扣回報衰減因子
        
        % =================================================================
        % 7. 工程衛生與確定性隨機數管理
        % =================================================================
        RNG_Seed      = 42          % 全域主隨機種子 (保證實驗 100% 可重現)
        RNG_Generator = 'mrg32k3a'  % 平行隨機數生成器 (支援 2^127 獨立無重疊子串流)
    end
    
    methods
        % =========================================================
        % 建構子：初始化路徑、建立目錄、自動防呆連動、載入參數與鎖定 RNG
        % =========================================================
        function obj = Config()
            % 解析專案根目錄 (動態層級回溯)
            currentPath = fileparts(mfilename('fullpath'));
            if isempty(currentPath), currentPath = pwd; end
            obj.ProjectRoot = fileparts(currentPath);
            
            if ~exist(fullfile(obj.ProjectRoot, 'data'), 'dir')
                obj.ProjectRoot = fileparts(obj.ProjectRoot);
            end
            
            % 映射全域標準化路徑 (含相容別名 ProjectDir)
            obj.ProjectDir  = obj.ProjectRoot;
            obj.DataDir     = fullfile(obj.ProjectRoot, 'data');
            obj.CacheDir    = fullfile(obj.ProjectRoot, 'data', 'cache');
            obj.DataLakeDir = fullfile(obj.ProjectRoot, 'data', 'data_lake');
            obj.ModelDir    = fullfile(obj.ProjectRoot, 'results', 'models');
            obj.ResultDir   = fullfile(obj.ProjectRoot, 'results', 'reports');
            
            % 自動創建基礎與結果目錄
            folders = {obj.DataDir, obj.CacheDir, obj.DataLakeDir, obj.ModelDir, obj.ResultDir};
            for i = 1:length(folders)
                if ~exist(folders{i}, 'dir'), mkdir(folders{i}); end
            end
            
            % ★ 剪枝邏輯自動防呆連動 (Auto-Interlocking)[cite: 1]
            % 若空間專家訓練為 false，強制關閉建圖與空間別名，並將時序權重鎖定為 1.0[cite: 1]
            if ~obj.EnableSpaceExpertTraining
                obj.EnableSpaceExpert       = false;
                obj.EnableGraphConstruction = false;
                obj.Expert_Time_Weight      = 1.0000;
            end
            
            % 啟動大宇宙解析與 BO 最佳化參數注入
            obj.loadUniverse();
            obj.loadBOParams();
            
            % 統一初始化主執行緒隨機數生成器 (Substream = 1)
            obj.initRNG(1);
        end
        
        % =========================================================
        % 函數：getRandStream (統一生產支援獨立子串流的 RandStream 物件)
        % =========================================================
        function stream = getRandStream(obj, substream_idx)
            if nargin < 2 || isempty(substream_idx)
                substream_idx = 1;
            end
            stream = RandStream(obj.RNG_Generator, 'Seed', obj.RNG_Seed);
            stream.Substream = substream_idx;
        end
        
        % =========================================================
        % 函數：initRNG (初始化當前執行緒/Worker 之全域隨機串流)
        % =========================================================
        function stream = initRNG(obj, substream_idx)
            if nargin < 2 || isempty(substream_idx)
                substream_idx = 1;
            end
            stream = obj.getRandStream(substream_idx);
            RandStream.setGlobalStream(stream);
        end
        
        % =========================================================
        % 函數：loadUniverse (True Point-in-Time 大宇宙動態解析器)
        % =========================================================
        function loadUniverse(obj)
            universePath = fullfile(obj.DataDir, 'crawlers', 'us_universe.csv');
            
            if isfile(universePath)
                opts = detectImportOptions(universePath);
                
                if ismember('Type', opts.VariableNames)
                    opts = setvartype(opts, 'Type', 'char');
                end
                
                universeTable = readtable(universePath, opts);
                
                if ismember('Type', universeTable.Properties.VariableNames)
                    valid_mask = strcmp(strtrim(universeTable.Type), 'Equity');
                    microTable = universeTable(valid_mask, :);
                else
                    microTable = universeTable(~contains(universeTable.Ticker, {'^VIX', 'CL=F', '^TNX', 'TLT', 'GLD', 'IEF'}), :);
                end
                
                obj.IdxTickers = strtrim(microTable.Ticker');
                obj.IdxNames   = strtrim(microTable.Ticker');
                obj.NumTickers = length(obj.IdxTickers);
                
                fprintf(' 🔄 [Config] 動態載入 True PiT 宇宙清單 (共 %d 檔純 Equity 標的)\n', obj.NumTickers);
            else
                warning('⚠️ 找不到 us_universe.csv，系統退回預設防呆五星宇宙。');
                obj.IdxNames   = {'SOXX', 'QQQ', 'SPY', 'DIA', 'IEF'};
                obj.IdxTickers = {'SOXX', 'QQQ', 'SPY', 'DIA', 'IEF'};
                obj.NumTickers = 5;
            end
        end
        
        % =================================================================
        % 函數：loadBOParams (貝氏尋優參數熱更新與 DSR 統計熔斷校驗)
        % =================================================================
        function loadBOParams(obj)
            boPath = fullfile(obj.ModelDir, 'BO_Hyperparameters.mat');
            
            if exist(boPath, 'file')
                try
                    data = load(boPath);
                    
                    % 縱深防禦 (Defense-in-Depth) 檢查 DSR 統計顯著性
                    if isfield(data, 'dsr_val') && isfield(data, 'best_robust_score')
                        if data.dsr_val < 0.95 || data.best_robust_score <= 0.0
                            warning(['⚠️ [Config] 檢測到 BO 參數未達統計顯著性 (DSR = %.4f < 0.95 或 Score = %.4f <= 0)！\n' ...
                                     '⚠️ 拒絕載入病態邊界參數，強制退回生產基準配置。'], ...
                                     data.dsr_val, data.best_robust_score);
                            obj.applyNeutralDefaults();
                            return;
                        end
                    end
                    
                    % 僅在通過檢定時注入最佳化參數
                    if isfield(data, 'best_params')
                        bp = data.best_params;
                        if istable(bp), bp = table2struct(bp); end
                        
                        fprintf(' ⚡ [Config] 成功注入通過 DSR 顯著性檢定之 Phase 4 BO 最佳化參數：\n');
                        
                        if isfield(bp, 'Guardrail_CrashProb')
                            obj.Guardrail_CrashProb = bp.Guardrail_CrashProb;
                            fprintf('    - 崩盤護欄硬熔斷閾值 : %.4f\n', obj.Guardrail_CrashProb);
                        end
                        
                        if isfield(bp, 'Expert_Time_Weight')
                            if ~obj.EnableSpaceExpertTraining
                                % 剪枝架構下強制鎖定 1.0，防止歷史尋優殘留的微小偏差污染[cite: 1]
                                obj.Expert_Time_Weight = 1.0000;
                                fprintf('    - 時序專家權重 (Time_W): %.4f (空間專家已剪枝，強制鎖定 1.0)\n', obj.Expert_Time_Weight);
                            else
                                obj.Expert_Time_Weight = bp.Expert_Time_Weight;
                                fprintf('    - 時序專家權重 (Time_W): %.4f\n', obj.Expert_Time_Weight);
                            end
                        end
                        
                        if isfield(bp, 'Top_K_Assets')
                            obj.Top_K_Assets = round(bp.Top_K_Assets);
                            fprintf('    - 集中度標的數 (Top_K) : %d\n', obj.Top_K_Assets);
                        end
                    end
                catch ME
                    warning('⚠️ 讀取 BO 參數失敗，強制退回生產基準配置。錯誤: %s', ME.message);
                    obj.applyNeutralDefaults();
                end
            else
                fprintf(' ℹ️ [Config] 未檢測到 BO 生產參數檔，啟用生產基準配置：\n');
                obj.applyNeutralDefaults();
            end
        end
    end
    
    methods (Access = private)
        % =========================================================
        % 私有輔助函數：套用生產基準參數 (防範病態解或訊號稀釋)
        % =========================================================
        function applyNeutralDefaults(obj)
            obj.Guardrail_CrashProb = 0.0978;
            obj.Top_K_Assets        = 10;
            
            % 空間專家剪枝防呆判定
            if ~obj.EnableSpaceExpertTraining
                obj.Expert_Time_Weight = 1.0000;
                fprintf('    - 崩盤護欄硬熔斷閾值 : %.4f (生產基準)\n', obj.Guardrail_CrashProb);
                fprintf('    - 時序專家權重 (Time_W): %.4f (空間專家已剪枝，鎖定 100%%)\n', obj.Expert_Time_Weight);
                fprintf('    - 集中度標的數 (Top_K) : %d (標準集中度)\n', obj.Top_K_Assets);
            else
                obj.Expert_Time_Weight = 0.5000;
                fprintf('    - 崩盤護欄硬熔斷閾值 : %.4f (中立基準)\n', obj.Guardrail_CrashProb);
                fprintf('    - 時序專家權重 (Time_W): %.4f (時空各 50%% 等權)\n', obj.Expert_Time_Weight);
                fprintf('    - 集中度標的數 (Top_K) : %d (標準集中度)\n', obj.Top_K_Assets);
            end
        end
    end
end
