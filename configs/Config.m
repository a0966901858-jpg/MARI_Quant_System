classdef Config < handle
    % =========================================================================
    % 類別: Config (系統全域超參數與路徑配置中心)
    % 升級: Phase 15.5 生產基準版 (★ 統一 60D 基準線全域參數單一來源、
    %       PurgeEmbargo/HACLag 阻斷長週期重疊標籤洩漏、RebalanceStride 平抑換手摩擦、
    %       mrg32k3a 獨立子串流隨機數引擎、DSR 熔斷防禦載入、SpaceExpertMixMode 空間模式)
    % 職責: 作為 MARI 量化系統的超參數與組態單一真理來源 (Single Source of Truth)
    % =========================================================================
    
    % ---------------------------------------------------------
    % 系統路徑 (設定為對外唯讀 SetAccess = private，防止腳本意外覆寫)
    % ---------------------------------------------------------
    properties (SetAccess = private)
        ProjectRoot      % 專案根目錄路徑
        ProjectDir       % 專案根目錄路徑 (相容性別名)
        DataDir          % 靜態與基礎資料夾
        CacheDir         % 存放降噪與特徵工程後的 .mat 快取
        DataLakeDir      % 存放 Python 爬蟲輸出的 Parquet 原始長表
        ModelDir         % 存放 VQ-VAE, GBDT, RL 大腦的網路權重檔
        ResultDir        % 存放最終回測績效與 SHAP 視覺化圖表
    end
    
    % ---------------------------------------------------------
    % 動態大宇宙定義 (Dynamic Universe)
    % ---------------------------------------------------------
    properties
        IdxNames         % 標的名稱陣列 (供 UI 或報告顯示)
        IdxTickers       % 標的代號陣列 (如 'AAPL', 'SPY')
        NumTickers       % 實際進入橫截面運算的有效股票總數 (純量)
    end
    
    % ---------------------------------------------------------
    % 系統全域超參數 (Hyperparameters - 統一控管防禦 Hardcoding)
    % ---------------------------------------------------------
    properties
        % --- 0. 模組執行開關 (Module Execution Flags) ---
        EnableSpaceExpertTraining = true  % 空間專家訓練開關
        SpaceExpertMixMode = 'gcn_only'   % 空間專家混合模式 ('gcn_only' | 'dynamic')
        
        % --- 0.5 全域訊號與預測週期基準 (★ Unified 60D Baseline) ---
        Horizon         = 60     % 全域超額報酬預測跨度 (Direction 2 基準線: 60 日)
        PurgeEmbargo    = 60     % 時序交叉驗證隔離期 (Embargo >= Horizon，杜絕標籤洩漏)
        HACLag          = 60     % Newey-West HAC 滯後階數 (強制 lag >= Horizon 校正自相關)
        RebalanceStride = 20     % 回測調倉步進天數 (建議 20 日滾動月換手，避免每日雜訊換手吞噬收益)
        
        % --- 1. 特徵工程與雙軌萃取器 (Phase 1 & 2) ---
        NumMacroFeatures = 10    % 宏觀特徵維度 (VIX_Proxy, R20, R60, Breadth, Real_VIX, VRP, T10Y2Y, HY, DGS10, UNRATE)
        NumMicroFeatures = 15    % 微觀特徵維度 (含特質波動度、Amihud流動性、52週新高、MACD柱狀圖)
        NumCointFeatures = 3     % 協整相對特徵 (Beta, Corr, Relative Strength)
        
        SeqLen = 60              % LSTM 時序專家回溯視窗長度 (對齊 60 個交易日)
        Lookback = 60            % 相關性圖譜與 IC 檢定的歷史滾動視窗
        
        % 雙軌萃取器正則化超參數與變異數保底
        DL_DropoutRate = 0.2             % Transformer-LSTM Dropout 機率
        DL_L2_Regularization = 1e-5      % 降低 L2 懲罰，避免主導弱梯度訊號
        DL_VarianceFloorLambda = 0.05    % 表徵變異數保底正則化係數
        DL_VarianceFloorTarget = 1.0     % 每個 embedding 維度跨樣本標準差的目標下限
        DL_EarlyStoppingPatience = 5     % 早停容忍輪數
        
        % --- 2. VQ-VAE 向量量化降噪器 (Phase 1) ---
        VQ_DLatent = 3           % 潛在空間維度
        VQ_KCodebook = 256       % 編碼簿大小
        VQ_Gamma = 0.99          % EMA 衰減率
        VQ_DHidden = 128         % 隱藏層神經元數量
        
        % --- 3. 顯性風險預測流 (GBDT Experts - Phase 3) ---
        GBDT_NumCycles = 50      % 決策樹森林基學習器數量
        GBDT_LearnRate = 0.1     % LogitBoost / LSBoost 學習率
        GBDT_MaxDepth = 5        % 單一決策樹最大深度
        
        % --- 4. HRL 總管狀態與決策空間 (Phase 5) ---
        CIO_SeqLen = 10          % 總管大腦的歷史記憶長度
        CIO_StateDim = 5         % 總管宏觀狀態維度 [P_crash, SPY_Ret20, Vol20, MDD252, PrevCash]
        CIO_ActionDim = 3        % 總管動作輸出維度 [w_time, w_space, target_cash]
        
        % --- 5. 交易摩擦與護欄閾值 (實盤物理環境) ---
        HRL_LR = 0.0005          % 強化學習大腦的基礎學習率
        MoE_FrictionMask = 0.005 % 固定機構級慣性摩擦力 0.5%
        
        % 全域統一交易成本模型係數
        BaseFrictionFee = 0.0005 % 基礎固定手續費 0.05%
        SlippageVolCoeff = 0.10  % 波動率動態衝擊成本係數 (日頻波動度基礎)
        
        % 學術中立基準超參數 (當 BO 未達 DSR 顯著時的強制 Fallback 配置)
        Guardrail_CrashProb = 0.0850 % 中立崩盤護欄硬熔斷閾值 (8.5%)
        Expert_Time_Weight  = 0.5000 % 中立時空專家等權 (50% / 50%)
        Top_K_Assets        = 20     % 中立標準持股分散度 (20 檔)
        
        % --- 6. 強化學習演算法 (RL Hyperparameters) ---
        HRL_Epochs = 500         % 強化學習平行滾動訓練總 Epoch 數
        HRL_Gamma = 0.96
        
        % --- 7. 工程衛生與隨機種子 ---
        RNG_Seed      = 42          % 全域隨機種子 (確保實驗可重現)
        RNG_Generator = 'mrg32k3a'  % 支援 2^127 獨立子串流之平行隨機數生成器
    end
    
    methods
        % =========================================================
        % 建構子：初始化路徑、建立目錄、載入宇宙與超參數、鎖定 RNG
        % =========================================================
        function obj = Config()
            currentPath = fileparts(mfilename('fullpath'));
            if isempty(currentPath), currentPath = pwd; end
            obj.ProjectRoot = fileparts(currentPath);
            
            if ~exist(fullfile(obj.ProjectRoot, 'data'), 'dir')
                obj.ProjectRoot = fullfile(obj.ProjectRoot, '..');
            end
            
            % 映射全域標準化路徑 (含相容別名 ProjectDir)
            obj.ProjectDir  = obj.ProjectRoot;
            obj.DataDir     = fullfile(obj.ProjectRoot, 'data');
            obj.CacheDir    = fullfile(obj.ProjectRoot, 'data', 'cache');
            obj.DataLakeDir = fullfile(obj.ProjectRoot, 'data', 'data_lake');
            obj.ModelDir    = fullfile(obj.ProjectRoot, 'results', 'models');
            obj.ResultDir   = fullfile(obj.ProjectRoot, 'results', 'reports');
            
            % 自動創建目錄
            folders = {obj.DataDir, obj.CacheDir, obj.DataLakeDir, obj.ModelDir, obj.ResultDir};
            for i = 1:length(folders)
                if ~exist(folders{i}, 'dir'), mkdir(folders{i}); end
            end
            
            % 啟動初始化程序
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
        % 函數：loadUniverse (動態大宇宙解析器)
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
        % 函數：loadBOParams (貝氏尋優參數熱更新與 DSR 熔斷雙重校驗)
        % =================================================================
        function loadBOParams(obj)
            boPath = fullfile(obj.ModelDir, 'BO_Hyperparameters.mat');
            
            if exist(boPath, 'file')
                try
                    data = load(boPath);
                    
                    % 縱深防禦 (Defense-in-Depth) 檢查 DSR 顯著性
                    if isfield(data, 'dsr_val') && isfield(data, 'best_robust_score')
                        if data.dsr_val < 0.95 || data.best_robust_score <= 0.0
                            warning(['⚠️ [Config] 檢測到 BO 參數未達統計顯著性 (DSR = %.4f < 0.95 或 Score = %.4f <= 0)！\n' ...
                                     '⚠️ 拒絕載入病態邊界參數，強制退回學術中立基準 (Time_W=0.50, TopK=20, Guard=0.0850)。'], ...
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
                            obj.Expert_Time_Weight = bp.Expert_Time_Weight;
                            fprintf('    - 時序專家權重 (Time_W): %.4f\n', obj.Expert_Time_Weight);
                        end
                        
                        if isfield(bp, 'Top_K_Assets')
                            obj.Top_K_Assets = round(bp.Top_K_Assets);
                            fprintf('    - 集中度標的數 (Top_K) : %d\n', obj.Top_K_Assets);
                        end
                    end
                catch ME
                    warning('⚠️ 讀取 BO 參數失敗，強制退回學術中立基準。錯誤: %s', ME.message);
                    obj.applyNeutralDefaults();
                end
            else
                fprintf(' ℹ️ [Config] 未檢測到 BO 生產參數檔，啟用學術中立基準：\n');
                obj.applyNeutralDefaults();
            end
        end
    end
    
    methods (Access = private)
        % =========================================================
        % 私有輔助函數：套用學術中立基準參數 (防止病態解傳導)
        % =========================================================
        function applyNeutralDefaults(obj)
            obj.Guardrail_CrashProb = 0.0850;
            obj.Expert_Time_Weight  = 0.5000;
            obj.Top_K_Assets        = 20;
            fprintf('    - 崩盤護欄硬熔斷閾值 : %.4f (中立基準)\n', obj.Guardrail_CrashProb);
            fprintf('    - 時序專家權重 (Time_W): %.4f (時空各 50%% 等權)\n', obj.Expert_Time_Weight);
            fprintf('    - 集中度標的數 (Top_K) : %d (標準分散度)\n', obj.Top_K_Assets);
        end
    end
end
