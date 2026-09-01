classdef Config < handle
    % =========================================================================
    % 類別: Config (系統全域超參數與路徑配置中心)
    % 升級: Phase 15 (★ 表徵變異數保底超參數注入、調降 L2 正則化係數防坍縮)
    % 職責: 作為 MARI 量化系統的超參數與組態單一真理來源 (Single Source of Truth)
    % =========================================================================
    
    % ---------------------------------------------------------
    % 系統路徑 (設定為對外唯讀 SetAccess = private，防止腳本意外覆寫)
    % ---------------------------------------------------------
    properties (SetAccess = private)
        ProjectRoot      % 專案根目錄路徑
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
        % --- 1. 特徵工程與雙軌萃取器 (Phase 1 & 2) ---
        NumMacroFeatures = 10    % 宏觀特徵維度 (VIX_Proxy, R20, R60, Breadth, Real_VIX, VRP, T10Y2Y, HY, DGS10, UNRATE)
        NumMicroFeatures = 15    % 微觀特徵維度 (含特質波動度、Amihud流動性、52週新高、MACD柱狀圖)
        NumCointFeatures = 3     % 協整相對特徵 (Beta, Corr, Relative Strength)
        % 總節點特徵數 = 10 + 15 + 3 = 28 維
        
        SeqLen = 60              % LSTM 時序專家回溯視窗長度 (對齊一季 60 個交易日)
        Lookback = 60            % 相關性圖譜與 IC 檢定的歷史滾動視窗
        
        % ★ Phase 15 雙軌萃取器正則化超參數與變異數保底
        DL_DropoutRate = 0.2             % Transformer-LSTM Dropout 機率 (抑制過擬合)
        DL_L2_Regularization = 1e-5      % ★ 由 1e-4 調降一個數量級，避免主導弱梯度訊號
        DL_VarianceFloorLambda = 0.05    % ★ 新增：表徵變異數保底正則化係數 (VICReg 風格)
        DL_VarianceFloorTarget = 1.0     % ★ 新增：每個 embedding 維度跨樣本標準差的目標下限
        DL_EarlyStoppingPatience = 5     % 早停容忍輪數
        
        % --- 2. VQ-VAE 向量量化降噪器 (Phase 2) ---
        VQ_DLatent = 3           % 潛在空間維度
        VQ_KCodebook = 256       % 編碼簿大小
        VQ_Gamma = 0.99          % EMA 衰減率
        VQ_DHidden = 128         % 隱藏層神經元數量
        
        % --- 3. 顯性風險預測流 (GBDT Experts - Phase 3) ---
        GBDT_NumCycles = 50      % 決策樹森林基學習器數量
        GBDT_LearnRate = 0.1     % LogitBoost 學習率
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
        
        % BO 最佳化超參數 (將在腳本初始化時被 loadBOParams 動態覆寫)
        Guardrail_CrashProb = 0.85 
        Expert_Time_Weight = 0.50  % 預設時序專家權重 50%
        Top_K_Assets = 20          % 預設持股集中度 20 檔
        
        % --- 6. 強化學習演算法 (RL Hyperparameters) ---
        HRL_Epochs = 500         % 強化學習平行滾動訓練總 Epoch 數
        HRL_Gamma = 0.96
        
        % --- 7. 工程衛生與隨機種子 ---
        RNG_Seed = 42            % 全域隨機種子 (確保實驗可重現)
    end
    
    methods
        % =========================================================
        % 建構子：初始化路徑、建立目錄、載入宇宙與超參數
        % =========================================================
        function obj = Config()
            currentPath = fileparts(mfilename('fullpath'));
            if isempty(currentPath), currentPath = pwd; end
            obj.ProjectRoot = fileparts(currentPath);
            
            if ~exist(fullfile(obj.ProjectRoot, 'data'), 'dir')
                obj.ProjectRoot = fullfile(obj.ProjectRoot, '..');
            end
            
            % 映射全域標準化路徑
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
        % 函數：loadBOParams (貝氏尋優參數熱更新)
        % =================================================================
        function loadBOParams(obj)
            boPath = fullfile(obj.ModelDir, 'BO_Hyperparameters.mat');
            
            if exist(boPath, 'file')
                try
                    data = load(boPath, 'best_params');
                    if isfield(data, 'best_params')
                        bp = data.best_params;
                        
                        if istable(bp)
                            bp = table2struct(bp);
                        end
                        
                        fprintf(' ⚡ [Config] 成功注入 Phase 4 BO 最佳化參數：\n');
                        
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
                    warning('⚠️ 讀取 BO 參數失敗，維持預設值。錯誤: %s', ME.message);
                end
            end
        end
    end 
end
