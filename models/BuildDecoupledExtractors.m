classdef BuildDecoupledExtractors
    % =========================================================================
    % 模組：BuildDecoupledExtractors.m (雙軌時空特徵萃取)
    % 升級：Phase 14.5 (★ 確立 Late Fusion 架構，引入 Transformer Attention 解決記憶衰退)
    % 職責：構建混合時序專家 (Attention+LSTM) 與空間專家 (GCN)，產出獨立的 [64, 878] 表徵
    % =========================================================================
    
    properties
        ConfigObj        % 全域設定檔參考
        NumTickers       % 大宇宙清單標的數 (如 878)
        SeqLen           % 時序特徵回溯長度 (如 60 天)
        EmbedDim         % 降噪表徵瓶頸維度 (鎖定 64 維)
        
        TotalNodeFeats   % 單一節點的特徵總數 (Relative 3 + Micro 15 + Macro 4 = 22)
        FlattenedFeatDim % 展平後的空間特徵維度 (22 * 878 = 19316)
        FlattenedAdjDim  % 展平後的鄰接矩陣維度 (878 * 878 = 770884)
    end
    
    methods
        % ---------------------------------------------------------
        % 建構子：初始化工廠參數與維度斷言
        % ---------------------------------------------------------
        function obj = BuildDecoupledExtractors(config)
            disp(' ⚙️ [NetworkFactory] 啟動雙軌特徵萃取器構建工廠...');
            
            obj.ConfigObj = config;
            obj.NumTickers = config.NumTickers;
            obj.SeqLen = config.SeqLen;
            obj.EmbedDim = 64; % 嚴格鎖定 64 維 Embedding，保留足夠高階語意給後續 CIO 決策
            
            % ---------------------------------------------------------
            % 1. 維度嚴格對齊 (Alignment Assertion)
            % ---------------------------------------------------------
            numRel = 3; % 相對大盤特徵 (Beta, Corr, RS)
            numMicro = config.NumMicroFeatures; % 15 維微觀特徵
            numMacro = config.NumMacroFeatures; % 4 維宏觀特徵
            
            % 計算單一節點特徵數 (確保為 22)
            obj.TotalNodeFeats = numRel + numMicro + numMacro;
            
            % 空間專家的 Input Layer 需要接收展平後的 2D 張量
            % [警告] 這是 MATLAB 框架限制下的 Dense 展平，內部需依賴客製化 Layer 解開以防 OOM
            obj.FlattenedFeatDim = obj.TotalNodeFeats * obj.NumTickers;
            obj.FlattenedAdjDim = obj.NumTickers * obj.NumTickers;
            
            fprintf('  -> 預期輸入設定：\n');
            fprintf('     [時序專家] 單一節點特徵: %d 維 | 序列長度: %d (Time-Distributed 共享權重)\n', obj.TotalNodeFeats, obj.SeqLen);
            fprintf('     [空間專家] 展平特徵維度: %d | 展平圖譜維度: %d (Late Fusion 拓樸)\n', obj.FlattenedFeatDim, obj.FlattenedAdjDim);
        end
        
        % =========================================================
        % 核心建構引擎：產出雙軌神經網路
        % =========================================================
        function [net_time, net_space] = buildNetworks(obj)
            % ---------------------------------------------------------
            % 1. 構建時序專家 (Time Expert) - Transformer + LSTM 混合架構
            % ---------------------------------------------------------
            % 物理意義：這是一個 Time-Distributed 的混合網路。
            % 透過外部迴圈將維度壓扁為 (B*N, SeqLen, Feats)，迫使模型學習「放諸四海皆準」
            % 的普適動能法則，而非死背個別股票代碼，杜絕過擬合。
            
            layers_time = [
                % 接收 [22, 60] 的時間序列特徵
                sequenceInputLayer(obj.TotalNodeFeats, 'Name', 'in_time')
                
                % ★ 核心修復 1：特徵維度投影 (Projection)
                % 將原始 22 維特徵投影至 128 維，這是為了完美整除後方的 4 個 Attention Heads
                fullyConnectedLayer(128, 'Name', 'proj_fc')
                
                % ★ 核心修復 2：引入 Multi-Head Self-Attention 機制 (Transformer 靈魂)
                % 4 個 Heads，每個 Head 負責 32 維 (4 * 32 = 128)。
                % 讓模型能無視時間距離，動態回顧 60 天內最重要的轉折點 (如財報發布、跳空缺口)
                selfAttentionLayer(4, 32, 'Name', 'self_attn')
                
                % ★ 核心修復 3：混合 LSTM 進行時序記憶壓縮
                % Attention 負責抓取「全域重點」，LSTM 負責梳理「時序因果律」。
                % 'OutputMode' 設為 'last'，將 60 天的序列徹底壓縮為一個靜態的 128 維向量
                lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm_1')
                
                % 將隱藏層壓縮至 64 維的特徵瓶頸 (Embedding)
                fullyConnectedLayer(obj.EmbedDim, 'Name', 'E_time') 
            ];
            
            % 將 Layer 陣列轉為 dlnetwork 以支援自定義訓練迴圈
            net_time = dlnetwork(layers_time);
            disp('✅ 時序專家網路拓樸 (Transformer-LSTM Hybrid) 構建完畢。');
            
            % ---------------------------------------------------------
            % 2. 構建空間專家 (Space Expert) - DyGAT 動態圖注意力網路
            % ---------------------------------------------------------
            lgraph_space = layerGraph();
            
            % 2.1 雙輸入節點 (接收全市場當日的橫截面特徵與動態相關圖譜)
            feat_input = featureInputLayer(obj.FlattenedFeatDim, 'Name', 'in_space_feat');
            adj_input  = featureInputLayer(obj.FlattenedAdjDim, 'Name', 'in_space_adj');
            
            % 2.2 核心自定義 GAT 卷積層
            % ★ 核心設計：拔除 GAP (Global Average Pooling)。
            % 它內部重塑回 3D 後，直接輸出 [64, 878, B] 的節點級別表徵矩陣，
            % 確保下游決策網路能精準分辨出「每一檔獨立的股票」，而非全市場的模糊均值。
            gat_layer = GraphSpatialFusionLayer('gat_1', obj.NumTickers, obj.EmbedDim);
            
            % 2.3 組合與連線 (Wiring)
            lgraph_space = addLayers(lgraph_space, feat_input);
            lgraph_space = addLayers(lgraph_space, adj_input);
            lgraph_space = addLayers(lgraph_space, gat_layer);
            
            % 將兩個 Input Layer 連接到自定義 GAT 層的雙輸入埠 (in1, in2)
            lgraph_space = connectLayers(lgraph_space, 'in_space_feat', 'gat_1/in1');
            lgraph_space = connectLayers(lgraph_space, 'in_space_adj', 'gat_1/in2');
            
            % 將 LayerGraph 封裝為可支援 dlfeval 的 dlnetwork
            net_space = dlnetwork(lgraph_space);
            disp('✅ 空間專家網路拓樸 (DyGAT) 構建完畢 (雙輸入，展平 2D 輸出)。');
        end
    end
end