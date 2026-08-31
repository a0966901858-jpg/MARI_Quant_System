classdef BuildDecoupledExtractors
    % =========================================================================
    % 模組：BuildDecoupledExtractors.m
    % 升級：Phase 14.25 (★ 剝離 Macro 特徵，解決維度權重失衡導致的表徵坍縮)
    % 職責：僅使用 18 維微觀與相對特徵萃取 [64, NumTickers] 節點表徵
    % =========================================================================
    
    properties
        ConfigObj        
        NumTickers       
        SeqLen           
        EmbedDim         
        DropoutRate      
        
        TotalNodeFeats   % 單一節點特徵總數 (Relative 3 + Micro 15 = 18)
        FlattenedFeatDim 
        FlattenedAdjDim  
    end
    
    methods
        function obj = BuildDecoupledExtractors(config, totalFeats)
            disp(' ⚙️ [NetworkFactory] 啟動雙軌特徵萃取器構建工廠 (Macro 剝離版)...');
            
            obj.ConfigObj = config;
            obj.NumTickers = config.NumTickers;
            obj.SeqLen = config.SeqLen;
            obj.EmbedDim = 64; 
            
            if isprop(config, 'DL_DropoutRate')
                obj.DropoutRate = config.DL_DropoutRate;
            else
                obj.DropoutRate = 0.2;
            end
            
            % ★ Phase 14.25 核心修復：強制剝離 Macro 特徵，只保留會隨個股變動的 18 維
            if nargin >= 2 && ~isempty(totalFeats)
                obj.TotalNodeFeats = totalFeats;
            else
                numRel = 3;
                numMicro = config.NumMicroFeatures;
                obj.TotalNodeFeats = numRel + numMicro; % 18 維
            end
            
            obj.FlattenedFeatDim = obj.TotalNodeFeats * obj.NumTickers;
            obj.FlattenedAdjDim  = obj.NumTickers * obj.NumTickers;
            
            fprintf('  -> 預期輸入設定：\n');
            fprintf('     [時序專家] 單一節點特徵: %d 維 (已剝離總經) | 序列長度: %d | Dropout: %.2f\n', ...
                obj.TotalNodeFeats, obj.SeqLen, obj.DropoutRate);
            fprintf('     [空間專家] 展平特徵維度: %d | 展平圖譜維度: %d\n', ...
                obj.FlattenedFeatDim, obj.FlattenedAdjDim);
        end
        
        function [net_time, net_space] = buildNetworks(obj)
            layers_time = [
                sequenceInputLayer(obj.TotalNodeFeats, 'Name', 'in_time')
                fullyConnectedLayer(128, 'Name', 'proj_fc')
                selfAttentionLayer(4, 32, 'Name', 'self_attn')
                dropoutLayer(obj.DropoutRate, 'Name', 'drop_attn')
                lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm_1')
                dropoutLayer(obj.DropoutRate, 'Name', 'drop_lstm')
                fullyConnectedLayer(obj.EmbedDim, 'Name', 'E_time') 
            ];
            net_time = dlnetwork(layers_time);
            disp('✅ 時序專家網路拓撲構建完畢。');
            
            lgraph_space = layerGraph();
            feat_input = featureInputLayer(obj.FlattenedFeatDim, 'Name', 'in_space_feat');
            adj_input  = featureInputLayer(obj.FlattenedAdjDim, 'Name', 'in_space_adj');
            
            gat_layer = GraphSpatialFusionLayer('gat_1', obj.NumTickers, obj.EmbedDim, obj.TotalNodeFeats);
            
            lgraph_space = addLayers(lgraph_space, feat_input);
            lgraph_space = addLayers(lgraph_space, adj_input);
            lgraph_space = addLayers(lgraph_space, gat_layer);
            
            lgraph_space = connectLayers(lgraph_space, 'in_space_feat', 'gat_1/in1');
            lgraph_space = connectLayers(lgraph_space, 'in_space_adj', 'gat_1/in2');
            
            net_space = dlnetwork(lgraph_space);
            disp('✅ 空間專家網路拓撲構建完畢。');
        end
    end
end
