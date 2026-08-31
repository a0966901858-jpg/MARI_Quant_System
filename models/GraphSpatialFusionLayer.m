classdef GraphSpatialFusionLayer < nnet.layer.Layer & nnet.layer.Formattable
    % =========================================================================
    % 類別：GraphSpatialFusionLayer (動態圖卷積空間融合層)
    % 升級：Phase 14.24 (★ 動態特徵維度注入、Masked Softmax、凸組合拓撲融合)
    % 職責：結合圖拓撲遮罩與動態 Attention 執行鄰居聚合，自適應任意輸入特徵維度
    % =========================================================================
    
    properties
        NumNodes        % 股票數量 (例如 60 / 504)
        NumFeatures     % 單一節點特徵數 (由工廠動態注入，支援 22/24 維)
        EmbedDim        % 輸出表徵維度 (64)
    end
    
    properties (Learnable)
        Weights         % 卷積權重矩陣
        Bias            % 偏差向量
        W_attn_1        % 注意力 Query 權重
        W_attn_2        % 注意力 Key 權重
        MixLogit        % 凸組合動態混合權重 Logit
    end
    
    methods
        function obj = GraphSpatialFusionLayer(name, numNodes, embedDim, numFeatures)
            obj.Name = name;
            obj.NumNodes = numNodes;
            obj.EmbedDim = embedDim;
            
            % ★ 動態支援輸入特徵維度 (防禦硬編碼引發的 reshape 崩潰)
            if nargin >= 4 && ~isempty(numFeatures)
                obj.NumFeatures = numFeatures;
            else
                obj.NumFeatures = 22; % 預設兜底
            end
            
            obj.InputNames = {'in1', 'in2'}; 
            obj.Description = 'Masked Attention-Augmented Dynamic Graph Convolution';
            
            % 1. GCN 本體卷積參數 (He 初始化)
            fanIn = obj.NumFeatures;
            obj.Weights = dlarray(randn(obj.EmbedDim, obj.NumFeatures, 'single') * sqrt(2/fanIn));
            obj.Bias = dlarray(zeros(obj.EmbedDim, 1, 'single'));
            
            % 2. 空間注意力機制 (Xavier 初始化)
            attnDim = obj.NumFeatures; 
            obj.W_attn_1 = dlarray(randn(attnDim, obj.NumFeatures, 'single') * sqrt(1/obj.NumFeatures));
            obj.W_attn_2 = dlarray(randn(attnDim, obj.NumFeatures, 'single') * sqrt(1/obj.NumFeatures));
            
            % 3. 可學習拓撲混合權重 (初始為 sigmoid(0) = 0.5)
            obj.MixLogit = dlarray(single(0));
        end
        
        function Z = predict(obj, X_flat, A_flat)
            X = stripdims(X_flat);
            A = stripdims(A_flat);
            B = size(X, 2); 
            
            % 1. 解壓縮回 3D 空間張量 (依據動態 NumFeatures 重組)
            X_3D = reshape(X, obj.NumFeatures, obj.NumNodes, B);
            A_3D = reshape(A, obj.NumNodes, obj.NumNodes, B);
            
            % 2. 基礎度數正規化 (Column-Stochastic, dim=1)
            deg = sum(A_3D, 1); 
            deg(deg == 0) = 1;
            A_norm_3D = A_3D ./ deg; 
            
            % 3. ASTGCN 空間注意力引擎 (Masked Attention)
            Q = pagemtimes(obj.W_attn_1, X_3D);
            K = pagemtimes(obj.W_attn_2, X_3D);
            
            S_raw = pagemtimes(Q, 'transpose', K, 'none') / sqrt(obj.NumFeatures);
            
            % Masked Softmax：非真實圖邊強制賦予 -inf
            neg_inf = single(-1e9);
            mask = (A_3D == 0);
            S_raw(mask) = neg_inf;
            
            S_exp = exp(S_raw - max(S_raw, [], 1));
            S_attn = S_exp ./ (sum(S_exp, 1) + 1e-8);
            
            % 凸組合拓撲融合
            alpha_mix = sigmoid(obj.MixLogit);
            A_dynamic = alpha_mix * A_norm_3D + (1 - alpha_mix) * S_attn;
            
            % 4. 圖卷積鄰居聚合與映射
            Agg_3D = pagemtimes(X_3D, A_dynamic); 
            Agg_flat = reshape(Agg_3D, obj.NumFeatures, []); 
            Z_flat = obj.Weights * Agg_flat + obj.Bias;
            Z_flat = relu(Z_flat);
            
            % 5. 重組輸出並封裝
            Z_3D_out = reshape(Z_flat, obj.EmbedDim, obj.NumNodes, B);
            Z_out = reshape(Z_3D_out, obj.EmbedDim * obj.NumNodes, B);
            
            Z = dlarray(Z_out, 'CB');
        end
    end
end
