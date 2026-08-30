classdef GraphSpatialFusionLayer < nnet.layer.Layer & nnet.layer.Formattable
    % =========================================================================
    % 類別：GraphSpatialFusionLayer (動態圖卷積空間融合層)
    % 升級：Phase 14.20 (★ 修正度數正規化為 Column-Stochastic，對齊空間注意力與特徵聚合方向)
    % 職責：接收展平特徵，結合靜態度數正規化與動態 Attention 執行鄰居聚合
    % =========================================================================
    
    properties
        NumNodes        % 股票數量 (例如 878)
        NumFeatures     % 單一節點特徵數 (綁定 MARI 的 22 維)
        EmbedDim        % 輸出表徵維度 (例如 64)
    end
    
    properties (Learnable)
        Weights         % 卷積權重矩陣
        Bias            % 偏差向量
        W_attn_1        % 注意力 Query 權重
        W_attn_2        % 注意力 Key 權重
    end
    
    methods
        function obj = GraphSpatialFusionLayer(name, numNodes, embedDim)
            obj.Name = name;
            obj.NumNodes = numNodes;
            obj.NumFeatures = 22; 
            obj.EmbedDim = embedDim;
            
            % 明確宣告多輸入埠的名稱 (in1: 節點特徵, in2: 靜態圖譜)
            obj.InputNames = {'in1', 'in2'}; 
            
            % 學術正名：Hybrid Spatial Attention GCN
            obj.Description = 'Attention-Augmented Dynamic Graph Convolution';
            
            % 1. GCN 本體卷積參數 (Xavier 初始化)
            fanIn = obj.NumFeatures;
            obj.Weights = dlarray(randn(obj.EmbedDim, obj.NumFeatures, 'single') * sqrt(2/fanIn));
            obj.Bias = dlarray(zeros(obj.EmbedDim, 1, 'single'));
            
            % 2. ★ 空間注意力機制 (Spatial Attention) 參數
            % 刻意使用極小的初始值 (0.001)，讓模型初期依賴靜態圖譜，防禦梯度消失
            attnDim = obj.NumFeatures; 
            obj.W_attn_1 = dlarray(randn(attnDim, obj.NumFeatures, 'single') * 0.001);
            obj.W_attn_2 = dlarray(randn(attnDim, obj.NumFeatures, 'single') * 0.001);
        end
        
        function Z = predict(obj, X_flat, A_flat)
            % 剝除格式標籤，確保底層線性代數運算暢通
            X = stripdims(X_flat);
            A = stripdims(A_flat);
            B = size(X, 2); 
            
            % 1. 解壓縮回 3D 空間張量
            % X_3D: [Features(22), Nodes(878), Batch]
            % A_3D: [Nodes(878), Nodes(878), Batch]
            X_3D = reshape(X, obj.NumFeatures, obj.NumNodes, B);
            A_3D = reshape(A, obj.NumNodes, obj.NumNodes, B);
            
            % 2. ★ 計畫書修正 (問題 5-1 🟡 P2)：基礎度數正規化 (Column-Stochastic)
            % 改為對「接收方向」(Column, dim=1) 進行度數加總與正規化，與 X * A 的聚合方向嚴格對齊
            deg = sum(A_3D, 1); 
            deg(deg == 0) = 1;  % 防止孤立節點導致除以零
            A_norm_3D = A_3D ./ deg; 
            
            % =========================================================
            % 3. ★ ASTGCN 空間注意力引擎 (Spatial Attention Engine)
            % =========================================================
            % A. 透過線性投影計算 Query 與 Key 
            % Q, K 維度: [AttnDim, Nodes, Batch]
            Q = pagemtimes(obj.W_attn_1, X_3D);
            K = pagemtimes(obj.W_attn_2, X_3D);
            
            % B. 計算未正規化的注意力分數 (Scaled Dot-Product)
            % Q^T * K -> 產出 [Nodes, Nodes, Batch] 關聯矩陣
            S_raw = pagemtimes(Q, 'transpose', K, 'none') / sqrt(obj.NumFeatures);
            
            % C. 手動 Softmax (對 Dimension 1 執行，維持 Column-Stochastic 性質)
            % 防止浮點數溢出，扣除最大值
            S_exp = exp(S_raw - max(S_raw, [], 1));
            S_attn = S_exp ./ (sum(S_exp, 1) + 1e-8);
            
            % D. 動態圖譜融合 (Hadamard Product)
            % 兩者皆為 Column-Stochastic，點乘後尺度受控，主動壓制 Ghost Connections
            A_dynamic = A_norm_3D .* S_attn;
            
            % =========================================================
            % 4. 圖卷積鄰居聚合與映射
            % =========================================================
            % 將特徵與「帶有注意力的動態圖譜」相乘：Agg(:, j) = Σ_i X(:, i) * A_dynamic(i, j)
            Agg_3D = pagemtimes(X_3D, A_dynamic); 
            
            % 準備線性映射 (展平節點特徵) -> [Features, Nodes * Batch]
            Agg_flat = reshape(Agg_3D, obj.NumFeatures, []); 
            
            % 執行特徵映射 -> [EmbedDim, Nodes * Batch]
            Z_flat = obj.Weights * Agg_flat + obj.Bias;
            
            % 非線性啟動
            Z_flat = relu(Z_flat);
            
            % 5. 維持真正的 Batch Size (B)，避開拓樸熔斷
            Z_3D_out = reshape(Z_flat, obj.EmbedDim, obj.NumNodes, B);
            Z_out = reshape(Z_3D_out, obj.EmbedDim * obj.NumNodes, B);
            
            % 手動貼回標籤 (CB)
            Z = dlarray(Z_out, 'CB');
        end
    end
end