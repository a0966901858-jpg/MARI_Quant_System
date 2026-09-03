classdef GraphSpatialFusionLayer < nnet.layer.Layer
    % =========================================================================
    % 類別：GraphSpatialFusionLayer (時空圖神經網路動態/靜態空間拓撲融合卷積層)
    % 升級：Phase 15.5 Stage 4 規範版 (★ 強制單位自環 A + I_N 杜絕節點自身特徵抹除、
    %       Kipf-Welling 對稱正規化 D^-1/2 * A_tilde * D^-1/2、He 權重初始化、
    %       MixMode = 'gcn_only' | 'dynamic' 嚴格架構隔離)
    % 職責：對截面 N 檔標的之關聯圖譜進行空間資訊聚合，輸出節點級 EmbedDim 表徵
    % =========================================================================

    properties
        NumNodes            % 節點數量 (N 檔股票)
        EmbedDim            % 輸出表徵維度 (預設 64)
        NumFeatures         % 輸入特徵維度 (預設 18)
        MixMode             % 空間混合模式 ('gcn_only' | 'dynamic')
    end

    properties (Learnable)
        Weights             % 節點特徵轉換權重矩陣 [EmbedDim, NumFeatures]
        Bias                % 節點特徵偏置向量 [EmbedDim, 1]
        W_attn_1            % ASTGCN Query 轉換權重 [EmbedDim, NumFeatures]
        W_attn_2            % ASTGCN Key 轉換權重 [EmbedDim, NumFeatures]
        MixLogit            % 靜態圖與動態注意力融合權重純量 (Logit 空間)
    end

    methods
        function obj = GraphSpatialFusionLayer(name, numNodes, embedDim, numFeatures, mixMode)
            obj.Name = name;
            obj.Description = "Symmetric Normalized Spatial Graph Convolution with Self-Loop";
            
            if nargin < 5 || isempty(mixMode), mixMode = 'gcn_only'; end
            if nargin < 4 || isempty(numFeatures), numFeatures = 18; end
            if nargin < 3 || isempty(embedDim), embedDim = 64; end
            if nargin < 2 || isempty(numNodes), numNodes = 60; end

            obj.NumNodes = numNodes;
            obj.EmbedDim = embedDim;
            obj.NumFeatures = numFeatures;
            obj.MixMode = validatestring(mixMode, {'gcn_only', 'dynamic'});
            obj.Type = "Graph Spatial Fusion";

            % He (Kaiming Normal) 初始化權重
            std_w = sqrt(2.0 / numFeatures);
            obj.Weights  = dlarray(randn(embedDim, numFeatures, 'single') * std_w);
            obj.Bias     = dlarray(zeros(embedDim, 1, 'single'));
            obj.W_attn_1 = dlarray(randn(embedDim, numFeatures, 'single') * std_w);
            obj.W_attn_2 = dlarray(randn(embedDim, numFeatures, 'single') * std_w);
            
            % 初始化 MixLogit 偏向靜態 GCN (sigmoid(2.0) ≈ 0.88)
            obj.MixLogit = dlarray(single(2.0));
        end

        function Z = predict(obj, X_flat, A_flat)
            X = stripdims(X_flat);
            A = stripdims(A_flat);
            B = size(X, 2);

            % 1. 解壓縮為 3D 空間張量
            X_3D = reshape(X, obj.NumFeatures, obj.NumNodes, B);
            A_3D = reshape(A, obj.NumNodes, obj.NumNodes, B);

            % ★ 核心修復 1：顯式疊加單位自環 (Self-Loop: A_tilde = A + I_N)
            % 徹底解決節點自身技術特徵遺失 (Self-Feature Amnesia)
            I_N = eye(obj.NumNodes, 'like', A_3D);
            A_tilde = A_3D + I_N;

            % ★ 核心修復 2：Kipf-Welling 對稱拉普拉斯正規化 (D^-1/2 * A_tilde * D^-1/2)
            deg = sum(abs(A_tilde), 2); % [NumNodes, 1, B]
            deg_inv_sqrt = 1.0 ./ sqrt(deg + 1e-6);
            A_norm_3D = deg_inv_sqrt .* A_tilde .* permute(deg_inv_sqrt, [2, 1, 3]);

            % 2. 依據 MixMode 分支確定拓撲矩陣
            if strcmp(obj.MixMode, 'gcn_only')
                A_spatial = A_norm_3D;
            else
                % Dynamic 模式：ASTGCN 注意力機制
                Q = pagemtimes(obj.W_attn_1, X_3D);
                K = pagemtimes(obj.W_attn_2, X_3D);

                S_raw = pagemtimes(Q, 'transpose', K, 'none') / sqrt(single(obj.EmbedDim));

                % 拓撲遮罩（無邊處負無窮遮蔽，保留自環）
                mask_no_edge = (A_tilde == 0);
                S_raw(mask_no_edge) = single(-1e9);

                S_exp = exp(S_raw - max(S_raw, [], 1));
                S_attn = S_exp ./ (sum(S_exp, 1) + 1e-8);

                alpha = sigmoid(obj.MixLogit);
                A_spatial = alpha * A_norm_3D + (1 - alpha) * S_attn;
            end

            % 3. 鄰居特徵聚合 (X_3D: [F, N, B], A_spatial: [N, N, B])
            % pagemtimes(X_3D, A_spatial) 計算: Agg(:, j) = sum_k X(:, k) * A(k, j)
            Agg_3D = pagemtimes(X_3D, A_spatial);
            Agg_flat = reshape(Agg_3D, obj.NumFeatures, []);

            % 4. 線性轉換與激活
            Z_flat = obj.Weights * Agg_flat + obj.Bias;
            Z_flat = relu(Z_flat);

            % 5. 重組輸出張量 [EmbedDim * NumNodes, B]
            Z_3D_out = reshape(Z_flat, obj.EmbedDim, obj.NumNodes, B);
            Z_out = reshape(Z_3D_out, obj.EmbedDim * obj.NumNodes, B);

            Z = dlarray(Z_out, 'CB');
        end

        function [loss, dLdX, dLdA] = backward(~, ~, ~, ~, ~, ~)
            % 啟用自動微分
            loss = []; dLdX = []; dLdA = [];
        end
    end
end