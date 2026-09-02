classdef EMAQuantizer < handle
    % =========================================================================
    % 類別：EMAQuantizer (指數移動平均向量量化器)
    % 升級：Phase 14.3 (引入全域平滑死向量判定與浮點數防爆)
    % 職責：維護 VQ-VAE 的動態離散字典，負責穩定特徵降噪與防範 Codebook Collapse
    % =========================================================================
    
    properties
        LatentDim         % 潛在空間維度 (對齊 MARI 系統的 VQ_DLatent)
        NumCodes          % 字典向量總數 (對齊 MARI 系統的 VQ_KCodebook)
        Gamma             % EMA 衰減率 (預設 0.99，控制字典更新的平滑度)
        Epsilon           % 拉普拉斯平滑常數 (防止除以零的數值保護)
        
        Codebook          % [LatentDim, NumCodes] 的量化字典矩陣
        EMA_Cluster_Size  % [1, NumCodes] 記錄每個字典向量的 EMA 使用頻率
        EMA_Codebook_Sum  % [LatentDim, NumCodes] 記錄分配到該字典向量的特徵 EMA 總和
        
        IsFrozen = false      % 物理結界：在 Out-of-Sample 推論期必須為 true，禁止更新
        IsInitialized = false % 標記是否已進行「資料依賴初始化 (Data-Dependent Init)」
    end
    
    methods
        % ---------------------------------------------------------
        % 建構子：初始化量化器屬性與 GPU 友善的單精度矩陣
        % ---------------------------------------------------------
        function obj = EMAQuantizer(latentDim, numCodes, gamma)
            obj.LatentDim = latentDim;
            obj.NumCodes = numCodes;
            obj.Gamma = gamma;
            obj.Epsilon = 1e-5;
            
            % 預配置單精度陣列 (single)，不僅節省記憶體，也有利於後續 GPU 運算對齊
            obj.Codebook = zeros(latentDim, numCodes, 'single');
            obj.EMA_Cluster_Size = zeros(1, numCodes, 'single');
            obj.EMA_Codebook_Sum = zeros(latentDim, numCodes, 'single');
        end
        
        % ---------------------------------------------------------
        % 凍結與解凍機制 (防範未來數據洩漏 Look-ahead Bias)
        % ---------------------------------------------------------
        function freeze(obj)
            obj.IsFrozen = true;
            disp(' 🔒 [EMAQuantizer] 字典已物理凍結。禁止任何記憶體狀態寫入。');
        end
        
        function unfreeze(obj)
            obj.IsFrozen = false;
        end
        
        % =========================================================
        % 1. 前向推論與「真實資料依賴初始化 (Data-Dependent Init)」
        % =========================================================
        function [z_q, b_idx, perplexity] = forward(obj, z_e)
            % 剝除 dlarray 的神經網路計算圖標籤，還原為純數值矩陣，避免框架報錯
            if isa(z_e, 'dlarray')
                z_e_data = extractdata(z_e);
            else
                z_e_data = z_e;
            end
            
            % 取得當前 Batch 樣本數 (z_e_data 的維度應為 [LatentDim, B])
            B = size(z_e_data, 2);
            
            % ---------------------------------------------------------
            % 首次 Forward 攔截：執行真實資料依賴抽樣初始化
            % 避免 Codebook 初始為全零導致梯度一開始就困在鞍點
            % ---------------------------------------------------------
            if ~obj.IsInitialized
                fprintf(' 🌟 [EMAQuantizer] 首次 Forward 攔截：執行真實資料依賴抽樣初始化...\n');
                
                % 從當下這批真實特徵中，隨機抽取 NumCodes 個向量作為字典初始值 (允許重複抽樣)
                pick_idx = randsample(B, obj.NumCodes, true);
                initial_vectors = z_e_data(:, pick_idx);
                
                obj.Codebook = initial_vectors;
                obj.EMA_Codebook_Sum = initial_vectors;
                
                % 初始使用量設為 1，避免第一回合就被 EMA 機制判定為死向量
                obj.EMA_Cluster_Size = ones(1, obj.NumCodes, 'single'); 
                
                obj.IsInitialized = true;
            end
            
            % ---------------------------------------------------------
            % 計算編碼特徵與 Codebook 的平方歐式距離
            % 展開公式：(a - b)^2 = a^2 - 2ab + b^2
            % 維度廣播說明：
            % sum(z_e_data.^2, 1)'     -> [B, 1]
            % (z_e_data' * obj.Codebook) -> [B, LatentDim] * [LatentDim, NumCodes] = [B, NumCodes]
            % sum(obj.Codebook.^2, 1)  -> [1, NumCodes]
            % [B, 1] - [B, NumCodes] + [1, NumCodes] 會在 MATLAB R2016b+ 自動擴張(Broadcast)為 [B, NumCodes]
            % ---------------------------------------------------------
            dist = sum(z_e_data.^2, 1)' - 2 * (z_e_data' * obj.Codebook) + sum(obj.Codebook.^2, 1);
            
            % ★ 核心修復：浮點數防爆。強制將精度誤差導致的微小負數截斷為 0
            dist = max(dist, 0);
            
            % 尋找最近鄰字典向量 (Quantization)，b_idx 維度為 [B, 1]
            [~, b_idx] = min(dist, [], 2); 
            
            % 根據索引提取離散化後的特徵 z_q，維度為 [LatentDim, B]
            z_q = obj.Codebook(:, b_idx); 
            
            % 計算困惑度 (Perplexity, 衡量 Codebook 活躍分佈的資訊熵)
            % 數值越接近 NumCodes，代表字典的使用率越均勻，模型越健康
            counts = histcounts(b_idx, 0.5:(obj.NumCodes+0.5)); 
            probs = counts / B;
            valid_probs = probs(probs > 0);
            perplexity = exp(-sum(valid_probs .* log(valid_probs)));
        end
        
        % =========================================================
        % 2. EMA 更新與死向量喚醒 (Global Smoothing Reset)
        % =========================================================
        function update(obj, z_e, b_idx)
            % 若處於 OOS 盲測期或物理結界中，絕對禁止更新內部記憶狀態
            if obj.IsFrozen
                return;
            end
            
            if isa(z_e, 'dlarray')
                z_e_data = extractdata(z_e);
            else
                z_e_data = z_e;
            end
            
            B = size(z_e_data, 2);
            
            % ---------------------------------------------------------
            % 建立 One-Hot 編碼矩陣
            % 將 b_idx [B, 1] 轉換為 [NumCodes, B] 的稀疏分佈矩陣
            % ---------------------------------------------------------
            encodings = zeros(obj.NumCodes, B, 'single');
            linearIdx = sub2ind([obj.NumCodes, B], b_idx', 1:B);
            encodings(linearIdx) = 1;
            
            % 統計當前 Batch 每個 Codebook 向量的使用次數 [1, NumCodes]
            curr_cluster_size = sum(encodings, 2)'; 
            
            % 將分配到該 Codebook 的連續特徵 z_e 加總起來 [LatentDim, NumCodes]
            curr_codebook_sum = z_e_data * encodings'; 
            
            % ---------------------------------------------------------
            % 執行 EMA (Exponential Moving Average) 全域平滑衰減
            % 這是替代傳統梯度下降更新字典的關鍵，能大幅提升收斂穩定性
            % ---------------------------------------------------------
            obj.EMA_Cluster_Size = obj.Gamma * obj.EMA_Cluster_Size + (1 - obj.Gamma) * curr_cluster_size;
            obj.EMA_Codebook_Sum = obj.Gamma * obj.EMA_Codebook_Sum + (1 - obj.Gamma) * curr_codebook_sum;
            
            % ---------------------------------------------------------
            % ★ 核心修復：死向量喚醒 (Dead Vector Resuscitation)
            % 判定基準為「EMA 全域使用率」跌破 0.5，代表該特徵已被市場淘汰
            % ---------------------------------------------------------
            dead_threshold = 0.5;
            dead_indices = find(obj.EMA_Cluster_Size < dead_threshold);
            
            if ~isempty(dead_indices)
                num_dead = length(dead_indices);
                
                % ★ 教授修復 1：改用 randsample 並開啟 replace=true
                % 杜絕極端市場狀況下 Batch 樣本數(B) 小於 num_dead 導致的 randperm 崩潰
                rand_pick = randsample(B, num_dead, true);
                replacement_vectors = z_e_data(:, rand_pick);
                
                % ★ 教授修復 2：加入微小的高斯白雜訊 (Gaussian Noise)
                % 避免在大崩盤日抽到的特徵過度同質化，打亂空間以防止二次坍塌
                noise = randn(size(replacement_vectors), 'single') * 1e-4;
                replacement_vectors = replacement_vectors + noise;
                
                % 覆蓋死向量，給予它重生去探索新 Alpha 訊號的機會
                obj.Codebook(:, dead_indices) = replacement_vectors;
                obj.EMA_Codebook_Sum(:, dead_indices) = replacement_vectors;
                
                % 賦予基礎存活權重 (1.0)，避免下一回合又立刻被 EMA 判死刑
                obj.EMA_Cluster_Size(dead_indices) = 1.0; 
            end
            
            % ---------------------------------------------------------
            % 拉普拉斯平滑與最終字典更新
            % 加入 Epsilon 防止除以零，並更新物理字典矩陣
            % 公式：Codebook = EMA_Codebook_Sum / Smoothed_Cluster_Size
            % ---------------------------------------------------------
            n = sum(obj.EMA_Cluster_Size);
            smoothed_cluster_size = (obj.EMA_Cluster_Size + obj.Epsilon) ./ (n + obj.NumCodes * obj.Epsilon) * n;
            
            % 運用 MATLAB 的廣播機制，[LatentDim, NumCodes] 對應除以 [1, NumCodes]
            obj.Codebook = obj.EMA_Codebook_Sum ./ smoothed_cluster_size;
        end
    end
end