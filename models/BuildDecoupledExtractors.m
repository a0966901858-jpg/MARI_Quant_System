classdef BuildDecoupledExtractors
    % =========================================================================
    % 模組：BuildDecoupledExtractors.m
    % 升級：Phase 15.5 Stage 2 (★ 排序型/回歸型損失函數庫、純量截面 Soft-IC、
    %       架構多態切換、LayerNorm 輸出鎖定、無 Linter 警告)
    % 職責：構建雙軌時空特徵萃取網路，並提供支援因果律訓練之可微分排序與回歸損失引擎
    % =========================================================================
    
    properties
        ConfigObj        
        NumTickers       
        SeqLen           
        EmbedDim         
        DropoutRate      
        ArchType         % 'trans_lstm' (預設) 或 'pure_lstm' (Round 8b 輕量版)
        
        TotalNodeFeats   % 單一節點特徵總數 (Relative 3 + Micro 15 = 18)
        FlattenedFeatDim 
        FlattenedAdjDim  
    end
    
    methods
        function obj = BuildDecoupledExtractors(config, totalFeats, archType)
            disp(' ⚙️ [NetworkFactory] 啟動雙軌特徵萃取器構建工廠 (Stage 2 排序/回歸支援版)...');
            
            obj.ConfigObj  = config;
            obj.NumTickers = config.NumTickers;
            obj.SeqLen     = config.SeqLen;
            obj.EmbedDim   = 64; 
            
            if isprop(config, 'DL_DropoutRate')
                obj.DropoutRate = config.DL_DropoutRate;
            else
                obj.DropoutRate = 0.2;
            end
            
            if nargin >= 3 && ~isempty(archType)
                obj.ArchType = validatestring(lower(archType), {'trans_lstm', 'pure_lstm'});
            else
                obj.ArchType = 'trans_lstm';
            end
            
            % 剝離 Macro 特徵，預設保留 18 維微觀/相對特徵
            if nargin >= 2 && ~isempty(totalFeats)
                obj.TotalNodeFeats = totalFeats;
            else
                numRel = 3;
                numMicro = config.NumMicroFeatures;
                obj.TotalNodeFeats = numRel + numMicro; % 18 維
            end
            
            obj.FlattenedFeatDim = obj.TotalNodeFeats * obj.NumTickers;
            obj.FlattenedAdjDim  = obj.NumTickers * obj.NumTickers;
            
            fprintf('  -> 網路拓撲設定：\n');
            fprintf('     [時序專家] 架構體制: %s | 單節點輸入: %d 維 | 序列長度: %d | Dropout: %.2f\n', ...
                upper(obj.ArchType), obj.TotalNodeFeats, obj.SeqLen, obj.DropoutRate);
            fprintf('     [空間專家] 展平特徵維度: %d | 展平圖譜維度: %d | 輸出維度: %d\n', ...
                obj.FlattenedFeatDim, obj.FlattenedAdjDim, obj.EmbedDim);
        end
        
        function [net_time, net_space] = buildNetworks(obj)
            % -------------------------------------------------------------
            % 1. 時序專家拓撲構建
            % -------------------------------------------------------------
            switch obj.ArchType
                case 'pure_lstm'
                    % Round 8b 驗證之輕量化無注意力架構 (節省 VRAM、無過擬合瓶頸)
                    layers_time = [
                        sequenceInputLayer(obj.TotalNodeFeats, 'Name', 'in_time', 'Normalization', 'none')
                        fullyConnectedLayer(64, 'Name', 'fc_in', 'WeightsInitializer', 'he')
                        reluLayer('Name', 'relu_in')
                        lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm_core')
                        layerNormalizationLayer('Name', 'ln_post_lstm')
                        dropoutLayer(obj.DropoutRate, 'Name', 'drop_lstm')
                        fullyConnectedLayer(obj.EmbedDim, 'Name', 'E_time', 'WeightsInitializer', 'he')
                        layerNormalizationLayer('Name', 'ln_time_out')
                    ];
                otherwise
                    % 原版 Transformer-LSTM 架構 (掛載雙層 LayerNorm)
                    layers_time = [
                        sequenceInputLayer(obj.TotalNodeFeats, 'Name', 'in_time')
                        fullyConnectedLayer(128, 'Name', 'proj_fc')
                        selfAttentionLayer(4, 32, 'Name', 'self_attn')
                        dropoutLayer(obj.DropoutRate, 'Name', 'drop_attn')
                        lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm_1')
                        layerNormalizationLayer('Name', 'ln_pre_embed')
                        dropoutLayer(obj.DropoutRate, 'Name', 'drop_lstm')
                        fullyConnectedLayer(obj.EmbedDim, 'Name', 'E_time') 
                        layerNormalizationLayer('Name', 'ln_time_out')
                    ];
            end
            net_time = dlnetwork(layers_time);
            fprintf('✅ 時序專家網路拓撲構建完畢 (%s + 雙重 LayerNorm)。\n', obj.ArchType);
            
            % -------------------------------------------------------------
            % 2. 空間專家拓撲構建 (雙輸入接口)
            % -------------------------------------------------------------
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
            disp('✅ 空間專家網路拓撲構建完畢 (DyGAT 雙輸入解耦版)。');
        end
        
        % 提供向後相容別名函式
        function [net_time, net_space] = build(obj)
            [net_time, net_space] = obj.buildNetworks();
        end
    end
    
    %% =====================================================================
    % 靜態方法：Stage 2 排序型 / 連續回歸型可微分損失引擎
    % =====================================================================
    methods (Static)
        % -----------------------------------------------------------------
        % 1. 每日橫截面 Soft-IC 損失 (Differentiable Cross-Sectional IC Loss)
        % 核心邏輯：在每個 Batch 內部按日分割，對當日活躍標的計算預測值與真實報酬之
        %           去中心化餘弦相似度 (Pearson IC)，回傳負均值以最大化選股相關性。
        % -----------------------------------------------------------------
        function loss = compute_soft_ic_loss(y_pred, y_true, act_mask, sample_T, B)
            y_p_mat = reshape(y_pred, sample_T, B);
            y_t_mat = reshape(y_true, sample_T, B);
            act_mat = reshape(act_mask, sample_T, B);
            
            ic_sum = dlarray(0, 'single');
            valid_days = 0;
            
            for b = 1:B
                m_b = act_mat(:, b);
                n_act = sum(m_b);
                if n_act >= 5
                    yp = y_p_mat(m_b, b);
                    yt = y_t_mat(m_b, b);
                    
                    yp_c = yp - mean(yp);
                    yt_c = yt - mean(yt);
                    
                    cov_xy  = sum(yp_c .* yt_c);
                    norm_xy = sqrt(sum(yp_c.^2) * sum(yt_c.^2) + 1e-6);
                    
                    ic_b = cov_xy / norm_xy;
                    ic_sum = ic_sum + ic_b;
                    valid_days = valid_days + 1;
                end
            end
            
            if valid_days > 0
                loss = -(ic_sum / valid_days); % 最小化負 IC 等價於最大化 IC
            else
                loss = dlarray(0, 'single');
            end
        end
        
        % -----------------------------------------------------------------
        % 2. 每日橫截面 Pairwise Ranking 損失 (RankNet Smooth Logistic Loss)
        % 核心邏輯：消除中位數硬切雜訊，若同一天內標的 i 實質跑贏標的 j，
        %           則推動預測分數差 (pred_i - pred_j) 向上，採用平滑 Softplus 梯度。
        % -----------------------------------------------------------------
        function loss = compute_pairwise_ranking_loss(y_pred, y_true, act_mask, sample_T, B, margin)
            if nargin < 6 || isempty(margin)
                margin = 0.01; % 報酬差值顯著性防死區門檻 (1%)
            end
            
            y_p_mat = reshape(y_pred, sample_T, B);
            y_t_mat = reshape(y_true, sample_T, B);
            act_mat = reshape(act_mask, sample_T, B);
            
            total_rank_loss = dlarray(0, 'single');
            total_pairs = 0;
            
            for b = 1:B
                m_b = act_mat(:, b);
                n_act = sum(m_b);
                if n_act >= 4
                    yp = y_p_mat(m_b, b);
                    yt = y_t_mat(m_b, b);
                    
                    % 構建截面配對差分矩陣
                    diff_t = yt - yt';
                    diff_p = yp - yp';
                    
                    % 僅對真實報酬差異顯著之對象計算排序損失 (避免雜訊對沖)
                    pair_mask = diff_t > margin;
                    n_pairs = sum(pair_mask, 'all');
                    
                    if n_pairs > 0
                        % RankNet 平滑損失: log(1 + exp(-(pred_i - pred_j)))
                        pair_losses = log(1 + exp(-diff_p(pair_mask)));
                        total_rank_loss = total_rank_loss + sum(pair_losses);
                        total_pairs = total_pairs + n_pairs;
                    end
                end
            end
            
            if total_pairs > 0
                loss = total_rank_loss / total_pairs;
            else
                loss = dlarray(0, 'single');
            end
        end
        
        % -----------------------------------------------------------------
        % 3. 複合連續回歸損失 (Huber Continuous Return + Soft-IC Regularizer)
        % 核心邏輯：直接以連續未來報酬為目標，融合 Huber 穩健振幅損失與
        %           橫截面單調性 (1 - IC) 正則項，兼顧極端值防禦與選股排序能力。
        % -----------------------------------------------------------------
        function loss = compute_continuous_return_loss(y_pred, y_true, act_mask, sample_T, B, ic_weight)
            if nargin < 6 || isempty(ic_weight)
                ic_weight = 0.5;
            end
            
            yp = y_pred(act_mask);
            yt = y_true(act_mask);
            yp = yp(:);
            yt = yt(:);
            
            % 1. 穩健 Huber 損失 (平滑 L1，過濾肥尾異常值)
            delta = 0.05; % 5% 報酬門檻
            err = abs(yp - yt);
            is_small = err <= delta;
            huber = mean(is_small .* (0.5 * err.^2) + (~is_small) .* (delta * (err - 0.5 * delta)));
            
            % 2. 截面 Soft-IC 損失
            ic_loss = BuildDecoupledExtractors.compute_soft_ic_loss(y_pred, y_true, act_mask, sample_T, B);
            
            loss = huber + ic_weight * (1.0 + ic_loss); % 使整體損失 >= 0
        end
    end
end