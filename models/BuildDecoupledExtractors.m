classdef BuildDecoupledExtractors
    % =========================================================================
    % 模組：BuildDecoupledExtractors.m
    % 升級：Phase 15.5 正則化防禦與空間混合升級版 (★ 支援 Feature/Attention/Variational Dropout、
    %       輸入動態高斯噪聲注入、SpaceExpertMixMode 動態注入、GCN-only/Dynamic 雙模式切換、
    %       Huber + Continuous Soft-IC 複合損失引擎、VICReg 變異數保底防坍縮)
    % 職責：構建雙軌時空特徵萃取網路，並提供支援因果律訓練之可微分排序、回歸損失與資料正則化引擎
    % =========================================================================
    
    properties
        ConfigObj        
        NumTickers       
        SeqLen           
        EmbedDim         
        DropoutRate      
        ArchType         % 'trans_lstm' (預設) 或 'pure_lstm' (Round 8b 輕量版)
        SpaceMixMode     % 'gcn_only' (預設凍結注意力) 或 'dynamic' (動態注意力)
        
        % ★ 正則化與特徵噪聲控制參數 (自 Config.m 動態注入)
        FeatureDropoutRate   % 特徵維度隨機丟棄率 (Feature Dropout)
        InputNoiseStd        % 輸入特徵高斯動態噪聲標準差 (Gaussian Noise)
        VariationalDropRate  % 時序序列固定 Drop Mask 機率 (Variational Dropout)
        AttentionDropRate    % 自注意力權重丟棄率 (Attention Dropout)
        HuberDelta           % Huber 損失線性過渡門檻
        ICLossWeight         % Continuous Soft-IC 損失權重
        
        TotalNodeFeats   % 單一節點特徵總數 (Relative 3 + Micro 15 = 18)
        FlattenedFeatDim 
        FlattenedAdjDim  
    end
    
    methods
        function obj = BuildDecoupledExtractors(config, totalFeats, archType)
            disp(' ⚙️ [NetworkFactory] 啟動雙軌特徵萃取器構建工廠 (深度正則化與空間混合模式支援版)...');
            
            obj.ConfigObj  = config;
            obj.NumTickers = config.NumTickers;
            obj.SeqLen     = config.SeqLen;
            obj.EmbedDim   = 64; 
            
            % 讀取密集層標準 Dropout 率
            if isprop(config, 'DL_DropoutRate') && ~isempty(config.DL_DropoutRate)
                obj.DropoutRate = config.DL_DropoutRate;
            else
                obj.DropoutRate = 0.2;
            end
            
            % ★ 正則化與特徵噪聲超參數動態注入 (與 Config.m 嚴格同步)
            if isprop(config, 'FeatureDropoutRate') && ~isempty(config.FeatureDropoutRate)
                obj.FeatureDropoutRate = config.FeatureDropoutRate;
            else
                obj.FeatureDropoutRate = 0.15;
            end
            
            if isprop(config, 'InputNoiseStd') && ~isempty(config.InputNoiseStd)
                obj.InputNoiseStd = config.InputNoiseStd;
            else
                obj.InputNoiseStd = 0.02;
            end
            
            if isprop(config, 'VariationalDropRate') && ~isempty(config.VariationalDropRate)
                obj.VariationalDropRate = config.VariationalDropRate;
            else
                obj.VariationalDropRate = 0.20;
            end
            
            if isprop(config, 'AttentionDropRate') && ~isempty(config.AttentionDropRate)
                obj.AttentionDropRate = config.AttentionDropRate;
            else
                obj.AttentionDropRate = 0.10;
            end
            
            if isprop(config, 'DL_HuberDelta') && ~isempty(config.DL_HuberDelta)
                obj.HuberDelta = config.DL_HuberDelta;
            else
                obj.HuberDelta = 0.1;
            end
            
            if isprop(config, 'DL_ICLossWeight') && ~isempty(config.DL_ICLossWeight)
                obj.ICLossWeight = config.DL_ICLossWeight;
            else
                obj.ICLossWeight = 0.5;
            end
            
            % 架構模式選擇
            if nargin >= 3 && ~isempty(archType)
                obj.ArchType = validatestring(lower(archType), {'trans_lstm', 'pure_lstm'});
            else
                obj.ArchType = 'trans_lstm';
            end
            
            % 空間專家混合模式組態 (預設 'gcn_only')
            if isprop(config, 'SpaceExpertMixMode') && ~isempty(config.SpaceExpertMixMode)
                obj.SpaceMixMode = config.SpaceExpertMixMode;
            else
                obj.SpaceMixMode = 'gcn_only';
            end
            
            if nargin >= 2 && ~isempty(totalFeats)
                obj.TotalNodeFeats = totalFeats;
            else
                numRel = 3;
                numMicro = config.NumMicroFeatures;
                obj.TotalNodeFeats = numRel + numMicro; % 18 維
            end
            
            obj.FlattenedFeatDim = obj.TotalNodeFeats * obj.NumTickers;
            obj.FlattenedAdjDim  = obj.NumTickers * obj.NumTickers;
            
            fprintf('  -> 網路拓撲與正則化設定：\n');
            fprintf('     [時序專家] 架構: %s | 輸入: %d 維 | 序列長度: %d | Dropout: %.2f | AttnDrop: %.2f\n', ...
                upper(obj.ArchType), obj.TotalNodeFeats, obj.SeqLen, obj.DropoutRate, obj.AttentionDropRate);
            fprintf('     [空間專家] 展平特徵維度: %d | 展平圖譜維度: %d | 輸出維度: %d | 混合模式: %s\n', ...
                obj.FlattenedFeatDim, obj.FlattenedAdjDim, obj.EmbedDim, upper(obj.SpaceMixMode));
            fprintf('     [數據增強] 特徵遮蔽 (FeatureDrop): %.2f | 高斯噪聲 (NoiseStd): %.4f | 循環遮蔽 (VarDrop): %.2f\n', ...
                obj.FeatureDropoutRate, obj.InputNoiseStd, obj.VariationalDropRate);
        end
        
        function [net_time, net_space] = buildNetworks(obj)
            switch obj.ArchType
                case 'pure_lstm'
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
                    layers_time = [
                        sequenceInputLayer(obj.TotalNodeFeats, 'Name', 'in_time')
                        fullyConnectedLayer(128, 'Name', 'proj_fc')
                        % ★ 注入 Attention Dropout 抑制對歷史特定 K 線模式的死記硬背
                        selfAttentionLayer(4, 32, 'Dropout', obj.AttentionDropRate, 'Name', 'self_attn')
                        dropoutLayer(obj.DropoutRate, 'Name', 'drop_attn')
                        lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm_1')
                        layerNormalizationLayer('Name', 'ln_pre_embed')
                        dropoutLayer(obj.DropoutRate, 'Name', 'drop_lstm')
                        fullyConnectedLayer(obj.EmbedDim, 'Name', 'E_time') 
                        layerNormalizationLayer('Name', 'ln_time_out')
                    ];
            end
            net_time = dlnetwork(layers_time);
            fprintf('✅ 時序專家網路拓撲構建完畢 (%s + Attention Dropout + 雙重 LayerNorm)。\n', obj.ArchType);
            
            lgraph_space = layerGraph();
            feat_input = featureInputLayer(obj.FlattenedFeatDim, 'Name', 'in_space_feat');
            adj_input  = featureInputLayer(obj.FlattenedAdjDim, 'Name', 'in_space_adj');
            
            % 將 SpaceMixMode 顯式傳入 GraphSpatialFusionLayer
            gat_layer = GraphSpatialFusionLayer('gat_1', obj.NumTickers, obj.EmbedDim, ...
                obj.TotalNodeFeats, obj.SpaceMixMode);
            
            lgraph_space = addLayers(lgraph_space, feat_input);
            lgraph_space = addLayers(lgraph_space, adj_input);
            lgraph_space = addLayers(lgraph_space, gat_layer);
            
            lgraph_space = connectLayers(lgraph_space, 'in_space_feat', 'gat_1/in1');
            lgraph_space = connectLayers(lgraph_space, 'in_space_adj', 'gat_1/in2');
            
            net_space = dlnetwork(lgraph_space);
            fprintf('✅ 空間專家網路拓撲構建完畢 (DyGAT 雙輸入解耦版, 模式: %s)。\n', upper(obj.SpaceMixMode));
        end
        
        function [net_time, net_space] = build(obj)
            [net_time, net_space] = obj.buildNetworks();
        end
    end
    
    %% =====================================================================
    % 靜態方法：深度學習動態正則化與特徵資料增強引擎
    % =====================================================================
    methods (Static)
        % -----------------------------------------------------------------
        % 1. 輸入特徵層正則化 (Feature Dropout + 動態高斯噪聲)
        % -----------------------------------------------------------------
        function x_aug = apply_input_regularization(x, feat_drop_rate, noise_std, is_training)
            if nargin < 4 || ~is_training
                x_aug = x;
                return;
            end
            
            sz = size(x);
            numFeats = sz(1);
            batchSize = sz(2);
            
            % Feature Dropout: 以特徵欄位為單位進行隨機遮蔽 (維持整條 Sequence 一致)
            if nargin >= 2 && ~isempty(feat_drop_rate) && feat_drop_rate > 0
                if length(sz) == 3
                    feat_mask = single(rand(numFeats, batchSize, 1) > feat_drop_rate) / (1.0 - feat_drop_rate);
                else
                    feat_mask = single(rand(numFeats, batchSize) > feat_drop_rate) / (1.0 - feat_drop_rate);
                end
                if isgpuarray(x)
                    feat_mask = gpuArray(feat_mask);
                end
                x = x .* feat_mask;
            end
            
            % 動態高斯噪聲注入: 模擬盤面真實滑價與微結構抖動
            if nargin >= 3 && ~isempty(noise_std) && noise_std > 0
                if isgpuarray(x)
                    noise = gpuArray.randn(size(x), 'single') * single(noise_std);
                else
                    noise = randn(size(x), 'single') * single(noise_std);
                end
                x = x + noise;
            end
            
            x_aug = x;
        end
        
        % -----------------------------------------------------------------
        % 2. 時序循環 Variational Dropout (Sequence 共享同一個 Drop Mask)
        % -----------------------------------------------------------------
        function x_drop = apply_variational_dropout(x, drop_rate, is_training)
            if nargin < 3 || ~is_training || isempty(drop_rate) || drop_rate <= 0
                x_drop = x;
                return;
            end
            
            sz = size(x);
            if length(sz) == 3
                % 廣播至所有時間步 [Dim, Batch, 1]
                mask = single(rand(sz(1), sz(2), 1) > drop_rate) / (1.0 - drop_rate);
            else
                mask = single(rand(sz) > drop_rate) / (1.0 - drop_rate);
            end
            
            if isgpuarray(x)
                mask = gpuArray(mask);
            end
            x_drop = x .* mask;
        end
        
        % -----------------------------------------------------------------
        % 3. 每日橫截面 Soft-IC 損失 (可微分連續排序代理)
        % -----------------------------------------------------------------
        function loss = compute_soft_ic_loss(y_pred, y_true, act_mask, sample_T, B)
            if isgpuarray(y_pred) && ~isgpuarray(y_true)
                y_true = gpuArray(y_true);
            end
            
            y_p_mat = reshape(y_pred, sample_T, B);
            y_t_mat = reshape(y_true, sample_T, B);
            act_mat = reshape(act_mask, sample_T, B);
            
            ic_sum = 0;
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
                loss = -(ic_sum / valid_days);
            else
                loss = sum(y_pred, 'all') * 0;
            end
        end
        
        % -----------------------------------------------------------------
        % 4. 每日橫截面 Pairwise Ranking 損失 (排序間距優化)
        % -----------------------------------------------------------------
        function loss = compute_pairwise_ranking_loss(y_pred, y_true, act_mask, sample_T, B, margin)
            if nargin < 6 || isempty(margin)
                margin = 0.01;
            end
            if isgpuarray(y_pred) && ~isgpuarray(y_true)
                y_true = gpuArray(y_true);
            end
            
            y_p_mat = reshape(y_pred, sample_T, B);
            y_t_mat = reshape(y_true, sample_T, B);
            act_mat = reshape(act_mask, sample_T, B);
            
            total_rank_loss = 0;
            total_pairs = 0;
            
            for b = 1:B
                m_b = act_mat(:, b);
                n_act = sum(m_b);
                if n_act >= 4
                    yp = y_p_mat(m_b, b);
                    yt = y_t_mat(m_b, b);
                    
                    diff_t = yt - yt';
                    diff_p = yp - yp';
                    
                    pair_mask = diff_t > margin;
                    n_pairs = sum(pair_mask, 'all');
                    
                    if n_pairs > 0
                        pair_losses = log(1 + exp(-diff_p(pair_mask)));
                        total_rank_loss = total_rank_loss + sum(pair_losses, 'all');
                        total_pairs = total_pairs + n_pairs;
                    end
                end
            end
            
            if total_pairs > 0
                loss = total_rank_loss / total_pairs;
            else
                loss = sum(y_pred, 'all') * 0;
            end
        end
        
        % -----------------------------------------------------------------
        % 5. 複合連續回歸損失 (Huber Continuous Return + Soft-IC Regularizer)
        % -----------------------------------------------------------------
        function loss = compute_continuous_return_loss(y_pred, y_true, act_mask, sample_T, B, ic_weight, delta)
            if nargin < 6 || isempty(ic_weight)
                ic_weight = 0.5;
            end
            if nargin < 7 || isempty(delta)
                delta = 0.1;
            end
            if isgpuarray(y_pred) && ~isgpuarray(y_true)
                y_true = gpuArray(y_true);
            end
            
            yp = y_pred(act_mask);
            yt = y_true(act_mask);
            yp = yp(:);
            yt = yt(:);
            
            err = abs(yp - yt);
            is_small = err <= delta;
            huber = mean(is_small .* (0.5 * err.^2) + (~is_small) .* (delta * (err - 0.5 * delta)), 'all');
            
            ic_loss = BuildDecoupledExtractors.compute_soft_ic_loss(y_pred, y_true, act_mask, sample_T, B);
            
            loss = huber + ic_weight * (1.0 + ic_loss);
        end
        
        % -----------------------------------------------------------------
        % 6. VICReg 變異數保底正則化懲罰項 (防範金融高雜訊引發表徵坍縮)
        % -----------------------------------------------------------------
        function var_penalty = compute_vicreg_penalty(emb, target_std)
            if nargin < 2 || isempty(target_std)
                target_std = 1.0;
            end
            mu_dim = mean(emb, 2);
            var_per_dim = mean((emb - mu_dim).^2, 2);
            std_per_dim = sqrt(var_per_dim + 1e-4);
            var_penalty = mean(max(0, target_std - std_per_dim));
        end
    end
end
