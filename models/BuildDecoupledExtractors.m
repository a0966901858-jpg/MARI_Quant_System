classdef BuildDecoupledExtractors
    % =========================================================================
    % 模組：BuildDecoupledExtractors.m
    % 升級：Phase 15.5 生產基準版 (★ 支援空間專家 Pure-Time 剪枝開關、
    %       原生 'like' 語法 GPU/dlarray 無縫張量正則化、時序注意力維度對齊、
    %       Huber + Continuous Soft-IC 複合損失引擎、VICReg 變異數保底防坍縮)
    % 職責：構建時空解耦特徵萃取網路（支援純時序單軌），並提供可微分排序與回歸損失引擎
    % =========================================================================
    
    properties
        ConfigObj               % 全域配置物件 (Config 實例)
        NumTickers              % 股票池標的數量
        SeqLen                  % 時序歷史回看視窗長度
        EmbedDim                % 降維目標 Embedding 維度 (預設 64)
        DropoutRate             % 密集層標準 Dropout 比率
        ArchType                % 'trans_lstm' (預設) 或 'pure_lstm'
        SpaceMixMode            % 'gcn_only' 或 'dynamic'
        EnableSpaceExpert       % ★ 空間專家啟用開關 (布林值，連動剪枝)
        
        % 正則化與特徵噪聲控制參數 (自 Config.m 動態注入)
        FeatureDropoutRate      % 特徵維度隨機遮蔽率 (Feature Dropout)
        InputNoiseStd           % 輸入特徵動態高斯噪聲標準差 (Gaussian Noise)
        VariationalDropRate     % 時序循環全時間步共享遮蔽率 (Variational Dropout)
        AttentionDropRate       % 自注意力權重丟棄率 (Attention Dropout)
        HuberDelta              % Huber 損失線性過渡門檻
        ICLossWeight            % Continuous Soft-IC 損失權重
        
        TotalNodeFeats          % 單節點特徵維度 (Relative 3 + Micro 15 = 18)
        FlattenedFeatDim        % 空間專家展平特徵維度
        FlattenedAdjDim         % 空間專家展平圖譜維度
    end
    
    methods
        function obj = BuildDecoupledExtractors(config, totalFeats, archType)
            disp(' ⚙️ [NetworkFactory] 啟動特徵萃取器構建工廠 (支援 Pure-Time 剪枝與深度正則化)...');
            
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
            
            % ★ 讀取正則化與特徵噪聲超參數 (與 Config.m 保持 SSOT 一致)
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
            
            % 空間專家混合模式組態
            if isprop(config, 'SpaceExpertMixMode') && ~isempty(config.SpaceExpertMixMode)
                obj.SpaceMixMode = config.SpaceExpertMixMode;
            else
                obj.SpaceMixMode = 'gcn_only';
            end
            
            % ★ 空間專家剪枝決策判定 (落實消融實驗與貝氏最佳化結論)
            if isprop(config, 'EnableSpaceExpertTraining') && ~config.EnableSpaceExpertTraining
                obj.EnableSpaceExpert = false;
            elseif isprop(config, 'EnableSpaceExpert') && ~config.EnableSpaceExpert
                obj.EnableSpaceExpert = false;
            elseif strcmpi(obj.SpaceMixMode, 'none') || strcmpi(obj.SpaceMixMode, 'off')
                obj.EnableSpaceExpert = false;
            else
                obj.EnableSpaceExpert = true;
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
            
            fprintf('  -> 網路拓撲與正則化組態：\n');
            fprintf('     [時序專家] 架構: %s | 輸入: %d 維 | 視窗: %d | Dropout: %.2f | AttnDrop: %.2f\n', ...
                upper(obj.ArchType), obj.TotalNodeFeats, obj.SeqLen, obj.DropoutRate, obj.AttentionDropRate);
            if obj.EnableSpaceExpert
                fprintf('     [空間專家] 狀態: 啟用 | 展平特徵: %d 維 | 圖譜: %d 維 | 輸出: %d 維 | 模式: %s\n', ...
                    obj.FlattenedFeatDim, obj.FlattenedAdjDim, obj.EmbedDim, upper(obj.SpaceMixMode));
            else
                fprintf('     [空間專家] 狀態: ⏩ 已依實驗結論全面剪枝 (Pure-Time 模式，杜絕過度平滑)\n');
            end
            fprintf('     [動態增強] 特徵遮蔽 (FeatureDrop): %.2f | 噪聲注入 (NoiseStd): %.4f | 循環遮蔽 (VarDrop): %.2f\n', ...
                obj.FeatureDropoutRate, obj.InputNoiseStd, obj.VariationalDropRate);
        end
        
        function [net_time, net_space] = buildNetworks(obj)
            % 1. 構建時序專家網路
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
                    % Trans-LSTM: 輸入投影 64 維 -> 多頭自注意力 (4 頭 x 16D = 64D) -> LSTM -> 64D
                    layers_time = [
                        sequenceInputLayer(obj.TotalNodeFeats, 'Name', 'in_time', 'Normalization', 'none')
                        fullyConnectedLayer(64, 'Name', 'proj_fc', 'WeightsInitializer', 'he')
                        selfAttentionLayer(4, 16, 'Dropout', obj.AttentionDropRate, 'Name', 'self_attn')
                        dropoutLayer(obj.DropoutRate, 'Name', 'drop_attn')
                        lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm_1')
                        layerNormalizationLayer('Name', 'ln_pre_embed')
                        dropoutLayer(obj.DropoutRate, 'Name', 'drop_lstm')
                        fullyConnectedLayer(obj.EmbedDim, 'Name', 'E_time', 'WeightsInitializer', 'he') 
                        layerNormalizationLayer('Name', 'ln_time_out')
                    ];
            end
            net_time = dlnetwork(layers_time);
            fprintf('✅ 時序專家網路拓撲構建完畢 (%s + Attention Dropout + 雙重 LayerNorm)。\n', obj.ArchType);
            
            % 2. 構建空間專家網路 (若剪枝則安全回傳空值)
            if ~obj.EnableSpaceExpert
                net_space = [];
                disp('⏩ 空間專家 (DyGAT) 已依組態剪枝跳過構建，節省顯存與前向運算開銷。');
                return;
            end
            
            lgraph_space = layerGraph();
            feat_input = featureInputLayer(obj.FlattenedFeatDim, 'Name', 'in_space_feat');
            adj_input  = featureInputLayer(obj.FlattenedAdjDim, 'Name', 'in_space_adj');
            
            gat_layer = GraphSpatialFusionLayer('gat_1', obj.NumTickers, obj.EmbedDim, ...
                obj.TotalNodeFeats, obj.SpaceMixMode);
            
            lgraph_space = addLayers(lgraph_space, feat_input);
            lgraph_space = addLayers(lgraph_space, adj_input);
            lgraph_space = addLayers(lgraph_space, gat_layer);
            
            lgraph_space = connectLayers(lgraph_space, 'in_space_feat', 'gat_1/in1');
            lgraph_space = connectLayers(lgraph_space, 'in_space_adj', 'gat_1/in2');
            
            net_space = dlnetwork(lgraph_space);
            fprintf('✅ 空間專家網路拓撲構建完畢 (DyGAT 雙輸入版, 模式: %s)。\n', upper(obj.SpaceMixMode));
        end
        
        function [net_time, net_space] = build(obj)
            [net_time, net_space] = obj.buildNetworks();
        end
    end
    
    %% =====================================================================
    % 靜態方法：深度學習動態正則化與可微分連續排序損失引擎
    % =====================================================================
    methods (Static)
        % -----------------------------------------------------------------
        % 1. 輸入特徵層動態正則化 (Feature Dropout + 高斯動態噪聲)
        % -----------------------------------------------------------------
        function x_aug = apply_input_regularization(x, feat_drop_rate, noise_std, is_training)
            if nargin < 4 || ~is_training
                x_aug = x;
                return;
            end
            
            sz = size(x);
            numFeats = sz(1);
            batchSize = sz(2);
            raw_data = extractdata(x);
            
            % Feature Dropout: 以特徵維度為單位進行整條序列共享遮蔽
            if nargin >= 2 && ~isempty(feat_drop_rate) && feat_drop_rate > 0
                keep_prob = single(1.0 - feat_drop_rate);
                if length(sz) == 3
                    mask_raw = rand(numFeats, batchSize, 1, 'like', raw_data);
                else
                    mask_raw = rand(numFeats, batchSize, 'like', raw_data);
                end
                feat_mask = dlarray(cast(mask_raw > feat_drop_rate, 'like', raw_data) / keep_prob);
                x = x .* feat_mask;
            end
            
            % 動態高斯噪聲注入: 模擬盤面真實滑價與微結構抖動
            if nargin >= 3 && ~isempty(noise_std) && noise_std > 0
                noise = dlarray(randn(sz, 'like', raw_data) * cast(noise_std, 'like', raw_data));
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
            raw_data = extractdata(x);
            keep_prob = single(1.0 - drop_rate);
            
            if length(sz) == 3
                mask_raw = rand(sz(1), sz(2), 1, 'like', raw_data);
            else
                mask_raw = rand(sz, 'like', raw_data);
            end
            
            mask = dlarray(cast(mask_raw > drop_rate, 'like', raw_data) / keep_prob);
            x_drop = x .* mask;
        end
        
        % -----------------------------------------------------------------
        % 3. 每日橫截面 Soft-IC 損失 (可微分連續排序代理)
        % -----------------------------------------------------------------
        function loss = compute_soft_ic_loss(y_pred, y_true, act_mask, sample_T, B)
            yp = stripdims(y_pred);
            yt = stripdims(y_true);
            m  = logical(stripdims(act_mask));
            
            y_p_mat = reshape(yp, sample_T, B);
            y_t_mat = reshape(yt, sample_T, B);
            act_mat = reshape(m,  sample_T, B);
            
            ic_sum = 0;
            valid_days = 0;
            
            for b = 1:B
                m_b = act_mat(:, b);
                n_act = sum(m_b);
                if n_act >= 5
                    yp_b = y_p_mat(m_b, b);
                    yt_b = y_t_mat(m_b, b);
                    
                    yp_c = yp_b - mean(yp_b);
                    yt_c = yt_b - mean(yt_b);
                    
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
                loss = sum(yp, 'all') * 0;
            end
        end
        
        % -----------------------------------------------------------------
        % 4. 每日橫截面 Pairwise Ranking 損失 (排序間距優化)
        % -----------------------------------------------------------------
        function loss = compute_pairwise_ranking_loss(y_pred, y_true, act_mask, sample_T, B, margin)
            if nargin < 6 || isempty(margin)
                margin = 0.01;
            end
            
            yp = stripdims(y_pred);
            yt = stripdims(y_true);
            m  = logical(stripdims(act_mask));
            
            y_p_mat = reshape(yp, sample_T, B);
            y_t_mat = reshape(yt, sample_T, B);
            act_mat = reshape(m,  sample_T, B);
            
            total_rank_loss = 0;
            total_pairs = 0;
            
            for b = 1:B
                m_b = act_mat(:, b);
                n_act = sum(m_b);
                if n_act >= 4
                    yp_b = y_p_mat(m_b, b);
                    yt_b = y_t_mat(m_b, b);
                    
                    diff_t = yt_b - yt_b';
                    diff_p = yp_b - yp_b';
                    
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
                loss = sum(yp, 'all') * 0;
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
            
            yp = stripdims(y_pred);
            yt = stripdims(y_true);
            m  = logical(stripdims(act_mask));
            
            yp_m = yp(m);
            yt_m = yt(m);
            
            % 元素級 Huber 殘差損失
            err = abs(yp_m - yt_m);
            is_small = err <= delta;
            huber = mean(is_small .* (0.5 * err.^2) + (~is_small) .* (delta * (err - 0.5 * delta)), 'all');
            
            % 橫截面 Soft-IC 排序正規化
            ic_loss = BuildDecoupledExtractors.compute_soft_ic_loss(yp, yt, m, sample_T, B);
            
            loss = huber + ic_weight * (1.0 + ic_loss);
        end
        
        % -----------------------------------------------------------------
        % 6. VICReg 變異數保底正則化懲罰項 (防範金融高雜訊引發表徵坍縮)
        % -----------------------------------------------------------------
        function var_penalty = compute_vicreg_penalty(emb, target_std)
            if nargin < 2 || isempty(target_std)
                target_std = 1.0;
            end
            e = stripdims(emb);
            mu_dim = mean(e, 2);
            var_per_dim = mean((e - mu_dim).^2, 2);
            std_per_dim = sqrt(var_per_dim + 1e-4);
            var_penalty = mean(max(0, target_std - std_per_dim));
        end
    end
end
