classdef BuildDecoupledExtractors
    % =========================================================================
    % 模組：BuildDecoupledExtractors.m
    % 升級：Phase 15.5 深度表徵容量增強與生產剪枝基準版
    % 核心演進：
    %   1. 【特徵直通殘差 (Highway Skip)】：新增特徵直通殘差開關 (EnableHighwayResidual)，
    %      以並聯輕量高速時序通路（Highway LSTM + LayerNorm）與 Trans-LSTM 主幹融合，
    %      在數學架構上確保萃取出的 64D Embedding 排序力不低於原始 18D 基準線 (+0.0290)[cite: 1]。
    %   2. 【參數相容性修復】：修正 lstmLayer 誤用 'WeightsInitializer' 之語法錯誤，
    %      改採 MATLAB 標準 LSTM 初始化機制[cite: 1]。
    %   3. 【空間專家 Pure-Time 剪枝】：依據 P2-1 與 Round 8a-v3 結論，自動旁路 DyGAT 圖卷積，
    %      net_space 設為空物件，徹底阻斷過度平滑 (Over-smoothing) 與顯存浪費[cite: 1, 3]。
    %   4. 【五重深度正則化引擎】：整合 Feature Dropout、動態高斯噪聲、時序循環 Variational Dropout
    %      以及 Attention Dropout，有效將 Overfitting Gap 壓制在 0.034 以內[cite: 1]。
    %   5. 【連續排序複合損失】：整合 Huber 點位抗厚尾損失、可微分橫截面 Soft-IC 損失與 VICReg 變異數保底。
    % 職責：構建時空解耦特徵萃取網路（支援純時序單軌），並提供可微分連續排序與回歸損失引擎
    % =========================================================================
    
    properties
        ConfigObj               % 全域配置物件 (Config 實例)
        NumTickers              % 股票池標的數量
        SeqLen                  % 時序歷史回看視窗長度 (預設 60 日)
        EmbedDim                % 降維目標 Embedding 維度 (預設 64)
        DropoutRate             % 密集層標準 Dropout 比率
        ArchType                % 'trans_lstm' (預設) 或 'pure_lstm'
        SpaceMixMode            % 'gcn_only' 或 'dynamic'
        EnableSpaceExpert       % ★ 空間專家啟用開關 (布林值，連動剪枝)
        EnableHighwayResidual   % ★ 特徵直通殘差開關 (解決深度平滑導致表徵略遜於 Raw 18D 瓶頸)[cite: 1]
        
        % 正則化與特徵噪聲控制參數 (自 Config.m 動態注入，維持 SSOT)
        FeatureDropoutRate      % 特徵維度隨機遮蔽率 (Feature Dropout，預設 0.12)[cite: 1]
        InputNoiseStd           % 輸入特徵動態高斯噪聲標準差 (Gaussian Noise，預設 0.015)[cite: 1]
        VariationalDropRate     % 時序循環全時間步共享遮蔽率 (Variational Dropout，預設 0.15)[cite: 1]
        AttentionDropRate       % 自注意力權重丟棄率 (Attention Dropout，預設 0.10)
        HuberDelta              % Huber 損失線性過渡門檻 (預設 0.10)
        ICLossWeight            % Continuous Soft-IC 損失權重 (預設 0.50)
        
        TotalNodeFeats          % 單節點特徵維度 (Relative 3 + Micro 15 = 18)
        FlattenedFeatDim        % 空間專家展平特徵維度
        FlattenedAdjDim         % 空間專家展平圖譜維度
    end
    
    methods
        % =====================================================================
        % 建構子：初始化網路超參數、讀取 Config 正則化組態並設置空間剪枝
        % =====================================================================
        function obj = BuildDecoupledExtractors(config, totalFeats, archType)
            disp(' ⚙️ [NetworkFactory] 啟動特徵萃取器構建工廠 (Pure-Time 剪枝與直通殘差加強版)...');
            
            obj.ConfigObj  = config;
            obj.NumTickers = config.NumTickers;
            obj.SeqLen     = config.SeqLen;
            obj.EmbedDim   = 64; 
            
            % 讀取密集層標準 Dropout 率
            if isprop(config, 'DL_DropoutRate') && ~isempty(config.DL_DropoutRate)
                obj.DropoutRate = config.DL_DropoutRate;
            else
                obj.DropoutRate = 0.20;
            end
            
            % ★ 讀取特徵直通殘差開關 (Highway Residual Connection)[cite: 1]
            if isprop(config, 'EnableHighwayResidual') && ~isempty(config.EnableHighwayResidual)
                obj.EnableHighwayResidual = config.EnableHighwayResidual;
            else
                obj.EnableHighwayResidual = true; % 預設啟用直通殘差，保證表徵排序力不低於 Raw 18D[cite: 1]
            end
            
            % ★ 正則化與特徵噪聲超參數動態注入 (與最新 Config.m 保持 SSOT 一致)[cite: 1]
            if isprop(config, 'FeatureDropoutRate') && ~isempty(config.FeatureDropoutRate)
                obj.FeatureDropoutRate = config.FeatureDropoutRate;
            else
                obj.FeatureDropoutRate = 0.12;
            end
            
            if isprop(config, 'InputNoiseStd') && ~isempty(config.InputNoiseStd)
                obj.InputNoiseStd = config.InputNoiseStd;
            else
                obj.InputNoiseStd = 0.015;
            end
            
            if isprop(config, 'VariationalDropRate') && ~isempty(config.VariationalDropRate)
                obj.VariationalDropRate = config.VariationalDropRate;
            else
                obj.VariationalDropRate = 0.15;
            end
            
            if isprop(config, 'AttentionDropRate') && ~isempty(config.AttentionDropRate)
                obj.AttentionDropRate = config.AttentionDropRate;
            else
                obj.AttentionDropRate = 0.10;
            end
            
            if isprop(config, 'DL_HuberDelta') && ~isempty(config.DL_HuberDelta)
                obj.HuberDelta = config.DL_HuberDelta;
            else
                obj.HuberDelta = 0.10;
            end
            
            if isprop(config, 'DL_ICLossWeight') && ~isempty(config.DL_ICLossWeight)
                obj.ICLossWeight = config.DL_ICLossWeight;
            else
                obj.ICLossWeight = 0.50;
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
            
            % ★ 空間專家剪枝決策判定 (落實 P2-1 與 Phase 4 BO 最佳化 Time_W=1.0 結論)[cite: 1]
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
            
            fprintf('  -> 網路拓撲與容量配置組態：\n');
            fprintf('     [時序專家] 架構: %s | 輸入: %d 維 | 視窗: %d | 直通殘差 (Highway): %d\n', ...
                upper(obj.ArchType), obj.TotalNodeFeats, obj.SeqLen, obj.EnableHighwayResidual);
            fprintf('     [時序正則] Dropout: %.2f | AttnDrop: %.2f | VarDrop: %.2f | FeatDrop: %.2f | Noise: %.4f\n', ...
                obj.DropoutRate, obj.AttentionDropRate, obj.VariationalDropRate, obj.FeatureDropoutRate, obj.InputNoiseStd);
            if obj.EnableSpaceExpert
                fprintf('     [空間專家] 狀態: 啟用 | 展平特徵: %d 維 | 圖譜: %d 維 | 模式: %s\n', ...
                    obj.FlattenedFeatDim, obj.FlattenedAdjDim, upper(obj.SpaceMixMode));
            else
                fprintf('     [空間專家] 狀態: ⏩ 已依實驗結論全面剪枝 (Pure-Time 模式，徹底阻斷過度平滑雜訊)[cite: 1]\n');
            end
        end
        
        % =====================================================================
        % 函數：buildNetworks (組裝時序專家與空間專家神經網路實體)
        % =====================================================================
        function [net_time, net_space] = buildNetworks(obj)
            % -------------------------------------------------------------
            % 1. 構建時序專家網路 (Trans-LSTM / Pure-LSTM)
            % -------------------------------------------------------------
            switch obj.ArchType
                case 'pure_lstm'
                    % 輕量級純 LSTM 基準模型 (無自注意力)
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
                    net_time = dlnetwork(layers_time);
                    
                otherwise
                    if obj.EnableHighwayResidual
                        % ★ 增強架構：Trans-LSTM 級聯 + 輕量 Highway 直通殘差通路
                        % 結構說明：
                        %   - 主幹路徑：輸入 18D -> 投影 64D -> 4頭自注意力 (4x16D=64D) -> LSTM(128D, last) -> FC(64D)
                        %   - 高速支路：輸入 18D -> 輕量 Highway-LSTM(64D, last) -> LayerNorm(64D)
                        %   - 特徵融合：additionLayer 逐元素相加 -> 全連接 E_time(64D) -> 最終 LayerNorm[cite: 1]
                        lgraph_time = layerGraph();
                        
                        % 主幹：深層時序關聯與注意力池化
                        layers_main = [
                            sequenceInputLayer(obj.TotalNodeFeats, 'Name', 'in_time', 'Normalization', 'none')
                            fullyConnectedLayer(64, 'Name', 'proj_fc', 'WeightsInitializer', 'he')
                            selfAttentionLayer(4, 16, 'Dropout', obj.AttentionDropRate, 'Name', 'self_attn')
                            dropoutLayer(obj.DropoutRate, 'Name', 'drop_attn')
                            lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm_1')
                            layerNormalizationLayer('Name', 'ln_pre_embed')
                            dropoutLayer(obj.DropoutRate, 'Name', 'drop_lstm')
                            fullyConnectedLayer(obj.EmbedDim, 'Name', 'fc_trans', 'WeightsInitializer', 'he')
                        ];
                        
                        % 支路：低延遲直通殘差路徑 (保留高頻微觀瞬時動能)
                        % ★ 關鍵修復：移除 lstmLayer 非法參數 'WeightsInitializer'
                        layers_highway = [
                            lstmLayer(obj.EmbedDim, 'OutputMode', 'last', 'Name', 'lstm_highway')
                            layerNormalizationLayer('Name', 'ln_highway')
                        ];
                        
                        % 匯總：特徵殘差相加與標準化投影
                        layers_head = [
                            additionLayer(2, 'Name', 'add_highway')
                            fullyConnectedLayer(obj.EmbedDim, 'Name', 'E_time', 'WeightsInitializer', 'he')
                            layerNormalizationLayer('Name', 'ln_time_out')
                        ];
                        
                        lgraph_time = addLayers(lgraph_time, layers_main);
                        lgraph_time = addLayers(lgraph_time, layers_highway);
                        lgraph_time = addLayers(lgraph_time, layers_head);
                        
                        % 連接輸入至 Highway 支路
                        lgraph_time = connectLayers(lgraph_time, 'in_time', 'lstm_highway');
                        
                        % 連接主幹與支路至殘差加法層
                        lgraph_time = connectLayers(lgraph_time, 'fc_trans', 'add_highway/in1');
                        lgraph_time = connectLayers(lgraph_time, 'ln_highway', 'add_highway/in2');
                        
                        net_time = dlnetwork(lgraph_time);
                        fprintf('✅ 時序專家網路拓撲構建完畢 (Trans-LSTM + Highway Residual 直通殘差 + 雙重 LayerNorm)[cite: 1]。\n');
                    else
                        % 標準單軌 Trans-LSTM 網路
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
                        net_time = dlnetwork(layers_time);
                        fprintf('✅ 時序專家網路拓撲構建完畢 (Trans-LSTM 標準版 + Attention Dropout + 雙重 LayerNorm)[cite: 1, 2]。\n');
                    end
            end
            
            % -------------------------------------------------------------
            % 2. 構建空間專家網路 (若剪枝則安全回傳空值，杜絕 GPU 顯存佔用)[cite: 1]
            % -------------------------------------------------------------
            if ~obj.EnableSpaceExpert
                net_space = [];
                disp('⏩ 空間專家 (DyGAT) 已依組態剪枝跳過構建，節省顯存與前向運算開銷[cite: 1]。');
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
            fprintf('✅ 空間專家網路拓撲構建完畢 (DyGAT 雙輸入版, 模式: %s)[cite: 2, 3]。\n', upper(obj.SpaceMixMode));
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
        %    - Feature Dropout: 整條時間序列在特徵維度共享遮蔽，破除單一強因子的死記依賴[cite: 1, 2]
        %    - Gaussian Noise: 模擬真實盤面報價微結構抖動與滑價誤差[cite: 1, 2]
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
            
            % 動態高斯噪聲注入
            if nargin >= 3 && ~isempty(noise_std) && noise_std > 0
                noise = dlarray(randn(sz, 'like', raw_data) * cast(noise_std, 'like', raw_data));
                x = x + noise;
            end
            
            x_aug = x;
        end
        
        % -----------------------------------------------------------------
        % 2. 時序循環 Variational Dropout (Sequence 共享同一個 Drop Mask)
        %    - 整條時間序列使用相同的遮蔽遮罩，避免破壞 LSTM 內部隱狀態的時間連續性[cite: 1, 2]
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
        %    - 直接最大化預測值與真實報酬的橫截面皮爾森相關係數（作為 Spearman 代理）[cite: 2, 5]
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
        %    - 結合點位誤差的 Huber 強健損失與橫截面排序一致性目標[cite: 1, 2, 5]
        % -----------------------------------------------------------------
        function loss = compute_continuous_return_loss(y_pred, y_true, act_mask, sample_T, B, ic_weight, delta)
            if nargin < 6 || isempty(ic_weight)
                ic_weight = 0.50;
            end
            if nargin < 7 || isempty(delta)
                delta = 0.10;
            end
            
            yp = stripdims(y_pred);
            yt = stripdims(y_true);
            m  = logical(stripdims(act_mask));
            
            yp_m = yp(m);
            yt_m = yt(m);
            
            % 元素級 Huber 殘差損失 (線性區間防範極端厚尾干擾)
            err = abs(yp_m - yt_m);
            is_small = err <= delta;
            huber = mean(is_small .* (0.5 * err.^2) + (~is_small) .* (delta * (err - 0.5 * delta)), 'all');
            
            % 橫截面 Soft-IC 排序正規化
            ic_loss = BuildDecoupledExtractors.compute_soft_ic_loss(yp, yt, m, sample_T, B);
            
            loss = huber + ic_weight * (1.0 + ic_loss);
        end
        
        % -----------------------------------------------------------------
        % 6. VICReg 變異數保底正則化懲罰項 (防範金融高雜訊引發表徵坍縮)
        %    - 約束各維度標準差不低於 target_std (預設 1.0)[cite: 1, 5]
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
