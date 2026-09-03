% 定義 VQVAEAgent 類別，繼承自 handle (傳址參考類別)，確保網路權重更新時不會消耗額外記憶體
classdef VQVAEAgent < handle
    % =========================================================================
    % 類別：VQVAEAgent (向量量化降噪代理人)
    % 升級：Phase 14.20 (★ 核心修復：修正 STE 梯度穿透數值分離、徹底根絕表徵坍塌)
    % 職責：針對高頻動能特徵 (R1, R5, R20) 進行離散化降噪，消除市場微觀雜訊
    % =========================================================================
    
    % 宣告類別屬性 (Properties)
    properties
        ConfigObj       % 全域設定檔參考
        Encoder         % 編碼器神經網路 (將連續特徵壓縮至潛在空間)
        RandStream
        Decoder         % 解碼器神經網路 (將離散化後的潛在特徵還原)
        Quantizer       % 向量量化器 (負責維護 Codebook/字典，並將連續向量離散化)
        
        % Adam 優化器的狀態變數，用於儲存梯度的一階動量 (AvgG) 與二階動量 (AvgSG)
        AvgG_Enc = []; AvgSG_Enc = [];
        AvgG_Dec = []; AvgSG_Dec = [];
        Iter = 0;       % 全域訓練迭代次數計數器
        
        NumTargetFeats = 3;  % 目標降噪特徵的數量 (固定為 3 維：1日、5日、20日報酬率)
        TargetFeatIdx        % 目標特徵在 3D 面板張量中的通道索引
    end
    
    methods
        % 建構子：初始化代理人與外部依賴
        function obj = VQVAEAgent(configObj)
            obj.ConfigObj = configObj;
            % 映射到 FeatureEngineer 輸出的矩陣，微觀特徵 1~3 (R1, R5, R20) 對應 3D 張量的第 4 到 6 維通道
                obj.TargetFeatIdx = 4:6; 
            if nargin >= 2 && ~isempty(stream)
                obj.RandStream = stream;
            elseif ~isempty(configObj) && ismethod(configObj, 'getRandStream')
                obj.RandStream = configObj.getRandStream(1);
            else
                obj.RandStream = RandStream.getGlobalStream();
            end
            % 實例化外部的 EMA 向量量化器 (EMAQuantizer)
            % 傳入潛在空間維度 (DLatent)、字典大小 (KCodebook) 以及 EMA 平滑衰減係數 (Gamma)
            obj.Quantizer = EMAQuantizer(configObj.VQ_DLatent, ...
                                         configObj.VQ_KCodebook, ...
                                         configObj.VQ_Gamma);
            
            % 呼叫內部方法建立 Encoder 與 Decoder 的網路架構
            obj.build_networks();
        end
        
        % 建立神經網路架構
        function build_networks(obj)
            d_feat = obj.NumTargetFeats;          % 輸入特徵維度 (3)
            d_hid  = obj.ConfigObj.VQ_DHidden;    % 隱藏層神經元數量
            d_lat  = obj.ConfigObj.VQ_DLatent;    % 潛在空間維度 (Latent Space)
            
            % 構建 Encoder (編碼器)：特徵輸入 -> 全連接層 -> ReLU 激活 -> 全連接層輸出潛在向量 (z_e)
            encLayers = [
                featureInputLayer(d_feat, 'Name', 'in_enc')
                fullyConnectedLayer(d_hid, 'Name', 'enc_fc1'); reluLayer('Name', 'relu1')
                fullyConnectedLayer(d_lat, 'Name', 'enc_fc2')
            ];
            obj.Encoder = dlnetwork(encLayers); % 封裝為可支援自動微分的 dlnetwork 物件
            
            % 構建 Decoder (解碼器)：潛在特徵輸入 (z_q) -> 全連接層 -> ReLU 激活 -> 全連接層還原為原特徵
            decLayers = [
                featureInputLayer(d_lat, 'Name', 'in_dec')
                fullyConnectedLayer(d_hid, 'Name', 'dec_fc1'); reluLayer('Name', 'relu2')
                fullyConnectedLayer(d_feat, 'Name', 'dec_fc2')
            ];
            obj.Decoder = dlnetwork(decLayers); % 封裝為 dlnetwork 物件
            
            % 若系統環境支援 GPU 運算，則將網路參數送入顯示卡記憶體中加速
            if canUseGPU()
                obj.Encoder = dlupdate(@gpuArray, obj.Encoder);
                obj.Decoder = dlupdate(@gpuArray, obj.Decoder);
            end
        end
        
        % 模型預訓練函數
        % 強制傳入 Expert_Active_IS 作為訓練集真實遮罩，避免模型學習到停牌補 0 的無效特徵
        function train(obj, X_norm_3D_IS, Expert_Active_IS, epochs)
            if nargin < 4, epochs = 30; end % 預設訓練週期為 30 Epochs
            disp('--- 啟動 VQ-VAE 降噪器預訓練 (STE 梯度與 EMA 字典聯合更新) ---');
            
            % 從 3D 原始張量中，萃取出指定的 3 維高頻動能特徵
            X_target_3D = X_norm_3D_IS(:, obj.TargetFeatIdx, :);
            
            % 維度重構與展平：將 [Days, Feats, Tickers] 轉置為 [Feats, Days, Tickers]
            % 再展平為 [Feats, Days*Tickers] 的 2D 矩陣，符合深度學習 Batch 訓練格式
            X_flat = reshape(permute(X_target_3D, [2, 1, 3]), obj.NumTargetFeats, []);
            
            % 將活躍遮罩 Expert_Active_IS [Days, Tickers] 同步展平為 1D 布林陣列 [1, Days*Tickers]
            active_mask = reshape(Expert_Active_IS, 1, []);
            
            % 絕對結界：利用布林遮罩，僅提取「真實活躍且非停牌」的數據點進行訓練
            X_train = X_flat(:, active_mask);
            total_samples = size(X_train, 2); % 計算有效樣本總數
            
            % 防呆機制：確保訓練集非空
            if total_samples == 0
                error('❌ 致命錯誤：有效樣本數為 0！請檢查 Expert_Active 遮罩是否異常。');
            end
            
            fprintf(' -> 提取高頻動能特徵完畢。有效樣本數: %d 筆。\n', total_samples);
            
            batch_size = 4096; % 設定批次大小，大 Batch 有助於梯度穩定與矩陣運算加速
            num_iters = max(1, floor(total_samples / batch_size)); % 計算每個 Epoch 需要跑多少個 Iteration
            lr = 1e-3; % 設定 Adam 優化器的學習率
            obj.Iter = 0; % 重置迭代計數器
            beta_commit = 0.25; % Commitment Loss 權重
            
            % 開始 Epoch 迴圈
            for ep = 1:epochs
                idx_shuffle = randperm(s, total_samples);
                ep_recon_loss = 0;  % 記錄當前 Epoch 的總重構損失
                ep_commit_loss = 0; % 記錄當前 Epoch 的總承諾損失 (Commitment Loss)
                
                % 開始 Iteration 迴圈 (Batch 級別)
                for i = 1:num_iters
                    obj.Iter = obj.Iter + 1;
                    % 取出當前 Batch 的資料索引
                    batch_idx = idx_shuffle((i-1)*batch_size + 1 : min(i*batch_size, total_samples));
                    X_batch = X_train(:, batch_idx); % 擷取 Batch 實體資料
                    
                    % 轉換為 dlarray，'CB' 代表維度格式為 [Channel (特徵數), Batch (樣本數)]
                    X_dl = dlarray(X_batch, 'CB');
                    if canUseGPU(), X_dl = gpuArray(X_dl); end % 將 Batch 送上 GPU
                    
                    % 呼叫自定義的 dlfeval (允許自動微分求導的執行函數)
                    [~, recon_L, commit_L, gEnc, gDec, z_e_data, b_idx] = ...
                        dlfeval(@obj.vqvae_loss, obj.Encoder, obj.Decoder, obj.Quantizer, X_dl, beta_commit);
                    
                    % 梯度裁剪 (Gradient Clipping)，防止梯度爆炸
                    gEnc = dlupdate(@(g) obj.clip_gradient(g, 1.0), gEnc);
                    gDec = dlupdate(@(g) obj.clip_gradient(g, 1.0), gDec);
                    
                    % 使用 Adam 優化演算法更新 Encoder 與 Decoder 的網路權重
                    [obj.Encoder, obj.AvgG_Enc, obj.AvgSG_Enc] = adamupdate(obj.Encoder, gEnc, obj.AvgG_Enc, obj.AvgSG_Enc, obj.Iter, lr);
                    [obj.Decoder, obj.AvgG_Dec, obj.AvgSG_Dec] = adamupdate(obj.Decoder, gDec, obj.AvgG_Dec, obj.AvgSG_Dec, obj.Iter, lr);
                    
                    % 指數移動平均 (EMA) 字典更新
                    obj.Quantizer.update(z_e_data, b_idx);
                    
                    % 累加該 Batch 的損失以計算 Epoch 平均
                    ep_recon_loss = ep_recon_loss + recon_L;
                    ep_commit_loss = ep_commit_loss + commit_L;
                end
                
                % 每 10 個 Epoch 或第 1 個 Epoch 時，印出訓練進度與損失數值
                if mod(ep, 10) == 0 || ep == 1
                    fprintf('  -> Epoch %2d/%d | Recon Loss: %.4f | Commit Loss: %.4f\n', ...
                        ep, epochs, ep_recon_loss/num_iters, ep_commit_loss/num_iters);
                end
            end
            disp('✅ VQ-VAE 降噪器訓練與 EMA 字典收斂完成。');
        end
        
        % 前向推論與降噪函數
        % 強制傳入 Expert_Active 進行推論後的「物理結界抹除」
        function X_denoised_3D = denoise(obj, X_norm_3D, Expert_Active)
            disp('--- 執行全域特徵解耦前向推論 (VRAM Safe Vectorized Inference) ---');
            [numDays, ~, numTickers] = size(X_norm_3D);
            X_denoised_3D = X_norm_3D; % 先複製一份原始 3D 面板資料
            
            % 提取並展平目標特徵 (與 Train 階段相同邏輯)
            X_target_3D = X_norm_3D(:, obj.TargetFeatIdx, :);
            X_flat = reshape(permute(X_target_3D, [2, 1, 3]), obj.NumTargetFeats, []);
            
            % 在輸入神經網路前，將 NaN 或 Inf 強制補 0，避免深度學習前向傳播時報錯崩潰
            X_flat(isnan(X_flat) | isinf(X_flat)) = 0;
            total_items = size(X_flat, 2);
            X_recon_flat = zeros(size(X_flat), 'single'); % 預配置降噪後的容器
            
            % 啟動 VRAM 防爆分塊推論 (Chunking)
            chunk_size = 200000; 
            fprintf(' -> 啟動記憶體分塊推論 (Chunk Size: %d, 總筆數: %d)...\n', chunk_size, total_items);
            
            for start_idx = 1 : chunk_size : total_items
                end_idx = min(start_idx + chunk_size - 1, total_items);
                chunk_data = X_flat(:, start_idx:end_idx);
                
                % 轉換格式並送上 GPU
                X_dl = dlarray(chunk_data, 'CB');
                if canUseGPU(), X_dl = gpuArray(X_dl); end
                
                % 1. Encoder 推論：壓縮為連續潛在空間向量 z_e
                z_e = predict(obj.Encoder, X_dl);
                
                % 2. 量化器推論：尋找最近的字典編碼，輸出離散化向量 z_q
                [z_q_data, ~, ~] = obj.Quantizer.forward(z_e); 
                z_q_dl = dlarray(z_q_data, 'CB');
                if canUseGPU(), z_q_dl = gpuArray(z_q_dl); end
                
                % 3. Decoder 推論：將離散化後的乾淨潛在向量還原為實體特徵
                X_recon_dl = predict(obj.Decoder, z_q_dl);
                
                % 將推論結果從 GPU 拉回 CPU，並填入預配置的容器中
                X_recon_flat(:, start_idx:end_idx) = gather(extractdata(X_recon_dl));
            end
            
            % 將展平的 2D 降噪矩陣，重塑並轉置回原本的 3D 張量維度 [Days, Feats, Tickers]
            X_recon_3D = permute(reshape(X_recon_flat, obj.NumTargetFeats, numDays, numTickers), [2, 1, 3]);
            
            % 物理結界抹除：斬斷殭屍股復活
            inactive_mask_3D = repmat(reshape(~Expert_Active, [numDays, 1, numTickers]), [1, obj.NumTargetFeats, 1]);
            X_recon_3D(inactive_mask_3D) = 0;
            
            % 將降噪並清洗完畢的高頻特徵，寫回原面板資料對應的通道中
            X_denoised_3D(:, obj.TargetFeatIdx, :) = X_recon_3D;
            disp('✅ 高頻動能降噪與平滑趨勢拼裝完畢 (VRAM Safe & 零雜訊外洩)！');
        end
    end
    
    methods (Static)
        % =========================================================================
        % 函數：vqvae_loss (VQ-VAE 前向推論、量化、STE 梯度穿透與損失計算)
        % ★ 升級：Phase 14.20 (修正 STE 梯度穿透公式，確保 Encoder 完整接收 Recon Loss)
        % =========================================================================
        function [loss, reconLoss_val, commitLoss_val, gEnc, gDec, z_e_data, batch_indices] = vqvae_loss(encoder, decoder, quantizer, X, beta_commit)
            % 預設的 Commit Loss 權重
            if nargin < 5
                beta_commit = 0.25;
            end
            
            % ---------------------------------------------------------------------
            % 1. 特徵編碼 (Encoder Forward)
            % ---------------------------------------------------------------------
            z_e = forward(encoder, X); 
            
            % ---------------------------------------------------------------------
            % 2. 安全數值提取
            % ---------------------------------------------------------------------
            if isa(z_e, 'dlarray')
                z_e_data = extractdata(z_e); 
            else
                z_e_data = z_e;
            end
            
            % ---------------------------------------------------------------------
            % 3. 量化字典前向推論 (純數值運算)
            % ---------------------------------------------------------------------
            [z_q_data, batch_indices, ~] = quantizer.forward(z_e_data); 
            
            % 封裝 z_q 為 dlarray 用於計算 Commit Loss
            if isa(z_e, 'dlarray') && ~isempty(dims(z_e))
                z_q_dl = dlarray(z_q_data, dims(z_e));
            else
                z_q_dl = dlarray(z_q_data, 'CB');
            end
            if canUseGPU(), z_q_dl = gpuArray(z_q_dl); end
            
            % ---------------------------------------------------------------------
            % 4. ★ 核心修復：Straight-Through Estimator (STE) 梯度穿透
            % ---------------------------------------------------------------------
            % 差值必須在「已切斷追蹤的純數值」層級計算，再包回 dlarray 作為常數項。
            % 確保 d(diff_detached)/d(z_e) = 0，從而實現真正的 Stop-Gradient。
            diff_numeric = z_q_data - z_e_data; 
            
            if isa(z_e, 'dlarray') && ~isempty(dims(z_e))
                diff_detached = dlarray(diff_numeric, dims(z_e));
            else
                diff_detached = dlarray(diff_numeric, 'CB');
            end
            if canUseGPU(), diff_detached = gpuArray(diff_detached); end
            
            % STE 梯度直通表徵：
            % 前向數值：z_e + (z_q - z_e) = z_q (離散特徵)
            % 反向梯度：d(z_q_ste)/d(z_e) = 1 + 0 = 1 (重構梯度直通 Encoder)
            z_q_ste = z_e + diff_detached;
            
            % ---------------------------------------------------------------------
            % 5. 特徵解碼與損失計算 (Decoder Forward & Losses)
            % ---------------------------------------------------------------------
            X_recon = forward(decoder, z_q_ste);
            
            reconLoss = mean((X - X_recon).^2, 'all');
            commitLoss = beta_commit * mean((z_e - z_q_dl).^2, 'all');
            loss = reconLoss + commitLoss;
            
            % ---------------------------------------------------------------------
            % 6. 計算梯度與指標匯出
            % ---------------------------------------------------------------------
            [gEnc, gDec] = dlgradient(loss, encoder.Learnables, decoder.Learnables);
            
            % 安全匯出純數值，供外部紀錄日誌使用 (拔除計算圖，釋放 GPU 記憶體)
            reconLoss_val = extractdata(reconLoss);
            commitLoss_val = extractdata(commitLoss);
        end
        
        % 梯度裁剪函數 (防止梯度爆炸)
        function g_clipped = clip_gradient(g, threshold)
            g_norm = sqrt(sum(g.^2, 'all') + 1e-8);
            if g_norm > threshold
                g_clipped = g .* (threshold / g_norm);
            else
                g_clipped = g;
            end
        end
    end
end