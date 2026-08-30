function [loss, reconLoss_val, commitLoss_val, gEnc, gDec, z_e_data, batch_indices] = vqvaeLoss(encoder, decoder, quantizer, X, beta_commit)
% =========================================================================
% 函數：vqvaeLoss (VQ-VAE 前向推論、量化、STE 梯度穿透與損失計算)
% 升級：Phase 14.6 (★ 解決架構耦合，完美呼叫 EMAQuantizer 並實作安全 STE)
% =========================================================================

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
    % 3. ★ 核心修正：物件封裝呼叫 (Encapsulation)
    % 絕對禁止徒手算距離！直接呼叫我們建構好的 EMAQuantizer，
    % 享受它的「資料依賴初始化」與「浮點數防爆」機制。
    % ---------------------------------------------------------------------
    [z_q_data, batch_indices, ~] = quantizer.forward(z_e_data); 
    
    % ---------------------------------------------------------------------
    % 4. 實作 Straight-Through Estimator (STE) 梯度穿透
    % ---------------------------------------------------------------------
    % 將量化後的數值轉回帶有維度標籤的 dlarray
    if isa(z_e, 'dlarray') && ~isempty(dims(z_e))
        z_q_dl = dlarray(z_q_data, dims(z_e));
    else
        z_q_dl = dlarray(z_q_data, 'CB');
    end
    
    % STE 核心魔法：
    % 前向傳播時，數值等於 z_q (離散特徵)。
    % 反向傳播時，梯度會繞過不可微的 z_q，直接流向 z_e (連續特徵)。
    diff_dl = z_q_dl - z_e;
    z_q_ste = z_e + stripdims(diff_dl); % 必須 stripdims 才能與原張量相加
    
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