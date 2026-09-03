classdef loss_functions
    methods (Static)
        % 方案 A：可微分 Pearson / Spearman 近似 IC 損失 (最大化 IC 等價於最小化負相關)
        function loss = negative_ic_loss(y_pred, y_true, act_mask)
            yp = y_pred(act_mask);
            yt = y_true(act_mask);
            
            yp = yp(:);
            yt = yt(:);
            
            % 中心化
            yp_c = yp - mean(yp);
            yt_c = yt - mean(yt);
            
            % 餘弦相似度 / 相關係數
            cov_xy = sum(yp_c .* yt_c);
            norm_xy = sqrt(sum(yp_c.^2) * sum(yt_c.^2) + 1e-8);
            
            ic = cov_xy / norm_xy;
            loss = -ic; % 最小化負 IC
        end
        
        % 方案 B：Pairwise Margin Ranking Loss (配對排序損失)
        function loss = pairwise_ranking_loss(y_pred, y_true, act_mask, margin)
            if nargin < 4, margin = 0.05; end
            yp = y_pred(act_mask);
            yt = y_true(act_mask);
            
            yp = yp(:);
            yt = yt(:);
            
            % 構建截面配對差分矩陣: (y_i - y_j)
            diff_true = yt - yt';
            diff_pred = yp - yp';
            
            % 只保留真實報酬有顯著區隔的對象標籤 (sign 矩陣)
            target_sign = sign(diff_true);
            valid_pairs = abs(diff_true) > 1e-4;
            
            % Hinge-like ranking error: max(0, -sign * (pred_i - pred_j) + margin)
            pair_loss = max(0, -target_sign .* diff_pred + margin);
            loss = sum(pair_loss(valid_pairs)) / max(1, sum(valid_pairs(:)));
        end
    end
end