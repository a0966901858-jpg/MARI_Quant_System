function [t_stat, p_value, lrv] = hac_significance_test(x, max_lag)
    % Newey-West HAC 校正之單樣本顯著性檢定
    % 適用於日頻金融序列（IC序列、報酬序列、損失差序列）普遍存在的序列自相關
    x = x(:); x = x(~isnan(x));
    n = length(x);
    if nargin < 2
        max_lag = max(1, floor(4 * (n/100)^(2/9)));  % Newey-West(1994)自動頻寬
        % 注意：若特徵本身是長窗滾動構造(如SMA60)，建議手動將max_lag設為
        % 至少對齊該特徵的lookback長度，自動公式可能低估所需lag
    end
    x_mean = mean(x); x_dm = x - x_mean;
    gamma0 = mean(x_dm.^2);
    lrv = gamma0;
    for k = 1:max_lag
        w_k = 1 - k / (max_lag + 1);  % Bartlett kernel
        gamma_k = mean(x_dm(1:end-k) .* x_dm(1+k:end));
        lrv = lrv + 2 * w_k * gamma_k;
    end
    lrv = max(lrv, 1e-12);
    se_hac = sqrt(lrv / n);
    t_stat = x_mean / se_hac;
    p_value = 2 * (1 - normcdf(abs(t_stat)));
end
