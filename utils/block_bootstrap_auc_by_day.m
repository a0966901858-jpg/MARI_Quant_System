function [ci_lower, ci_upper, auc_point, boot_aucs] = block_bootstrap_auc_by_day(y_true, y_prob, day_indices, n_boot, ci_level)
% =========================================================================
% 函數：block_bootstrap_auc_by_day
% 職責：以「交易日」為單位進行 Day-Level Block Bootstrap，計算橫截面 AUC 與 95% 信賴區間
% 說明：解決同交易日內多檔股票共享市場衝擊（Beta / 總經事件）之橫截面依賴問題，
%       避免逐筆重抽樣導致假性精確與信賴區間過窄之方法論漏洞。
%
% 輸入參數：
%   y_true      : [N x 1] 二元真實標籤 (0 或 1，或 logical)
%   y_prob      : [N x 1] 預測正類機率或決策分數 (continuous scores)
%   day_indices : [N x 1] 各樣本所屬之交易日識別碼 (如日期整數、Date ID)
%   n_boot      : 自助重抽樣次數 (可選，預設 = 1000)
%   ci_level    : 信賴水準 (可選，預設 = 0.95，對應 2.5% 與 97.5% 百分位數)
%
% 輸出參數：
%   ci_lower    : 信賴區間下界 (例如 2.5% 百分位數)
%   ci_upper    : 信賴區間上界 (例如 97.5% 百分位數)
%   auc_point   : 全樣本點估計 AUC (Point Estimate)
%   boot_aucs   : [n_boot x 1] 每次 Bootstrap 之 AUC 分布序列
% =========================================================================

    if nargin < 4 || isempty(n_boot)
        n_boot = 1000;
    end
    if nargin < 5 || isempty(ci_level)
        ci_level = 0.95;
    end

    % 1. 展平與有效資料清洗
    y_true = double(y_true(:));
    y_prob = double(y_prob(:));
    day_indices = day_indices(:);

    valid_mask = ~isnan(y_true) & ~isnan(y_prob) & ~isnan(day_indices) & ...
                 ~isinf(y_true) & ~isinf(y_prob) & ~isinf(day_indices);

    y_true = y_true(valid_mask);
    y_prob = y_prob(valid_mask);
    day_indices = day_indices(valid_mask);

    if isempty(y_true) || length(unique(y_true)) < 2
        warning('⚠️ 輸入有效資料不足或僅包含單一類別標籤，無法計算 AUC。');
        ci_lower = NaN; ci_upper = NaN; auc_point = NaN; boot_aucs = NaN(n_boot, 1);
        return;
    end

    % 2. 計算全樣本 AUC 點估計值
    [~, ~, ~, auc_point] = perfcurve(y_true, y_prob, 1);

    % 3. 依交易日建立樣本索引映射表 (提升重抽樣效能)
    [unique_days, ~, day_group_ids] = unique(day_indices);
    num_unique_days = length(unique_days);

    % 建立每個交易日對應的資料列索引 Cell 陣列
    day_to_row_idx = accumarray(day_group_ids, (1:length(y_true))', [num_unique_days, 1], @(x) {x});

    % 4. 執行 Day-Level Block Bootstrap
    boot_aucs = zeros(n_boot, 1, 'double');
    alpha_tail = (1 - ci_level) / 2;

    for b = 1:n_boot
        % 以交易日為單位進行有放回重抽樣 (With Replacement)
        sampled_day_ids = randi(num_unique_days, [num_unique_days, 1]);
        sampled_row_indices = vertcat(day_to_row_idx{sampled_day_ids});

        y_b_true = y_true(sampled_row_indices);
        y_b_prob = y_prob(sampled_row_indices);

        % 檢查抽樣樣本是否具備兩類標籤
        n_pos = sum(y_b_true == 1);
        n_neg = length(y_b_true) - n_pos;

        if n_pos > 0 && n_neg > 0
            % 採用極速 Mann-Whitney 秩轉換計算 AUC，大幅降低 1000 次 perfcurve 的開銷
            ranks = tiedrank(y_b_prob);
            sum_ranks_pos = sum(ranks(y_b_true == 1));
            boot_aucs(b) = (sum_ranks_pos - (n_pos * (n_pos + 1)) / 2) / (n_pos * n_neg);
        else
            boot_aucs(b) = 0.50; % 類別退化防呆
        end
    end

    % 5. 提取分位數信賴區間
    ci_lower = prctile(boot_aucs, alpha_tail * 100);
    ci_upper = prctile(boot_aucs, (1 - alpha_tail) * 100);
end