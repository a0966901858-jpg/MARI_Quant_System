function [ci_lower, ci_upper, ic_point, boot_ics] = block_bootstrap_ic_by_day(daily_ic, n_boot, ci_level)
% =========================================================================
% 函式：block_bootstrap_ic_by_day (日層級 IC 聚合序列 Bootstrap 信賴區間估計)
% 依據：《Stage 2.5 – 4 Code Review 計劃書》 §3.3 規範
% 職責：
%   對已經依「日」橫截面聚合好的 Spearman Rank IC 向量實施 Case-Resampling Bootstrap，
%   估算日層級 IC 期望值的雙尾信賴區間，與 HAC 檢定形成同統計問題的獨立交叉估計。
%
% 輸入參數：
%   daily_ic  : 逐日橫截面 Rank IC 向量 [N x 1] (可含 NaN/Inf，內部自動濾除)
%   n_boot    : Bootstrap 重抽樣次數 (預設 = 1000)
%   ci_level  : 信賴區間水準 (預設 = 0.95，對應 95% CI)
%
% 輸出參數：
%   ci_lower  : 信賴區間下界 (百分位法)
%   ci_upper  : 信賴區間上界 (百分位法)
%   ic_point  : 原始樣本 IC 點估計值 (均值)
%   boot_ics  : 各次 Bootstrap 抽樣計算所得之 IC 均值分布 [n_boot x 1]
% =========================================================================

    % 預設引數處理
    if nargin < 3 || isempty(ci_level)
        ci_level = 0.95;
    end
    if nargin < 2 || isempty(n_boot)
        n_boot = 1000;
    end

    % 剔除缺失值與無效值
    valid_ic = daily_ic(~isnan(daily_ic) & ~isinf(daily_ic));
    valid_ic = valid_ic(:);
    n = length(valid_ic);

    % 防禦極小樣本 (樣本數小於 5 時無法估算可靠信賴區間)
    if n < 5
        ic_point = mean(valid_ic);
        ci_lower = NaN;
        ci_upper = NaN;
        boot_ics = [];
        return;
    end

    % 樣本點估計 (Sample Mean)
    ic_point = mean(valid_ic);

    % 以「日」為獨立單位進行 Case-Resampling Bootstrap[cite: 1]
    boot_ics = zeros(n_boot, 1, 'like', valid_ic);
    for b = 1:n_boot
        idx = randi(n, [n, 1]); % 均勻抽後放回[cite: 1]
        boot_ics(b) = mean(valid_ic(idx)); %[cite: 1]
    end

    % 雙尾百分位數切割計算信賴區間[cite: 1]
    alpha_tail = (1 - ci_level) / 2; %[cite: 1]
    ci_lower = prctile(boot_ics, alpha_tail * 100); %[cite: 1]
    ci_upper = prctile(boot_ics, (1 - alpha_tail) * 100); %[cite: 1]
end