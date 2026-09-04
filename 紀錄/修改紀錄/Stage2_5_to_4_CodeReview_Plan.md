# Stage 2.5 – 4 Code Review 計劃書

版本：v1（規劃階段，尚未動工）
範圍：`hac_significance_test` max_lag 稽核 → 多子集穩健性重跑 → GBDT 迴歸替代驗證 → Round 9 方法論修正 → Stage 3 文件可定稿判準 → Stage 4 程式碼變更規格

---

## 0. 總覽與依賴關係

```
Task A (max_lag 稽核＋重跑)  ─┐
Task B (多子集/多種子重跑)   ─┼─→ 決定 Direction 2/3 最終措辭 ─→ Stage 3 §7 定稿
Task B'(GBDT 迴歸驗證)       ─┘                                      │
Round 9 方法論修正 ──────────────────────────────────────────────────┘
                                                                      ▼
                                                          Stage 4 程式碼變更範圍
                                                    （是否需要 Soft-IC 訓練管線 /
                                                     GCN-only 簡化二選一或並存）
```

**核心結論先講在前面**：doc 42 認為 Task A「幾乎零成本」的前提，在我逐行核對程式碼後**不成立**。原因見 §1.3。這會影響你對整體時程的預期，請優先看完 §1 再決定要不要照原計畫走。

---

## 1. Task A：`hac_significance_test` 的 max_lag 傳遞稽核

### 1.1 函式本身的設計假設

```matlab
function [t_stat, p_value, lrv] = hac_significance_test(x, max_lag)
    ...
    if nargin < 2
        max_lag = max(1, floor(4 * (n/100)^(2/9)));  % Newey-West(1994)自動頻寬
        % 注意：若特徵本身是長窗滾動構造(如SMA60)，建議手動將max_lag設為
        % 至少對齊該特徵的lookback長度，自動公式可能低估所需lag
    end
```

函式作者自己已經在註解裡預警了這個風險，但**全專案沒有任何一個呼叫點真正落實這個建議**。以下是逐檔逐行稽核。

### 1.2 呼叫點盤點表（Call-site Inventory）

| # | 檔案 | 呼叫對象（序列性質） | 序列長度 n（估） | 隱含所需 max_lag | 目前是否顯式傳入 | 風險等級 |
|---|------|---------------------|------------------|------------------|------------------|----------|
| 1 | `Run_Extractor_Pretrain_SMOKE.m` → `evaluate_probe_and_hac` 內部 | Group3/4 `daily_ic`（5D/60D OOF Rank IC） | G3≈780, G4≈726 | G3≥5, **G4≥60** | ❌ 否 | 🔴 **極高（Direction 3 生死點）** |
| 2 | `Run_Extractor_Pretrain_SMOKE.m` §6 報告迴圈 | `diff_d`（Group g vs Base 的逐日 loss 差，含 60D 標籤重疊） | 同上 | 同上 | ❌ 否 | 🔴 高（"對比 Base" 欄位同樣失真） |
| 3 | `Run_CeilingScan_MultiHorizon.m` → `execute_single_task_stage2` | `Daily_IC(:,f)`（原始特徵 1~90D IC） | ≈8300 | H=1~**90** | ❌ 否 | 🟠 中高（僅在 n 很大時被自動頻寬≈10 部分緩解，但 H=40/60/90 仍不足） |
| 4 | `Run_CeilingScan_MultiHorizon.m` → 同上 | `daily_oof_ic`（OOF Rank IC） | 遠小於上者 | 同上 | ❌ 否 | 🟠 中高 |
| 5 | `models/FeatureEvaluator.m` → `report_icir_ranking` | 特徵 IC 序列（目前僅被以 horizon=1, 5 呼叫） | ≈8391 | ≤5 | ❌ 否 | 🟢 低（目前用法下自動頻寬≈10 已足夠，但函式介面本身沒有 horizon 參數，未來若被人拿去跑 H=60 會重蹈覆轍——這是設計缺陷，不是當前錯誤） |
| 6 | `Run_Ablation_VQVAE.m` | 5D 特徵 IC；日報酬差 | 前者≈8391，後者為日頻報酬 | ≤5 | ❌ 否 | 🟢 低（同上，剛好夠用） |
| 7 | `Round10_FeatureSubsetTrim.m` | `Daily_IC_5D` | ≈8391 | ≤5 | ❌ 否 | 🟢 低 |
| 8 | `Run_Ablation_SpaceExpert.m` / `Run_Ablation_CrashGuard.m` / `Run_Ablation_PPO_Ensemble.m` | 日頻**報酬差**序列（非重疊 horizon 標籤） | 千級 | 無結構性下限 | ❌ 否 | ⚪ 不適用（這類序列不是重疊標籤構造出來的，auto 頻寬本來就是為這種情境設計，不算缺陷） |

**結論**：這不是「Group3/4 個案問題」，而是**專案級的呼叫慣例缺失**——只是剛好在 H=60（Group 4）與 Workstream A 的 H≥20 才會被放大到肉眼可見的程度。#1 是 doc 42 點名的重點，也是本計劃書優先修的項目；#3、#4 是我審核時額外發現、doc 42 沒提到但同樣需要處理的，否則 Stage 3 文字結論（NO-GO）跟附表（HAC_SigCount 非零）會自相矛盾，口委很容易抓到。

### 1.3 為什麼「零成本重跑統計」這個假設不成立

```matlab
function [daily_losses, probe_metric, p_hac] = evaluate_probe_and_hac(net, X_in, Y_in, Expert, val_days, seqLen, ~, target_type, use_gpu)
    ...
    daily_ic = daily_ic(1:v_idx);
    probe_metric = mean(daily_ic, 'omitnan');
    if length(daily_ic) >= 20
        [~, p_hac] = hac_significance_test(daily_ic);
    else
        p_hac = 1.0;
    end
end
```

`daily_ic` 是函式內部區域變數，**從未被回傳、也未被外層腳本存檔**。`history_results(g).daily_loss` 有存（因為 `daily_losses` 有回傳），但 IC 序列本身沒有。也就是說：

- 已經跑過的那次 SMOKE TEST（log 檔裡 Direction 2/3 的結果）**現在已經無法重新分析**，因為原始 IC 向量不存在於任何 `.mat` 檔案裡，只剩下聚合後的純量（`p_hac`, `metric`）留在文字 log。
- 若要用正確的 max_lag 重新檢定，**必須至少重跑一次訓練**才能重新取得 `daily_ic`。

**好消息**：這個重跑本身確實便宜。log 顯示 Group1/2/3/4 分別耗時 38.83s / 38.02s / 76.39s / 72.33s（15 epochs），全部重跑一次約 4 分鐘。真正的成本是「補程式碼」而非「補算力」。

### 1.4 修正規格

**(a) `evaluate_probe_and_hac` 簽名變更**（僅此檔案內的 local function，未被其他腳本呼叫，可安全修改）：

```matlab
function [daily_losses, probe_metric, p_hac, daily_ic_out, hac_lag_used] = ...
    evaluate_probe_and_hac(net, X_in, Y_in, Expert, val_days, seqLen, ~, target_type, use_gpu, H_horizon)
```

新增第 10 個輸入 `H_horizon`（呼叫時傳入 `grp.H`），內部改為：

```matlab
if strcmp(target_type, 'binary')
    daily_ic_out = [];
    p_hac = 1.0;
else
    daily_ic_out = daily_ic;
    if length(daily_ic) >= 20
        hac_lag_used = max(H_horizon, floor(4 * (length(daily_ic)/100)^(2/9)));
        [~, p_hac] = hac_significance_test(daily_ic, hac_lag_used);
    else
        p_hac = 1.0; hac_lag_used = NaN;
    end
end
```

`max(H_horizon, auto_bandwidth)` 的用意：H 較小（如 5D）時沿用原本自動頻寬即可（不會比目前結果更保守），H 較大（60D）時強制 floor 到 H。**同時建議雙欄輸出**（auto-only p 值 vs 修正後 p 值並列），這本身可以寫進 Stage 3 當一個小節的方法論反思素材（見 §5）。

**(b) 呼叫端更新**：

```matlab
[daily_val_losses, probe_metric, p_hac, daily_ic_g, hac_lag_g] = ...
    evaluate_probe_and_hac(net_t, X_in, Y_in, Expert_sub, val_days, seqLen, sample_T, grp.type, use_gpu, grp.H);
```

**(c) §6 報告迴圈的 `diff_d` 檢定**：

```matlab
% 原本
[~, p_pair] = hac_significance_test(diff_d);
% 修正
[~, p_pair] = hac_significance_test(diff_d, exp_groups{g}.H);
```

**(d) `Run_CeilingScan_MultiHorizon.m` 的 #3、#4**：在 `execute_single_task_stage2(t_info, ...)` 內，`t_info.H` 已存在，直接套用：

```matlab
% Daily_IC 逐特徵 HAC（#3）
[~, p_hac_vec(f)] = hac_significance_test(Daily_IC(:, f), H);

% OOF Rank IC HAC（#4）
if length(daily_oof_ic) >= 20
    [~, p_hac_oof_ic] = hac_significance_test(daily_oof_ic, H);
else
    p_hac_oof_ic = 1.0;
end
```

由於 Workstream A 主結論是 AUC/CI 而非 HAC p 值（NO-GO 判定條件是 `oof_auc >= 0.52 且 CI_lower > 0.505`，HAC 只影響 `sig_count`/`max|ICIR|` 這兩個輔助欄位），**預期主結論不會翻盤**，但這兩欄若要放進論文附表就必須先訂正，否則正文寫 NO-GO、附表卻暗示某些特徵 HAC 顯著，會是內部矛盾。

**(e) `FeatureEvaluator.report_icir_ranking`**：建議新增 `horizon` 輸入參數並貫穿到內部呼叫，即使目前用法沒踩雷，這是把「地雷解除」而非「補救已炸的雷」，屬於低優先但值得順手做的一致性修正。

### 1.5 新腳本設計：`Stage2_5_HAC_Relag_Revalidation.m`

- **輸入**：與原 SMOKE 腳本相同的資料載入區塊（可直接複製 §0~§3）
- **執行**：僅重跑 Group 1（Base）、Group 3（Soft-IC）、Group 4（60D）三組（Group 2 已被 Direction 1 定案，略過以省時間）
- **核心產出表**（這是 Stage 2.5 唯一必須交付的表）：

| Group | H | n | Auto max_lag | Auto p_hac | Forced max_lag=H | Corrected p_hac | 結論是否改變 |
|-------|---|---|---------------|-------------|--------------------|--------------------|----------------|
| G3 (Soft-IC) | 5 | ~780 | 6 | (待填) | 5 | (待填) | 預期：不變（auto 已足夠） |
| G4 (60D) | 60 | ~726 | 6 | 0.0000（原始log） | 60 | (待填) | **預期：極可能由顯著轉為不顯著** |

- **附帶產出**：`diff_d`（vs Base）同步用 `max_lag=H` 重算，因為這是 Direction 3 判定式裡「對比 Base」欄位的來源。
- **完成判準**：這張表填滿即可回答 doc 42 提出的問題，且此表**直接決定 Stage 3 §7 是否能寫「Direction 3 CONFIRMED」**。

---

## 2. Task B：多子集 / 多種子穩健性重跑

### 2.1 對 doc 42 原文的一個修正

doc 42 建議「比照 Round 8a-v2 用不同種子抽 3–5 組 60 檔子集」。**我核對過 Round 8a-v2 的抽樣邏輯，它其實跟 SMOKE 腳本用的是同一套規則**：

```matlab
active_sums = sum(Expert_Active(valid_idx, :), 1);
[~, top_active_tickers] = sort(active_sums, 'descend');
sub_tickers = top_active_tickers(1:sample_T);
```

這是**完全確定性**的「取成交活躍度前 60 名」，全專案目前沒有任何一支診斷腳本做過真正隨機的子集抽樣。也就是說「比照 Round 8a-v2」這個說法本身不成立——Round 8a-v2 沒有多種子的先例可循，這個能力需要**新建**。我把這點寫出來是因為如果直接照抄「比照」字面意思去找 Round 8a-v2 的多種子邏輯，會找不到東西，白費時間。

### 2.2 新腳本設計：`Stage2_5_MultiSeed_DirectionCompare.m`

**抽樣設計**：
```matlab
seeds = [101, 102, 103, 104, 105];
% 候選池限制：只從「活躍天數佔比 > 50%」的標的中抽，避免抽到近乎全 NaN 的子集
candidate_pool = find(active_sums / numDaysRaw >= 0.5);
for s = 1:length(seeds)
    rng(seeds(s), 'twister');
    sub_tickers_s = randsample(candidate_pool, sample_T);
    ...
end
```

**每個 seed 內部**：
- 訓練 Group1（Base BCE）、Group3（Soft-IC）、Group4（60D）
- Epoch 數可從 15 降到 **10**（原始 log 顯示 Group3 在 epoch 5 已收斂到 ~0.704，epoch 10 幾乎沒再變化，10 epochs 足以捕捉分離趨勢，換取 5 組 seed 的時間預算）
- 收集：`final_val_loss(Group1 vs Group3)`、`OOF_IC(Group4)`、以及套用 §1.4 修正後的 `p_hac`

**推論單位轉換（重要）**：跨 seed 比較時，統計檢定單位從「日」變成「seed」，**不再需要 HAC 校正**（seed 之間彼此獨立，不存在序列自相關），改用：
- n=5 建議用 **Wilcoxon signed-rank test**（`signrank` in MATLAB Statistics Toolbox）而非 paired t-test，因為 n=5 時常態性假設很難站得住腳，wilcoxon 更保守也更好辯護
- 同時報告 5 組原始差值本身（不要只給 p 值，樣本這麼小時讀者需要看到全部資料點）

**成本估計**：5 seeds × (Group1+Group3+Group4，每組≈10/15×原時間) ≈ 5 × (26+51+48)s ≈ 625s ≈ **~10-11 分鐘**。

**誠實聲明（要寫進 Stage 3）**：n=5 seed 仍然是小樣本，只能把結論從「n=1，完全無法排除運氣」升級到「初步穩健性證據」，不能宣稱「已證明穩健」。這個措辭邊界很重要。

---

## 3. Task B'：GBDT 迴歸驗證 Direction 2（建議執行，非必要）

### 3.1 `RawBaselineTrainer.m` 可重用性稽核

| 區塊 | 現況 | 迴歸版本是否可直接重用 |
|------|------|------------------------|
| `NumFolds`, `EmbargoDays`, `MaxTrainSamples` 屬性 | 通用超參數 | ✅ 全部重用 |
| Purged/Embargoed fold 切分（`edges`, `fold_indices`） | 純粹依 `day_array` 切分，與 Y 型態無關 | ✅ 完全重用，不用改一行 |
| `t_tree = templateTree(...)` | `templateTree` 本身跨分類/迴歸通用 | ✅ 重用 |
| `Y_cat = categorical(Y_flat); mdl = fitcensemble(...,'LogitBoost',...)` | 分類專用 | ❌ 需替換為 `fitrensemble(...,'LSBoost',...)`，Y 保持 double/single 不轉 categorical |
| `[~, score] = predict(mdl, ...); oof_scores(...) = score(:,2)` | 分類 predict 回傳兩欄機率 | ❌ 迴歸 predict 只回傳單一連續值，需改 |
| `val_probs = 1./(1+exp(-val_scores))` | Sigmoid 轉機率，分類專用 | ❌ 不需要，迴歸直接用預測值 |
| `block_bootstrap_auc_by_day(...)` | AUC 專用（內部用 Mann-Whitney tiedrank 技巧） | ❌ 不可重用，需要新函式（見 3.3） |
| `metrics.AUC_Point / PosClassRatio` 等輸出欄位 | 分類語意 | ❌ 需替換為 `OOF_IC / ICIR / HAC_p` |

**發現的專案內部重複**：`RawBaselineTrainer.m` 的 fold 切分邏輯，跟 `Run_CeilingScan_MultiHorizon.m` 裡 `execute_single_task_stage2` 內部自己重新實作的一套幾乎一模一樣（都是 `edges = linspace(...)` + `fold_indices` 賦值）。這代表**目前有兩份平行維護的 Purged CV 邏輯**。本次新增迴歸版本時，建議至少把 fold 切分抽成一個共用 private method（如 `compute_purged_folds`），不強求立刻收斂 CeilingScan 那邊（那是更大範圍的重構，列入 Stage 4 加分項，非 Stage 2.5 必要項）。

### 3.2 新增方法規格：`RawBaselineTrainer.train_and_eval_regression`

```matlab
function [metrics, oof_preds, eval_mask] = train_and_eval_regression(obj, X_flat, Y_cont_flat, day_array, H_horizon, n_boot)
    % 與 train_and_eval 共用 fold 切分（建議抽出 compute_purged_folds 共用）
    ...
    for k = 2:K
        ...
        mdl = fitrensemble(X_flat(tr_idx,:), Y_cont_flat(tr_idx), ...
            'Method', 'LSBoost', 'Learners', t_tree, ...
            'NumLearningCycles', obj.NumLearningCycles, 'LearnRate', obj.LearnRate);
        oof_preds(va_idx) = predict(mdl, X_flat(va_idx, :));
    end

    % 逐日橫截面 Spearman IC（沿用 execute_single_task_stage2 已驗證過的邏輯）
    [daily_ic, ~] = compute_daily_cross_sectional_ic(oof_preds, Y_cont_flat, day_array, eval_mask);

    hac_lag = max(H_horizon, floor(4*(length(daily_ic)/100)^(2/9)));
    [~, p_hac] = hac_significance_test(daily_ic, hac_lag);

    [ci_lower, ci_upper, ic_point, boot_ics] = block_bootstrap_ic_by_day(daily_ic, n_boot, 0.95);

    metrics = struct('OOF_IC', ic_point, 'CI_Lower', ci_lower, 'CI_Upper', ci_upper, ...
        'HAC_p', p_hac, 'HAC_MaxLag', hac_lag, 'NumDays', length(daily_ic));
end
```

### 3.3 新函式規格：`utils/block_bootstrap_ic_by_day.m`

**這裡要澄清 doc 42 的一個隱含假設**：`block_bootstrap_auc_by_day` 是對**逐筆觀測值**（每天每檔股票一列）依日期分組後重抽樣、再對重抽樣後**混池**的觀測值算單一 AUC；IC 序列本身已經是**逐日聚合完的純量**（一天一個 Spearman 值），性質不同，不能照搬同一套機制，否則會把「每日橫截面相關性」跟「跨日混池相關性」這兩件事搞混（後者會把時間序列雜訊也算進相關性裡，稀釋掉真正的橫截面訊號）。

正確設計是對**已經逐日聚合好的 `daily_ic` 向量本身**做 case-resampling bootstrap：

```matlab
function [ci_lower, ci_upper, ic_point, boot_ics] = block_bootstrap_ic_by_day(daily_ic, n_boot, ci_level)
    daily_ic = daily_ic(~isnan(daily_ic));
    n = length(daily_ic);
    ic_point = mean(daily_ic);
    boot_ics = zeros(n_boot, 1);
    for b = 1:n_boot
        idx = randi(n, [n, 1]);          % 以「日」為單位重抽樣
        boot_ics(b) = mean(daily_ic(idx));
    end
    alpha_tail = (1 - ci_level) / 2;
    ci_lower = prctile(boot_ics, alpha_tail*100);
    ci_upper = prctile(boot_ics, (1-alpha_tail)*100);
end
```

這個函式比 `block_bootstrap_auc_by_day`簡單很多（不需要 `day_to_row_idx` 映射，因為輸入已經是逐日聚合值），本質上是對「日層級 IC 均值」做 bootstrap CI，跟修正後的 HAC 檢定是**同一個統計問題的兩種獨立估計方式**——兩者若互相印證（bootstrap CI 不含 0，且 HAC p<0.05），證據力會比單一方法強很多，這點建議寫進 Stage 3。

### 3.4 新腳本設計：`Stage2_5_GBDT_SoftIC_Validation.m`

- **關鍵設計決定（建議採用，但列為待你確認的選項）**：用**全 504 檔宇宙**而非 60 檔子集跑，因為 GBDT 比 DL 便宜很多，且全宇宙結果比 SMOKE test 的 60 檔子集更有說服力，可以當作「獨立於 DL pipeline 的交叉驗證」而不只是同一套資料的另一次重複。
- 流程：載入 `features_denoised.mat` → 取 18D raw features（不做 GICS 中性化，因為 Direction 1 已否證其增量，維持基準線乾淨）→ 構建 `Y_bin_5D`（既有）與 `Y_cont_5D`（既有，`(r-mean)/std` z-score 版本）→ 攤平成 `(X_flat, Y_flat, day_array)`（沿用 `GBDTExpertAgent` 攤平邏輯的寫法）
- 同時跑 `train_and_eval`（既有分類版本，Y_bin_5D）與新的 `train_and_eval_regression`（Y_cont_5D），**共用同一組 fold 切分**，確保 apple-to-apple
- 輸出對照表：OOF AUC (binary) vs OOF IC (regression)，兩者的 day-block bootstrap CI，以及修正後 HAC p 值

**成本估計**：GBDT 5-fold 在全宇宙上（參考 `GBDTExpertAgent` log，總樣本 ~180萬筆，單次 fold 訓練約數十秒到一兩分鐘），全部跑完（分類+迴歸各 4 folds）預估 **10-20 分鐘**，遠低於任何 DL 重跑。

---

## 4. Round 9 驗證程序問題的具體定位

逐行核對後，我把可能被稱為「驗證集抽樣」問題的地方拆成三個獨立問題：

### 4.1 RNG 未在組間重置（confounding 問題，不是抽樣問題，但影響公平性）

```matlab
rng(configObj.RNG_Seed, 'twister');   % 只在腳本最開頭設一次
...
for r = 1:num_regimes
    reg = lr_regimes{r};
    ...
    extractorFactory = BuildDecoupledExtractors(smokeConfig);
    [net_curr, ~] = extractorFactory.buildNetworks();   % 權重初始化消耗隨機流
    ...
    shuffled_days = train_days(randperm(...));           % 每個 epoch 都消耗隨機流
```

三組排程（Step Decay / Constant / Cosine）依序執行，**後面的排程繼承了前面排程已經消耗掉的隨機數流**，導致三組的網路初始權重、mini-batch 洗牌順序全部不同且不可控——排程效果與「隨機流位置」完全混淆（confounded），不是乾淨的控制實驗。

**修正**：

```matlab
for r = 1:num_regimes
    reg = lr_regimes{r};
    rng(configObj.RNG_Seed + r, 'twister');   % ★ 每組獨立可重現的種子
    ...
```

### 4.2 `baseline_threshold = 0.6938` 是跨腳本硬編碼比較基準

```matlab
baseline_threshold = 0.6938; % 15-Epoch 最低基準點
```

這個數字的來源是註解裡寫的「15-Epoch」，但 Round9 本身跑的是 100 epochs、且是另一組（可能是更早的 Smoke Test 或 Round7）的結果，**不是在 Round9 這次執行內部用同一個 seed/同一個資料子集算出來的**。用這個數字去判定「TRAINING EXTENSION BENEFICIAL」，本質上是拿別次實驗的結果當作這次實驗的對照組，違反控制變因原則。

**修正**：改為動態計算，例如用 Schedule 1 自己在 epoch 15 的 val loss 當基準：

```matlab
baseline_threshold = regime_results(1).val_loss(15);  % 用同次執行、同一子集的內部基準
```

### 4.3 決定性 Top-60 子集（與 §2.1 同一問題，共用 Task B 的修正）

不重複贅述，見 §2.2。

**這三點的處理方式**：4.1、4.2 是最小修正（幾行程式碼），建議在**重跑 Round9 之前**先修好，不需要另開新腳本；4.3 已經被 Task B 的多種子設計涵蓋。**是否要重新完整跑一次 Round9（100 epochs × 3 schedule × 修正後）** 是一個時間/必要性的取捨——如果 Round9 的結論只是輔助性質（訓練體制非瓶頸），可以考慮在 Stage 3 用「已知方法論限制，結論列為參考而非定論」的方式處理，不一定要重跑；但若 Direction 2/3 的討論會引用 Round9 的收斂行為佐證，就建議至少修正後跑一次 Schedule 2（Constant Small，log 顯示是三組中最快突破基準且最穩定的）驗證 4.1/4.2 修正後結論是否還成立。**這點需要你決定優先順序**（見 §7）。

---

## 5. Stage 3 文件——各節「可定稿判準」檢查表

| Stage 3 章節 | 內容 | 定稿前置條件 | 目前狀態 |
|---|---|---|---|
| §1 研究動機與假說 | Val Loss 卡在 ln(2) 附近，四個對立假說 | 無依賴，可現在定稿 | ✅ 可定稿 |
| §2 診斷方法論 | Round6 陽性對照設計、Purged CV、HAC、day-block bootstrap 介紹 | 建議補一小節「為何同時看 Val Loss 與探針 AUC」+ 本次發現的 HAC max_lag 慣例問題（見下） | ✅ 可定稿，但建議加一段方法論反思 |
| §3.1 工程機制排除（Round6） | AUC 0.977/0.975 | 無依賴 | ✅ 可定稿 |
| §3.2 架構容量排除（Round8b） | 簡化架構表現相當 | 無依賴 | ✅ 可定稿 |
| §3.3 訓練時長/LR 排除（Round9） | 目前寫法引用 0.6938 magic number | **待 §4.2 修正後重跑，或明確標註方法論限制** | 🟡 待補 |
| §3.4 IC 閘門/特徵精簡（Round7/10） | 用「發現2」框架改寫措辭 | 無新依賴，改寫用詞即可 | ✅ 可改寫定稿 |
| §3.5 空間專家診斷（Round8a-v2） | Attention 分支梯度消失 vs 張量重組缺陷 | 無新依賴 | ✅ 可定稿 |
| §4 主要證偽結論（Workstream A） | AUC≈0.50-0.52 全 NO-GO | **HAC_SigCount / max\|ICIR\| 兩欄需用 §1.4(d) 修正後數字** | 🟡 待補（主結論不變，附表數字要換） |
| §5 產業中性化排除（Direction 1） | GICS 無增量 | 無依賴 | ✅ 可定稿 |
| §6 訓練目標重構探索性發現（Direction 2/3） | Soft-IC / 60D | **完全依賴 Task A + Task B（+ 可選 Task B'）結果** | 🔴 **不可定稿，等 Stage 2.5** |

**寫作建議**：§1-§5（除 §3.3、§4 的附表數字）可以現在就開始寫草稿，§6 保持空白或用「待驗證假說」的方式先寫一版最保守的敘述框架，等 Stage 2.5 數字出來再填入結論句——這樣不會卡住整體寫作進度。

---

## 6. Stage 4 程式碼變更規格（先定規格，不預設實作）

### 6.1 `models/GraphSpatialFusionLayer.m`

**目標**：讓「純靜態 GCN」與「動態 Attention 混合」可以透過開關切換，對應 Round 8a-v2 對 attention 分支梯度消失的精確診斷。

**變更範圍**：
- 建構子新增一個**非 Learnable** 屬性，例如 `MixMode`（字串 `'dynamic'` | `'gcn_only'`，預設 `'dynamic'` 以維持向後相容）
- `predict()` 方法內：

```matlab
if strcmp(obj.MixMode, 'gcn_only')
    A_dynamic = A_norm_3D;   % 完全繞過 Q/K/S_attn 計算，省算力
else
    % 現有 attention 分支邏輯不變
    ...
    alpha_mix = sigmoid(obj.MixLogit);
    A_dynamic = alpha_mix * A_norm_3D + (1 - alpha_mix) * S_attn;
end
```

- **限制說明**：MATLAB 自訂 layer 的 `Learnable` 屬性在類別定義時就固定聲明，無法依實例動態增減。`W_attn_1`/`W_attn_2`/`MixLogit` 在 `gcn_only` 模式下仍會被聲明為 Learnable，但因為 predict 路徑完全不使用它們，`dlgradient` 對它們的梯度會精確為 0（不是「趨近於 0」，是數學上真正的 0），等同凍結，這比目前 attention 分支「梯度消失但非零」更乾淨、更可解釋。
- **回歸相容性風險**：已存的 `DL_Extractors.mat`（含訓練好的 `net_space`）是用舊版類別定義存的物件，類別定義變更後**大機率無法直接載入**（MATLAB 物件序列化對類別結構變更很敏感）。這代表 Stage 4 必然需要**重新跑一次 Phase 2 預訓練**，不是單純換開關就好——這點要不要做、要花多久，是決策點（見 §7）。

### 6.2 `models/BuildDecoupledExtractors.m`

`buildNetworks()` 內建立 `GraphSpatialFusionLayer` 的呼叫需要多傳一個 mix mode 參數，來源建議從 `obj.ConfigObj` 讀取（見 6.3），而非寫死在 factory 裡：

```matlab
gat_layer = GraphSpatialFusionLayer('gat_1', obj.NumTickers, obj.EmbedDim, obj.TotalNodeFeats, ...
    obj.ConfigObj.SpaceExpertMixMode);
```

### 6.3 `configs/Config.m`

新增屬性：

```matlab
SpaceExpertMixMode = 'gcn_only'   % 'gcn_only' | 'dynamic'，預設依 Round8a-v2 診斷結果採 GCN-only
```

`EnableSpaceExpertTraining` 維持可開（不像 doc 42 原始建議「整條路徑關」），因為 GCN-only 模式下空間專家仍是一個獨立、有意義的對照組（P2-1 消融的 Group C「純空間」仍然需要這條路徑跑得動）。

### 6.4 訓練管線腳本（依 Task B' 結果二選一，此處先各自定規格）

**若 Direction 2（Soft-IC）確認有效**：
- `Run_Extractor_Pretrain.m` 的訓練目標從 §5 `aux_loss_time`/`aux_loss_space`（純 BCE）改為呼叫 `BuildDecoupledExtractors.compute_continuous_return_loss`（此函式已存在於 `BuildDecoupledExtractors.m`，目前只在 Stage2 SMOKE 用過，尚未接進正式 Phase 2 管線）
- 連動：`Run_GBDT_and_SHAP.m` 的 Y_Labels_3D 建構需同步改成連續版本，GBDT 訓練方法從 `fitcensemble` 換 `fitrensemble`（可直接複用 3.2 的 `RawBaselineTrainer` 迴歸邏輯，不需要重新設計）

**若未確認**：
- 退回 doc 42 原計畫：簡化為 GBDT-on-raw，拿掉 DL 萃取層，`EnableSpaceExpertTraining` 視情況整條關閉

**PPO 集成 / 崩盤護欄相關腳本**：不受 Stage 2.5 結果影響，維持 doc 42 原建議不變（P2-3 簡化為單一 agent；崩盤護欄維持系統賣點定位）。

---

## 7. 風險與待決事項（需要你決定）

1. **Task A 是否要重跑訓練，還是接受無法重新分析舊結果的事實直接往前走？**（我的建議：跑，4 分鐘成本極低，且這是 Direction 3 生死攸關的檢定）
2. **Task B' 的驗證規模**：全 504 檔宇宙（我建議）vs 維持 60 檔子集以求跟 DL smoke test 直接可比？兩者都做也是選項，但會拉長時間。
3. **Round 9 是否要在 §4.1/4.2 修正後完整重跑一次**，還是在 Stage 3 用方法論限制的方式帶過不重跑？
4. **Stage 4 的 `GraphSpatialFusionLayer` 變更會迫使 Phase 2 重新訓練**——這件事要不要現在就排進時程，還是等 Stage 2.5/3 文件穩定後再啟動（我傾向後者，因為 Stage 4 是大改版，不該在 Stage 2.5 結論還沒定案前就動）？
5. **`RawBaselineTrainer.m` 與 `Run_CeilingScan_MultiHorizon.m` 的 fold 邏輯重複**是否要在這次順手收斂成共用函式，還是列入技術債，Stage 4 再處理？

---

## 8. 建議執行順序與時程

| 順序 | 工作項 | 預估時間 | 產出 |
|---|---|---|---|
| 1 | §1.4 修正 `evaluate_probe_and_hac` + 重跑 Task A | 0.5 天 | Direction 3 存活/死亡判定表 |
| 2 | §4.1/4.2 Round9 最小修正 | 0.5 小時 | 修正後的 Round9 腳本（跑不跑視 §7-3 決定） |
| 3 | Task B 多種子腳本撰寫 + 執行 | 1 天 | Direction 2/3 的 n=5 穩健性表 |
| 4 | Task B'（RawBaselineTrainerRegression + block_bootstrap_ic_by_day + 驗證腳本） | 1-1.5 天 | 獨立於 DL pipeline 的 Direction 2 交叉驗證 |
| 5 | 彙整 Stage 2.5 全部結果，回填 Stage 3 §6/§3.3/§4 | 0.5-1 天 | Stage 3 文件定稿版 |
| 6 | Stage 4 規格審閱 + 決定是否啟動重訓 | 待 §7 決定 | Stage 4 實作排程 |

Stage 2.5 全部完成預估 **3-4 個工作天**（不含等待你確認 §7 決策點的時間）。

---

## 附錄：本文件未涵蓋但建議之後追蹤的項目

- `FeatureEvaluator.report_icir_ranking` 的 horizon 參數化（§1.4-e，低優先一致性修正）
- `RawBaselineTrainer` 與 `Run_CeilingScan_MultiHorizon` 的 fold 邏輯收斂（§7-5）
- Hodrick(1992) 或非重疊抽樣作為 HAC 在極長 lag（如 H=90）下的穩健性替代方案（若 Stage 2.5 發現 max_lag=59-89 時 HAC 估計本身開始不穩定，可以考慮這條路，屬於加分項而非必要項）
