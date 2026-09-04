# Phase 2 修根診斷實驗設計方案
### 從「表徵坍縮已解」到「表徵無資訊」的根因定位與驗證協定

---

## 0. 文件使用說明

本文件是給你自己動手寫 MATLAB 腳本用的**實驗設計規格書**，不含程式碼本體。每一輪實驗都包含：目的（要驗證哪個假說）、精確設計（改什麼、不改什麼）、需要新建/重用哪些檔案、判讀門檻（多少算過、多少算沒過）、以及結果會把你導向哪個下一步。

所有輪次都**不改變現有 5 日 Beat-the-Median 標籤方法論**（依你的決定），Workstream A 唯一允許變動的維度是 Horizon 天數；Workstream B 完全固定在 5 日標籤上，只動架構與訓練 regime。

建議命名慣例延續你現有風格：新腳本放在 `scripts/diagnostics/` 下，檔名格式 `RoundN_<主題>.m`。每輪跑完後，把結果數字回填進本文件對應的「預期輸出」表格，之後可以直接整包貼回 Claude 的 memory 或論文 5.2 節素材。

---

## 1. 現況重述：兩個必須分開處理的問題

| | 現象 | 證據 | 已解決？ |
|---|---|---|---|
| **問題一：表徵坍縮 (Representation Collapse)** | Embedding 退化成近乎常數向量，variance 趨近 0 | Round 1 之前 Dense Softmax 稀釋、BCE 卡死在梯度死鎖 | ✅ 已解決（Masked Softmax + VICReg，Phase 15.5 全量跑 E_Var(T)=1.21~1.66、E_Var(S)=1.1~2.2，全程無坍縮警告） |
| **問題二：表徵無資訊 (Representation Non-Informativeness)** | Embedding variance 健康，但 BCE Loss 仍停在 ln(2)≈0.6931 附近 | Phase 15.5 全量跑 Val Loss 最終 0.6938 / 0.6943；GBDT 下游 OOF AUC：Raw 18D=0.5021、Time=0.5029、Space=0.5020，三者統計打平 | ❌ **未解決，這是本方案要攻的目標** |

**關鍵線索**：GBDT 直接吃原始 18D+10D 特徵（LogitBoost，非線性、對表徵坍縮/梯度消失免疫）也只做到 AUC 0.5021。這代表問題大機率不在 DL 萃取器的訓練機制本身，而在於「這組特徵在 5 日 Horizon 下，可萃取的橫截面資訊含量本來就趨近於 0」——但這個結論目前只在單一 Horizon（5 日）上被驗證過，其他 Horizon 是否也一樣沒被系統性掃過。這是 Workstream A 要補的洞。

同時，GBDT 的 Raw Baseline 用的是**未經 IC 注意力閘門加權**的特徵，而 DL 萃取器吃的是**加權後**的版本，兩者不是嚴格對照組。這是 Workstream B 要控制掉的混淆變因之一。

---

## 2. 執行總覽與優先順序

| 順序 | Round | Workstream | 目的 | 是否需要 GPU | 預估規模/耗時（參考量級） | 依賴 |
|---|---|---|---|---|---|---|
| 1 | Round 6 | B | 合成訊號 Positive Control：排除 pipeline 本身有 bug | 需要（沿用 smoke 規模） | 60檔×3200天，15~30 epoch，量級同你既有 smoke test | 無，最優先 |
| 2 | Workstream A | A | 多 Horizon 天花板掃描（GBDT + FeatureEvaluator） | 不需要，純 CPU parfor | 504檔全量，每個 Horizon 一輪 IC + 5-Fold GBDT，可與 Round 6 平行跑 | 無 |
| 3 | Round 8a | B | 過擬合能力檢測（小樣本記憶力測試） | 需要 | 20~40 個訓練樣本，長 epoch | Round 6 通過後 |
| 4 | Round 7 | B | IC 注意力閘門開關消融 | 需要 | 60檔×3200天 ×2組 | Round 6、8a 通過後，可與 Round 8b 平行 |
| 5 | Round 8b | B | 架構複雜度消融（拿掉 Self-Attention） | 需要 | 60檔×3200天 ×2組 | 同上 |
| 6 | Round 9 | B | 訓練 Regime 排查（LR/Epoch/Early Stopping） | 需要 | 60檔×3200天 ×3~4組，長 epoch | Round 6~8 都排除後才做，成本較高，優先度最低 |
| 7 | Round 10 | B | 特徵子集精簡消融 | 需要 | 依 Workstream A 結果篩選特徵後重跑 | 需等 Workstream A 出結果 |

**第一步永遠是 Round 6。** 它便宜、快、而且是唯一能回答「我後面做的所有事情有沒有意義」的檢查——如果 pipeline 本身有 bug，後面 7~10 輪全部白做。

---

## 3. Workstream A：多 Horizon 天花板掃描

### 3.1 目的

在**不改變 Beat-the-Median 標籤方法論**的前提下，橫向掃描不同 Horizon，找出：(a) 是否存在某個 Horizon 讓橫截面訊號明顯強於目前的 5 日；(b) 微觀 18D 特徵與宏觀 10D 特徵誰扛住了訊號（這會回答「DL 萃取器把 Macro 剝離掉是不是也把訊號一起剝掉了」）。

### 3.2 實驗設計

**Horizon 候選集合**：`H = [1, 3, 5, 10, 15, 20, 30, 60]` 天（8 個點，涵蓋短期反轉到中期動能的常見窗口）。

**特徵子集**：對每個 Horizon 各跑三個版本：
- (a) 18D（Rel 3 + Micro 15，即現有 DL 萃取器吃的子集）
- (b) 10D（純 Macro）
- (c) 28D 全量（18D+10D）

**每個 (Horizon, 特徵子集) 組合的步驟**：

1. 用現有 `FeatureEvaluator.compute_confidence(X_subset_3D, Prices_Active, Expert_Active, horizon)` 直接算 Daily_IC，`report_icir_ranking` 印出 HAC-ICIR 排行榜（這兩個函式已支援任意 horizon 參數，**不需要改動**）。
2. 用你現有的 5 日 Beat-the-Median 標籤邏輯（`median(R_fwd(t, active_mask))` 判斷高於/低於中位數），只把 `horizon` 換成 H 中的值，重新產生 `Y_Labels_3D`。
3. 訓練一個**獨立於 GBDTExpertAgent.m 的輕量 Raw-Baseline 訓練函式**（見 3.5 節說明為何要獨立），沿用 Phase 3 現有配方：`fitcensemble`、`Method='LogitBoost'`、`templateTree('MaxNumSplits',20,'MinLeafSize',50)`、`NumLearningCycles=30`、Purged Expanding Window 5-Fold、20 天 Embargo。
4. 記錄：OOF AUC 點估計、**Day-Level Block Bootstrap 95% CI**（見 3.3 節，這是新增的統計修正，不是既有 Bootstrap 的原樣重用）、正類比例（應接近 50%，作為 sanity check）、該 Horizon 下 HAC 顯著特徵數與最大 |ICIR|。

### 3.3 統計方法修正：為什麼不能直接沿用崩盤護欄的 Bootstrap

你現有的崩盤護欄 AUC 用 `n_boot=1000` 對**個別觀測值**重抽樣，這在崩盤護欄那個情境下沒問題，因為崩盤標籤是**逐日**的 1D 序列，每個觀測值天然獨立（一天一筆）。

但 Beat-the-Median 的橫截面 AUC 是在 `(day, ticker)` 這個二維上算的——同一天內的 ~490 檔股票會共享同一個大盤日內衝擊（市場 beta、當日總經事件），彼此高度相關。如果直接對這些 `(day,ticker)` 列逐筆重抽樣，會嚴重低估真實不確定性、做出過窄的假信賴區間，這正是你在 4.3 節 Newey-West HAC 想解決的同一類問題，只是換了個統計量（AUC 而非 IC 均值）。

**修正做法（需要你新寫一個小工具函式，例如 `block_bootstrap_auc_by_day.m`）**：
- 以「交易日」為重抽樣單位，而非 `(day,ticker)` 列。
- 從評估 fold 中所有出現過的唯一交易日，做 `n=1000` 次「連同該日全部股票列」的整批重抽樣（with replacement）。
- 每次重抽樣後，把選中的日子對應的所有 `(day,ticker)` 列彙整起來，重新丟進 `perfcurve` 算一次 AUC。
- 取 1000 次結果的 2.5 / 97.5 百分位數作為 95% CI 下界/上界。

這個修正本身就是一個可以寫進論文 4.5 節的方法論補強——現有崩盤護欄的 Bootstrap 沒有這個問題，但如果未來有其他橫截面層級的信賴區間估計，都應該用 Day-Level Block Bootstrap，這點值得在論文裡明確講清楚兩者的差異與適用時機。

### 3.4 判讀標準

以現有 5 日、28D、AUC=0.5021 作為基準錨點：

| 結果 | 判定 | 意涵 |
|---|---|---|
| 點估計 ≥ 0.52 **且** Day-Block Bootstrap 95% CI 下界 > 0.505 | **GO**：該 Horizon 具備可萃取訊號 | 優先把這個 Horizon 拿去挑戰 DL 萃取器（Workstream B 之後的輪次可以考慮切過去，但這超出本輪「不改標籤」的範圍，屬於下一階段決策） |
| 點估計落在 [0.505, 0.52) 且 CI 下界 > 0.50 | **臨界**：訊號存在但弱 | 記錄下來，排序取前 1~2 名 Horizon 保留觀察，但論文中需誠實標註「臨界顯著，非決定性」 |
| 點估計落在 [0.495, 0.515] 且 CI 涵蓋 0.50 | **NO-GO** | 排除該 Horizon |
| 全部 8 個 Horizon 都落在 NO-GO | **系統性結論**：現有 28 維特徵集在任何常見 Horizon 下都缺乏穩定橫截面訊號 | 修根重心應轉向**特徵工程擴充**（見第 6 節），而非繼續在現有特徵上調 DL 架構 |

另外單獨比較 (a)18D vs (b)10D vs (c)28D 三個子集在同一 Horizon 下的表現：若 (b) 明顯優於 (a)，代表 DL 萃取器現行「剝離 Macro 只留 18D」的設計決策本身可能就是問題的一部分（把訊號留在了被丟掉的那 10 維裡）。

### 3.5 需要新建/重用的檔案

- **重用、不改動**：`FeatureEvaluator.m`（`compute_confidence` 已支援任意 horizon，`report_icir_ranking` 已支援任意 horizon 標籤）。
- **新建**：`scripts/diagnostics/Run_CeilingScan_MultiHorizon.m` —— 外層迴圈跑 8 個 Horizon × 3 個特徵子集，呼叫下面這個新函式。
- **新建**：`agents/RawBaselineTrainer.m`（或直接寫成函式，不做成 class 也可以）—— 把 `GBDTExpertAgent.train_and_predict_oof_cross_sectional` 裡專門訓練 Raw Baseline 的那段邏輯（`X_raw_flat`、5-Fold Purged CV、`fitcensemble`）抽出來，做成一個不依賴 DL Embedding 輸入的獨立函式。**不要直接修改 `GBDTExpertAgent.m` 本體**，避免影響 Phase 3 正式流程——這是延續你自己一貫「單一變因隔離」的做法。
- **新建**：`utils/block_bootstrap_auc_by_day.m` —— 3.3 節描述的日層級區塊自助法工具。

### 3.6 預期輸出格式（範例骨架，供你填數字）

```
Horizon(天) | 特徵子集 | OOF AUC | 95% CI (Day-Block) | 正類比例 | HAC顯著特徵數 | max|ICIR|
1           | 18D      |         |                     |          |               |
1           | 10D      |         |                     |          |               |
1           | 28D      |         |                     |          |               |
3           | 18D      |         |                     |          |               |
...(依此類推至 60 天)
```

這張表填完後可以直接對應論文表 5-1 的擴充版本（原本只有 1D/5D 兩欄，現在有完整 Horizon 掃描），是一張獨立於 DL 結果、單靠 GBDT+統計檢定就能產出的高價值表格。

---

## 4. Workstream B：DL Pipeline 架構與訓練 Regime 稽核

固定：5 日 Beat-the-Median 標籤、60 檔標的、3200 天子集（沿用你現有的 smoke test 抽樣規模與 `Run_Extractor_Pretrain_SMOKE.m` 作為模板起點）。

### 4.1 Round 6：合成訊號 Positive Control（最優先）

**目的**：在懷疑「特徵沒訊號」之前，先排除「pipeline 有 bug 導致什麼都學不到」——這兩種失敗模式在 BCE 停在 ln(2) 這個現象上長得一模一樣，必須用一個「答案已知」的任務把它們分開。

**設計**：
1. 複製 `Run_Extractor_Pretrain_SMOKE.m` 為新檔案，**跳過步驟 1.5 的 IC 注意力閘門**（保持乾淨，不引入額外混淆變因）。
2. 把步驟 2 的 `Y_Labels_3D` 改成一個**已知可由當日輸入特徵直接算出**的合成標籤，而不是未來報酬。例如：`Y_synthetic(t,i) = 1{X(t, R1的通道索引, i) > median_across_active_tickers(X(t, R1通道, :))}`——也就是把「預測未來報酬贏過中位數」換成「判斷今天的 R1 有沒有贏過今天的橫截面中位數」，答案 100% 由當天輸入決定，理論上任何正常運作的網路都應該能學到接近完美。
3. 時序專家（net_time）與空間專家（net_space）都各自用這個合成標籤單獨測一次。
4. （選配，若時間允許可加做）**Round 6b**：針對空間專家做一個「必須用到圖結構才能答對」的合成任務，例如標籤依賴鄰居節點的特徵值而非自身，專門驗證 `GraphSpatialFusionLayer` 的 adjacency 使用邏輯有沒有接對——GAT 類架構的 bug historically 很常出在鄰接矩陣方向、自環處理、masked softmax 邊界這幾個地方。

**判讀門檻**：
- **PASS**：Val BCE < 0.15 **且**（用凍結後的 embedding 接一個簡單 logistic regression 在 held-out 天數上測）AUC ≥ 0.90，在 ≤ 30 epoch 內達成。→ 判定 pipeline 機制正確，後續 ln(2) 停滯可以放心歸因於「真實標籤在現有特徵下缺乏訊號」，不是程式錯誤。這本身是一段可以寫進論文 5.2 節的重要方法論佐證——即使最終結果仍是 null，這段證明了你不是因為 bug 沒做出來，而是誠實窮盡了診斷。
- **FAIL**：Val BCE 仍 > 0.5 或 AUC < 0.7 → 判定 pipeline 存在阻斷梯度或資料對齊的實作錯誤，**必須先除錯才能繼續**。建議依序排查：`prepareBatchData` 裡的 `permute`/`reshape` 索引是否對齊（尤其 `[Days,Feats,Tickers]→[Feats,Tickers,SeqLen]` 這幾次維度重排最容易出錯位）、`dlgradient` 是否真的對 `net.Learnables` 完整回傳梯度（可以用 `norm` 檢查梯度是否為全零或异常小）、Adam 更新是否套用到正確的網路物件、`W_aux` 線性頭本身能不能單獨收斂（先只訓練線性頭、凍結網路權重測一次）。

### 4.2 Round 7：IC 注意力閘門開關消融

**目的**：測試「時變 IC 加權」是否本身在給 LSTM/GAT 的輸入分佈引入非平穩雜訊，讓優化變困難（GBDT 用的 Raw Baseline 沒經過這個閘門，DL 萃取器有，這是目前兩者比較不乾淨的地方）。

**設計**：兩組平行跑，除了「是否套用步驟 1.5 的 IC 閘門」之外，其他設定（子集規模、架構、LR、正則化、epoch 數）完全相同。

**判讀門檻**：若「不加閘門」組的最終 Val Loss 比「加閘門」組低超過 0.005（以你既有輸出的小數位精度為參考），視為閘門有負面影響，建議之後移除，或改成非時變的靜態版本（例如固定用全樣本期的平均 |IC| 排序，而非逐日滾動 60 天重算）。若兩組差異在雜訊範圍內，排除此假說。

### 4.3 Round 8a：過擬合能力檢測（小樣本記憶力測試）

**目的**：這是 Round 6 之外的第二道 bug 排除機制，直接在真實架構、真實標籤上做，而非合成任務——如果連刻意讓模型過度參數化都無法讓訓練損失趨近 0，代表問題出在梯度流或優化本身，而不是資料的統計性質。

**設計**：沿用完整 3200 天子集（因為每個訓練樣本仍需要 60 天回溯窗才能構成序列），但把 `train_idx_valid` 硬性限縮到只有 20~40 個時間點，`Dropout=0`、`L2=0`、關掉 Early Stopping（或把 patience 設到很大），跑 150~200 epoch，只看 **Train Loss**（不是 Val Loss）。

**判讀門檻**：Train BCE 應可降到 < 0.05。若怎麼跑都下不去，同樣指向梯度流/optimizer 綁定問題，需要跟 Round 6 的除錯清單一起排查。

### 4.4 Round 8b：架構複雜度消融（拿掉 Self-Attention）

**目的**：3020 個訓練樣本（IS 天數）對一個 Self-Attention + LSTM + LayerNorm 的組合來說，容量可能明顯過剩。測試簡化架構是否表現持平甚至更好——如果是，代表複雜度不是瓶頸（強化「無資訊」而非「架構不夠強」的結論），而且順便給你一個訓練更快、更省 VRAM 的替代架構選項（對 4060 8GB 更友善）。

**設計**：`net_time` 拿掉 `selfAttentionLayer`，只留 `fullyConnectedLayer→lstmLayer(128)→layerNormalizationLayer→dropout→fullyConnectedLayer(64)`，其餘一切（Horizon、標籤、正則化、LR、epoch）與現行 smoke test 配置相同，跑一組對照。

**判讀門檻**：若簡化版與現行複雜版的最終 Val Loss 差距在 ±0.003 內，判定「架構複雜度非瓶頸」，建議之後全量訓練直接採用簡化版本。若簡化版明顯更差，則複雜度本身可能是有用的，需要另尋方向解決「容量過剩 vs 樣本不足」的問題（例如更強的正則化、更多資料、或分階段預訓練）。

### 4.5 Round 9：訓練 Regime 排查（LR / Epoch / Early Stopping）

**目的**：排除「訓練不夠久」這個最基本的假說。現有證據（Epoch 1 到 15 Val Loss 幾乎不動）已經讓這個假說看起來不太可能，但為了診斷鏈的完整性，值得明確測過一次再排除，而不是憑印象跳過。

**設計**：固定現行架構與 5 日標籤，關掉 Early Stopping，跑到 100 epoch，同時把 Loss 記錄頻率從「每 epoch 一次」加密到「每 5~10 個 mini-batch 一次」，藉此觀察 Epoch 1 最初幾步是否**完全沒有任何下降**（這代表初始梯度信號可能太弱或被某個飽和的激活函數卡住），還是有下降但很快就在某個水平線附近震盪（這比較像是真正的資訊天花板）。可以額外測 2~3 組不同 LR schedule（現行的分段衰減 vs. 固定小 LR vs. Cosine Decay）互相對照。

**判讀門檻**：若整個 100+ epoch 過程中，Val Loss 從未低於現有 15-epoch 版本觀察到的最低點（0.6938），排除「訓練不足」假說。

### 4.6 Round 10：特徵子集精簡消融

**目的**：如果 Workstream A 掃出來只有少數幾個特徵（例如既有 HAC-ICIR 表中通過顯著性檢定的 R1、R5、RelStrength、SMA20、SMA60、MACD_Hist、RSI、Dist_H20、OBV20、Amihud20 等）真正帶有訊號，而 Beta、Corr、Dist_H252、HL_Spread、Vol20 這類目前看起來不顯著的特徵，可能只是在稀釋優化訊號、增加模型要學會忽略的雜訊維度。

**設計**：需等 Workstream A 結果出爐後才能決定精確的保留清單。用篩選後的精簡特徵子集，重跑一次現行架構（含 Self-Attention 或不含，依 Round 8b 結論而定），與全 18 維版本比較。

**判讀門檻**：若精簡版 Val Loss 相較全特徵版改善超過 0.005，且 E_Var 仍在健康範圍，判定精簡特徵有實質幫助，納入後續全量訓練設計；否則排除，維持全特徵。

---

## 5. 整合決策樹

```
Round 6 合成訊號測試
├─ FAIL → 停下來除錯 pipeline（依 4.1 節清單排查），
│         不進行後續任何輪次，先修好再回頭跑
│
└─ PASS → pipeline 機制確認無誤，繼續
    │
    ├─ Workstream A 天花板掃描（可與 Round 6 平行跑）
    │   ├─ 找到 GO/臨界 Horizon → 記錄下來，作為未來擴大驗證的候選
    │   │                          （本輪範圍內不切換標籤，留給下一階段決策）
    │   └─ 全部 NO-GO → 傾向「現有特徵集資訊含量不足」的系統性結論，
    │                    修根重心應轉往特徵工程擴充
    │
    └─ Round 8a 過擬合能力測試
        ├─ FAIL（連小樣本都學不會）→ 回頭跟 Round 6 一起除錯
        └─ PASS → 繼續 Round 7 與 Round 8b（可平行）
            ├─ Round 7：閘門有害 → 移除/改設計
            ├─ Round 7：閘門無害 → 排除此假說
            ├─ Round 8b：簡化架構持平或更好 → 換用簡化架構，繼續 Round 9/10
            └─ Round 8b：簡化架構明顯更差 → 保留複雜架構，另尋容量/樣本平衡方案
                │
                └─ 若 Round 6~9 全數指向「pipeline 正確、訊號本身極弱」
                    → 最終根因定位為「現有 28 維特徵在測試過的 Horizon 範圍內
                       資訊含量趨近於零」，這是本輪修根工作的誠實終點，
                       下一階段決策點：(a) 擴充正交特徵家族，或
                       (b) 接受此為方法論核心發現，轉向論文 5.11/6.4 的敘事重心
```

---

## 6. 與論文大綱的對應

| 本方案產出 | 對應章節 |
|---|---|
| Workstream A 多 Horizon 天花板表 | 5.1 節（可取代/擴充現有僅 1D/5D 的表 5-1） |
| Round 6 合成訊號 PASS/FAIL 結果 | 5.2 節新增一段「pipeline 正確性驗證」，強化診斷嚴謹度 |
| Round 7~10 各項消融結果 | 5.2 節收斂診斷討論的具體佐證數據 |
| 「表徵坍縮」vs「表徵無資訊」的區分本身 | 建議在 5.2 節開頭明確點出這是兩個不同層次的問題，Round 1 解決的是前者，本輪解決/排除的是後者——這個區分本身就是一個值得寫出來的方法論貢獻 |
| Day-Level Block Bootstrap 修正 | 4.5 節可以補充一段，說明橫截面層級信賴區間與崩盤護欄逐日信賴區間在依賴結構上的差異，以及為何需要不同的重抽樣單位 |
| 若最終仍全數 NO-GO | 直接支撐 6.3 研究限制與 6.4 未來工作方向（特徵擴充建議），敘事上與現有大綱完全相容，不需要重寫任何章節骨架 |

（附帶一提：現有 6.3 節寫「CPU-only 訓練規模限制了模型容量」，但 Phase 2 執行日誌顯示有抓到 RTX 4060 GPU，只有 Agent_PPO 刻意鎖 CPU。這句話最終定稿前建議修正得更精確，避免跟執行紀錄對不上。）

---

## 7. 附錄：本輪需要新建/修改的檔案總表

| 檔案 | 動作 | 說明 |
|---|---|---|
| `scripts/diagnostics/Run_CeilingScan_MultiHorizon.m` | 新建 | Workstream A 主控腳本 |
| `agents/RawBaselineTrainer.m`（或函式檔） | 新建 | 從 `GBDTExpertAgent.m` 抽出的獨立 Raw-Baseline 訓練邏輯 |
| `utils/block_bootstrap_auc_by_day.m` | 新建 | 日層級區塊自助法 AUC 信賴區間工具 |
| `scripts/diagnostics/Round6_SyntheticSanityCheck.m` | 新建（複製 SMOKE 腳本修改） | Positive Control |
| `scripts/diagnostics/Round7_ICGateAblation.m` | 新建 | 閘門開關對照 |
| `scripts/diagnostics/Round8a_OverfitCapacityCheck.m` | 新建 | 小樣本記憶力測試 |
| `scripts/diagnostics/Round8b_ArchSimplification.m` | 新建 | 簡化架構對照 |
| `scripts/diagnostics/Round9_TrainingRegimeSweep.m` | 新建 | LR/Epoch 排查 |
| `scripts/diagnostics/Round10_FeatureSubsetTrim.m` | 新建，待 Workstream A 結果 | 精簡特徵重訓 |
| `GBDTExpertAgent.m`、`Config.m`、`FeatureEvaluator.m` | **不修改** | 保持 Phase 3 正式流程不受影響，維持單一變因隔離原則 |

---

*建議每輪跑完後，把「判讀門檻」欄位實際數字回填進本文件，累積成你自己的第 6 輪起 smoke test 紀錄，之後可以整批貼給 Claude 更新 memory，或直接作為論文附錄 C 的延伸素材。*
