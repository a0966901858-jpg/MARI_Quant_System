# MARI Quant System 修正計劃書
### Phase 14.23 — 大幅重構版：從「Loss 卡在 0.6931」到「獲利翻正」的根因鏈路修復

---

## 0. 方法論說明

這份計畫書不是逐檔案平均用力的清單，而是**根因鏈路分析（Root-Cause Chain Analysis）**。理由很簡單：你的系統是一條嚴格的資料流水線，下游模組的品質上限由上游模組決定。如果 Phase 2 的 Embedding 是雜訊，那麼不管 Phase 3 的 GBDT 調得多好、Phase 4 的 BO 搜索多細、Phase 5 的 RL 訓練多久，都只是在雜訊上蓋房子。

**優先級標記說明：**
- 🔴 **P0（阻斷級）**：不修就沒有意義往下走
- 🟡 **P1（重要）**：會直接影響能否「獲利翻正」，但不阻斷驗證流程
- 🟢 **P2（次要）**：程式碼衛生/穩健性，優化空間但非燃眉之急

---

## 1. 執行摘要：根因鏈路圖

```
Phase 2 (BuildDecoupledExtractors + GraphSpatialFusionLayer)
  └─ 🔴 Train/Val BCE Loss 30 epoch 全程釘死在 0.6931 (=ln2，等於亂猜)
     └─ E_time_all / E_space_all (64維 Embedding) = 純雜訊
        │
        ▼
Phase 3 (GBDTExpertAgent)
  └─ 🟡 P_time / P_space 實質上只能靠共用的 4 維 Macro 特徵撐訊號
     └─ P_time ≈ P_space（虛假多樣性 False Diversification）
        │
        ▼
Phase 4 (Run_BO_Hyperparameter_Tuning)
  └─ 🟡 在兩個幾乎相同的雜訊訊號間尋優 → Expert_Time_Weight=0.8998（邊界解）
  └─ 🔴 Guardrail 下界從 p75 放寬到 p50 → 護欄閾值只比"平均崩盤機率"高一點點
     └─ 25 次評估的代理目標函數 IR=6.12 是雜訊過擬合的假象
        │
        ▼
Phase 5 (Agent_PPO + Run_CIO_Awakening)
  └─ 🟡 500 epoch 訓練，三個 Agent 的 Reward 全程負值不收斂
  └─ 🟡 非對稱懲罰 Reward (下檔平方懲罰 vs 上檔線性獎勵) 可能誘導"退化為全現金"策略
        │
        ▼
Phase 6 (Run_WalkForward_Backtest)
  └─ IS: 總報酬 -95.03%, MDD -95.03%, Sharpe -2.99 (護欄頻繁觸發、現金鎖死)
  └─ OOS: 總報酬 0.00%, MDD 0.00% (幾乎全程 100% 現金，等同沒有交易)
```

**結論：Phase 2 是唯一的「真因」，Phase 3~6 的異常都是它的下游症狀。** 因此本計畫書把最大篇幅放在 Phase 2，但為了達成「大幅重構」與「獲利翻正」的目標，Phase 4~6 的設計缺陷也必須同步修正——即使 Phase 2 修好了，護欄閾值與 Reward 設計問題依然會讓系統傾向於「不交易」。

---

## 2. 🔴【P0】Phase 2 深度診斷與重構

檔案：`models/GraphSpatialFusionLayer.m`、`models/BuildDecoupledExtractors.m`、`scripts/Run_Extractor_Pretrain.m`

### 2.1 空間專家（DyGAT）訊號衰減——可證明的數學根因

**問題程式碼**（`GraphSpatialFusionLayer.m` 建構子）：
```matlab
% 刻意使用極小的初始值 (0.001)，讓模型初期依賴靜態圖譜，防禦梯度消失
obj.W_attn_1 = dlarray(randn(attnDim, obj.NumFeatures, 'single') * 0.001);
obj.W_attn_2 = dlarray(randn(attnDim, obj.NumFeatures, 'single') * 0.001);
```

以及 `predict()` 中的 softmax：
```matlab
S_exp = exp(S_raw - max(S_raw, [], 1));
S_attn = S_exp ./ (sum(S_exp, 1) + 1e-8);   % ← 對「全部 504 檔」做 dense softmax
A_dynamic = A_norm_3D .* S_attn;             % ← Hadamard 乘積
```

**推導（可自行代入驗證）：**

1. `Q(d,j) = Σ_f W_attn_1(d,f)·X(f,j)`，其中 22 個 f 的 X 已做橫截面 Z-score（≈N(0,1)），`W_attn_1 ~ N(0, 0.001²)`。所以 `Q(d,j) ~ N(0, 22×0.001²)`，標準差 ≈ 0.0047。K 同理。

2. `S_raw(i,j) = Σ_d Q(d,i)·K(d,j) / √22`，22 個乘積項相加後 `S_raw` 量級 ≈ **1×10⁻⁴**。

3. 這麼小的 `S_raw`，`exp(S_raw - max)` 對所有 (i,j) 都近似等於 1（因為指數的自變數幾乎是 0）。於是 softmax 結果 `S_attn(i,j) ≈ 1/504`——**對全部 504 個節點近乎均勻分布**，而不是只集中在真正的圖譜鄰居上（因為 softmax 是對整個 504 維做的，沒有用鄰接矩陣做 mask）。

4. `A_norm_3D` 本身是 column-stochastic（每個節點的入邊權重加總為 1，但只分布在約 20~50 個真實鄰居上，所以每個非零項 ≈ 1/degree）。

5. `A_dynamic = A_norm_3D .* S_attn`：每個真實邊的權重 ≈ `(1/degree) × (1/504)`。**也就是說，不管訓練到第幾個 epoch、不管 attention 邏輯設計得多精巧，一開始就把每條邊的訊號強度砍到只剩原本的 1/504 ≈ 0.2%。**

6. 這個近乎全零的 `A_dynamic` 直接餵進 `Agg_3D = pagemtimes(X_3D, A_dynamic)`，聚合後的特徵量級被壓到幾乎消失。後續 `Weights * Agg_flat + Bias` 在 Bias 初始為 0 的情況下，輸出也幾乎是常數 0（經過 ReLU 後大概率仍是 0 或極小值）。

7. **後果**：`E_space` 這個 64 維 Embedding 對「不同股票、不同天」幾乎輸出同一組值（低變異數）。線性探測頭 `logits = W_aux_space * E_unfmt + b_aux_space` 因此對所有樣本輸出近乎相同的 logit → `Y_pred ≈ 常數 p`。當預測機率是常數時，BCE 的期望值恰好在 `p=0.5` 時取得 `-log(0.5) = ln2 = 0.6931`——**這正是你在 log 裡看到的數字，且完全不會隨 epoch 改變（因為梯度本身就被這個結構性的 500 倍衰減鎖死了）**。

> 📌 這條推導直接命中了「防禦梯度消失」這個 comment 本身的自我矛盾：**設計者原本想避免梯度消失，但 dense softmax 的正規化機制導致這個極小初始值反而製造出更嚴重的梯度消失。**

**時序專家（Transformer-LSTM）沒有這個結構性 bug**，`selfAttentionLayer` 用的是 MATLAB 內建標準初始化，沒有人為縮小 1000 倍。但它的 Loss 也同樣釘死在 0.6931，這代表還有第二個共同因素在起作用（見 2.3、2.4）。

### 2.2 修正碼：`GraphSpatialFusionLayer.m`

**修正原則：把 dense softmax 改成 masked softmax（只在真實圖邊上分配注意力質量），並改用凸組合取代 Hadamard 乘積（避免兩個「和為1」的分布相乘導致連乘衰減）。**

```matlab
% ============ predict() 內部，取代原本的 C~D 區塊 ============

% C. Masked Softmax：只允許真實圖結構(含自環)上分配注意力質量
%    這是標準 GAT (Graph Attention Network) 的做法，避免 dense softmax
%    把注意力質量稀釋到全部 504 個節點上
neg_inf = single(-1e9);
mask = (A_3D == 0);                 % [Nodes, Nodes, Batch] 邏輯遮罩
S_raw(mask) = neg_inf;              % 非鄰居直接打成 -inf，softmax 後為 0

S_exp = exp(S_raw - max(S_raw, [], 1));
S_attn = S_exp ./ (sum(S_exp, 1) + 1e-8);

% D. ★ 修正：改用凸組合而非 Hadamard 乘積
%    原因：A_norm_3D 與 S_attn 都是「和為1」的機率分布，逐項相乘等於
%    做二次收縮 (每項變成原本的平方量級)，數值上會持續偏小。
%    凸組合可以讓網路自己學習靜態圖結構與動態注意力之間的權重。
alpha_mix = 0.5;   % 可設為 obj.Learnable 讓網路自己學(見下方進階選項)
A_dynamic = alpha_mix * A_norm_3D + (1 - alpha_mix) * S_attn;
```

**初始化也要跟著調整**（因為現在有 mask 保護，不再需要用極小值人工抑制訊號）：
```matlab
% 建構子內，改用標準 Xavier/He 尺度，而非 *0.001
attnDim = obj.NumFeatures;
obj.W_attn_1 = dlarray(randn(attnDim, obj.NumFeatures, 'single') * sqrt(1/obj.NumFeatures));
obj.W_attn_2 = dlarray(randn(attnDim, obj.NumFeatures, 'single') * sqrt(1/obj.NumFeatures));
```

**進階選項（若要更嚴謹，可將 `alpha_mix` 設為可學習純量參數）：**
```matlab
properties (Learnable)
    ...
    MixLogit    % 純量，經 sigmoid 轉換為 [0,1] 的 alpha_mix，讓網路自己決定要多信任靜態圖 vs 動態注意力
end
% 建構子中：obj.MixLogit = dlarray(single(0));   % sigmoid(0)=0.5，初始為對半
% predict() 中：alpha_mix = sigmoid(obj.MixLogit);
```

### 2.3 順帶發現的次要正確性問題（🟡P1）

在檢視 `A_3D` 構建邏輯時發現：**圖結構的有效節點判定（`FeatureEngineer.m` 裡的 `valid_nodes`，基於相關性視窗無 NaN）跟特徵活躍遮罩（`Expert_Active`，基於流動性/成分股）是兩套不同的 mask**。這代表可能存在「某節點在圖上有邊，但其 `X_norm_3D` 特徵已被 Expert_Active 強制歸零」的情況。雖然這類節點對聚合結果的**貢獻**是 0（乘出來就是 0），但它仍然佔用了 `deg = sum(A_3D,1)` 的度數正規化分母，稀釋了真正活躍鄰居應得的權重份額。

**建議修正**（`FeatureEngineer.m` 的圖構建迴圈中）：
```matlab
% 在 valid_nodes 判定時，額外交集 Expert_Active(t,:)，確保圖結構與特徵活躍遮罩完全一致
valid_nodes = all(~isnan(window_rets), 1) & all(~isnan(window_logP), 1);
% ★ 新增：與當前計算錨點 t 當天的活躍遮罩交集(在 parfor 迴圈內可傳入 Expert_Active(t,:))
```
> 這不是造成 Loss 卡死的主因，但屬於資料一致性的正確性瑕疵，建議一併修正。

### 2.4 標籤設計問題：單日中位數切分訊噪比過低

**問題**：目前的輔助預訓練標籤是「次日報酬是否高於當日橫截面中位數」。在效率接近的大型美股市場中，**單日報酬幾乎完全由雜訊主導**（財經文獻中，日頻個股報酬的可預測性極弱，這是動量因子研究普遍在 1~12 個月週期才有穩定訊號的原因）。這解釋了為什麼**時序專家**（沒有結構性 bug）依然學不動——不是模型的錯，是任務本身訊噪比太低。

**修正**（`Run_Extractor_Pretrain.m` 步驟 2）：
```matlab
% 舊：單日 forward return 中位數切分
% 新：改用 5 日遠期報酬，訊噪比顯著提升，且仍嚴格無未來函數
horizon = 5;
R_fwd = NaN(numDaysRaw, numT, 'single');
R_fwd(1:end-horizon, :) = (Prices_Active(1+horizon:end, :) - Prices_Active(1:end-horizon, :)) ...
                          ./ Prices_Active(1:end-horizon, :);
R_fwd(isinf(R_fwd)) = NaN;

Y_Labels_3D = zeros(numDaysRaw, numT, 'single');
for t = 1:numDaysRaw-horizon
    active_mask = Expert_Active(t, :);
    if sum(active_mask) > 10
        med_ret = median(R_fwd(t, active_mask), 'omitnan');
        Y_Labels_3D(t, active_mask) = single(R_fwd(t, active_mask) > med_ret);
    end
end
```
> ⚠️ 注意：這個 5 日窗口的標籤**只用於 Phase 2 的輔助預訓練任務**，目的是強迫 Embedding 空間學出有結構的表徵。Phase 3 的 GBDT 最終選股標籤（`Run_GBDT_and_SHAP.m`）可以維持原本的 1 日標籤（因為那是實際交易執行頻率），兩者不需要一致——預訓練任務只是要讓 Embedding「學會辨認出哪些股票長得像贏家」，不必是最終決策任務本身。

### 2.5 輔助預測頭訊號強度不足

```matlab
% 舊：初始化過小，梯度訊號從一開始就微弱
W_aux_time  = dlarray(randn(1, 64, 'single') * 0.01);

% 新：改用標準 He 初始化尺度
W_aux_time  = dlarray(randn(1, 64, 'single') * sqrt(2/64));
b_aux_time  = dlarray(zeros(1, 1, 'single'));
W_aux_space = dlarray(randn(1, 64, 'single') * sqrt(2/64));
b_aux_space = dlarray(zeros(1, 1, 'single'));
```

**進階選項（建議一併做，屬於「大幅重構」範疇）**：把單一線性探測頭升級為 2 層 MLP，避免探測頭本身表達能力不足掩蓋了 Embedding 是否真的學到東西：
```matlab
% 新增中間隱藏層 (64 -> 16 -> 1)
W_aux_time_h = dlarray(randn(16, 64, 'single') * sqrt(2/64));
b_aux_time_h = dlarray(zeros(16, 1, 'single'));
W_aux_time_o = dlarray(randn(1, 16, 'single') * sqrt(2/16));
b_aux_time_o = dlarray(zeros(1, 1, 'single'));

% aux_loss_time 內部改為：
% h = relu(W_aux_time_h * E_unfmt + b_aux_time_h);
% logits = W_aux_time_o * h + b_aux_time_o;
% [grad_net, grad_Wh, grad_bh, grad_Wo, grad_bo] = dlgradient(loss, net.Learnables, W_aux_time_h, b_aux_time_h, W_aux_time_o, b_aux_time_o);
```
> 需同步修改 `Run_Extractor_Pretrain.m` 中的梯度累積與 `adamupdate` 呼叫（多兩組參數），工程量不大但要小心別漏改驗證迴圈與最終推論迴圈的呼叫介面。

### 2.6 新增診斷埋點（強烈建議，直接對應你的「減少嘗試次數」需求）

在 `Run_Extractor_Pretrain.m` 訓練迴圈中加入以下診斷輸出，這樣**下次跑完不用等到全部 30 epoch 才知道有沒有救**：

```matlab
% 在每個 epoch 結尾，Loss 列印那行之前加入：
if mod(epoch, 1) == 0
    % 1. 梯度範數健檢：確認梯度真的非零、沒有爆炸/消失
    gnorm_t = sqrt(sum(cellfun(@(x) sum(x(:).^2), grad_t_accum.Value)));
    gnorm_s = sqrt(sum(cellfun(@(x) sum(x(:).^2), grad_s_accum.Value)));
    
    % 2. Embedding 變異數健檢：確認輸出不是常數坍縮
    e_t_sample = extractdata(predict(net_time, X_batch_time));
    e_s_sample = extractdata(reshape(predict(net_space, X_batch_space, A_batch_space), 64, []));
    fprintf('    [診斷] GradNorm(T:%.2e, S:%.2e) | E_Var(T:%.4f, S:%.4f)\n', ...
        gnorm_t, gnorm_s, var(e_t_sample(:)), var(e_s_sample(:)));
end
```
- 若 `E_Var(S)` 長期趨近 0 → 空間專家仍有坍縮，回頭檢查 2.2 的修正是否確實生效。
- 若 `GradNorm` 非常小（<1e-6）→ 代表梯度傳遞仍有問題，需要進一步排查（而不是又跑一次 30 epoch 才發現）。

### 2.7 建議先做「小規模驗證」，再決定要不要跑全量 Pipeline

你提到每次全跑很久，這裡給一個**低成本驗證流程**：

1. **不用重跑 Phase 1**（`features_denoised.mat` 已經快取好了，直接沿用）。
2. 複製一份 `Run_Extractor_Pretrain.m` 為 `Run_Extractor_Pretrain_SMOKE.m`，加入：
   ```matlab
   % 在載入完 X_norm_3D 後，立刻做子集抽樣
   sub_tickers = randsample(numT, min(60, numT));
   sub_days_range = round(numDaysRaw*0.6) : round(numDaysRaw*0.6) + 1500;  % 抓連續1500天
   
   X_norm_3D = X_norm_3D(sub_days_range, :, sub_tickers);
   AdjMatrix_3D = AdjMatrix_3D(sub_tickers, sub_tickers, sub_days_range);
   Prices_Active = Prices_Active(sub_days_range, sub_tickers);
   Expert_Active = Expert_Active(sub_days_range, sub_tickers);
   Dates_Active = Dates_Active(sub_days_range);
   ```
   並把 `epochs` 從 30 降到 5~8，`configObj.NumTickers` 暫時覆寫為 60。
3. 這樣的規模在純 CPU 上大概幾分鐘內就能跑完 5~8 epoch。**只要看到 Loss 從 0.6931 開始明顯下降（哪怕只降到 0.68~0.65），就代表梯度流動已修復，可以放心跑全量**；如果小樣本測試 5 epoch 後 Loss 仍然死釘在 0.6931，代表還有其他 bug 沒抓到，不需要浪費時間跑全量，直接回來加診斷埋點debug。

---

## 3. 🟡【P0連動】Phase 3：GBDT 訊號可信度驗證

檔案：`agents/GBDTExpertAgent.m`、`scripts/Run_GBDT_and_SHAP.m`

### 3.1 虛假多樣性（False Diversification）問題

`P_time` 和 `P_space` 目前都是「64維(可能是雜訊的)Embedding + 4維Macro特徵」餵進獨立訓練的 GBDT。**如果 Embedding 是雜訊，那 GBDT 實質上只能從共用的 4 維 Macro 特徵中榨取訊號**——這代表 `P_time` 和 `P_space` 事實上高度相關，被當成兩個獨立訊號源疊加，是一種「假多樣性」。這正好解釋了 Phase 4 的 BO 為什麼會選出 `Expert_Time_Weight=0.8998` 這種邊界解：兩個幾乎相同的訊號在配權重時，BO 找到的只是 IS 樣本內哪個「雜訊版本」剛好跟歷史雜訊對得更準，沒有任何實質意義。

**建議新增 Sanity Baseline**（修 Phase 2 後、跑 Phase 3 之前先做）：
```matlab
% 在 Run_GBDT_and_SHAP.m 中新增一段：純 Macro 4 維 baseline 模型
% 用來對照「加入 Embedding 後 AUC 有沒有顯著提升」
mdl_macro_only = fitcensemble(Macro_IS, Y_categorical_IS, 'Method', 'LogitBoost', ...
    'Learners', t_tree, 'NumLearningCycles', 30, 'LearnRate', 0.1);
[~, score_macro] = predict(mdl_macro_only, Macro_OOS_or_valfold);
auc_macro_only = ... % 計算 AUC

fprintf('📊 [Sanity Check] 純 Macro Baseline AUC: %.4f | 含 Embedding 版本 AUC: %.4f\n', ...
    auc_macro_only, overall_auc_t);
```
若兩者幾乎相同（差距 < 0.01~0.02），代表 Embedding 完全沒貢獻，必須回頭確認 Phase 2 修正是否真的生效。**這是判斷「Phase 2 有沒有真的修好」最直接的驗收指標，比看 Loss 數字更貼近最終目標。**

### 3.2 補充：務必記錄並提供完整 OOF AUC 輸出

你先前提供的執行過程 log 剛好在 Phase 3 印出「全域 OOF AUC」那幾行之前被截斷。**強烈建議下次執行後把這幾行完整記錄下來**（`GBDTExpertAgent.m` 第 122 行附近的 `fprintf`），這是判斷目前 GBDT 到底有沒有撈到任何訊號的關鍵證據。

---

## 4. 🔴【P0連動】Phase 4：BO 護欄閾值設計錯誤與過擬合風險

檔案：`scripts/Run_BO_Hyperparameter_Tuning.m`

### 4.1 護欄閾值下界設計錯誤（直接導致 OOS 0% 報酬）

```matlab
% ★ 二次優化修正 1：放寬下界至 p_50（中位數），避免搜索空間被卡死在 75 百分位數
p_50 = prctile(P_crash_IS, 50);
p_99 = prctile(P_crash_IS, 99);
var_guard  = optimizableVariable('Guardrail_CrashProb', [p_50, p_99], 'Type', 'real');
```

**問題**：`P_crash` 若是一個經過良好校準的機率（0=不會崩盤, 1=會崩盤），那麼把閾值下界設在「中位數」意味著**系統有將近 50% 的交易日會被判定為潛在崩盤日**。這在物理意義上不合理——真正的市場崩盤（如你的標籤定義：次日跌幅 >0.5%）發生頻率遠低於 50%。BO 最終選中 `Guardrail_CrashProb=0.5306`，幾乎等於 `p_50=0.5302`（下界邊界解），這代表 BO 在代理模擬器裡發現「幾乎永遠持有現金」在雜訊訊號的環境下反而是相對安全（IS 損失較小）的策略——**這不是找到好策略，是在一個沒有真實 Alpha 的環境裡找到「少犯錯」的策略，而少犯錯的方法就是不交易**。

**修正建議**：
```matlab
% 下界改回合理範圍（例如 p80~p85），避免護欄長期觸發
p_80 = prctile(P_crash_IS, 80);
p_99 = prctile(P_crash_IS, 99);

if isnan(p_80) || isnan(p_99) || p_80 >= p_99
    p_80 = 0.70;
    p_99 = 0.95;
end
var_guard  = optimizableVariable('Guardrail_CrashProb', [p_80, p_99], 'Type', 'real');
```
**同時建議在 BO 結果輸出後加入合理性檢查**：
```matlab
% 在步驟5之後新增：
pct_days_guarded = mean(P_crash_IS > best_params.Guardrail_CrashProb) * 100;
fprintf('  > 護欄閾值觸發比例（IS期間）: %.1f%% 的交易日會被鎖現金\n', pct_days_guarded);
if pct_days_guarded > 15
    warning('⚠️ 護欄觸發比例過高，可能導致系統長期空手，建議提高閾值下界！');
end
```

### 4.2 BO 目標函數容易過擬合到雜訊

25 次評估、3 個變數、單一 IS 大窗口——這個搜索方式對「訊號本身很弱或是雜訊」的情境特別脆弱，因為 BO 很容易找到剛好跟這一段歷史雜訊對得上的參數組合，而這個組合對 OOS 完全沒有泛化能力（你已經在 Phase 6 親眼看到這個落差：BO 宣稱 IR=6.12，真實 IS 回測卻是 -95%）。

**建議修正**（提升 BO 目標函數的穩健性）：
```matlab
% 將 evaluate_hrl_proxy 目標函數改為「多個滾動子窗口的平均 IR」而非單一大窗口
% 例如把 IS 區間切成 4 個不重疊的 4 年子窗口，分別算 IR 後取平均(或懲罰變異數)
function neg_IR = evaluate_hrl_proxy_robust(params, Prices, Opens, Expert, P_time, P_space, P_crash, config)
    numDays = size(Prices, 1);
    n_folds = 4;
    fold_edges = round(linspace(1, numDays, n_folds+1));
    fold_IRs = zeros(n_folds, 1);
    for f = 1:n_folds
        rng_f = fold_edges(f):fold_edges(f+1);
        % ... 呼叫原本的模擬邏輯，但只在 rng_f 子區間內跑 ...
        fold_IRs(f) = compute_IR_for_subrange(...);
    end
    % 懲罰跨窗口不穩定的參數組合(標準差大代表不穩健)
    neg_IR = -( mean(fold_IRs) - 0.5*std(fold_IRs) );
end
```
這樣選出來的參數組合會傾向「在多段不同市況下都表現穩定」，而不是「剛好完美擬合某一段歷史」，能顯著降低過擬合風險。

### 4.3 追加建議：BO 迭代次數與邊界解警示

`MaxObjectiveEvaluations=25` 對 3 維空間來說偏少。且如果重跑後**又**選中邊界值（如 `Top_K` 貼著 10 或 40，`Expert_Time_Weight` 貼著 0 或 1），這是一個明確的警示訊號，代表搜索空間邊界設定不合理，或訊號依然不可信，建議加入自動化檢查：
```matlab
boundary_tol = 0.02;
if abs(best_params.Expert_Time_Weight - 0) < boundary_tol || abs(best_params.Expert_Time_Weight - 1) < boundary_tol
    warning('⚠️ Expert_Time_Weight 落在搜索邊界，可能代表 P_time/P_space 高度相關(虛假多樣性)，請檢查 Phase2/3！');
end
if best_params.Top_K_Assets <= 11 || best_params.Top_K_Assets >= 39
    warning('⚠️ Top_K_Assets 落在搜索邊界，可能代表選股訊號不穩健，過度集中或過度分散！');
end
```

---

## 5. 🟡【P1】Phase 5：RL 獎勵函數設計與現金鎖死疊加效應

檔案：`agents/Agent_PPO.m`、`scripts/Run_CIO_Awakening.m`

### 5.1 非對稱獎勵可能誘導「退化為全現金」策略

```matlab
rew_cio(pos_mask) = excess_return(pos_mask) * 100.0;
raw_penalty = (abs(excess_return(neg_mask)) * 100.0).^2;
rew_cio(neg_mask) = -min(raw_penalty, 100.0);
```

上檔是**線性**給分，下檔是**平方**懲罰（且封頂在 -100，但 -100 這個懲罰量級遠大於一般線性獎勵所能給的量級，例如 1% 超額報酬只給 +1 分，但 -1% 超額報酬要扣 -1 分、-5% 超額報酬要扣 -25 分）。這種設計在訊號本身微弱、模型還沒學會辨識真正機會的訓練早期，會讓「不冒險、保持現金」成為局部最優策略——因為每一次錯誤的下注都被放大懲罰，而正確下注的獎勵沒有對等放大。這與你在 log 中觀察到的「500 epoch 後三個 Agent 的 Reward 依然全數為負」高度吻合。

**修正建議**：把懲罰改為更溫和的凸函數（例如 Huber-like，小損失線性、大損失才二次方），並加入「現金持倉懲罰」避免退化解：
```matlab
% 溫和化懲罰：小損失線性，只有超過某個閾值才二次方放大
loss_threshold = 0.02;  % 2%
small_loss_mask = neg_mask & (abs(excess_return) <= loss_threshold);
large_loss_mask = neg_mask & (abs(excess_return) > loss_threshold);

rew_cio(small_loss_mask) = excess_return(small_loss_mask) * 100.0;  % 線性，跟正報酬對稱
rew_cio(large_loss_mask) = -( loss_threshold*100 + (abs(excess_return(large_loss_mask)) - loss_threshold*100).^2 );
rew_cio(large_loss_mask) = max(rew_cio(large_loss_mask), -100);  % 保留封頂防爆

% ★ 新增：懲罰長期空手，避免 RL 學到"什麼都不做最安全"的退化解
cash_penalty_coeff = 0.05;
rew_cio = rew_cio - cash_penalty_coeff * (w_cash > 0.95);  % 現金比例>95%時扣分
```

### 5.2 訓練迭代數與收斂性

500 epoch 但 Reward 沒有明確收斂趨勢（只是在負值區間震盪），這可能是：(a) 訊號太弱（Phase 2 修好後應該會改善）(b) `Early Stopping` 的 `patience_limit=30` 搭配 `ep>100` 才觸發，可能過早鎖定了一個次優快照。修 Phase 2 之後，建議先觀察 Reward 曲線是否出現明確上升趨勢，若依然持平，再考慮加大 `HRL_Epochs` 或調整 `ActionStd`(探索幅度)。

---

## 6. 🟡【P1】Phase 6：回測邏輯檢視

檔案：`scripts/Run_WalkForward_Backtest.m`

好消息是：**這支腳本本身的因果律邏輯是嚴謹的**（T+1 執行、摩擦成本統一讀取 Config、IS/OOS 嚴格分離、破產防護機制都有做對），這部分不需要大改。問題全部來自餵給它的上游訊號與護欄閾值（已在第 2~4 節處理）。

**唯一建議新增的診斷可視化**：在子圖區塊加入「崩盤機率 vs 護欄線」疊圖，方便未來肉眼快速診斷是否又發生鎖死現金的情況：
```matlab
% 在子圖 3 之後新增子圖 4
subplot(4,1,4);
plot(Dates_Active(valid_start_t:end), P_crash_smooth(valid_start_t:end), 'Color', '#7E2F8E', 'LineWidth', 1); hold on;
yline(opt_guard, '--r', sprintf('Guardrail=%.3f', opt_guard), 'LineWidth', 1.5);
title('崩盤機率 vs 護欄閾值（診斷用）');
ylabel('P(crash)'); grid on;
```

---

## 7. 🟡 Phase 1 特徵工程：增減建議

檔案：`data/FeatureEngineer.m`

### 7.1 建議移除／合併（共線性高、邊際資訊低）

| 特徵 | 問題 | 建議 |
|---|---|---|
| `SMA20`, `SMA60`（均線乖離率） | 本質上是動量的移動平均版本，與 `R20` 高度線性相關 | 保留一個，或改用 `SMA20 - SMA60`（乖離差，資訊密度更高） |
| `MACD_Line`, `MACD_Sig` | Sig 是 Line 的 EMA 平滑，兩者天生高相關 | 改用 `MACD_Histogram = MACD_Line - MACD_Sig`，只留這一維 |
| `RSI`, `MFI` | 分別是價格動能、資金流動能的相對強弱，方向性高度重疊 | 保留其一，或做正交化（用 `MFI - RSI` 殘差保留獨立資訊） |
| `OBV_20`, `VPT_20` | 都是量價關係指標，功能重疊 | 保留其一 |

移除/合併後預期把 22 維壓到約 17~18 維，可降低模型過擬合風險，也能減輕橫截面標準化時的雜訊放大效應。

### 7.2 建議新增（提升訊噪比）

| 新特徵 | 理由 |
|---|---|
| **60日、120日動量**（中長期報酬率） | 學術上（Jegadeesh & Titman 經典動量效應）美股中期動量（3~12個月）比日頻報酬穩健得多，訊噪比遠優於現有的 R1/R5 |
| **Beta中性化後的殘差動量（Alpha Momentum）** | 用市場模型剔除 Beta 暴露後的殘差報酬動量，比原始報酬更接近「選股能力」而非「隨大盤漲跌」 |
| **隔夜跳空 vs 日內報酬分解** | Overnight gap 與 Intraday return 在文獻中有不同的訊息含量與均值回歸特性，分開建模資訊量更豐富 |
| （若能取得財報資料）**估值因子**（P/E, P/B） | Fama-French 因子研究顯示，價值因子在長期選股上比純技術指標穩定，可與技術面互補 |

---

## 8. 🟢【P2】VQ-VAE 與 Archive 檔案

- `agents/VQVAEAgent.m`：訓練後 Recon Loss 0.055 / Commit Loss 0.048，數值收斂正常，非本次問題主因，暫不需要修改，但建議在下次跑完 Phase 2 修正後，順手檢查降噪前後 R1/R5/R20 三維特徵的橫截面 IC 是否有變好或變壞（用 `FeatureEvaluator.m` 現有工具即可），確認 VQ-VAE 沒有把有用的高頻訊號一併磨掉。
- `Archive/HRLEnvironment.m`、`Archive/vqvaeLoss.m`、`Archive/HRLManagerAgent.m`：確認不在六個 Phase 腳本的實際執行路徑中（已核對 `addpath` 只加了 `configs/data/envs/models/agents`，而 `envs/` 底下實際使用的是 `Agent_PPO.m` 的向量化模擬邏輯，非 `HRLEnvironment.m`），**本次不需要修改**，建議之後直接刪除或移到 `_deprecated/` 更明確標示，避免路徑混淆。

---

## 9. 驗證 Checklist（防止修完又陷入新的過擬合/無效學習）

- [ ] 小樣本 smoke test（第 2.7 節）：Loss 是否從 0.6931 明顯下降？
- [ ] 診斷埋點（第 2.6 節）：GradNorm 是否非零？Embedding 變異數是否 > 0？
- [ ] Phase 3 Sanity Baseline（第 3.1 節）：加入 Embedding 後 AUC 是否顯著優於純 Macro baseline（差距應 > 0.02~0.03）？
- [ ] Phase 4 邊界解警示（第 4.3 節）：BO 選出的參數是否又貼著搜索邊界？
- [ ] Phase 4 護欄觸發比例（第 4.1 節）：IS 期間護欄觸發比例是否 < 15%？
- [ ] Phase 5 Reward 曲線：500 epoch 訓練是否出現明確上升趨勢（而非持平震盪）？
- [ ] Phase 6：IS 回測是否不再出現連續多年「現金:100%, 持股:0檔」？
- [ ] Phase 6：OOS 報酬是否不再是精確的 0.00%？

---

## 10. 建議執行順序（分階段驗證，降低重跑成本）

```
第一階段（分鐘級，CPU 即可）
  1. 套用 2.2、2.3、2.4、2.5 的程式碼修正
  2. 執行 2.7 的小樣本 smoke test，確認 Loss 能脫離 0.6931
     └─ 若失敗：加開 2.6 診斷埋點，找出剩餘的梯度阻塞點，不要跳過這步直接跑全量

第二階段（全量 Phase 1~3，約需你原本 Phase1-3 的時間）
  3. 全量重跑 Phase 1（若特徵工程有改第7節的特徵增減，才需要重跑；否則可沿用快取）
  4. 全量重跑 Phase 2（含修正後架構）
  5. 執行 3.1 的 Sanity Baseline 比對，確認 Embedding 真的有貢獻
  6. 全量重跑 Phase 3，記錄完整 OOF AUC 輸出

第三階段（Phase 4~6）
  7. 套用 4.1、4.2、4.3 的護欄與 BO 目標函數修正後執行 Phase 4
     └─ 檢查 4.3 的邊界解警示是否觸發
  8. 套用 5.1 的 Reward 函數修正後執行 Phase 5
     └─ 觀察訓練曲線是否收斂到正值區間
  9. 執行 Phase 6，對照第 9 節 Checklist 全數確認
```

---

## 11. 檔案修改總覽表

| 檔案 | 優先級 | 修改原因 | 修改摘要 |
|---|---|---|---|
| `models/GraphSpatialFusionLayer.m` | 🔴P0 | Dense softmax 稀釋注意力訊號約500倍，數學可證 | Masked softmax + 凸組合取代 Hadamard 乘積 + 調整初始化尺度 |
| `scripts/Run_Extractor_Pretrain.m` | 🔴P0 | 標籤訊噪比過低、探測頭初始化過小、缺乏診斷埋點 | 標籤改5日窗口、探測頭升級、新增梯度/變異數診斷 |
| `data/FeatureEngineer.m` | 🟡P1 | 圖結構mask與特徵活躍mask不一致；特徵共線性高 | valid_nodes交集Expert_Active；特徵增減（第7節） |
| `scripts/Run_BO_Hyperparameter_Tuning.m` | 🔴P0 | 護欄閾值下界不合理導致現金鎖死；目標函數易過擬合 | 下界改回p80；改多子窗口平均IR目標函數；加邊界解警示 |
| `agents/GBDTExpertAgent.m` / `scripts/Run_GBDT_and_SHAP.m` | 🟡P1 | 無法驗證Embedding是否真的貢獻訊號 | 新增純Macro Baseline比對 |
| `agents/Agent_PPO.m` / `scripts/Run_CIO_Awakening.m` | 🟡P1 | 非對稱Reward可能誘導退化為全現金策略 | 懲罰函數溫和化 + 加入現金持倉懲罰項 |
| `scripts/Run_WalkForward_Backtest.m` | 🟢P2 | 邏輯本身嚴謹，僅需新增診斷可視化 | 新增崩盤機率vs護欄線疊圖 |
| `agents/VQVAEAgent.m` | 🟢P2 | 目前無明確問題，列為觀察項 | 修完Phase2後順手檢查降噪前後IC變化 |

---

*本計畫書基於現有程式碼與執行日誌的靜態分析與數學推導完成。第2.1節的訊號衰減推導具高確信度；第2.4節（標籤訊噪比）對時序專家的解釋力則需透過2.6節診斷埋點與2.7節smoke test實際驗證後才能完全確認，建議依第10節順序逐步驗證，避免一次性大改後難以定位問題。*
