# MARI Quant System 修正計劃書 — 附錄：Phase 14.24 大規模優化擴充計畫

> 本文件承接 `MARI_修正計劃書_Phase14.23.md`，記錄 Smoke Test 驗證結果，並依據使用者確認的三項條件（絕對報酬>0為初階目標／打敗SPY為終極目標、資料源可自由擴充、時間充裕可大規模優化）展開後續工作。

---

## 1. Smoke Test 結果存檔與判讀

| 指標 | 時序專家 (T) | 空間專家 (S) |
|---|---|---|
| Train Loss (Ep1→Ep6) | 0.6943 → 0.6778 (↓2.38%) | 0.8695 → 0.6961 (↓19.94%) |
| Val Loss (Ep1→Ep6) | 0.6970 → **0.7091 (↑ 單調上升)** | 0.7634 → 0.6980 (↓，趨勢正常) |
| GradNorm (末epoch) | 1.99e-01 | 9.82e-02 |
| E_Var (末epoch) | 0.0745 | 1.1399 |

**結論：**
1. ✅ **核心根因已解除**：兩分支 GradNorm 皆非零、E_Var 皆非坍縮量級，證實 Phase 14.23 的 Masked Softmax + 5日標籤 + 探測頭修正確實讓梯度重新流動。這是本次驗證最重要的目標，**已達成**。
2. ⚠️ **新發現：時序專家在小樣本下出現訓練/驗證曲線分岔**（過擬合訊號）。這在 60檔×1500天 的小樣本規模下部分可歸因於「有效樣本數遠低於表面樣本數」（同期間 60 檔股票的滾動窗口高度重疊、宏觀特徵逐日共用），但仍需在正式全量訓練前加上正則化機制以策安全。
3. ℹ️ **空間專家 Loss 絕對值仍貼近 ln2**：需考慮 60 檔股票下 Engle-Granger 共整合圖譜遠比 504 檔稀疏，此為 smoke test 規模效應，非修正失敗；GradNorm/E_Var 兩項核心指標已驗證通過。

**下一步（第 2 節）**：加正則化，重跑 15~20 epoch 冒煙測試（仍是分鐘級），確認 Val(T) 不再單調上升、Val(S) 能穩定跌破 ln2 後，才進入全量 504 檔訓練。

---

## 2. 正則化修正（阻斷全量訓練前必做）

### 2.1 Dropout（`models/BuildDecoupledExtractors.m`）
見對話中程式碼（時序專家 self-attention 後與 LSTM 後各加一層 dropout）。

### 2.2 L2 權重衰減（`scripts/Run_Extractor_Pretrain.m`）
兩個分支的 `aux_loss_time` / `aux_loss_space` 皆加入 `l2_lambda=1e-4` 的權重懲罰項（見對話程式碼）。

### 2.3 建議新增：Early Stopping（目前 Phase 2 訓練沒有這個機制，Phase 5 才有）
```matlab
% 在 Run_Extractor_Pretrain.m 的 epoch 迴圈中，仿照 Run_CIO_Awakening.m 的做法
best_val_loss = inf; patience = 0; patience_limit = 5;
best_net_time = []; best_net_space = [];

% 每個 epoch 結尾：
combined_val = val_loss_time(epoch) + val_loss_space(epoch);
if combined_val < best_val_loss - 1e-4
    best_val_loss = combined_val;
    patience = 0;
    best_net_time = net_time; best_net_space = net_space;
else
    patience = patience + 1;
end
if patience >= patience_limit
    fprintf('🛑 Early Stopping 於 Epoch %d 觸發，回滾至最佳快照。\n', epoch);
    net_time = best_net_time; net_space = best_net_space;
    break;
end
```
> 這對全量訓練（8453 天、504 檔）特別重要——你已經在小樣本上看到過擬合訊號，全量訓練參數量更大，沒有 Early Stopping 的話風險只會更高。

---

## 3. 免費資料源擴充計畫

現況：系統目前**只用了 OHLCV 價量資料**（22維全部由價格/成交量衍生），連已經抓進資料湖的 `^VIX`、`^TNX`、`CL=F` 都沒有被拿來當特徵用（`FeatureEngineer.m` 的 `VIX_Proxy` 是用 SPY 自己的已實現波動率去「模擬」VIX，不是真正的 VIX 期貨隱含波動率）。這是修正優先級最高的一項——**不用新爬蟲，只是把已經有的資料接上**。

### 3.1 🔴 立即可做：真實 VIX 與波動率風險溢酬 (Volatility Risk Premium)

`data/DataFetcher.m` 已經抓了 `^VIX` 價格（只是目前 `is_macro_ticker` 把它排除在停牌檢查外，但沒被 `FeatureEngineer.m` 拿來用作特徵）。

**修正 `data/FeatureEngineer.m` 的 `calc_macro_features`：**
```matlab
function Macro = calc_macro_features(~, P, Expert, spy_idx, vix_idx, numDays)
    % ... 原本 4 維不變，新增：
    real_vix = P(:, vix_idx);  % 真實 VIX 指數收盤價
    
    % 新特徵 5：VIX 水位本身（校準過的市場恐慌指標，比自算 proxy 更可信）
    Macro(:, 5) = real_vix;
    
    % 新特徵 6：波動率風險溢酬 VIX - RealizedVol（文獻上穩健的風險情緒指標）
    Macro(:, 6) = real_vix - vix_proxy;  % vix_proxy 是原本 SPY 已實現波動率
end
```
需同步修改 `Config.m` 的 `NumMacroFeatures` 從 4 改為視最終選定新增數量而定（見 3.4 總表），並在 `FeatureEngineer.process()` 呼叫時傳入 VIX 在 `ticker_list` 中的索引。

### 3.2 🟡 建議新增：FRED 總經資料（免費，需申請免費 API Key）

殖利率曲線倒掛與信用利差是總經文獻中最穩健的**衰退/崩盤領先指標**，目前系統完全沒有使用，而護欄模型正是靠 Macro 特徵撐訊號的關鍵模組——這是投資報酬率最高的擴充項。

| 序列代碼 | 說明 | 用途 |
|---|---|---|
| `T10Y2Y` | 10年期減2年期公債利差 | 殖利率曲線倒掛，經典衰退領先指標 |
| `BAMLH0A0HYM2` | 美銀美林高收益債利差 | 信用風險/流動性壓力指標 |
| `DGS10` | 10年期公債殖利率 | 利率環境 |
| `UNRATE` | 失業率 | 總經動能 |

**實作方式（新增 Python 爬蟲）：**
```python
# 新增 data/crawlers/fred_crawler.py
from fredapi import Fred
fred = Fred(api_key=os.getenv('FRED_API_KEY'))  # 免費申請
series_ids = ['T10Y2Y', 'BAMLH0A0HYM2', 'DGS10', 'UNRATE']
for sid in series_ids:
    df = fred.get_series(sid)
    # ⚠️ 務必用「實際公布日期」而非「數據所屬期間」對齊時間軸，避免未來函數
    # FRED API 回傳的日期通常是數據所屬期間起始日，UNRATE/CPI等月頻數據
    # 實際公布通常有1個月左右落後，需要額外處理 release lag
```
> ⚠️ **PiT 風險提醒**：月頻/季頻總經數據最容易犯的錯誤就是用「數據所屬期間」當作可得日期，但實際上 CPI/失業率等數據公布會落後 1 個月左右。務必查證每個序列的官方發布時間表，加上合理的 lag（保守起見可以直接用「所屬期間結束日 + 30天」作為可得日期）。

### 3.3 🟢 進階選項：SEC EDGAR 基本面 / 內部人交易（免費）

- 基本面：`https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`（財報XBRL，免費無需金鑰）
- Form 4 內部人交易：內部人買超在文獻中有一定的預測力，且是免費公開資料
- 13F 機構持倉：季頻，可看機構增減碼趨勢
- **注意**：這些是低頻資料（季/月），建議作為「橫截面靜態特徵」疊加，而非取代現有高頻價量特徵；且務必以**實際申報日期**對齊時間軸（10-Q/10-K 申報日通常落後財報所屬季度 30~45 天）

### 3.4 特徵集擴充後總表

| 類別 | 現況 | 建議新增 | 優先級 |
|---|---|---|---|
| 相對大盤 (3維) | Beta, Corr, RS | 不變 | - |
| 微觀技術 (15維) | 見 Phase14.23 第7節建議壓縮至 ~11維 | 60日/120日中期動量、Overnight/Intraday分解 | 🟡 |
| 宏觀 (4維→8~10維) | VIX_Proxy(自算), R20, R60, Breadth | 真實VIX、VRP、殖利率曲線利差、信用利差 | 🔴 |
| 基本面 (新) | 無 | 選配：估值因子(P/E,P/B)、內部人買超 | 🟢 |

---

## 4. 崩盤護欄模型 (`MdlCrash`) 專項強化

這是本次擴充計畫中**優先級最高的單一改動**，因為 Phase 6 已實證證明現有護欄邏輯是造成系統長期空手、報酬無法轉正的直接原因。

**修正 `agents/GBDTExpertAgent.m` 的 `train_and_predict_oof_crash`：**
- 輸入從目前 4 維（全部衍生自SPY自身）擴充為第3節的 8~10 維（含真實VIX、VRP、殖利率曲線倒掛、信用利差）
- **新增機率校準驗證**：目前 `1./(1+exp(-score))` 只是原始 LogitBoost 分數的通用 sigmoid 轉換，不保證是良好校準的機率。護欄閾值邏輯完全依賴機率的絕對數值有意義（不只是排序），建議加入信賴度診斷：
```matlab
% 在 train_and_predict_oof_crash 結尾加入校準檢查（Reliability / Brier Score）
[brier, cal_bins] = compute_calibration_diagnostics(P_crash_oof, Y_Crash_1D);
fprintf('  📊 [護欄機率校準檢查] Brier Score: %.4f (越接近0越好)\n', brier);
% 建議：若校準不佳，考慮用 Platt Scaling 或 Isotonic Regression 對 P_crash 做後校準
```

---

## 5. 兩階段獲利驗收標準（正式納入 Checklist）

| 階段 | 驗收條件 | 對應檢查方式 |
|---|---|---|
| **短期（初階）** | IS 與 OOS 絕對報酬皆 > 0% | 且**非因「長期空手」達成**——需檢查平均持股天數與集中度是否合理（不能是100%現金導致報酬剛好卡在0附近） |
| **長期（終極）** | OOS 報酬 > SPY OOS 報酬（目前基準：71.55%，`Run_WalkForward_Backtest.m` log） | 且風險調整後報酬（Sharpe/Sortino）需優於或至少不劣於SPY，避免"用槓桿風險換報酬"式的假勝利 |

---

## 6. 更新後執行順序

```
第零階段（延續 Phase14.23，分鐘級）
  0. 加入 2.1、2.2 節正則化，重跑 smoke test 15~20 epoch
     └─ 確認 Val(T) 不再單調上升，Val(S) 穩定跌破 0.6931

第一階段（資料擴充，可與模型訓練並行準備）
  1. 實作 3.1（真實VIX，立即可做）與 3.2（FRED總經）爬蟲
  2. 更新 FeatureEngineer.m / Config.m 維度與 PiT 對齊邏輯
  3. 全量重跑 Phase 1（因特徵集有變動，這次無法沿用舊快取）

第二階段（Phase 2 全量 + 驗證）
  4. 全量重跑 Phase 2（含正則化 + Early Stopping）
  5. Sanity Baseline 比對（Phase14.23 第3.1節）

第三階段（護欄強化 + Phase 3~6）
  6. 套用第4節護欄模型強化，重跑 Phase 3
  7. 套用 Phase14.23 第4節 BO 修正，重跑 Phase 4
  8. 套用 Phase14.23 第5節 Reward 修正，重跑 Phase 5
  9. 執行 Phase 6，對照第5節「兩階段獲利驗收標準」逐項確認
```
