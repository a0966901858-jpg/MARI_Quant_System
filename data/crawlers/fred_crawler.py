import os
import logging
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

# =========================================================================
# 模組：fred_crawler.py (MARI Phase 1 總經資料爬蟲微服務)
# 升級：Phase 15.5 生產基準版 (★ SPY 交易日曆精準錨定、
#       階層式利差回填 Local CSV + BAA10Y OLS 動態校準、bfill 邊界防禦、
#       R² 擬合優度品質稽核與 MATLAB 100% 相容落地)
# =========================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH)

# 自動定位 data_lake 目錄
CANDIDATE_DATA_LAKES = [
    os.path.abspath(os.path.join(BASE_DIR, '..', 'data_lake')),
    os.path.abspath(os.path.join(BASE_DIR, 'data_lake')),
    os.path.abspath('data/data_lake'),
    os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'data_lake'))
]
DATA_LAKE_DIR = next((p for p in CANDIDATE_DATA_LAKES if os.path.exists(p)), CANDIDATE_DATA_LAKES[0])
os.makedirs(DATA_LAKE_DIR, exist_ok=True)

FRED_API_KEY = os.getenv('FRED_API_KEY')

# ★ 總經特徵設定 (納入發布延遲 lag_days 與長週期代理序列)
FRED_SERIES_CONFIG = {
    'VIXCLS': {
        'desc': 'CBOE 波動率指數 (VIX)',
        'lag_days': 1
    },
    'T10Y2Y': {
        'desc': '10年期減2年期公債殖利率利差',
        'lag_days': 1
    },
    'BAMLH0A0HYM2': {
        'desc': '美銀美林高收益債期權調整利差 (ICE BofA OAS)',
        'lag_days': 1,
        'local_csv': 'BAMLH0A0HYM2.csv',
        'proxy_series': 'BAA10Y' # 穆迪 Baa 信用利差 (自 1953 年起全覆蓋)
    },
    'DGS10': {
        'desc': '10年期美國國債基準殖利率',
        'lag_days': 1
    },
    'UNRATE': {
        'desc': '美國官方失業率',
        'lag_days': 35 # 官方失業率約次月第一個週五發布，嚴格設置 35 日防前視
    }
}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def fetch_fred_raw(series_id, api_key, start_date='1990-01-01'):
    """調用 FRED API 取得觀測序列並清洗空值標記"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date
    }
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    data = response.json()
    
    if 'observations' not in data:
        raise ValueError(f"❌ FRED API 未回傳 observations ({series_id})")
        
    records = []
    for obs in data['observations']:
        raw_val = obs.get('value', '.')
        val = np.nan if raw_val == '.' else float(raw_val)
        records.append({'Date': obs['date'], series_id: val})
        
    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date').sort_index()

def fetch_series_with_hierarchical_imputation(series_id, api_key, conf):
    """
    抓取總經序列，執行【本地真實 CSV 注入】與【BAA10Y OLS 動態校準回填】，
    並施加發布延遲 lag_days 以阻絕前視偏差。
    """
    df = fetch_fred_raw(series_id, api_key)
    
    # 1. 第一級防禦：若存在本地手動下載之全量真實 CSV 則優先融合
    local_csv_name = conf.get('local_csv')
    if local_csv_name:
        csv_candidates = [
            os.path.join(DATA_LAKE_DIR, local_csv_name),
            os.path.join(BASE_DIR, local_csv_name),
            os.path.abspath(local_csv_name)
        ]
        csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
        if csv_path:
            try:
                local_df = pd.read_csv(csv_path)
                local_df.columns = [str(c).strip().upper() for c in local_df.columns]
                date_col = next((c for c in local_df.columns if 'DATE' in c), None)
                val_col = next((c for c in local_df.columns if series_id.upper() in c), None)
                if date_col and val_col:
                    local_df['Date'] = pd.to_datetime(local_df[date_col])
                    local_df[series_id] = pd.to_numeric(local_df[val_col], errors='coerce')
                    local_df = local_df.set_index('Date')[[series_id]].sort_index()
                    df = df.combine_first(local_df)
                    logging.info(f"  📂 成功融合本地檔案 [{local_csv_name}]！")
            except Exception as ex:
                logging.warning(f"  ⚠️ 讀取本地檔案 [{local_csv_name}] 失敗: {ex}")

    # 2. 第二級防禦：若仍有歷史區間缺失，調用 BAA10Y 進行 OLS 線性迴歸擬合回填
    if conf.get('proxy_series'):
        proxy_id = conf['proxy_series']
        proxy_df = fetch_fred_raw(proxy_id, api_key)
        merged = df.join(proxy_df, how='outer')
        overlap = merged.dropna(subset=[series_id, proxy_id])
        
        if len(overlap) >= 60:
            p_vals = overlap[proxy_id].values
            y_vals = overlap[series_id].values
            slope, intercept = np.polyfit(p_vals, y_vals, 1)
            
            y_pred = intercept + slope * p_vals
            ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
            ss_res = np.sum((y_vals - y_pred) ** 2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
            
            impute_mask = merged[series_id].isna() & merged[proxy_id].notna()
            impute_count = impute_mask.sum()
            merged.loc[impute_mask, series_id] = intercept + slope * merged.loc[impute_mask, proxy_id]
            df = merged[[series_id]].copy()
            logging.info(f"  ✨ 成功透過 [{proxy_id}] 代理回填 {series_id} 共 {impute_count} 筆歷史資料 (斜率={slope:.3f}, 截距={intercept:.3f}, R²={r2:.4f})！")
        else:
            logging.warning(f"  ⚠️ [{proxy_id}] 重疊有效樣本不足，略過動態迴歸回填。")

    # 3. 施加發布延遲 (Publication Lag) 阻斷前視偏差
    lag_days = conf.get('lag_days', 0)
    if lag_days > 0:
        df.index = df.index + pd.Timedelta(days=lag_days)
        
    return df

def main():
    if not FRED_API_KEY:
        raise ValueError(f"❌ 未在環境變數或 {ENV_PATH} 中偵測到 FRED_API_KEY！")
        
    logging.info(f"🔑 成功讀取 FRED API 金鑰，準備抓取 {len(FRED_SERIES_CONFIG)} 組總經序列...")
    
    dfs = []
    for series_id, conf in FRED_SERIES_CONFIG.items():
        logging.info(f"  📥 正在抓取: [{series_id}] {conf['desc']}...")
        s_df = fetch_series_with_hierarchical_imputation(series_id, FRED_API_KEY, conf)
        dfs.append(s_df)

    # 合併多個總經時間序列
    macro_df = pd.concat(dfs, axis=1).sort_index()
    
    # 讀取大宇宙交易日曆並精確對齊 SPY 基準
    master_path = os.path.join(DATA_LAKE_DIR, "master_universe.parquet")
    if os.path.exists(master_path):
        logging.info("📅 偵測到 master_universe.parquet，正在依據 SPY 交易日曆重採樣對齊...")
        try:
            # 優先嘗試以 SPY 作為絕對交易日曆基準 (與 DataFetcher.m 嚴格對齊)
            master_spy = pd.read_parquet(master_path, filters=[('Ticker', '==', 'SPY')], columns=['Date'])
            if not master_spy.empty:
                date_series = master_spy['Date']
            else:
                master_df = pd.read_parquet(master_path, columns=['Date'])
                date_series = master_df['Date']
        except Exception:
            master_df = pd.read_parquet(master_path, columns=['Date'])
            date_series = master_df['Date']
            
        raw_dates = pd.to_datetime(date_series.drop_duplicates())
        if raw_dates.dt.tz is not None:
            raw_dates = raw_dates.dt.tz_convert(None)
        trading_dates = pd.DatetimeIndex(raw_dates).normalize()
        trading_dates = trading_dates[trading_dates >= pd.Timestamp('1990-01-01')].sort_values()
        
        # 雙向傳遞填補：ffill 向後傳遞常態資訊，bfill 解決序列開頭因 lag_days 產生的空缺
        macro_df = macro_df.reindex(trading_dates).ffill().bfill()
        macro_df.index.name = 'Date'
    else:
        macro_df = macro_df[macro_df.index >= pd.Timestamp('1990-01-01')].ffill().bfill()
        
    macro_df = macro_df.reset_index()
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.tz_localize(None)
    
    # 強制型態轉型為 float64，確保 MATLAB parquetread 解析順暢
    for col in FRED_SERIES_CONFIG.keys():
        if col in macro_df.columns:
            macro_df[col] = macro_df[col].astype('float64')

    # 執行特徵品質全面稽核
    summary = []
    total = len(macro_df)
    target_cols = list(FRED_SERIES_CONFIG.keys())
    
    for col in target_cols:
        nans = macro_df[col].isna().sum()
        valid_mask = ~macro_df[col].isna()
        first_valid = macro_df.loc[valid_mask, 'Date'].iloc[0] if valid_mask.any() else "None"
        last_valid = macro_df.loc[valid_mask, 'Date'].iloc[-1] if valid_mask.any() else "None"
        summary.append({
            'Feature': col,
            'TotalRows': total,
            'NaNCount': nans,
            'NaNRatio(%)': round((nans / total) * 100, 2),
            'FirstValidDate': str(first_valid)[:10],
            'LastValidDate': str(last_valid)[:10]
        })
    
    audit_table = pd.DataFrame(summary)
    print("\n" + "="*80)
    print("                 FRED Macro 特徵矩陣品質稽核報告")
    print("="*80)
    print(audit_table.to_string(index=False))
    print("="*80 + "\n")
    
    # 斷言保護：特徵缺失率大於 1% 立即阻斷
    for col in target_cols:
        col_nan = audit_table.loc[audit_table['Feature'] == col, 'NaNRatio(%)'].values[0]
        if col_nan > 1.0:
            raise RuntimeError(f"🚨 致命錯誤：特徵 [{col}] 缺失率高達 {col_nan}%，資料寫入中止！")

    out_parquet = os.path.join(DATA_LAKE_DIR, "fred_macro.parquet")
    out_csv = os.path.join(DATA_LAKE_DIR, "fred_macro.csv")
    
    macro_df.to_parquet(out_parquet, engine='pyarrow', index=False)
    macro_df.to_csv(out_csv, index=False)
    
    logging.info(f"🏆 FRED 宏觀總經資料落地完成！")
    logging.info(f"   📦 Parquet: {out_parquet}")
    logging.info(f"   📄 CSV:     {out_csv}")

if __name__ == "__main__":
    main()
