import os
import logging
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

# =========================================================================
# 模組：fred_crawler.py (MARI Phase 1 總經資料爬蟲微服務)
# 升級：Phase 15.5 生產修復版 (日曆對齊至 1990+ 與 BAA10Y 代理動態回填)
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

# ★ 正式特徵配置
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
        'proxy_series': 'BAA10Y' # ★ 使用官方開放的穆迪利差自動補齊歷史
    },
    'DGS10': {
        'desc': '10年期美國國債基準殖利率',
        'lag_days': 1
    },
    'UNRATE': {
        'desc': '美國官方失業率',
        'lag_days': 35
    }
}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def fetch_fred_raw(series_id, api_key, start_date='1990-01-01'):
    """調用 FRED API 取得觀測序列"""
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

def fetch_series_with_lag(series_id, api_key, conf):
    """抓取特徵並施加發布延遲 lag_days 以阻斷前視偏差"""
    df = fetch_fred_raw(series_id, api_key)
    
    # 若為 BAMLH0A0HYM2 且受 3 年商業限制，自動調用 BAA10Y 代理校準
    if conf.get('proxy_series'):
        proxy_id = conf['proxy_series']
        logging.info(f"  🔗 正在拉取 [{proxy_id}] 作為 [{series_id}] 的長週期動態校準錨點...")
        proxy_df = fetch_fred_raw(proxy_id, api_key)
        
        merged = df.join(proxy_df, how='outer')
        overlap = merged.dropna(subset=[series_id, proxy_id])
        
        if len(overlap) >= 60:
            # 建立線性回歸校準模型 BAML = a + b * BAA10Y
            p_vals = overlap[proxy_id].values
            y_vals = overlap[series_id].values
            slope, intercept = np.polyfit(p_vals, y_vals, 1)
            
            # 對 BAML 歷史 NaN 區間進行高保真回填
            impute_mask = merged[series_id].isna() & (~merged[proxy_id].isna())
            merged.loc[impute_mask, series_id] = intercept + slope * merged.loc[impute_mask, proxy_id]
            df = merged[[series_id]].copy()
            logging.info(f"  ✨ 成功以 BAA10Y 代理補齊 {series_id} 歷史序列 (迴歸斜率={slope:.3f}, 截距={intercept:.3f})！")
        else:
            logging.warning(f"  ⚠️ 重疊樣本數不足 ({len(overlap)})，無法執行線性擬合回填。")

    lag_days = conf.get('lag_days', 0)
    if lag_days > 0:
        df.index = df.index + pd.Timedelta(days=lag_days)
    return df

def main():
    if not FRED_API_KEY:
        raise ValueError(f"❌ 未在環境變數或 {ENV_PATH} 中偵測到 FRED_API_KEY！")
        
    logging.info(f"🔑 成功讀取 FRED API 金鑰，抓取 {len(FRED_SERIES_CONFIG)} 組總經序列...")
    
    dfs = []
    for series_id, conf in FRED_SERIES_CONFIG.items():
        logging.info(f"  📥 正在抓取: [{series_id}] {conf['desc']}...")
        s_df = fetch_series_with_lag(series_id, FRED_API_KEY, conf)
        dfs.append(s_df)

    macro_df = pd.concat(dfs, axis=1).ffill()
    
    # 讀取大宇宙交易日曆並執行嚴格時間軸對齊
    master_path = os.path.join(DATA_LAKE_DIR, "master_universe.parquet")
    if os.path.exists(master_path):
        logging.info("📅 偵測到 master_universe.parquet，正在依據有效交易日曆重採樣對齊...")
        master_df = pd.read_parquet(master_path, columns=['Date'])
        
        # ★ 核心修正 1：過濾掉 1962~1989 年的古老股票日，將基準對齊在 1990 年之後
        trading_dates = pd.DatetimeIndex(master_df['Date'].drop_duplicates()).tz_localize(None).normalize()
        trading_dates = trading_dates[trading_dates >= pd.Timestamp('1990-01-01')].sort_values()
        
        macro_df = macro_df.reindex(trading_dates).ffill()
        macro_df.index.name = 'Date'
    else:
        # 若無大宇宙則直接過濾 1990 以後的日曆日
        macro_df = macro_df[macro_df.index >= pd.Timestamp('1990-01-01')].ffill()
        
    macro_df = macro_df.reset_index()
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.tz_localize(None)
    
    for col in FRED_SERIES_CONFIG.keys():
        if col in macro_df.columns:
            macro_df[col] = macro_df[col].astype('float64')

    # 執行品質全面稽核
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
    
    # 斷言保護：缺失率大於 5% 立即阻斷
    for col in target_cols:
        col_nan = audit_table.loc[audit_table['Feature'] == col, 'NaNRatio(%)'].values[0]
        if col_nan > 5.0:
            raise RuntimeError(f"🚨 特徵 [{col}] 缺失率為 {col_nan}% (高於門檻 5%)，寫入中止！")

    out_parquet = os.path.join(DATA_LAKE_DIR, "fred_macro.parquet")
    out_csv = os.path.join(DATA_LAKE_DIR, "fred_macro.csv")
    
    macro_df.to_parquet(out_parquet, engine='pyarrow', index=False)
    macro_df.to_csv(out_csv, index=False)
    
    logging.info(f"🏆 FRED 宏觀總經資料落地完成！")
    logging.info(f"   📦 Parquet: {out_parquet}")
    logging.info(f"   📄 CSV:     {out_csv}")

if __name__ == "__main__":
    main()
