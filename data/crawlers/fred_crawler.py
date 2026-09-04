import os                      
import logging                 
import requests                
import pandas as pd            
import numpy as np             
from dotenv import load_dotenv 
from tenacity import retry, stop_after_attempt, wait_fixed 

# =========================================================================
# 模組：fred_crawler.py (MARI Phase 1 總經資料爬蟲微服務)
# 升級：Phase 14.24 (★ 加入 VIXCLS 波動率、殖利率曲線、信用利差、失業率)
# =========================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH)

DATA_LAKE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data_lake'))
os.makedirs(DATA_LAKE_DIR, exist_ok=True)

FRED_API_KEY = os.getenv('FRED_API_KEY')

# ★ 擴充 VIXCLS
FRED_SERIES_CONFIG = {
    'VIXCLS': {
        'desc': 'CBOE 波動率指數 (VIX)',
        'freq': 'daily',
        'lag_days': 1
    },
    'T10Y2Y': {
        'desc': '10年期減2年期公債殖利率利差',
        'freq': 'daily',
        'lag_days': 1
    },
    'BAMLH0A0HYM2': {
        'desc': '美銀美林高收益債期權調整利差',
        'freq': 'daily',
        'lag_days': 1
    },
    'DGS10': {
        'desc': '10年期美國國債基準殖利率',
        'freq': 'daily',
        'lag_days': 1
    },
    'UNRATE': {
        'desc': '美國官方失業率',
        'freq': 'monthly',
        'lag_days': 35 
    }
}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def fetch_fred_series(series_id, api_key, config):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': '1990-01-01'
    }
    
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    
    if 'observations' not in data:
        raise ValueError(f"❌ FRED API 未回傳有效的 observations 欄位 ({series_id})")
        
    records = []
    for obs in data['observations']:
        raw_val = obs.get('value', '.')
        val = np.nan if raw_val == '.' else float(raw_val)
        records.append({
            'Date': obs['date'],
            series_id: val
        })
        
    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['Date'])
    
    lag_days = config.get('lag_days', 0)
    if lag_days > 0:
        df['Date'] = df['Date'] + pd.Timedelta(days=lag_days)
        
    df = df.set_index('Date').sort_index()
    return df

def main():
    if not FRED_API_KEY:
        raise ValueError(f"❌ 未在 {ENV_PATH} 中偵測到 FRED_API_KEY！")
        
    logging.info(f"🔑 成功讀取 FRED API 金鑰，準備抓取 {len(FRED_SERIES_CONFIG)} 組總經領先序列...")
    
    dfs = []
    for series_id, conf in FRED_SERIES_CONFIG.items():
        logging.info(f"  📥 正在抓取: [{series_id}] {conf['desc']}...")
        try:
            s_df = fetch_fred_series(series_id, FRED_API_KEY, conf)
            dfs.append(s_df)
        except Exception as e:
            logging.error(f"❌ 序列 {series_id} 抓取失敗: {e}")
            raise e

    macro_df = pd.concat(dfs, axis=1)
    macro_df = macro_df.ffill().dropna(how='all')
    
    master_path = os.path.join(DATA_LAKE_DIR, "master_universe.parquet")
    if os.path.exists(master_path):
        master_df = pd.read_parquet(master_path, columns=['Date'])
        trading_dates = pd.DatetimeIndex(master_df['Date'].drop_duplicates()).tz_localize(None).normalize()
        macro_df = macro_df.reindex(trading_dates).ffill()
        macro_df.index.name = 'Date'
        
    macro_df = macro_df.reset_index()
    
    out_parquet = os.path.join(DATA_LAKE_DIR, "fred_macro.parquet")
    macro_df.to_parquet(out_parquet, engine='pyarrow', index=False)
    
    logging.info(f"🏆 FRED 宏觀總經資料落地完成！({out_parquet})")

if __name__ == "__main__":
    main()