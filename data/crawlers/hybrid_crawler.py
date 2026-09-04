import yfinance as yf          # 用於獲取 Yahoo Finance 歷史 K 線資料
import pandas as pd            # 用於資料結構轉換與面板資料 (Panel Data) 處理
import os                      # 用於作業系統路徑與目錄管理
import logging                 # 用於系統日誌與執行狀態追蹤
import requests                # 用於發送 HTTP 請求抓取維基百科
import io                      # 用於字串與位元流轉換 (輔助 pandas 讀取 HTML)
import re                      # 用於清理維基百科引用標籤 (如 [1], [note])
from dotenv import load_dotenv # 用於讀取環境變數 (保護 API Keys)
from tenacity import retry, stop_after_attempt, wait_fixed # 用於爬蟲斷線自動重試防護
from concurrent.futures import ThreadPoolExecutor, as_completed # 用於多執行緒高併發下載

# =========================================================================
# 模組：hybrid_crawler.py (MARI Phase 1 混合爬蟲微服務)
# 升級：Phase 15.5 Stage 2 (★ GICS 產業別/次產業全面萃取、True PiT 歷史宇宙擴充)
# 職責：構建 Point-in-Time 歷史宇宙，萃取 GICS 板塊標籤，下載價格並輸出 Parquet 長表
# =========================================================================

# 1. 系統環境與日誌初始化
yf.shared._ERRORS = {}
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_PATH = os.path.join(BASE_DIR, 'us_universe.csv')
DATA_LAKE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data_lake'))
os.makedirs(DATA_LAKE_DIR, exist_ok=True)

logging.info("🚀 啟動 MARI 混合爬蟲微服務 [Phase 15.5 Stage 2 GICS 產業別萃取升級版]")

# =========================================================================
# 函數：build_historical_pit_universe
# 職責：動態解析維基百科，提取當前/歷史成分股變更及 GICS Sector / Sub-Industry
# =========================================================================
def build_historical_pit_universe():
    logging.info("🌐 正在從維基百科抓取 S&P 500 當前名單、GICS 產業分類與歷史變更紀錄...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        tables = pd.read_html(io.StringIO(response.text))

        df_current = None
        df_changes = None
        
        # 1. 雙模態表頭特徵探測
        for tbl in tables:
            if isinstance(tbl.columns, pd.MultiIndex):
                col_strings = ['_'.join([str(c) for c in col]).lower() for col in tbl.columns]
            else:
                col_strings = [str(col).lower() for col in tbl.columns]
            
            # 當前成分股特徵 (包含 symbol 與 gics/security)
            if df_current is None:
                if any('symbol' in c or 'ticker' in c for c in col_strings) and any('gics' in c or 'security' in c for c in col_strings):
                    df_current = tbl.copy()
                    continue
            
            # 歷史變更特徵 (包含 added 與 removed)
            if df_changes is None:
                if any('added' in c for c in col_strings) and any('removed' in c for c in col_strings):
                    df_changes = tbl.copy()
                    continue

        # 順序兜底
        if df_current is None and len(tables) >= 1:
            df_current = tables[0].copy()
        if df_changes is None and len(tables) >= 2:
            df_changes = tables[1].copy()
                
        if df_current is None or df_changes is None:
            raise ValueError("❌ 無法在維基百科頁面中定位到成分股或變更表格。")

        # ----------------- 提取當前成分股與 GICS 產業別 -----------------
        if isinstance(df_current.columns, pd.MultiIndex):
            df_current.columns = ['_'.join([str(p) for p in col if 'Unnamed' not in str(p)]).strip() for col in df_current.columns]
        
        sym_col = next((c for c in df_current.columns if 'symbol' in c.lower() or 'ticker' in c.lower()), None)
        sector_col = next((c for c in df_current.columns if 'gics sector' in c.lower() or ('sector' in c.lower() and 'sub' not in c.lower())), None)
        sub_sector_col = next((c for c in df_current.columns if 'sub-industry' in c.lower() or 'sub industry' in c.lower() or 'subindustry' in c.lower()), None)

        if not sym_col:
            raise ValueError("❌ 當前成分股表中找不到 Symbol/Ticker 欄位。")

        ticker_to_sector = {}
        ticker_to_subsector = {}
        current_tickers = []

        for _, row in df_current.iterrows():
            raw_sym = row[sym_col]
            if pd.notna(raw_sym):
                t = str(raw_sym).replace('.', '-').strip()
                if t and t.lower() != 'nan':
                    current_tickers.append(t)
                    
                    # 擷取並清理 GICS 主產業
                    if sector_col and pd.notna(row[sector_col]):
                        sec_val = re.sub(r'\[.*?\]', '', str(row[sector_col])).strip()
                    else:
                        sec_val = 'Unknown'
                    ticker_to_sector[t] = sec_val
                    
                    # 擷取並清理 GICS 次產業
                    if sub_sector_col and pd.notna(row[sub_sector_col]):
                        subsec_val = re.sub(r'\[.*?\]', '', str(row[sub_sector_col])).strip()
                    else:
                        subsec_val = 'Unknown'
                    ticker_to_subsector[t] = subsec_val

        current_tickers = sorted(list(set(current_tickers)))

        if sector_col:
            sector_dist = pd.Series(ticker_to_sector).value_counts()
            logging.info(f"🏷️ [GICS 板塊審計] 成功辨識產業欄位: [{sector_col}] (共 {len(sector_dist)} 個板塊)")
            for s_name, count in sector_dist.items():
                logging.info(f"   • {s_name:<28}: {count} 檔")
        else:
            logging.warning("⚠️ 未能識別 GICS Sector 欄位，產業別將標記為 Unknown。")

        # ----------------- 提取歷史變更名單 -----------------
        added_tickers = []
        removed_tickers = []
        
        if isinstance(df_changes.columns, pd.MultiIndex):
            for col in df_changes.columns:
                top_lvl = str(col[0]).lower()
                sub_lvl = str(col[1]).lower()
                if 'added' in top_lvl and ('ticker' in sub_lvl or 'symbol' in sub_lvl):
                    added_tickers.extend(df_changes[col].dropna().astype(str).str.replace('.', '-', regex=False).str.strip().tolist())
                elif 'removed' in top_lvl and ('ticker' in sub_lvl or 'symbol' in sub_lvl):
                    removed_tickers.extend(df_changes[col].dropna().astype(str).str.replace('.', '-', regex=False).str.strip().tolist())
        else:
            for col in df_changes.columns:
                c_low = str(col).lower()
                if 'added' in c_low and ('ticker' in c_low or 'symbol' in c_low):
                    added_tickers.extend(df_changes[col].dropna().astype(str).str.replace('.', '-', regex=False).str.strip().tolist())
                elif 'removed' in c_low and ('ticker' in c_low or 'symbol' in c_low):
                    removed_tickers.extend(df_changes[col].dropna().astype(str).str.replace('.', '-', regex=False).str.strip().tolist())

        # 清理並聯集所有歷史股票
        added_tickers = [t for t in set(added_tickers) if t and t.lower() != 'nan']
        removed_tickers = [t for t in set(removed_tickers) if t and t.lower() != 'nan']
        
        full_history_tickers = list(set(current_tickers + added_tickers + removed_tickers))

        # 強制注入宏觀指標與核心避險資產
        core_and_macro = ['SPY', 'IEF', 'TLT', 'GLD', '^VIX', 'CL=F', '^TNX']
        for t in core_and_macro:
            if t not in full_history_tickers:
                full_history_tickers.append(t)

        logging.info(f"📊 [大宇宙結構審計] 當前成分股: {len(current_tickers)} 檔 | "
                     f"歷史新增: {len(added_tickers)} 檔 | "
                     f"歷史剔除: {len(removed_tickers)} 檔 | "
                     f"聯集後大宇宙: {len(full_history_tickers)} 檔")

        if len(added_tickers) == 0 or len(removed_tickers) == 0:
            logging.warning("⚠️ 警告：歷史變更名單解析數為 0！可能導致大宇宙退化為靜態 500 檔，請檢查維基百科表格結構！")

        # ----------------- 輸出含 GICS 標籤之全宇宙清單 -----------------
        universe_rows = []
        for t in full_history_tickers:
            asset_type = 'Macro' if t in ['^VIX', 'CL=F', '^TNX'] else ('Safe Haven' if t in ['IEF', 'TLT', 'GLD'] else 'Equity')
            
            if t in ['^VIX', 'CL=F', '^TNX']:
                sec = 'Macro'
                subsec = 'Macro'
            elif t in ['IEF', 'TLT', 'GLD']:
                sec = 'Safe Haven'
                subsec = 'Safe Haven'
            elif t == 'SPY':
                sec = 'Broad Market Index'
                subsec = 'Large Cap Blend'
            else:
                sec = ticker_to_sector.get(t, 'Unknown')
                subsec = ticker_to_subsector.get(t, 'Unknown')

            universe_rows.append({
                'Ticker': t,
                'Type': asset_type,
                'GICS_Sector': sec,
                'GICS_SubIndustry': subsec
            })

        final_df = pd.DataFrame(universe_rows)
        final_df = final_df.sort_values(by=['Type', 'Ticker']).reset_index(drop=True)
        final_df.to_csv(UNIVERSE_PATH, index=False, encoding='utf-8')
        
        logging.info(f"✅ 成功重構歷史全宇宙清單 (含 GICS 產業別)，共 {len(final_df)} 檔標的已寫入: {UNIVERSE_PATH}")
        return full_history_tickers, df_current, df_changes

    except Exception as e:
        logging.error(f"❌ 歷史宇宙建構失敗: {e}")
        if os.path.exists(UNIVERSE_PATH):
            logging.warning("⚠️ 啟用本地 us_universe.csv 作為兜底歷史宇宙名單。")
            fallback_df = pd.read_csv(UNIVERSE_PATH)
            return fallback_df['Ticker'].tolist(), df_current, df_changes
        raise e

# =========================================================================
# 函數：generate_pit_membership_long
# 職責：逆向工程建構每日成分股資格，並輸出為長表格式
# =========================================================================
def generate_pit_membership_long(spy_dates, df_current, df_changes):
    logging.info("⏳ 正在逆向工程建構 True PiT 每日成分股資格 (長表輸出)...")

    if df_current is None or df_changes is None:
        logging.warning("⚠️ 缺少動態變更表格，成分股資格將預設全覆蓋。")
        records = [{'Date': d, 'Ticker': t, 'IsConstituent': 1} for d in spy_dates for t in ['SPY']]
        return pd.DataFrame(records)

    if isinstance(df_current.columns, pd.MultiIndex):
        df_current.columns = ['_'.join([str(p) for p in col if 'Unnamed' not in str(p)]).strip() for col in df_current.columns]
    
    sym_col = next((c for c in df_current.columns if 'symbol' in c.lower() or 'ticker' in c.lower()), df_current.columns[0])
    current_set = set(df_current[sym_col].dropna().astype(str).str.replace('.', '-', regex=False).str.strip().tolist())
    
    core_and_macro = {'SPY', 'IEF', 'TLT', 'GLD', '^VIX', 'CL=F', '^TNX'}
    current_set.update(core_and_macro)

    if isinstance(df_changes.columns, pd.MultiIndex):
        df_changes.columns = ['_'.join([str(p) for p in col if 'Unnamed' not in str(p)]).strip() for col in df_changes.columns]

    date_col = next((c for c in df_changes.columns if 'date' in c.lower()), None)
    add_col  = next((c for c in df_changes.columns if 'added' in c.lower() and ('ticker' in c.lower() or 'symbol' in c.lower())), None)
    rem_col  = next((c for c in df_changes.columns if 'removed' in c.lower() and ('ticker' in c.lower() or 'symbol' in c.lower())), None)

    if not date_col:
        logging.warning("⚠️ 變更表中未識別出 Date 欄位，維持全量成分股。")
        records = [{'Date': d, 'Ticker': t, 'IsConstituent': 1} for d in spy_dates for t in current_set]
        return pd.DataFrame(records)

    df_changes = df_changes.rename(columns={date_col: 'Date', add_col: 'Added', rem_col: 'Removed'})
    
    df_changes['Date'] = pd.to_datetime(df_changes['Date'], errors='coerce').dt.tz_localize('UTC')
    df_changes = df_changes.dropna(subset=['Date']).sort_values(by='Date', ascending=False)

    spy_dates_sorted = sorted(spy_dates, reverse=True)
    change_idx = 0
    num_changes = len(df_changes)
    records = []

    for current_date in spy_dates_sorted:
        while change_idx < num_changes and df_changes.iloc[change_idx]['Date'] >= current_date:
            row = df_changes.iloc[change_idx]
            add_t = str(row['Added']).replace('.', '-').strip() if 'Added' in row and pd.notna(row['Added']) else ''
            rem_t = str(row['Removed']).replace('.', '-').strip() if 'Removed' in row and pd.notna(row['Removed']) else ''

            if add_t and add_t in current_set: current_set.remove(add_t)
            if rem_t and rem_t != 'nan' and rem_t != '': current_set.add(rem_t)
            change_idx += 1

        for ticker in current_set:
            records.append({'Date': current_date, 'Ticker': ticker, 'IsConstituent': 1})

    return pd.DataFrame(records)

# =========================================================================
# 函數：download_single_ticker_panel
# 職責：下載單一股票並轉換為帶有 Ticker 欄位的標準化長表
# =========================================================================
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def download_single_ticker_panel(ticker):
    try:
        stock = yf.Ticker(ticker)
        raw_df = stock.history(period="max", auto_adjust=True)
        
        if raw_df is None or raw_df.empty:
            raise ValueError(f"{ticker} 回傳空資料，可能為 Rate Limit 或已下市")
            
        df_clean = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        idx = pd.to_datetime(df_clean.index)
        if idx.tz is None:
            idx = idx.tz_localize('UTC')
        else:
            idx = idx.tz_convert('UTC')
        df_clean.index = idx.normalize()
        
        df_clean['Ticker'] = ticker
        df_clean = df_clean.reset_index().rename(columns={'index': 'Date'})
        return df_clean
    except Exception as e:
        raise e

# =========================================================================
# 主程序
# =========================================================================
def main():
    all_tickers, df_current, df_changes = build_historical_pit_universe()

    logging.info(f"⚡ 啟動高併發下載 {len(all_tickers)} 檔歷史大宇宙 K 線資料 (長表轉換中)...")
    
    panel_dfs = []
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_single_ticker_panel, t): t for t in all_tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                res_df = future.result()
                if res_df is not None and not res_df.empty:
                    panel_dfs.append(res_df)
            except Exception as exc:
                logging.debug(f"標的 {ticker} 下載最終失敗或已下市: {exc}")

    if not panel_dfs:
        raise ValueError("❌ 無法下載任何資料，請檢查網路連線。")

    successful_tickers = {df['Ticker'].iloc[0] for df in panel_dfs}
    missing_tickers = set(all_tickers) - successful_tickers
    drop_rate = len(missing_tickers) / len(all_tickers)
    
    logging.warning(f"⚠️ {len(missing_tickers)}/{len(all_tickers)} 檔標的下載失敗 "
                    f"(缺漏率 {drop_rate:.1%})，清單已寫入 missing_tickers.csv")
    
    missing_df = pd.DataFrame({'Ticker': sorted(missing_tickers)})
    missing_df.to_csv(os.path.join(DATA_LAKE_DIR, 'missing_tickers.csv'), index=False)

    if drop_rate > 0.05:
        logging.error("❌ 缺漏率超過 5%，建議暫停並人工檢查，避免倖存者偏差回滲。")

    logging.info("🐼 正在垂直拼接所有 K 線特徵，構建全宇宙 Panel Data...")
    master_price_df = pd.concat(panel_dfs, ignore_index=True)
    
    spy_dates = master_price_df[master_price_df['Ticker'] == 'SPY']['Date'].unique()

    df_membership = generate_pit_membership_long(spy_dates, df_current, df_changes)

    logging.info("🐼 正在融合 (Merge) 價格長表與 PiT 成員矩陣...")
    master_final = pd.merge(master_price_df, df_membership, on=['Date', 'Ticker'], how='left')
    master_final['IsConstituent'] = master_final['IsConstituent'].fillna(0).astype('int8')

    master_final['Open']   = master_final['Open'].astype('float32')
    master_final['High']   = master_final['High'].astype('float32')
    master_final['Low']    = master_final['Low'].astype('float32')
    master_final['Close']  = master_final['Close'].astype('float32')
    master_final['Volume'] = master_final['Volume'].astype('float32')
    
    master_final = master_final.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)

    master_path = os.path.join(DATA_LAKE_DIR, "master_universe.parquet")
    master_final.to_parquet(master_path, engine='pyarrow', index=False)
    
    logging.info(f"🏆 True PiT 面板資料對齊完成！檔案已存至 {master_path}")
    logging.info(f"   總列數 (Rows): {len(master_final)} | 欄位 (Columns): {master_final.columns.tolist()}")

if __name__ == "__main__":
    main()