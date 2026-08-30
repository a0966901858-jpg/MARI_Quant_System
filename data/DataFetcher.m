% 定義 DataFetcher 類別，繼承自 handle (傳址參考類別)，確保物件在傳遞時不會產生多餘的記憶體拷貝
classdef DataFetcher < handle
    % =========================================================================
    % 模組：DataFetcher.m (MARI 數據湖泊載入與極速映射)
    % 升級：Phase 14.20 (★ 排除 Macro 指數停牌誤判、補齊 Opens<=0 異常防護、日誌拆分細化)
    % 職責：從 Parquet 讀取大宇宙長表，映射為嚴格對齊的 2D 矩陣
    % =========================================================================
    
    % 宣告類別屬性 (Properties)
    properties
        ConfigObj % 儲存全域設定檔物件 (Config) 的參考，用於存取如檔案路徑、股票清單等全域參數
    end
    
    % 宣告類別方法 (Methods)
    methods
        % 建構子 (Constructor)：在實例化 DataFetcher 物件時自動執行
        % 接收外部傳入的 Config 物件，確保此模組使用的參數與全域系統設定統一
        function obj = DataFetcher(configObj)
            obj.ConfigObj = configObj; % 將傳入的設定檔物件賦值給類別屬性 ConfigObj
        end
        
        % 核心資料抓取與處理函數：負責讀取資料湖並轉換為系統所需的矩陣格式
        function dataStruct = fetch_data(obj)
            % 在命令列印出提示訊息，表示開始執行資料載入與轉換
            disp('--- 啟動 Data Lake 面板資料載入與極速矩陣映射 ---');
            
            % 1. 定位並讀取 Python 爬蟲產出的 Parquet 檔案
            % 透過 fullfile 函數，結合 ConfigObj 中的 DataDir 路徑，組裝出 master_universe.parquet 的完整絕對路徑
            parquetPath = fullfile(obj.ConfigObj.DataDir, 'data_lake', 'master_universe.parquet');
            
            % 檢查該路徑下檔案是否存在，若不存在則拋出錯誤 (防呆機制)
            if ~exist(parquetPath, 'file')
                error('❌ 找不到 master_universe.parquet。請確認爬蟲是否執行成功。'); % 拋出致命錯誤並中斷程式
            end
            
            % 使用 MATLAB 內建的高效 parquetread 函數，將 Parquet 檔案讀取為 MATLAB 的 Table 資料結構 (長表格式)
            longTable = parquetread(parquetPath);
            
            % 嚴格檢查時間欄位 (Date) 的時區設定，防止夏月時間或跨國報價轉換導致的時序錯亂
            % 若 TimeZone 為空，或者不是 'UTC'，代表時間軸不乾淨，直接拋出錯誤
            if isempty(longTable.Date.TimeZone) || ~strcmp(longTable.Date.TimeZone, 'UTC')
                error('❌ 致命錯誤：資料庫非 UTC 時區，大宇宙絕對時間對齊失敗。');
            end
            
            % 在命令列印出提示訊息，表示準備建立時間軸
            disp(' -> 建立基準時間軸 (以 SPY 作為絕對交易日曆)...');
            
            % 2. ★ 核心修復：SPY 絕對日曆錨定
            % 建立布林遮罩 (Boolean Mask)，找出長表中 Ticker 欄位等於 "SPY" 的所有列
            spy_mask = (longTable.Ticker == "SPY");
            
            % 若遮罩內沒有任何 True，代表資料庫漏抓或遺失 SPY 資料，無法建立基準日曆，拋出錯誤
            if ~any(spy_mask)
                error('❌ 致命錯誤：數據庫中找不到基準標的 SPY，無法建立絕對交易日曆！');
            end
            
            % 提取 SPY 存在的所有日期，透過 unique 去重複並 sort 確保時間遞增排序，作為全市場的「絕對交易日曆」
            Dates_Active = sort(unique(longTable.Date(spy_mask)));
            
            % 取得時間維度參數：計算絕對交易日曆的總天數 (Rows)
            numDays = length(Dates_Active);
            % 從設定檔取得股票維度參數：計算系統要處理的股票總檔數 (Columns)
            numTickers = obj.ConfigObj.NumTickers;
            % 從設定檔取得股票清單 (Cell Array 或 String Array 格式)
            tickers = obj.ConfigObj.IdxTickers;
            
            % 印出提示訊息，準備進行長表 (Long Table) 轉寬表 (Wide Matrix) 的映射
            disp(' -> 執行極速矩陣映射 (Bypassing Unstack 效能優化)...');
            
            % 3. 預配置純淨 NaN 矩陣 (加入 Opens 確保回測可使用隔日開盤價)
            % 使用 NaN(列, 行, 資料型別) 預先配置記憶體，避免迴圈內動態擴充記憶體導致效能低落。'single' 可節省一半記憶體。
            Opens   = NaN(numDays, numTickers, 'single'); % 開盤價矩陣
            Prices  = NaN(numDays, numTickers, 'single'); % 收盤價矩陣 (此處命名為 Prices)
            Highs   = NaN(numDays, numTickers, 'single'); % 最高價矩陣
            Lows    = NaN(numDays, numTickers, 'single'); % 最低價矩陣
            Volumes = NaN(numDays, numTickers, 'single'); % 成交量矩陣
            IsConst = false(numDays, numTickers);         % 標記是否為標普成分股的布林矩陣，預設為 false
            
            % 建立時間軸的快速索引映射
            % ismember 會比對長表的 Date 是否存在於 Dates_Active (日曆) 中
            % is_in_calendar: 布林陣列，標記哪些資料列是在合法交易日中
            % date_idx_map: 數值陣列，將長表的每一列對應到 Dates_Active 的 Row Index (1 到 numDays)
            [is_in_calendar, date_idx_map] = ismember(longTable.Date, Dates_Active);
            
            % 印出提示，表示進入平行運算階段
            disp(' -> 啟動多核心 parfor 執行極速矩陣映射 (記憶體防爆版)...');
            
            % 【極度重要】將 Table 欄位拆解為獨立一維陣列
            % 由於 parfor (平行迴圈) 的變數廣播 (Broadcasting) 機制，如果直接將整個 longTable 傳入迴圈，
            % 會導致每個 CPU 核心都複製一份巨大的 Table，引發記憶體核爆 (Out of Memory)。
            lt_tickers = string(longTable.Ticker); % 獨立提取 Ticker 陣列
            lt_close   = longTable.Close;          % 獨立提取 Close 陣列
            lt_high    = longTable.High;           % 獨立提取 High 陣列
            lt_low     = longTable.Low;            % 獨立提取 Low 陣列
            lt_vol     = longTable.Volume;         % 獨立提取 Volume 陣列
            
            % 檢查 Table 中是否存在 'Open' 欄位 (相容性防呆)
            has_open   = ismember('Open', longTable.Properties.VariableNames);
            % 檢查 Table 中是否存在 'IsConstituent' 欄位 (相容性防呆)
            has_const  = ismember('IsConstituent', longTable.Properties.VariableNames);
            
            % 若有 Open 欄位則提取，否則設為空陣列
            if has_open,  lt_open  = longTable.Open; else, lt_open  = []; end
            % 若有 IsConstituent 欄位則提取，否則設為空陣列
            if has_const, lt_const = longTable.IsConstituent; else, lt_const = []; end
            
            % 4. 啟動多核心填裝矩陣 (完全避開 MATLAB 內建緩慢的 unstack 函數)
            % parfor 開啟平行運算，迭代每一檔股票 (i 為股票的 Column Index)
            parfor i = 1:numTickers
                % 將設定檔中的股票代號轉為 string 格式，以利後續字串比對
                target_ticker = string(tickers{i});
                
                % 建立過濾遮罩：找出「Ticker 相符」且「日期在合法日曆內」的所有列
                t_mask = (lt_tickers == target_ticker) & is_in_calendar;
                
                % 建立暫存的直行 (Column) 陣列，長度為 numDays
                % 這是為了滿足 parfor 的合法 Slicing 規則，避免多個核心同時寫入同一個二維矩陣造成資料競爭
                col_open  = NaN(numDays, 1, 'single');
                col_price = NaN(numDays, 1, 'single');
                col_high  = NaN(numDays, 1, 'single');
                col_low   = NaN(numDays, 1, 'single');
                col_vol   = NaN(numDays, 1, 'single');
                col_const = false(numDays, 1);
                
                % 若該檔股票在長表中有對應的資料
                if any(t_mask)
                    % 取得這些資料在基準日曆中對應的行索引 (Row Index)
                    row_idx = date_idx_map(t_mask);
                    
                    % 根據行索引，將長表中的一維陣列資料填入暫存的直行陣列中
                    if has_open, col_open(row_idx) = lt_open(t_mask); end % 填入開盤價
                    col_price(row_idx) = lt_close(t_mask);                % 填入收盤價
                    col_high(row_idx)  = lt_high(t_mask);                 % 填入最高價
                    col_low(row_idx)   = lt_low(t_mask);                  % 填入最低價
                    col_vol(row_idx)   = lt_vol(t_mask);                  % 填入成交量
                    
                    % 若有成分股標籤，大於 0 視為 true，轉換為布林值填入
                    if has_const, col_const(row_idx) = (lt_const(t_mask) > 0); end
                end
                
                % 將整理好的一維暫存陣列，一次性寫回主記憶體中對應的第 i 行 (Column i)
                % 這種寫法能將 CPU 鎖定競爭降到最低，是 MATLAB 官方建議的 parfor 最佳實踐
                Opens(:, i)   = col_open;
                Prices(:, i)  = col_price;
                Highs(:, i)   = col_high;
                Lows(:, i)    = col_low;
                Volumes(:, i) = col_vol;
                IsConst(:, i) = col_const;
            end
            
            % 印出提示訊息，準備進行資料清洗與異常值檢查
            disp('--- 執行大宇宙數據邊界與品質斷言檢查 ---');
            
            % 5. ★ 核心修復：矩陣級別的實體市場物理斷言與流動性枯竭防護
            
            % 條件 A: 物理報價不合理 (★ 計畫書修正：補齊 Opens <= 0 檢查)
            % 異常包含: 最高低於最低、收盤高於最高或低於最低、開盤高於最高或低於最低、收盤價或開盤價為零或負數
            invalid_price_mask = (Highs < Lows) | (Prices > Highs) | (Prices < Lows) | ...
                                 (Opens > Highs) | (Opens < Lows) | (Prices <= 0) | (Opens <= 0);
                             
            % 條件 B: 停牌或流動性枯竭 (★ 計畫書修正：排除 Macro 指數標的，防止正常零成交量被誤判為停牌)
            is_macro_ticker = ismember(strtrim(tickers), {'^VIX', '^TNX', 'CL=F'});
            macro_mask_2d   = repmat(is_macro_ticker, numDays, 1);
            halted_mask     = (Volumes <= 0) & ~macro_mask_2d;
            
            % 將條件 A 與條件 B 使用 OR (|) 邏輯聯集，產生「所有無效狀態」的最終布林遮罩矩陣
            total_invalid = invalid_price_mask | halted_mask;
            
            % 若無效矩陣中存在任何一個 true (異常值)
            if any(total_invalid, 'all')
                % 分類統計異常筆數
                invalid_count   = sum(total_invalid, 'all');
                price_err_count = sum(invalid_price_mask, 'all');
                halt_count      = sum(halted_mask, 'all');
                
                % 印出警告訊息與細化統計
                fprintf('⚠️ 攔截到 %d 筆異常 (報價異常: %d, 真實停牌枯竭: %d，已排除 Macro 指數)，已強制物理損毀交由下游 omitnan 濾除。\n', ...
                    invalid_count, price_err_count, halt_count);
                
                % 將這些發生異常或停牌的「日期-股票」節點，所有價格與成交量強制設為 NaN (物理損毀)
                % 此舉可避免機器學習模型與強化學習大腦學習到錯誤的「幽靈特徵」
                Opens(total_invalid)   = NaN;
                Highs(total_invalid)   = NaN;
                Lows(total_invalid)    = NaN;
                Prices(total_invalid)  = NaN;
                Volumes(total_invalid) = NaN;
            end
            
            % 6. 封裝輸出結構：將處理好的矩陣統一打包進 dataStruct 結構體中，準備回傳給下游模組 (如環境模擬器)
            dataStruct.Dates   = Dates_Active; % 基準交易日曆 (時間軸)
            dataStruct.Opens   = Opens;        % 開盤價矩陣
            dataStruct.Prices  = Prices;       % 收盤價矩陣
            dataStruct.Highs   = Highs;        % 最高價矩陣
            dataStruct.Lows    = Lows;         % 最低價矩陣
            dataStruct.Volumes = Volumes;      % 成交量矩陣
            dataStruct.IsConst = IsConst;      % True Point-in-Time 歷史成分股標籤矩陣
            
            % 執行完成，印出包含統計資訊的成功提示訊息
            fprintf('💾 成功以 SPY 基準時間軸對齊大宇宙資料 (共 %d 檔標的，%d 個真實交易日)。\n', numTickers, numDays);
        end
    end
end