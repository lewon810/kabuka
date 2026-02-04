import pandas as pd
import yfinance as yf
import time

import requests
from io import StringIO


def get_nikkei225_tickers():
    """
    Wikipediaから日経平均採用銘柄のリストを取得する
    Returns: dict { 'xxxx.T': 'Company Name' }
    """
    url = "https://en.wikipedia.org/wiki/Nikkei_225"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        
        import re
        
        def get_name_col(df):
            for col in ['Company', 'Constituent Name', 'Name', '銘柄']:
                if col in df.columns: return col
            # Fallback: find column with string length avg > 5 and not ticker
            for col in df.columns:
                if col in ['Symbol', 'Ticker', 'コード', '証券コード', 'Sector']: continue
                if df[col].dtype == object:
                     return col
            return None

        # 1. First, try to find a single big table
        def find_big_table(tables):
            for i, t in enumerate(tables):
                if len(t) >= 200:
                    if 'Symbol' in t.columns: return t, 'Symbol'
                    if 'Ticker' in t.columns: return t, 'Ticker'
                    if 'コード' in t.columns: return t, 'コード'
                    
                    for col in t.columns:
                        sample = t[col].astype(str).head(5)
                        if sample.apply(lambda x: bool(re.search(r'\d{4}', x))).all():
                            return t, col
            return None, None

        df, ticker_col = find_big_table(tables)
        ticker_map = {}

        if df is not None:
            # Single big table found
            name_col = get_name_col(df)
            print(f"Using big table. Ticker: {ticker_col}, Name: {name_col}")
            
            for _, row in df.iterrows():
                val = row[ticker_col]
                match = re.search(r'(\d{4})', str(val))
                if match: 
                    code = match.group(1) + ".T"
                    name = row[name_col] if name_col else code
                    ticker_map[code] = str(name)

        else:
            print("Trying JA Wikipedia aggregation...")
            url_ja = "https://ja.wikipedia.org/wiki/%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87%E6%A0%AA%E4%BE%A1"
            headers_ja = headers 
            
            try:
                response = requests.get(url_ja, headers=headers_ja)
                response.raise_for_status()
                tables_ja = pd.read_html(StringIO(response.text))
                
                for t in tables_ja:
                    found_ticker_col = None
                    if '証券コード' in t.columns: found_ticker_col = '証券コード'
                    elif 'コード' in t.columns:
                         sample = t['コード'].astype(str).head(5)
                         if sample.apply(lambda x: bool(re.search(r'\d{4}', x))).all():
                            found_ticker_col = 'コード'
                    
                    if found_ticker_col:
                        name_col = get_name_col(t)
                        for _, row in t.iterrows():
                             match = re.search(r'(\d{4})', str(row[found_ticker_col]))
                             if match:
                                 code = match.group(1) + ".T"
                                 name = row[name_col] if name_col else code
                                 ticker_map[code] = str(name)
                
            except Exception as e:
                print(f"JA Wikipedia fetch error: {e}")

        if not ticker_map:
            print("225銘柄を含むテーブルが見つかりませんでした。")
            return {}
            
        print(f"取得した銘柄数: {len(ticker_map)}")
        return ticker_map
    except Exception as e:
        print(f"銘柄リスト取得エラー: {e}")
        return {}

def fetch_stock_data(tickers, days=365):
    """
    yfinanceを使って株価データを取得する
    """
    if not tickers:
        return pd.DataFrame()
        
    print(f"{len(tickers)}銘柄のデータをダウンロード中... ({days}日分)")
    
    # yfinanceの一括ダウンロード
    # group_by='ticker' -> 銘柄ごとにまとめる (扱いやすい)
    # auto_adjust=True -> 分割併合などを調整済み
    try:
        data = yf.download(
            tickers, 
            period=f"{days}d", 
            group_by='ticker', 
            auto_adjust=True,
            progress=False,
            threads=True
        )
    except Exception as e:
        print(f"ダウンロードエラー: {e}")
        return pd.DataFrame()

    # データ整形: MultiIndex columns (Ticker, OHLCV) -> Long Format (Date, Ticker, OHLCV)
    # yfinanceのdownload結果は、Tickerがカラムのトップレベルに来る
    
    frames = []
    # 銘柄ごとにDataFrameを分割してリスト化
    # 注意: 一括DLの結果は、カラムが (Ticker, Open), (Ticker, Close)... と階層化されている
    for ticker in tickers:
        try:
            # その銘柄のデータがあるか確認
            if ticker in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else data.columns:
                 # MultiIndexの場合の抽出
                df_ticker = data[ticker].copy()
                
                # 全てNaN（データなし）の場合はスキップ
                if df_ticker.isnull().all().all():
                    continue
                    
                df_ticker['SecuritiesCode'] = ticker.replace('.T', '') # 数字のみにする習慣
                df_ticker.reset_index(inplace=True) # Dateをカラムに
                
                # 必要なカラムがあるか
                if 'Close' not in df_ticker.columns:
                    continue
                    
                frames.append(df_ticker)
        except Exception as e:
            continue
            
    if not frames:
        return pd.DataFrame()
        
    full_df = pd.concat(frames, ignore_index=True)
    
    # 整形
    # Date, SecuritiesCode, Open, High, Low, Close, Volume
    full_df.rename(columns={'Date': 'Date', 'SecuritiesCode': 'SecuritiesCode'}, inplace=True)
    
    # カラム名を統一 (Open, High, Low, Close, Volume)
    # yfinanceの一括DLだと Close等はそのままClose
    
    # Date型変換
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    
    # ソート
    full_df.sort_values(['SecuritiesCode', 'Date'], inplace=True)
    
    return full_df

def preprocess_data(df):
    """
    基本的な前処理（欠損値除去など）
    """
    # 欠損がある行を削除
    df.dropna(subset=['Close', 'Volume'], inplace=True)
    return df

if __name__ == "__main__":
    # テスト実行
    ticker_map = get_nikkei225_tickers()
    if ticker_map:
        tickers = list(ticker_map.keys())
        # テスト用に最初の3銘柄だけDL
        df = fetch_stock_data(tickers[:3], days=30)
        print(df.head())
        print(df.info())
