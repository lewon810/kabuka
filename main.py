import sys
import os
import argparse
from src.data_loader import get_nikkei225_tickers, fetch_stock_data, preprocess_data
from src.predictor import create_features, train_model, predict_next_week
from src.report_generator import generate_html

def main():
    parser = argparse.ArgumentParser(description='AI Stock Prediction Pipeline')
    parser.add_argument('--days', type=int, default=365*2, help='Days of historical data to fetch')
    parser.add_argument('--output', type=str, default='index.html', help='Output HTML path')
    args = parser.parse_args()

    print("=== 1. Data Loading ===")
    ticker_map = get_nikkei225_tickers()
    if not ticker_map:
        print("Error: No tickers found.")
        sys.exit(1)
        
    tickers = list(ticker_map.keys())
    # For testing, limit tickers if needed (or fetch all)
    # tickers = tickers[:20] 
    
    df = fetch_stock_data(tickers, days=args.days)
    if df.empty:
        print("Error: No data fetched.")
        sys.exit(1)
        
    df = preprocess_data(df)
    print(f"Loaded data: {df.shape}")

    print("\n=== 2. Prediction ===")
    df_features = create_features(df)
    
    # モデル学習
    # 実運用では、毎日学習するか、保存したモデルをロードするか選べるようにすべきだが、
    # ここでは毎回学習する仕様とする（データ量が少ないため高速）
    models_and_features = train_model(df_features)
    
    # 翌週予測
    predictions = predict_next_week(models_and_features, df_features)
    
    if predictions.empty:
        print("Error: Prediction failed.")
        sys.exit(1)
        
    print(f"Predicted for {len(predictions)} stocks.")
    print(predictions.head())

    print("\n=== 3. Reporting ===")
    generate_html(predictions, args.output, ticker_map=ticker_map)
    print(f"Report generated: {args.output}")

if __name__ == "__main__":
    main()
