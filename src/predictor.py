import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

def create_features(df):
    """
    予測に有効な特徴量を作成します。
    JPX方式の肝は「テクニカル」と「感情」の組み合わせです。
    """
    df = df.copy()
    
    # 銘柄ごとの処理
    # 処理速度向上のために、applyではなくベクトル化またはgroupby().transformを使用
    # しかし、rollingなどはgroupby().rollingだとMultiIndexになるので注意が必要
    
    # 簡単のため、demo.pyのループ処理を踏襲しつつ、warningが出ないように修正
    # groupby objectをループするのは遅いが、225銘柄なら許容範囲
    
    result_dfs = []
    
    for code, group in df.groupby('SecuritiesCode'):
        group = group.sort_values('Date').copy()
        
        # --- A. リターン系特徴量 (Lag Features) ---
        for lag in [1, 5, 21]:
            col_name = f'return_{lag}d'
            group[col_name] = group['Close'].pct_change(lag)
            
        # --- B. ボラティリティ (Volatility) ---
        group['volatility_20d'] = group['Close'].pct_change().rolling(20).std()
        
        # --- C. 移動平均乖離率 (Moving Average Divergence) ---
        for window in [5, 25, 75]:
            ma = group['Close'].rolling(window).mean()
            group[f'ma_divergence_{window}'] = (group['Close'] - ma) / ma

        # --- D. RSI (Relative Strength Index) ---
        delta = group['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        group['RSI'] = 100 - (100 / (1 + rs))

        # --- E. 感情分析特徴量の加工 (Sentiment) ---
        if 'SentimentScore' in group.columns:
            group['sentiment_ma_5'] = group['SentimentScore'].rolling(5).mean()
            group['sentiment_momentum'] = group['SentimentScore'].diff(3)
            
        result_dfs.append(group)
        
    df = pd.concat(result_dfs)

    # --- F. 市場全体に対する相対指標 (Market Relative Features) ---
    # JPX方式で最も重要な要素：その日の全銘柄平均との差分
    date_groups = df.groupby('Date')
    
    target_cols = ['return_5d', 'volatility_20d', 'RSI']
    if 'SentimentScore' in df.columns:
        target_cols.append('SentimentScore')
        
    for col in target_cols:
        if col in df.columns:
            # その日の平均値
            daily_mean = date_groups[col].transform('mean')
            # 平均からの乖離
            df[f'{col}_rel'] = df[col] - daily_mean

    return df

def train_model(df):
    """
    モデル学習 (LightGBM)
    """
    # ターゲット生成 (Target Engineering)
    # 5日後(1週間後)のリターン
    horizon = 5
    
    # 銘柄ごとにShiftしてターゲットを作る
    # ここでもループではなくgroupby.shiftを使う
    # dfは既にDate, SecuritiesCodeでソートされていると仮定したいが、念のため
    df.sort_values(['SecuritiesCode', 'Date'], inplace=True)
    df['Target_Return'] = df.groupby('SecuritiesCode')['Close'].shift(-horizon) / df['Close'] - 1
    
    # 学習に使用するデータ（ターゲットがNaNでないもの）
    train_df = df.dropna(subset=['Target_Return']).copy()
    
    # 特徴量リスト
    features = [
        'return_1d', 'return_5d', 'return_21d',
        'volatility_20d',
        'ma_divergence_5', 'ma_divergence_25', 'ma_divergence_75',
        'RSI',
        'return_5d_rel', 'RSI_rel'
    ]
    if 'SentimentScore' in df.columns:
        features.extend(['SentimentScore', 'sentiment_ma_5', 'sentiment_momentum', 'SentimentScore_rel'])
        
    target = 'Target_Return'
    
    # 時系列分割 (Time Series Split)
    tscv = TimeSeriesSplit(n_splits=3)
    
    models = []
    train_df.sort_values(['Date', 'SecuritiesCode'], inplace=True)
    
    print("Training LightGBM models...")
    for fold, (train_index, val_index) in enumerate(tscv.split(train_df)):
        X_train, X_val = train_df.iloc[train_index][features], train_df.iloc[val_index][features]
        y_train, y_val = train_df.iloc[train_index][target], train_df.iloc[val_index][target]
        
        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,
            'seed': 42
        }
        
        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        models.append(model)
        print(f"Fold {fold+1} RMSE: {model.best_score['valid_1']['rmse']:.5f}")
        
    return models, features # モデルリストと使用した特徴量を返す

def predict_next_week(models_and_features, df):
    """
    最新データに対する予測
    """
    if models_and_features is None:
        return pd.DataFrame()
        
    models, features = models_and_features
    
    # 最新の日付（銘柄ごとに最新）だけを取り出す
    # 実際には「予測時点」のデータが必要。
    # ここではデータセット全体の最新日付（たとえば今日）のデータを使う
    latest_date = df['Date'].max()
    target_df = df[df['Date'] == latest_date].copy()
    
    if target_df.empty:
        return pd.DataFrame()
        
    # 特徴量が計算されているか確認（create_features済みであること）
    # 欠損がある場合は予測できないので除去（RSIなどは初期データでNaNになる）
    target_df.dropna(subset=features, inplace=True)
    
    if target_df.empty:
        print("最新データの特徴量が欠損しているため予測できません。データ期間が短すぎる可能性があります。")
        return pd.DataFrame()

    # アンサンブル予測
    preds = np.zeros(len(target_df))
    for model in models:
        preds += model.predict(target_df[features]) / len(models)
        
    target_df['Predicted_Return'] = preds
    
    # ランキング
    target_df['Rank'] = target_df['Predicted_Return'].rank(ascending=False, method='first')
    
    # 結果を整理
    result = target_df[['Date', 'SecuritiesCode', 'Close', 'Predicted_Return', 'Rank']].sort_values('Rank')
    return result
