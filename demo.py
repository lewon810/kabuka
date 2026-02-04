#このコードは以下のKagleを参考に作成しています
#https://www.kaggle.com/code/ikeppyo/jpx-lightgbm-demo

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ==========================================
# 1. 設定 & 概要 (Configuration)
# ==========================================
"""
【JPX方式（ランキング学習）株価予測モデル】

■ 概要
このモデルは、個別銘柄の「価格」ではなく、市場全体における「相対的なリターンの順位（ランク）」を予測します。
市場全体が暴落する局面でも、相対的に強い銘柄（下落率が低い銘柄）を上位にランク付けすることで、
マーケットニュートラルなポートフォリオ構築を目指します。

■ アルゴリズムの流れ
1. データ準備: 複数銘柄の過去データを取得
2. 特徴量生成: テクニカル指標 + 感情スコア（外部データ想定）
3. ターゲット生成: 1週間後（5営業日後）のリターンを計算
4. 学習 (LightGBM): ターゲット（リターン）を回帰予測
5. ランキング化: 予測されたリターンに基づき、日付ごとに銘柄を順位付け
6. 評価: 上位銘柄と下位銘柄のパフォーマンス差（スプレッド）を確認
"""

CONFIG = {
    'num_tickers': 20,       # ダミーデータの銘柄数（実運用では2000程度推奨）
    'days': 500,             # 過去データの日数
    'prediction_horizon': 5, # 何日後を予測するか（5日＝1週間）
    'folds': 3,              # Cross Validationの分割数
    'random_seed': 42
}

# ==========================================
# 2. データ生成 / 取得 (Data Loading)
# ==========================================
def generate_dummy_data(num_tickers, days):
    """
    動作確認用のダミーデータを生成します。
    実運用の際は、ここで yfinance や CSV からデータを読み込んでください。
    必須カラム: ['Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume']
    オプション: ['SentimentScore'] (感情分析スコア)
    """
    np.random.seed(CONFIG['random_seed'])
    
    dates = pd.date_range(end=datetime.today(), periods=days, freq='B') # 営業日
    data = []
    
    for code in range(1000, 1000 + num_tickers):
        # ランダムウォークで価格生成
        price = 1000 + np.cumsum(np.random.randn(days)) * 10
        # 感情スコア（-1.0 ~ 1.0）をランダム生成（少し価格に相関を持たせる）
        sentiment = np.random.uniform(-1, 1, days) + (np.diff(price, prepend=price[0]) * 0.01)
        
        df_ticker = pd.DataFrame({
            'Date': dates,
            'SecuritiesCode': code,
            'Open': price + np.random.randn(days),
            'High': price + np.abs(np.random.randn(days)) * 5,
            'Low': price - np.abs(np.random.randn(days)) * 5,
            'Close': price,
            'Volume': np.random.randint(1000, 100000, days),
            'SentimentScore': sentiment # 感情分析の特徴量（FinBERT等の出力想定）
        })
        data.append(df_ticker)
        
    full_df = pd.concat(data).reset_index(drop=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df.sort_values(['SecuritiesCode', 'Date'], inplace=True)
    return full_df

print("データを生成中...")
df = generate_dummy_data(CONFIG['num_tickers'], CONFIG['days'])
print(f"データ生成完了: {df.shape}")

# ==========================================
# 3. 特徴量エンジニアリング (Feature Engineering)
# ==========================================
def create_features(df):
    """
    予測に有効な特徴量を作成します。
    JPX方式の肝は「テクニカル」と「感情」の組み合わせです。
    """
    df = df.copy()
    
    # 銘柄ごとの処理
    for code, group in df.groupby('SecuritiesCode'):
        
        # --- A. リターン系特徴量 (Lag Features) ---
        # 過去のリターンは将来のボラティリティを示唆する
        for lag in [1, 5, 21]:
            # pct_change() は変化率
            col_name = f'return_{lag}d'
            df.loc[group.index, col_name] = group['Close'].pct_change(lag)
            
        # --- B. ボラティリティ (Volatility) ---
        # 過去の変動幅 (Historical Volatility)
        df.loc[group.index, 'volatility_20d'] = group['Close'].pct_change().rolling(20).std()
        
        # --- C. 移動平均乖離率 (Moving Average Divergence) ---
        for window in [5, 25, 75]:
            ma = group['Close'].rolling(window).mean()
            df.loc[group.index, f'ma_divergence_{window}'] = (group['Close'] - ma) / ma

        # --- D. RSI (Relative Strength Index) ---
        # 簡易計算
        delta = group['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df.loc[group.index, 'RSI'] = 100 - (100 / (1 + rs))

        # --- E. 感情分析特徴量の加工 (Sentiment) ---
        # 単日のスコアだけでなく、移動平均（トレンド）を見る
        if 'SentimentScore' in df.columns:
            df.loc[group.index, 'sentiment_ma_5'] = group['SentimentScore'].rolling(5).mean()
            # 感情の変化（改善したか悪化したか）
            df.loc[group.index, 'sentiment_momentum'] = group['SentimentScore'].diff(3)

    # --- F. 市場全体に対する相対指標 (Market Relative Features) ---
    # JPX方式で最も重要な要素：その日の全銘柄平均との差分
    # これにより「市場全体が上がったから上がった」ノイズを除去できる
    date_groups = df.groupby('Date')
    for col in ['return_5d', 'volatility_20d', 'RSI', 'SentimentScore']:
        # その日の平均値
        daily_mean = date_groups[col].transform('mean')
        # 平均からの乖離
        df[f'{col}_rel'] = df[col] - daily_mean

    return df

print("特徴量を作成中...")
df = create_features(df)
# 欠損値処理（移動平均などでNaNが出るため削除）
df.dropna(inplace=True)
print("特徴量作成完了")

# ==========================================
# 4. ターゲット生成 (Target Engineering)
# ==========================================
def create_target(df, horizon):
    """
    学習させる正解データを作成します。
    JPX方式：未来のリターンそのものではなく、リターンを予測させ、後でランク付けします。
    """
    # horizon日後のClose / 今日のClose - 1
    # shift(-horizon) で未来のデータを現在の行に持ってくる
    df['Target_Return'] = df.groupby('SecuritiesCode')['Close'].shift(-horizon) / df['Close'] - 1
    
    # 学習に使えない（未来データがない）末尾の行を削除
    df.dropna(subset=['Target_Return'], inplace=True)
    return df

df = create_target(df, CONFIG['prediction_horizon'])

# ==========================================
# 5. モデル学習: LightGBM (Training)
# ==========================================
# 学習に使用する特徴量のリスト
features = [
    'return_1d', 'return_5d', 'return_21d',
    'volatility_20d',
    'ma_divergence_5', 'ma_divergence_25', 'ma_divergence_75',
    'RSI',
    'SentimentScore', 'sentiment_ma_5', 'sentiment_momentum', # 感情系
    'return_5d_rel', 'RSI_rel', 'SentimentScore_rel'         # 相対系
]

target = 'Target_Return'

# 時系列分割 (Time Series Split)
# 過去データで学習し、未来データでテストする（リーク防止）
tscv = TimeSeriesSplit(n_splits=CONFIG['folds'])

print("\n=== モデル学習開始 (LightGBM) ===")

models = []
importances = pd.DataFrame()

# 日付でソートされていることを確認（重要）
df.sort_values(['Date', 'SecuritiesCode'], inplace=True)

for fold, (train_index, val_index) in enumerate(tscv.split(df)):
    X_train, X_val = df.iloc[train_index][features], df.iloc[val_index][features]
    y_train, y_val = df.iloc[train_index][target], df.iloc[val_index][target]
    
    # データセット作成
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    
    # パラメータ設定 (回帰モデルとして設定)
    # Rankerを使う方法もあるが、リターン回帰→ソートの方が安定しやすい
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'random_state': CONFIG['random_seed'],
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0) # ログ出力を抑制
        ]
    )
    
    models.append(model)
    
    # 特徴量重要度の保存
    imp_df = pd.DataFrame({
        'feature': features,
        'gain': model.feature_importance(importance_type='gain'),
        'fold': fold + 1
    })
    importances = pd.concat([importances, imp_df])
    
    print(f"Fold {fold+1} Finished. Best RMSE: {model.best_score['valid_1']['rmse']:.5f}")

# ==========================================
# 6. ランキング予測と評価 (Prediction & Ranking)
# ==========================================
print("\n=== 予測とランキング生成 ===")

# 最新のテスト期間（最後のFoldのValidationデータ）を使ってシミュレーション
test_df = df.iloc[val_index].copy()

# アンサンブル予測（全Foldのモデルの平均）
preds = np.zeros(len(test_df))
for model in models:
    preds += model.predict(test_df[features]) / len(models)

test_df['Predicted_Return'] = preds

# ★ JPX方式の核心 ★
# 日付ごとに、予測リターンの高い順にランク付けする
# method='first' は同率の場合の処理
test_df['Rank'] = test_df.groupby('Date')['Predicted_Return'].rank(ascending=False, method='first')

# ポートフォリオ構築シミュレーション
# 毎日、予測上位3銘柄を買い(Long)、下位3銘柄を売り(Short)と仮定
top_k = 3
long_portfolio = test_df[test_df['Rank'] <= top_k]
short_portfolio = test_df[test_df['Rank'] > (CONFIG['num_tickers'] - top_k)]

# リターンの計算
long_return = long_portfolio.groupby('Date')['Target_Return'].mean()
short_return = short_portfolio.groupby('Date')['Target_Return'].mean()

# ロング・ショート戦略のリターン（買いの益 - 売りの益）
# ※売りの益は「価格が下がるとプラス」なので、(Long - Short) で計算
strategy_return = long_return - short_return
cumulative_return = (1 + strategy_return).cumprod()

# 市場平均（ベンチマーク）
market_return = test_df.groupby('Date')['Target_Return'].mean()
cumulative_market = (1 + market_return).cumprod()

# ==========================================
# 7. 結果の可視化 (Visualization)
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(cumulative_return.index, cumulative_return, label='JPX Strategy (Long-Short)', color='blue', linewidth=2)
plt.plot(cumulative_market.index, cumulative_market, label='Market Average', color='gray', linestyle='--')
plt.title(f'Cumulative Return Simulation (Top {top_k} vs Bottom {top_k})')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# グラフを表示せず保存する場合は plt.savefig('result.png')
# plt.show() 

# 特徴量重要度の表示
mean_imp = importances.groupby('feature')['gain'].mean().sort_values(ascending=False)
print("\n=== 特徴量重要度 (Top 10) ===")
print(mean_imp.head(10))

print("\n=== シミュレーション結果 ===")
print(f"戦略トータルリターン: {(cumulative_return.iloc[-1] - 1) * 100:.2f}%")
print(f"市場平均リターン: {(cumulative_market.iloc[-1] - 1) * 100:.2f}%")

# シャープレシオの簡易計算（年率換算）
# 日次リターンの平均 / 日次リターンの標準偏差 * sqrt(252営業日)
sharpe_ratio = strategy_return.mean() / strategy_return.std() * np.sqrt(252)
print(f"シャープレシオ (年率): {sharpe_ratio:.2f}")

print("\n完了。このコードをベースに、実データの読み込み部分を実装してください。")