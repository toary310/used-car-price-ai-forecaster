"""
中古車価格予測モデルの学習スクリプト
このファイルは、中古車のデータを使用して価格予測モデルを学習し、保存します。
"""

import pandas as pd  # データ処理用ライブラリ
from sklearn.model_selection import train_test_split  # データ分割用
from sklearn.ensemble import RandomForestRegressor  # ランダムフォレスト回帰モデル
from sklearn.metrics import mean_squared_error  # モデル評価用
from sklearn.preprocessing import LabelEncoder  # カテゴリ変数のエンコーディング用
import joblib  # モデルの保存・読み込み用
import os
from datetime import datetime

# --- 定数 ---
# ファイルパスの設定
DATA_FILE = 'data/car data.csv'  # 学習データのCSVファイル
MODEL_DIR = 'model'  # モデルを保存するディレクトリ
MODEL_FILE = os.path.join(MODEL_DIR, 'car_price_model.joblib')  # 学習済みモデルの保存先
FEATURES_FILE = os.path.join(MODEL_DIR, 'model_features.joblib')  # 特徴量リストの保存先

# --- ディレクトリ作成 ---


def ensure_dir(directory):
    """
    指定されたディレクトリが存在しない場合に作成する関数

    Parameters:
    - directory: 作成するディレクトリのパス
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


# モデル保存用ディレクトリの作成
ensure_dir(MODEL_DIR)

# --- データ読み込み ---
try:
    # CSVファイルからデータを読み込む
    df = pd.read_csv(DATA_FILE)
    print(f"Data loaded successfully from {DATA_FILE}")
    print("Dataset head:\n", df.head())
except FileNotFoundError:
    # データファイルが見つからない場合のエラー処理
    print(
        f"Error: Data file not found at {DATA_FILE}. Please download and place it correctly.")
    exit()

# --- 前処理 ---
print("Starting preprocessing...")

# 1. 不要な列の削除
# Car_Name（車名）は予測に使用しないため削除
df = df.drop('Car_Name', axis=1)

# 2. 車齢の計算
# 現在の年から製造年を引いて車齢を計算
current_year = datetime.now().year
df['Car_Age'] = current_year - df['Year']
df = df.drop('Year', axis=1)  # 元のYear列は削除（車齢に変換済み）

# 3. カテゴリ特徴量のエンコーディング
# カテゴリ変数を数値に変換（One-Hot Encoding）
# 例：Fuel_TypeがPetrol, Diesel, CNGの場合、3つの列に変換
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
df_processed = pd.get_dummies(
    df, columns=categorical_features, drop_first=True, dtype=int)

# コメントアウトされたLabel Encodingの実装例
# Owner（前オーナー数）は順序性があるため、Label Encodingを使用することも可能
# label_encoders = {}
# for col in ['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']:
#     le = LabelEncoder()
#     df_processed[col] = le.fit_transform(df_processed[col])
#     label_encoders[col] = le

print("Processed data head:\n", df_processed.head())

# 4. 特徴量とターゲットの分割
# X: 入力特徴量（価格以外のすべての列）
# y: 予測対象（Selling_Price列）
X = df_processed.drop('Selling_Price', axis=1)
y = df_processed['Selling_Price']

# 5. データの分割
# 学習データ（80%）とテストデータ（20%）に分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Preprocessing finished.")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- モデル学習 ---
print("Starting model training...")
# ランダムフォレスト回帰モデルの作成と学習
# n_estimators: 決定木の数
# random_state: 乱数シード（再現性のため）
# n_jobs: 並列処理の数（-1で全CPUコアを使用）
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Model training finished.")

# --- モデル評価 ---
# テストデータでの予測と評価
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  # 平均二乗誤差
rmse = mse**0.5  # 平方根平均二乗誤差
print(f"Model Evaluation (RMSE on test set): {rmse:.4f}")

# --- モデルと特徴量リストの保存 ---
# 学習済みモデルを保存
print(f"Saving model to {MODEL_FILE}")
joblib.dump(rf_model, MODEL_FILE)

# 予測時に必要な特徴量のリストを保存
# これにより、APIで予測する際に同じ特徴量を使用できる
model_features = list(X.columns)
print(f"Saving feature list to {FEATURES_FILE}")
joblib.dump(model_features, FEATURES_FILE)

print("Model and features saved successfully.")
