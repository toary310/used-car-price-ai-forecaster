import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder # LabelEncoderを追加
import joblib
import os
from datetime import datetime

# --- 定数 ---
DATA_FILE = 'data/car data.csv' # データセットのパス (ファイル名を確認してください)
MODEL_DIR = 'model'             # モデル保存用ディレクトリ
MODEL_FILE = os.path.join(MODEL_DIR, 'car_price_model.joblib')
FEATURES_FILE = os.path.join(MODEL_DIR, 'model_features.joblib')

# --- ディレクトリ作成 ---
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir(MODEL_DIR)

# --- データ読み込み ---
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Data loaded successfully from {DATA_FILE}")
    print("Dataset head:\n", df.head())
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}. Please download and place it correctly.")
    exit()

# --- 前処理 ---
print("Starting preprocessing...")

# 1. 不要な列の削除 (Car_Nameは今回使わない)
df = df.drop('Car_Name', axis=1)

# 2. 車齢 (Car_Age) の計算
current_year = datetime.now().year
df['Car_Age'] = current_year - df['Year']
df = df.drop('Year', axis=1) # 元の Year 列は削除

# 3. カテゴリ特徴量のエンコーディング (One-Hot Encoding)
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
df_processed = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=int)

# # 3. カテゴリ特徴量のエンコーディング (Label Encoding)
# # Owner は Label Encoding を使う (0, 1, 3 のままだと順序性があると誤解される可能性があるため)
# # label_encoders = {}
# # for col in ['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']:
# #     le = LabelEncoder()
# #     df_processed[col] = le.fit_transform(df_processed[col])
# #     label_encoders[col] = le # 後で使うために保存

# # # Owner は数値なのでそのまま使用 (または One-Hot も検討可能)
# # df_processed['Owner'] = df['Owner']

print("Processed data head:\n", df_processed.head())

# 4. 特徴量 (X) とターゲット (y) の分割
X = df_processed.drop('Selling_Price', axis=1)
y = df_processed['Selling_Price']

# 5. 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Preprocessing finished.")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- モデル学習 ---
print("Starting model training...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 で CPU コアを活用
rf_model.fit(X_train, y_train)
print("Model training finished.")

# --- モデル評価 (任意) ---
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print(f"Model Evaluation (RMSE on test set): {rmse:.4f}")

# --- モデルと特徴量リストの保存 ---
print(f"Saving model to {MODEL_FILE}")
joblib.dump(rf_model, MODEL_FILE)

# 予測時に同じ特徴量を使えるように、学習時の列名を保存
model_features = list(X.columns)
print(f"Saving feature list to {FEATURES_FILE}")
joblib.dump(model_features, FEATURES_FILE)

# # LabelEncoder を使った場合、エンコーダーも保存
# # joblib.dump(label_encoders, os.path.join(MODEL_DIR, 'label_encoders.joblib'))

print("Model and features saved successfully.")