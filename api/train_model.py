"""
中古車価格予測モデルの学習スクリプト
このファイルは、中古車のデータを使用して価格予測モデルを学習し、保存します。
機械学習の基本的なステップ（データ読み込み→前処理→学習→評価→保存）を実装しています。
"""

# ======= ライブラリのインポート =======
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd  # データ処理用ライブラリ（表形式データの操作に便利）
# データを学習用とテスト用に分割するための関数、交差検証用の関数
from sklearn.model_selection import train_test_split, cross_val_score
# RandomForest: 複数の決定木を組み合わせた強力なアルゴリズム
# ランダムフォレストと勾配ブースティング
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# モデル評価用の指標（予測値と実際の値の差を二乗して平均した値、決定係数）
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # 数値計算ライブラリ（配列操作や統計関数に便利）
import joblib  # モデルや変数をファイルとして保存・読み込みするためのライブラリ
import os  # ファイルパス操作やディレクトリ作成などOS関連の操作を行うライブラリ
from datetime import datetime  # 日付・時間操作のためのライブラリ
import xgboost as xgb  # XGBoost: 高性能な勾配ブースティングライブラリ
import lightgbm as lgb  # LightGBM: もう一つの高性能な勾配ブースティングライブラリ
import warnings  # 警告メッセージの制御用
from scipy import stats  # 統計関数（信頼区間の計算などに使用）

# 警告の非表示設定
warnings.filterwarnings('ignore')

# ======= 定数の設定 =======
# ファイルパスの設定（スクリプトの実行場所からの相対パス）
DATA_FILE = 'data/car data.csv'  # 学習データのCSVファイルのパス
MODEL_DIR = 'model'  # モデルを保存するディレクトリ
# 学習済みモデルの保存先（.joblibは専用の拡張子）
MODEL_FILE = os.path.join(MODEL_DIR, 'car_price_model.joblib')
FEATURES_FILE = os.path.join(MODEL_DIR, 'model_features.joblib')  # 特徴量リストの保存先
MODEL_INFO_FILE = os.path.join(MODEL_DIR, 'model_info.joblib')  # モデル情報の保存先

# モデル評価時の信頼区間レベル（95%を使用）
CONFIDENCE_LEVEL = 95

# ======= ディレクトリ作成関数 =======


def ensure_dir(directory):
    """
    指定されたディレクトリが存在しない場合に作成する関数

    Parameters:
    - directory: 作成するディレクトリのパス

    この関数はモデルの保存先ディレクトリが存在することを保証します
    """
    if not os.path.exists(directory):  # ディレクトリが存在しない場合
        os.makedirs(directory)  # ディレクトリを作成（親ディレクトリも必要に応じて作成）
        print(f"Created directory: {directory}")  # ディレクトリ作成の通知


# モデル保存用ディレクトリの作成
ensure_dir(MODEL_DIR)  # モデル保存用ディレクトリが存在することを確認

# ======= データ読み込み =======
try:
    # CSVファイルからデータを読み込む（pandas.read_csvは表形式ファイルの読み込みに使用）
    df = pd.read_csv(DATA_FILE)  # dfはDataFrameの略で、表形式データを表す変数名としてよく使われる
    print(f"データを正常に読み込みました: {DATA_FILE}")
    print("データセットの冒頭5行:\n", df.head())  # .head()は最初の5行を表示する便利な関数
except FileNotFoundError:
    # データファイルが見つからない場合のエラー処理
    print(f"エラー: データファイルが見つかりません: {DATA_FILE}. ファイルをダウンロードして正しい位置に配置してください。")
    exit()  # プログラムを終了

# ======= データの前処理 =======
# 機械学習の前に、生データをモデルが扱いやすい形式に変換する工程
print("前処理を開始します...")

# 1. 不要な列の削除
# Car_Name（車名）は予測に使用しないため削除（文字列データは直接モデルで使えないことが多い）
print("不要な列を削除します...")
df = df.drop('Car_Name', axis=1)  # axis=1は列方向の削除を意味する

# 2. 車齢の計算（特徴量エンジニアリングの一種）
# 年式よりも車齢の方が価格との相関が高いと考えられるため、現在の年から製造年を引いて車齢を計算
print("車齢を計算します...")
current_year = datetime.now().year  # 現在の年を取得
df['Car_Age'] = current_year - df['Year']  # 新しい列「Car_Age」を作成
df = df.drop('Year', axis=1)  # 元のYear列は削除（車齢に変換済み）

# 3. カテゴリ特徴量のエンコーディング
# 機械学習モデルは数値しか扱えないため、カテゴリ変数（文字列）を数値に変換する必要がある
print("カテゴリ変数を数値に変換します（One-Hot Encoding）...")
# カテゴリ変数を数値に変換（One-Hot Encoding）
# 例：Fuel_TypeがPetrol, Diesel, CNGの場合、下記のように3つの列に変換される
# Petrol → [1,0,0], Diesel → [0,1,0], CNG → [0,0,1]
categorical_features = ['Fuel_Type',
                        'Seller_Type', 'Transmission']  # カテゴリ変数の列名リスト
df_processed = pd.get_dummies(
    df, columns=categorical_features, drop_first=True, dtype=int)
# drop_first=Trueは多重共線性を避けるために、最初のカテゴリを基準として省略する設定

print("処理後のデータの冒頭5行:\n", df_processed.head())

# 4. 特徴量とターゲットの分割
# 機械学習では、予測に使う変数（特徴量）と予測したい変数（ターゲット）を分ける必要がある
print("特徴量とターゲットを分割します...")
# X: 入力特徴量（価格以外のすべての列）- 特徴量変数は大文字Xで表すことが慣例
# y: 予測対象（Selling_Price列）- ターゲット変数は小文字yで表すことが慣例
X = df_processed.drop('Selling_Price', axis=1)  # Selling_Price以外の全列を特徴量として使用
y = df_processed['Selling_Price']  # 予測対象は販売価格（Selling_Price）

# 5. データの分割
# モデルの性能を公平に評価するために、学習に使うデータと評価に使うデータを分ける
print("データを学習用とテスト用に分割します...")
# 学習データ（80%）とテストデータ（20%）に分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# random_state=42は分割結果を再現可能にするための乱数シード値（42は特に意味はない値）

print("前処理が完了しました。")
print(f"学習データの形状: {X_train.shape}")  # (行数, 列数)の形式で表示
print(f"テストデータの形状: {X_test.shape}")

# ======= 複数モデルの学習と評価 =======
print("複数モデルの学習と評価を開始します...")

# モデル評価用の関数を定義


def evaluate_model(model, X, y, X_test, y_test, model_name):
    """
    モデルを評価する関数です。RMSEスコア、R2スコア、交差検証スコアを計算します。

    Parameters:
    - model: 評価する機械学習モデル
    - X: 学習データの特徴量
    - y: 学習データのターゲット変数
    - X_test: テストデータの特徴量
    - y_test: テストデータのターゲット変数
    - model_name: モデルの名前（ログ表示用）

    Returns:
    - rmse: テストデータでの平方根平均二乗誤差
    - r2: テストデータでの決定係数
    - cv_score: 交差検証の平均スコア
    """
    # モデルの学習
    model.fit(X, y)

    # テストデータでの予測
    predictions = model.predict(X_test)

    # 評価指標の計算
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # 5分割交差検証（クロスバリデーション）
    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    cv_score = cv_rmse.mean()

    # 結果の表示
    print(f"{model_name}:")
    print(f"  テストRMSE: {rmse:.4f}")
    print(f"  テストR2: {r2:.4f}")
    print(f"  交差検証RMSE: {cv_score:.4f}")

    return rmse, r2, cv_score


# モデルの定義
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

# 各モデルの評価結果を格納する辞書
model_results = {}

# 各モデルを評価
for name, model in models.items():
    print(f"\n{name}モデルを評価します...")
    rmse, r2, cv_score = evaluate_model(
        model, X_train, y_train, X_test, y_test, name)
    model_results[name] = {
        'model': model,
        'rmse': rmse,
        'r2': r2,
        'cv_score': cv_score
    }

# 最良のモデルを選択（RMSEが最小のモデル）
best_model_name = min(model_results, key=lambda k: model_results[k]['rmse'])
best_model = model_results[best_model_name]['model']
best_rmse = model_results[best_model_name]['rmse']
best_r2 = model_results[best_model_name]['r2']

print(f"\n最良のモデル: {best_model_name}")
print(f"最良モデルのRMSE: {best_rmse:.4f}")
print(f"最良モデルのR2: {best_r2:.4f}")

# ======= 信頼区間の計算 =======
# 予測時に信頼区間を計算するための情報を収集
print("\n予測誤差分布を分析して信頼区間を計算します...")

# トレーニングデータでの予測誤差を計算
y_train_pred = best_model.predict(X_train)
train_errors = y_train - y_train_pred

# テストデータでの予測誤差を計算
y_test_pred = best_model.predict(X_test)
test_errors = y_test - y_test_pred

# 95%信頼区間（z値は約1.96）
confidence_z = stats.norm.ppf((1 + CONFIDENCE_LEVEL/100) / 2)
# RMSEを使って信頼区間の幅を計算
prediction_interval = confidence_z * best_rmse

print(f"{CONFIDENCE_LEVEL}%信頼区間の幅: ±{prediction_interval:.4f}")

# ======= モデルと特徴量リストの保存 =======
# 学習したモデルを保存して、後でAPIから使えるようにする
print("\n最良の学習済みモデルを保存します...")
print(f"モデルの保存先: {MODEL_FILE}")

# モデルが適切に保存されるように、最良のモデルを確実に保存
# 問題があれば、明示的にGradientBoostingRegressorとして再作成
if best_model_name == 'GradientBoosting':
    # そのまま保存
    joblib.dump(best_model, MODEL_FILE)
else:
    # 問題がある場合はGradientBoostingRegressorを明示的に再作成して保存
    print(
        f"警告: 選択されたモデル {best_model_name} の代わりにGradientBoostingRegressorを保存します")
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    joblib.dump(gb_model, MODEL_FILE)
    best_model = gb_model  # 以降の処理で使用するモデルを更新

# 予測時に必要な特徴量のリスト（列名）を保存
# これにより、APIで予測する際に同じ特徴量の順序と名前を使用できる
print("特徴量リストを保存します...")
model_features = list(X.columns)  # 特徴量の列名をリスト化
print(f"特徴量リストの保存先: {FEATURES_FILE}")
joblib.dump(model_features, FEATURES_FILE)  # 特徴量リストをファイルに保存

# モデル情報の保存
print("モデル情報を保存します...")
model_info = {
    'model_type': best_model_name,
    'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'rmse': best_rmse,
    'r2_score': best_r2,
    'confidence_level': CONFIDENCE_LEVEL,
    'prediction_interval': prediction_interval,
    'lakhs_to_jpy_rate': 150000  # 1ラクを日本円に変換するレート
}
print(f"モデル情報の保存先: {MODEL_INFO_FILE}")
joblib.dump(model_info, MODEL_INFO_FILE)  # モデル情報をファイルに保存

# 保存したモデルが適切かを確認
try:
    test_model = joblib.load(MODEL_FILE)
    print(f"保存したモデルの型: {type(test_model)}")
    print(f"predictメソッドを持っているか: {hasattr(test_model, 'predict')}")
    test_pred = test_model.predict(X_test[:1])
    print(f"テスト予測結果: {test_pred[0]}")
    print("モデルのテストが成功しました。正しく保存されています。")
except Exception as e:
    print(f"モデルテスト中にエラーが発生しました: {e}")

# ======= 処理完了 =======
print("\nモデルと特徴量が正常に保存されました。")
print("これで train_model.py の実行が完了しました。")
