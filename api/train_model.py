"""
中古車価格予測モデルの学習スクリプト
このファイルは、中古車のデータを使用して価格予測モデルを学習し、保存します。
機械学習の基本的なステップ（データ読み込み→前処理→学習→評価→保存）を実装しています。
"""

# ======= ライブラリのインポート =======
import pandas as pd  # データ処理用ライブラリ（表形式データの操作に便利）
from sklearn.model_selection import train_test_split  # データを学習用とテスト用に分割するための関数
# ランダムフォレスト回帰モデル（複数の決定木を組み合わせた強力なアルゴリズム）
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error  # モデル評価用の指標（予測値と実際の値の差を二乗して平均した値）
from sklearn.preprocessing import LabelEncoder  # カテゴリ変数を数値に変換するエンコーダー
import joblib  # モデルや変数をファイルとして保存・読み込みするためのライブラリ
import os  # ファイルパス操作やディレクトリ作成などOS関連の操作を行うライブラリ
from datetime import datetime  # 日付・時間操作のためのライブラリ

# ======= 定数の設定 =======
# ファイルパスの設定（スクリプトの実行場所からの相対パス）
DATA_FILE = 'data/car data.csv'  # 学習データのCSVファイルのパス
MODEL_DIR = 'model'  # モデルを保存するディレクトリ
# 学習済みモデルの保存先（.joblibは専用の拡張子）
MODEL_FILE = os.path.join(MODEL_DIR, 'car_price_model.joblib')
FEATURES_FILE = os.path.join(MODEL_DIR, 'model_features.joblib')  # 特徴量リストの保存先

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

# コメントアウトされたLabel Encodingの実装例
# Label Encodingは別のエンコーディング手法で、カテゴリを0,1,2...のような連番に変換する
# Owner（前オーナー数）は順序性があるため、Label Encodingを使用することも可能
# label_encoders = {}
# for col in ['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']:
#     le = LabelEncoder()
#     df_processed[col] = le.fit_transform(df_processed[col])
#     label_encoders[col] = le

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

# ======= モデル学習 =======
print("モデル学習を開始します...")
# ランダムフォレスト回帰モデルの作成と学習
# ランダムフォレストは複数の決定木を組み合わせた強力なアルゴリズムで、様々なデータに対して良い性能を発揮する
print("RandomForestRegressorモデルを初期化します...")
# モデルの主要パラメータ:
# n_estimators: 決定木の数（多いほど精度が上がるが、計算時間も増加）
# random_state: 乱数シード（再現性のため）
# n_jobs: 並列処理の数（-1で全CPUコアを使用して計算を高速化）
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# モデルの学習（fitメソッドを使用）
print("モデルの学習を実行します...")
rf_model.fit(X_train, y_train)  # 学習データを使ってモデルを訓練
print("モデル学習が完了しました。")

# ======= モデル評価 =======
print("モデルの評価を行います...")
# テストデータでの予測と評価（学習に使っていないデータで評価することが重要）
y_pred = rf_model.predict(X_test)  # テストデータを使って予測
mse = mean_squared_error(y_test, y_pred)  # 平均二乗誤差（値が小さいほど良いモデル）
rmse = mse**0.5  # 平方根平均二乗誤差（元の単位に戻すために平方根を取る）
print(f"モデル評価 (テストデータでのRMSE): {rmse:.4f}")
# RMSEは「平均的に予測値が実際の値からどのくらい離れているか」を示す指標

# ======= モデルと特徴量リストの保存 =======
# 学習したモデルを保存して、後でAPIから使えるようにする
print("学習済みモデルを保存します...")
print(f"モデルの保存先: {MODEL_FILE}")
joblib.dump(rf_model, MODEL_FILE)  # モデルをファイルに保存

# 予測時に必要な特徴量のリスト（列名）を保存
# これにより、APIで予測する際に同じ特徴量の順序と名前を使用できる
print("特徴量リストを保存します...")
model_features = list(X.columns)  # 特徴量の列名をリスト化
print(f"特徴量リストの保存先: {FEATURES_FILE}")
joblib.dump(model_features, FEATURES_FILE)  # 特徴量リストをファイルに保存

# ======= 処理完了 =======
print("モデルと特徴量が正常に保存されました。")
print("これで train_model.py の実行が完了しました。")
