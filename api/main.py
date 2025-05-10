"""
中古車価格予測APIのメインファイル
このファイルは、FastAPIを使用して中古車の価格を予測するためのWeb APIを提供します。
学習済みモデルを読み込み、HTTPリクエストを受け付けて、予測結果をJSONで返します。
FastAPIは最新のPython Web APIフレームワークで、高速で使いやすく、自動ドキュメント生成機能も備えています。
"""

# ======= ライブラリのインポート =======
import joblib  # モデルの保存・読み込みに使用（学習済みモデルをファイルから読み込むため）
import pandas as pd  # データ処理に使用（DataFrameでデータを扱いやすくするため）
import numpy as np  # 数値計算用（配列操作や統計計算に使用）
from fastapi import FastAPI, HTTPException  # Web APIフレームワーク（APIエンドポイントの作成と例外処理）
# クロスオリジンリソース共有の設定（異なるドメイン間でのリクエスト許可）
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field  # データバリデーション（入力データの検証と制約の設定）
from datetime import datetime  # 日付・時間操作（現在の年を取得するため）
import os  # ファイルパス操作（ファイルパスを結合するため）

# ======= 定数の設定 =======
# モデルファイルと特徴量リストの保存場所を指定
MODEL_DIR = 'model'  # モデルを保存するディレクトリ
MODEL_FILE = os.path.join(MODEL_DIR, 'car_price_model.joblib')  # 学習済みモデルのファイル
FEATURES_FILE = os.path.join(
    MODEL_DIR, 'model_features.joblib')  # モデルが使用する特徴量のリスト
MODEL_INFO_FILE = os.path.join(MODEL_DIR, 'model_info.joblib')  # モデル情報ファイル

# ======= モデルと特徴量リストの読み込み =======
try:
    # 学習済みモデルと特徴量リストを読み込む
    # joblibはscikit-learnのモデルを効率的に保存・読み込みするためのライブラリ
    print(f"モデルを読み込んでいます: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)  # 学習済みモデルを読み込み

    print(f"特徴量リストを読み込んでいます: {FEATURES_FILE}")
    model_features = joblib.load(FEATURES_FILE)  # 特徴量リストを読み込み

    # モデル情報を読み込む（存在すれば）
    # モデル情報には、モデルの種類や交差検証結果などが含まれる可能性がある
    model_info = None
    if os.path.exists(MODEL_INFO_FILE):
        print(f"モデル情報を読み込んでいます: {MODEL_INFO_FILE}")
        model_info = joblib.load(MODEL_INFO_FILE)

    print("モデルと特徴量を正常に読み込みました。")
    print(f"予測に必要な特徴量: {model_features}")
except FileNotFoundError:
    # モデルファイルが見つからない場合のエラー処理
    print(
        f"エラー: モデルまたは特徴量ファイルが {MODEL_DIR} に見つかりません。"
        "先に train_model.py を実行してモデルを学習してください。")
    model = None
    model_features = None
    model_info = None
except Exception as e:
    # その他のエラー処理
    print(f"モデルまたは特徴量の読み込み中にエラーが発生しました: {e}")
    model = None
    model_features = None
    model_info = None

# ======= FastAPI アプリケーションの初期化 =======
# FastAPIのインスタンスを作成し、APIの基本情報を設定
# この情報はSwagger UIなどの自動生成ドキュメントに表示される
app = FastAPI(
    title="中古車価格予測API",  # APIのタイトル
    description="中古車の特徴から販売価格を予測するためのAPI。機械学習モデルを使用して予測を行います。",
    version="0.1.0"  # APIのバージョン
)

# ======= CORS 設定 =======
# CORS(Cross-Origin Resource Sharing)は、異なるオリジン（ドメイン）間でのリクエストを許可するためのセキュリティ機構
# フロントエンドアプリケーションからのリクエストを許可するための設定
# WebブラウザはデフォルトでCORSを制限しているため、APIサーバー側で明示的に許可する必要がある
origins = [
    "http://localhost:3000",  # Next.jsの開発サーバー
    # 必要に応じて他のオリジンを追加（本番環境のドメインなど）
]

# CORSミドルウェアを追加（ミドルウェアはリクエスト/レスポンスの処理の途中で動作する仕組み）
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 許可するオリジンのリスト
    allow_credentials=True,  # クレデンシャル（Cookie等）の送信を許可
    allow_methods=["*"],  # すべてのHTTPメソッドを許可（GET, POST, PUT, DELETEなど）
    allow_headers=["*"],  # すべてのHTTPヘッダーを許可
)

# ======= リクエストボディの Pydantic モデル =======
# Pydanticは入力データの検証と型変換を行うライブラリ
# フロントエンドから送信されるデータの構造を定義（入力バリデーション）
# Fieldクラスを使用して各フィールドの制約を設定（例：最小値、最大値など）


class CarFeaturesInput(BaseModel):
    # 各フィールドに型とバリデーションルールを設定
    # ... は必須フィールドを意味する
    # gt, ge は「より大きい」、「以上」の制約（greater than, greater equal）
    # lt, le は「より小さい」、「以下」の制約（less than, less equal）
    year: int = Field(..., gt=1979, lt=datetime.now().year +
                      2, description="車両の製造年")
    present_price: float = Field(..., gt=0, description="現在のショールーム価格（万円）")
    kms_driven: int = Field(..., ge=0, description="走行距離（キロメートル）")
    fuel_type: str = Field(..., description="燃料タイプ（Petrol, Diesel, CNG）")
    seller_type: str = Field(..., description="販売者タイプ（Dealer, Individual）")
    transmission: str = Field(...,
                              description="トランスミッションタイプ（Manual, Automatic）")
    owner: int = Field(..., ge=0, lt=5, description="前オーナー数（0, 1, 3）")

    # APIドキュメント用のサンプルデータ（Swagger UIで表示される）
    # pydantic v2から導入されたmodel_configを使用
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "year": 2018,
                    "present_price": 5.59,
                    "kms_driven": 27000,
                    "fuel_type": "Petrol",
                    "seller_type": "Dealer",
                    "transmission": "Manual",
                    "owner": 0
                }
            ]
        }
    }

# ======= レスポンスボディの Pydantic モデル =======
# APIのレスポンスデータの構造を定義（出力の形式を明確にする）


class PredictionOutput(BaseModel):
    # 予測結果を表すフィールド
    predicted_price_jpy: float = Field(..., description="予測された販売価格（円）")
    lower_bound_jpy: float = Field(..., description="予測価格の下限（円）")
    upper_bound_jpy: float = Field(..., description="予測価格の上限（円）")
    confidence_level: float = Field(..., description="信頼区間のレベル（%）")

# ======= ルートエンドポイント (動作確認用) =======
# @app.get() デコレータはGETリクエストに対応するエンドポイントを定義
# / はルートパス（例：http://localhost:8000/）


@app.get("/")
async def read_root():
    """APIの動作確認用のエンドポイント（サーバーが稼働しているか確認するため）"""
    return {"message": "Welcome to the Car Price Prediction API!"}

# ======= 予測エンドポイント =======
# @app.post() デコレータはPOSTリクエストに対応するエンドポイントを定義
# /predict はパス（例：http://localhost:8000/predict）
# response_model で返すデータの型を指定（PydanticモデルのPredictionOutput）


@app.post("/predict", response_model=PredictionOutput)
async def predict_price(features: CarFeaturesInput):
    """
    中古車の価格を予測するエンドポイント

    Parameters:
    - features: 車両の特徴（年式、価格、走行距離など）

    Returns:
    - predicted_price_jpy: 予測された販売価格（円）
    - lower_bound_jpy: 予測価格の下限（円）
    - upper_bound_jpy: 予測価格の上限（円）
    - confidence_level: 信頼区間のレベル（%）
    """
    # ===== 1. 入力チェック =====
    # モデルが読み込まれていない場合のエラー処理
    if model is None or model_features is None:
        # HTTPExceptionでエラーレスポンスを返す（ステータスコード500: サーバー内部エラー）
        raise HTTPException(
            status_code=500,
            detail="モデルが読み込まれていないため、予測できません。サーバー管理者に連絡してください。"
        )

    # 受け取ったデータをログに出力（デバッグ用）
    print(f"予測のために受け取った特徴量: {features.model_dump()}")

    # ===== 2. データ変換 =====
    # 1. 受け取ったデータをDataFrameに変換（機械学習モデルの入力形式に合わせる）
    input_data = pd.DataFrame([features.model_dump()])

    # ===== 3. データの前処理 =====
    try:
        # 2a. 車齢の計算
        # 現在の年から製造年を引いて車齢を計算（年式よりも車齢の方が価格との相関が高いため）
        current_year = datetime.now().year
        input_data['Car_Age'] = current_year - input_data['year']
        # yearをモデルの特徴量に含めない場合は削除: input_data = input_data.drop('year', axis=1)

        # 2b. カテゴリ特徴量のOne-Hot Encoding
        # カテゴリ変数（文字列）を数値表現に変換
        # 例：'Petrol'→[1,0,0], 'Diesel'→[0,1,0], 'CNG'→[0,0,1]
        categorical_cols_to_encode = [
            'fuel_type', 'seller_type', 'transmission']

        # カテゴリカル列を明示的にcategory型に変換
        for col in categorical_cols_to_encode:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype('category')

        # One-Hot Encoding（pd.get_dummiesを使用）
        input_processed = pd.get_dummies(
            input_data, columns=categorical_cols_to_encode, drop_first=True, dtype=int)

        # 2c. モデルが期待する特徴量の形式に変換
        # モデル学習時と同じ特徴量の列名と順序に合わせる必要がある
        # - 不足している列を0で埋める
        # - 余分な列を削除
        # - 列の順序を学習時と一致させる

        # 学習時の特徴量リストで空のDataFrameを作成
        final_input = pd.DataFrame(columns=model_features)

        # 処理済みデータを結合（不足列はNaNになる）
        final_input = pd.concat([final_input, input_processed])

        # 不足列（NaN）を0で埋める
        final_input = final_input.fillna(0)

        # 列の順序を学習時と一致させ、余分な列を削除
        final_input = final_input[model_features]

        # 前処理後のデータをログに出力（デバッグ用）
        print(
            f"前処理後のデータ (形状: {final_input.shape}):\n{final_input.head()}")

    except Exception as e:
        # 前処理中のエラーをログに出力
        print(f"データ前処理中にエラーが発生しました: {e}")
        # HTTPExceptionでエラーレスポンスを返す（ステータスコード400: クライアントエラー）
        raise HTTPException(
            status_code=400, detail=f"入力特徴量の処理中にエラーが発生しました: {e}")

    # ===== 4. モデルで予測 =====
    try:
        # 4a. 基本予測
        # モデルのpredictメソッドを使って予測
        prediction = model.predict(final_input)
        predicted_price = prediction[0]  # 予測結果の最初の要素（単一予測の場合）

        # 4b. 信頼区間の計算
        confidence = 95  # 95%信頼区間（一般的な信頼水準）

        # モデルタイプに応じた信頼区間の計算方法
        if hasattr(model, 'estimators_'):
            # RandomForestなどアンサンブルモデルの場合：
            # 個々の決定木の予測から標準偏差を計算し、信頼区間を求める

            # すべての決定木から個別に予測値を取得
            tree_predictions = np.array(
                [tree.predict(final_input)[0] for tree in model.estimators_])

            # 予測値の標準偏差を計算
            std_dev = np.std(tree_predictions)

            # 95%信頼区間の計算（統計学の正規分布の性質を利用）
            # 1.96は標準正規分布の95%信頼区間に対応する値
            z_score = 1.96
            lower_bound = max(0, predicted_price - z_score *
                              std_dev)  # 下限は0未満にならないよう調整
            upper_bound = predicted_price + z_score * std_dev
        else:
            # その他のモデルの場合：簡易的に予測値の±10%を信頼区間とする
            lower_bound = predicted_price * 0.9
            upper_bound = predicted_price * 1.1

        # 4c. 日本円に変換
        # データセットがラクス単位（インドの通貨単位、1ラクス = 10万ルピー）のため
        # 日本円に変換（1ラクス = 約15万円と仮定）
        jpy_rate = 150000  # 1ラクス = 15万円の変換レート
        predicted_price_jpy = predicted_price * jpy_rate
        lower_bound_jpy = lower_bound * jpy_rate
        upper_bound_jpy = upper_bound * jpy_rate

        # 予測結果をログに出力
        print(
            f"予測成功: {predicted_price_jpy:.0f} 円 ({lower_bound_jpy:.0f} - {upper_bound_jpy:.0f})")
    except Exception as e:
        # 予測中のエラーをログに出力
        print(f"予測実行中にエラーが発生しました: {e}")
        # HTTPExceptionでエラーレスポンスを返す（ステータスコード500: サーバー内部エラー）
        raise HTTPException(
            status_code=500, detail=f"予測中にエラーが発生しました: {e}")

    # ===== 5. 予測結果を返す =====
    # PredictionOutputモデルのインスタンスを作成して返す
    # 値は浮動小数点数に変換（numpy.float64などの特殊な型を避けるため）
    return PredictionOutput(
        predicted_price_jpy=float(predicted_price_jpy),
        lower_bound_jpy=float(lower_bound_jpy),
        upper_bound_jpy=float(upper_bound_jpy),
        confidence_level=confidence
    )

# ======= サーバー起動コマンド =======
# このファイルを直接実行するのではなく、uvicornコマンドでサーバーを起動します
# 以下のコマンドをターミナルで実行してください：
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# コマンドの説明:
# - main:app → main.pyファイル内のappインスタンスを指定
# - --reload → コード変更時に自動的にサーバーを再起動（開発時に便利）
# - --host 0.0.0.0 → すべてのネットワークインターフェースでリッスン（他のマシンからもアクセス可能）
# - --port 8000 → ポート8000でサーバーを起動
#
# サーバー起動後、以下のURLでAPIドキュメントにアクセスできます：
# - Swagger UI (対話的ドキュメント): http://localhost:8000/docs
# - ReDoc (読みやすいドキュメント): http://localhost:8000/redoc
