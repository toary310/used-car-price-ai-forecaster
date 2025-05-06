"""
中古車価格予測APIのメインファイル
このファイルは、FastAPIを使用して中古車の価格を予測するためのWeb APIを提供します。
"""

import joblib  # モデルの保存・読み込みに使用
import pandas as pd  # データ処理に使用
from fastapi import FastAPI, HTTPException  # Web APIフレームワーク
from fastapi.middleware.cors import CORSMiddleware  # クロスオリジンリソース共有の設定
from pydantic import BaseModel, Field  # データバリデーション
from datetime import datetime
import os

# --- 定数 ---
# モデルファイルと特徴量リストの保存場所を指定
MODEL_DIR = 'model'  # モデルを保存するディレクトリ
MODEL_FILE = os.path.join(MODEL_DIR, 'car_price_model.joblib')  # 学習済みモデルのファイル
FEATURES_FILE = os.path.join(
    MODEL_DIR, 'model_features.joblib')  # モデルが使用する特徴量のリスト

# --- モデルと特徴量リストの読み込み ---
try:
    # 学習済みモデルと特徴量リストを読み込む
    model = joblib.load(MODEL_FILE)
    model_features = joblib.load(FEATURES_FILE)
    print("Model and features loaded successfully.")
    print(f"Expected features: {model_features}")
except FileNotFoundError:
    # モデルファイルが見つからない場合のエラー処理
    print(
        f"Error: Model or features file not found in {MODEL_DIR}. Run train_model.py first.")
    model = None
    model_features = None
except Exception as e:
    # その他のエラー処理
    print(f"Error loading model or features: {e}")
    model = None
    model_features = None

# --- FastAPI アプリケーションの初期化 ---
# FastAPIのインスタンスを作成し、APIの基本情報を設定
app = FastAPI(
    title="Used Car Price Prediction API",
    description="API to predict the selling price of used cars.",
    version="0.1.0"
)

# --- CORS 設定 ---
# フロントエンドアプリケーションからのリクエストを許可する設定
# CORS: Cross-Origin Resource Sharing（クロスオリジンリソース共有）
origins = [
    "http://localhost:3000",  # Next.jsの開発サーバー
    # 必要に応じて他のオリジンを追加
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 許可するオリジンのリスト
    allow_credentials=True,  # クレデンシャル（Cookie等）の送信を許可
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのHTTPヘッダーを許可
)

# --- リクエストボディの Pydantic モデル ---
# フロントエンドから送信されるデータの構造を定義
# Fieldクラスを使用して各フィールドの制約を設定


class CarFeaturesInput(BaseModel):
    year: int = Field(..., gt=1979, lt=datetime.now().year +
                      2, description="車両の製造年")
    present_price: float = Field(..., gt=0, description="現在のショールーム価格（ラクス単位）")
    kms_driven: int = Field(..., ge=0, description="走行距離（キロメートル）")
    fuel_type: str = Field(..., description="燃料タイプ（Petrol, Diesel, CNG）")
    seller_type: str = Field(..., description="販売者タイプ（Dealer, Individual）")
    transmission: str = Field(...,
                              description="トランスミッションタイプ（Manual, Automatic）")
    owner: int = Field(..., ge=0, lt=5, description="前オーナー数（0, 1, 3）")

    # APIドキュメント用のサンプルデータ
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

# --- レスポンスボディの Pydantic モデル ---
# APIのレスポンスデータの構造を定義


class PredictionOutput(BaseModel):
    predicted_price_lakhs: float = Field(..., description="予測された販売価格（ラクス単位）")

# --- ルートエンドポイント (動作確認用) ---


@app.get("/")
async def read_root():
    """APIの動作確認用のエンドポイント"""
    return {"message": "Welcome to the Car Price Prediction API!"}

# --- 予測エンドポイント ---


@app.post("/predict", response_model=PredictionOutput)
async def predict_price(features: CarFeaturesInput):
    """
    中古車の価格を予測するエンドポイント

    Parameters:
    - features: 車両の特徴（年式、価格、走行距離など）

    Returns:
    - predicted_price_lakhs: 予測された販売価格（ラクス単位）
    """
    # モデルが読み込まれていない場合のエラー処理
    if model is None or model_features is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. API cannot predict.")

    print(f"Received features for prediction: {features.model_dump()}")

    # 1. 受け取ったデータをDataFrameに変換
    input_data = pd.DataFrame([features.model_dump()])

    # 2. データの前処理
    try:
        # 2a. 車齢の計算
        current_year = datetime.now().year
        input_data['Car_Age'] = current_year - input_data['year']

        # 2b. カテゴリ特徴量のOne-Hot Encoding
        # カテゴリ変数を数値に変換（例：Petrol → [1,0,0], Diesel → [0,1,0]）
        categorical_cols_to_encode = [
            'fuel_type', 'seller_type', 'transmission']
        for col in categorical_cols_to_encode:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype('category')

        input_processed = pd.get_dummies(
            input_data, columns=categorical_cols_to_encode, drop_first=True, dtype=int)

        # 2c. モデルが期待する特徴量の形式に変換
        # - 不足している列を0で埋める
        # - 余分な列を削除
        # - 列の順序を学習時と一致させる
        final_input = pd.DataFrame(columns=model_features)
        final_input = pd.concat([final_input, input_processed])
        final_input = final_input.fillna(0)
        final_input = final_input[model_features]

        print(
            f"Processed features for prediction (shape: {final_input.shape}):\n{final_input.head()}")

    except Exception as e:
        print(f"Error during preprocessing input data: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error processing input features: {e}")

    # 3. モデルで予測を実行
    try:
        prediction = model.predict(final_input)
        predicted_price = prediction[0]  # 予測結果は配列なので最初の要素を取得
        print(f"Prediction successful: {predicted_price}")
    except Exception as e:
        print(f"Error during model prediction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error making prediction: {e}")

    # 予測結果を返す（ラクス単位）
    return PredictionOutput(predicted_price_lakhs=predicted_price)

# --- サーバー起動コマンド ---
# 以下のコマンドでサーバーを起動できます：
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
